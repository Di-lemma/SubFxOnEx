import copy
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import migrate_existing_ontology as migration


class FakeOntology:
    EFFECT_INDEX = {
        "joy": {
            "domain": "emotional",
            "effect": "joy",
            "parent_effect": "emotional change",
        },
        "visual imagery": {
            "domain": "visual",
            "effect": "visual imagery",
            "parent_effect": "visual distortions",
        },
        "visual acuity enhancement": {
            "domain": "visual",
            "effect": "visual acuity enhancement",
            "parent_effect": "visual distortions",
        },
        "spiritual experience": {
            "domain": "spiritual",
            "effect": "spiritual experience",
            "parent_effect": "spiritual experience",
        },
    }
    SAFE_DEPRECATED_EFFECT_REDIRECTS = {
        "closed-eye visuals": "visual imagery",
        "entity imagery": "visual imagery",
    }
    UNSAFE_DEPRECATED_EFFECT_REDIRECTS = {
        "mystical quality": "spiritual experience",
        "visual clarity": "visual acuity enhancement",
        "open-eye visuals": "visual imagery",
    }
    DEPRECATED_EFFECT_REDIRECTS = {
        **SAFE_DEPRECATED_EFFECT_REDIRECTS,
        **UNSAFE_DEPRECATED_EFFECT_REDIRECTS,
    }
    DEPRECATED_EFFECT_DETAILS = {
        "closed-eye visuals": "eyes closed",
        "entity imagery": "entity",
    }
    EFFECT_ALIASES = {"self esteem boost": "self-esteem elevation"}

    @staticmethod
    def normalize_raw_effect_label(value):
        if not isinstance(value, str):
            return None
        normalized = " ".join(value.strip().lower().replace("_", " ").split())
        return normalized or None

    @staticmethod
    def sanitize_extraction_payload(*args, **kwargs):
        raise AssertionError("migration must not call the extraction sanitizer")


ONTOLOGY = FakeOntology()


def transform_tag(tag):
    return migration.transform_tag(tag, ontology=ONTOLOGY)


def transform_document(document):
    return migration.transform_document(document, ontology=ONTOLOGY)


def verify_document_pair(before, after):
    return migration.verify_document_pair(before, after, ontology=ONTOLOGY)


def verify_document_sequences(before, after):
    return migration.verify_document_sequences(before, after, ontology=ONTOLOGY)


def make_tag(effect="joy", **overrides):
    tag = {
        "domain": "legacy-domain",
        "effect": effect,
        "subjective_effect": "legacy-parent",
        "parent_effect": "legacy-parent",
        "detail": None,
        "attribution": {
            "attribution_type": "unknown",
            "dose_refs": [],
            "attribution_note": None,
        },
        "text_detail": "I felt joyful",
        "confidence": 0.87,
    }
    tag.update(overrides)
    return tag


def make_document(tags, status="complete"):
    return {
        "_id": "source-object-id",
        "exp_id": 123,
        "title": "Example",
        "subjective_effect_tags": tags,
        "subjective_effect_extraction": {
            "status": status,
            "model_name": "historical-model",
            "tag_count": len(tags),
        },
    }


def make_snapshot(collection, collection_uuid, content_sha256):
    return migration.CollectionSnapshot(
        collection=collection,
        collection_uuid=collection_uuid,
        document_count=1,
        content_sha256=content_sha256,
        options_sha256="o" * 64,
        indexes_sha256="i" * 64,
        status_counts={"complete": 1},
        model_counts={"model": 1},
        tag_count=1,
        options={},
        indexes=[],
    )


def observed_snapshot(snapshot):
    return {
        "collection": snapshot.collection,
        "exists": True,
        "snapshot": snapshot.to_dict(),
    }


class CliSafetyTests(unittest.TestCase):
    def test_blank_mode_selectors_never_dispatch_forward_apply(self):
        for flag in ("--rollback-backup", "--repair-from-backup"):
            with self.subTest(flag=flag), patch.object(
                migration,
                "require_database_dependencies",
            ) as require_database, patch.object(
                migration,
                "MongoClient",
            ) as mongo_client, patch.object(
                migration,
                "run_apply",
            ) as run_apply, patch(
                "sys.stderr",
                new=io.StringIO(),
            ):
                with self.assertRaises(SystemExit) as raised:
                    migration.main(["--apply", flag, "   "])

                self.assertEqual(2, raised.exception.code)
                require_database.assert_not_called()
                mongo_client.assert_not_called()
                run_apply.assert_not_called()

    def test_nonfinite_quiescence_never_dispatches_to_database(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value), patch.object(
                migration,
                "require_database_dependencies",
            ) as require_database, patch.object(
                migration,
                "MongoClient",
            ) as mongo_client, patch(
                "sys.stderr",
                new=io.StringIO(),
            ):
                with self.assertRaises(SystemExit) as raised:
                    migration.main(
                        [f"--quiescence-seconds={value}"]
                    )

                self.assertEqual(2, raised.exception.code)
                require_database.assert_not_called()
                mongo_client.assert_not_called()

    def test_cutover_failure_status_and_conflict_exit_are_preserved(self):
        expected_statuses = {
            "committed": "failed_after_cutover_committed",
            "not_committed": "failed_cutover_not_committed",
            "indeterminate": "failed_cutover_indeterminate",
        }
        for outcome, expected in expected_statuses.items():
            with self.subTest(outcome=outcome):
                self.assertEqual(
                    expected,
                    migration.failure_status_for_manifest(
                        {
                            "cutover_attempted": True,
                            "cutover_outcome": outcome,
                        }
                    ),
                )

        self.assertEqual(
            3,
            migration.completion_exit_code(
                {"status": "repair_dry_run_conflicts"}
            ),
        )
        self.assertEqual(
            0,
            migration.completion_exit_code(
                {"status": "repair_dry_run_complete"}
            ),
        )

    def test_main_keeps_reconciliation_when_apply_raises(self):
        client = MagicMock()
        database = MagicMock()
        target = MagicMock()
        client.__getitem__.return_value = database
        database.__getitem__.return_value = target

        def failed_apply(*args, manifest, **kwargs):
            manifest.update(
                {
                    "cutover_attempted": True,
                    "cutover_outcome": "not_committed",
                    "writes_performed": True,
                    "cutover": {
                        "reconciliation": {
                            "outcome": "not_committed",
                            "observations": {"target": {"exists": True}},
                        }
                    },
                }
            )
            raise RuntimeError("post-attempt failure")

        with patch.object(
            migration,
            "require_database_dependencies",
        ), patch.object(
            migration,
            "require_extractor",
        ), patch.object(
            migration,
            "manifest_base",
            return_value={"status": "started"},
        ), patch.object(
            migration,
            "MongoClient",
            return_value=client,
        ), patch.object(
            migration,
            "collection_info",
        ), patch.object(
            migration,
            "run_apply",
            side_effect=failed_apply,
        ), patch.object(
            migration,
            "write_manifest",
        ), patch.object(
            migration,
            "emit_manifest",
        ) as emit_manifest:
            exit_code = migration.main(
                [
                    "--apply",
                    "--manifest",
                    "/tmp/test-cutover-manifest.json",
                ]
            )

        self.assertEqual(2, exit_code)
        emitted = emit_manifest.call_args.args[0]
        self.assertEqual("failed_cutover_not_committed", emitted["status"])
        self.assertTrue(emitted["writes_performed"])
        self.assertEqual(
            "not_committed",
            emitted["cutover"]["reconciliation"]["outcome"],
        )


class CutoverSafetyTests(unittest.TestCase):
    def setUp(self):
        shared_content = "a" * 64
        self.target_before = make_snapshot(
            "live",
            "target-before-uuid",
            shared_content,
        )
        self.replacement = make_snapshot(
            "shadow",
            "replacement-uuid",
            shared_content,
        )
        self.retained = make_snapshot(
            "retained",
            "retained-uuid",
            shared_content,
        )

    def observations(self, target, replacement, retained=None):
        return {
            "target": target,
            "replacement": replacement,
            "retained_backup": retained or observed_snapshot(self.retained),
        }

    def classify(self, observations):
        return migration.classify_cutover_outcome(
            observations,
            expected_target_before=self.target_before,
            expected_replacement=self.replacement,
            expected_retained_backup=self.retained,
        )[0]

    def test_rename_command_uses_majority_journal_write_concern(self):
        client = MagicMock()
        migration.atomic_replace_collection(
            client,
            database_name="tripindex",
            replacement_name="shadow",
            target_name="live",
        )
        client.admin.command.assert_called_once_with(
            {
                "renameCollection": "tripindex.shadow",
                "to": "tripindex.live",
                "dropTarget": True,
                "writeConcern": {"w": "majority", "j": True},
            }
        )

    def test_manifest_write_is_private_and_fsyncs_file_and_parent_directory(self):
        original_fsync = migration.os.fsync
        with TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text("old manifest\n", encoding="utf-8")
            manifest_path.chmod(0o644)
            with patch.object(
                migration.os,
                "fsync",
                wraps=original_fsync,
            ) as fsync:
                migration.write_manifest(
                    manifest_path,
                    {"status": "cutover_attempted"},
                )

            self.assertTrue(manifest_path.is_file())
            self.assertEqual(0o600, manifest_path.stat().st_mode & 0o777)
            self.assertEqual(2, fsync.call_count)

    def test_database_write_phase_is_persisted_before_collection_writes(self):
        manifest = {"status": "preflight_complete"}
        persisted = []
        with patch.object(
            migration,
            "write_manifest",
            side_effect=lambda path, value: persisted.append(
                copy.deepcopy(value)
            ),
        ):
            migration.persist_database_write_phase(
                manifest,
                Path("manifest.json"),
                "backup_creation_started",
            )

        self.assertTrue(persisted[0]["writes_performed"])
        self.assertIn("database_writes_started_at", persisted[0])
        self.assertEqual(
            "backup_creation_started",
            persisted[0]["last_database_write_phase"],
        )

    def test_cutover_classification_uses_uuid_and_collection_presence(self):
        absent_replacement = {"collection": "shadow", "exists": False}
        cases = {
            "committed": self.observations(
                observed_snapshot(self.replacement),
                absent_replacement,
            ),
            "not_committed": self.observations(
                observed_snapshot(self.target_before),
                observed_snapshot(self.replacement),
            ),
            "indeterminate_alien_target": self.observations(
                observed_snapshot(
                    make_snapshot("live", "alien-uuid", "a" * 64)
                ),
                absent_replacement,
            ),
            "indeterminate_read_error": self.observations(
                {
                    "collection": "live",
                    "exists": None,
                    "error": "read failed",
                },
                absent_replacement,
            ),
        }
        expected = {
            "committed": "committed",
            "not_committed": "not_committed",
            "indeterminate_alien_target": "indeterminate",
            "indeterminate_read_error": "indeterminate",
        }
        for name, observations in cases.items():
            with self.subTest(name=name):
                self.assertEqual(expected[name], self.classify(observations))

    def test_retained_backup_health_is_independent_of_committed_outcome(self):
        observations = [
            observed_snapshot(self.replacement),
            {"collection": "shadow", "exists": False},
            {"collection": "retained", "exists": False},
        ]
        database = MagicMock()
        with patch.object(
            migration,
            "observe_cutover_collection",
            side_effect=observations,
        ):
            reconciliation = migration.reconcile_cutover(
                database,
                target_name="live",
                replacement_name="shadow",
                retained_backup_name="retained",
                expected_target_before=self.target_before,
                expected_replacement=self.replacement,
                expected_retained_backup=self.retained,
                batch_size=10,
            )

        self.assertEqual("committed", reconciliation["outcome"])
        self.assertEqual("degraded", reconciliation["retained_backup_status"])

    def test_cutover_attempt_is_persisted_before_rename(self):
        events = []
        database = MagicMock()
        database.name = "tripindex"
        manifest = {"status": "shadow_verified"}

        def record_manifest(path, current):
            events.append(("manifest", copy.deepcopy(current)))

        def record_rename(*args, **kwargs):
            events.append(("rename", kwargs))

        def record_reconciliation(*args, **kwargs):
            events.append(("reconcile", kwargs["trigger"]))
            return {"outcome": "committed"}

        with patch.object(
            migration,
            "write_manifest",
            side_effect=record_manifest,
        ), patch.object(
            migration,
            "atomic_replace_collection",
            side_effect=record_rename,
        ), patch.object(
            migration,
            "persist_cutover_reconciliation",
            side_effect=record_reconciliation,
        ):
            result = migration.execute_verified_cutover(
                MagicMock(),
                database,
                target_name="live",
                replacement_name="shadow",
                retained_backup_name="retained",
                expected_target_before=self.target_before,
                expected_replacement=self.replacement,
                expected_retained_backup=self.retained,
                batch_size=10,
                manifest=manifest,
                manifest_path=Path("manifest.json"),
                verify_after=lambda: "verified",
            )

        self.assertEqual("verified", result)
        self.assertEqual("manifest", events[0][0])
        self.assertEqual("cutover_attempted", events[0][1]["status"])
        self.assertTrue(events[0][1]["cutover_attempted"])
        self.assertEqual("rename", events[1][0])
        self.assertEqual(
            self.replacement.stable_identity(),
            events[0][1]["cutover"]["expected"]
            ["replacement_before_and_target_after"],
        )

    def test_failed_attempt_manifest_write_prevents_rename(self):
        database = MagicMock()
        database.name = "tripindex"
        rename = MagicMock()
        with patch.object(
            migration,
            "write_manifest",
            side_effect=OSError("disk full"),
        ), patch.object(
            migration,
            "atomic_replace_collection",
            rename,
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                migration.execute_verified_cutover(
                    MagicMock(),
                    database,
                    target_name="live",
                    replacement_name="shadow",
                    retained_backup_name="retained",
                    expected_target_before=self.target_before,
                    expected_replacement=self.replacement,
                    expected_retained_backup=self.retained,
                    batch_size=10,
                    manifest={},
                    manifest_path=Path("manifest.json"),
                    verify_after=lambda: None,
                )
        rename.assert_not_called()

    def test_command_and_verification_failures_are_reconciled_and_reraised(self):
        cases = (
            (
                "command",
                RuntimeError("rename reply lost"),
                "not_committed",
                "rename_command_exception",
            ),
            (
                "verification",
                ValueError("post-check failed"),
                "committed",
                "post_cutover_verification_failure",
            ),
            (
                "interrupt",
                KeyboardInterrupt("operator interrupted"),
                "indeterminate",
                "rename_command_exception",
            ),
        )
        for phase, failure, outcome, trigger in cases:
            with self.subTest(phase=phase):
                database = MagicMock()
                database.name = "tripindex"
                manifest = {}
                rename = MagicMock()
                verify = MagicMock(return_value=None)
                if phase in {"command", "interrupt"}:
                    rename.side_effect = failure
                else:
                    verify.side_effect = failure
                reconciliation = {
                    "outcome": outcome,
                    "retained_backup_status": "verified",
                    "observations": {},
                }
                with patch.object(
                    migration,
                    "write_manifest",
                ), patch.object(
                    migration,
                    "atomic_replace_collection",
                    rename,
                ), patch.object(
                    migration,
                    "reconcile_cutover",
                    return_value=reconciliation,
                ):
                    with self.assertRaises(type(failure)) as raised:
                        migration.execute_verified_cutover(
                            MagicMock(),
                            database,
                            target_name="live",
                            replacement_name="shadow",
                            retained_backup_name="retained",
                            expected_target_before=self.target_before,
                            expected_replacement=self.replacement,
                            expected_retained_backup=self.retained,
                            batch_size=10,
                            manifest=manifest,
                            manifest_path=Path("manifest.json"),
                            verify_after=verify,
                        )

                self.assertIs(failure, raised.exception)
                self.assertEqual(outcome, manifest["cutover_outcome"])
                self.assertTrue(manifest["writes_performed"])
                self.assertEqual(
                    trigger,
                    manifest["cutover"]["reconciliation"]["trigger"],
                )
                self.assertEqual(
                    type(failure).__name__,
                    manifest["cutover"]["reconciliation"]["failure"]
                    ["error_type"],
                )

    def test_reconciliation_failure_never_masks_rename_exception(self):
        database = MagicMock()
        database.name = "tripindex"
        manifest = {}
        rename_error = RuntimeError("rename reply lost")
        with patch.object(
            migration,
            "write_manifest",
        ), patch.object(
            migration,
            "atomic_replace_collection",
            side_effect=rename_error,
        ), patch.object(
            migration,
            "persist_cutover_reconciliation",
            side_effect=OSError("cannot persist reconciliation"),
        ):
            with self.assertRaises(RuntimeError) as raised:
                migration.execute_verified_cutover(
                    MagicMock(),
                    database,
                    target_name="live",
                    replacement_name="shadow",
                    retained_backup_name="retained",
                    expected_target_before=self.target_before,
                    expected_replacement=self.replacement,
                    expected_retained_backup=self.retained,
                    batch_size=10,
                    manifest=manifest,
                    manifest_path=Path("manifest.json"),
                    verify_after=lambda: None,
                )

        self.assertIs(rename_error, raised.exception)
        self.assertEqual(
            "OSError",
            manifest["cutover_reconciliation_error"]["error_type"],
        )


class CompatibilityDetailTests(unittest.TestCase):
    def test_fill_merge_and_idempotence(self):
        filled, fill_action = migration.merge_compatibility_detail(
            None, "eyes closed"
        )
        self.assertEqual("eyes closed", filled)
        self.assertEqual("filled", fill_action)

        merged, merge_action = migration.merge_compatibility_detail(
            "blue rotating geometry", "eyes closed"
        )
        self.assertEqual("eyes closed; blue rotating geometry", merged)
        self.assertEqual("merged", merge_action)

        repeated, repeated_action = migration.merge_compatibility_detail(
            merged, "eyes closed"
        )
        self.assertEqual(merged, repeated)
        self.assertEqual("already_present", repeated_action)

    def test_multiclause_phrase_detection_is_token_aware(self):
        detail = "auditory inducer with a bright visual concurrent"
        self.assertTrue(
            migration.detail_contains_compatibility(
                detail,
                "auditory inducer; visual concurrent",
            )
        )
        self.assertFalse(
            migration.detail_contains_compatibility(
                "musical appreciation",
                "music",
            )
        )


class PureTagTransformTests(unittest.TestCase):
    def test_redirect_partition_must_be_explicit_disjoint_and_exact(self):
        with patch.object(
            ONTOLOGY,
            "SAFE_DEPRECATED_EFFECT_REDIRECTS",
            {"visual clarity": "visual acuity enhancement"},
        ), patch.object(
            ONTOLOGY,
            "UNSAFE_DEPRECATED_EFFECT_REDIRECTS",
            {"visual clarity": "visual acuity enhancement"},
        ):
            with self.assertRaisesRegex(
                migration.MigrationSafetyError,
                "overlap",
            ):
                transform_tag(make_tag("visual clarity"))

        with patch.object(
            ONTOLOGY,
            "DEPRECATED_EFFECT_REDIRECTS",
            {},
        ):
            with self.assertRaisesRegex(
                migration.MigrationSafetyError,
                "must equal",
            ):
                transform_tag(make_tag("visual clarity"))

    def test_current_canonical_repairs_hierarchy_only(self):
        original = make_tag()
        transformed, result = transform_tag(original)

        expected = ONTOLOGY.EFFECT_INDEX["joy"]
        self.assertEqual("joy", transformed["effect"])
        self.assertEqual(expected["domain"], transformed["domain"])
        self.assertEqual(expected["parent_effect"], transformed["parent_effect"])
        self.assertEqual(
            expected["parent_effect"], transformed["subjective_effect"]
        )
        self.assertEqual(original["text_detail"], transformed["text_detail"])
        self.assertEqual(original["attribution"], transformed["attribution"])
        self.assertEqual(set(original), set(transformed))
        self.assertEqual("changed", result.outcome)

    def test_safe_redirect_merges_context_without_overwriting_detail(self):
        original = make_tag(
            "closed-eye visuals",
            detail="blue rotating geometry",
            text_detail="With my eyes closed I saw blue geometry",
        )
        transformed, result = transform_tag(original)

        self.assertEqual("visual imagery", transformed["effect"])
        self.assertEqual("visual", transformed["domain"])
        self.assertEqual("visual distortions", transformed["parent_effect"])
        self.assertEqual("visual distortions", transformed["subjective_effect"])
        self.assertEqual(
            "eyes closed; blue rotating geometry",
            transformed["detail"],
        )
        self.assertEqual("merged", result.detail_action)
        self.assertEqual("changed", result.outcome)
        self.assertEqual(set(original), set(transformed))

    def test_safe_redirect_is_idempotent(self):
        document = make_document(
            [
                make_tag(
                    "closed-eye visuals",
                    detail="blue rotating geometry",
                )
            ]
        )
        first, _ = transform_document(document)
        second, results = transform_document(first)

        self.assertEqual(first, second)
        self.assertEqual("unchanged", results[0].outcome)
        self.assertIsNone(results[0].detail_action)

    def test_safe_contextual_visual_redirects_remain_distinct(self):
        common = {
            "detail": "color-filled visions",
            "text_detail": "visions of figures with my eyes closed",
        }
        document = make_document(
            [
                make_tag("entity imagery", **common),
                make_tag("closed-eye visuals", **common),
            ]
        )
        transformed, _ = transform_document(document)
        first, second = transformed["subjective_effect_tags"]

        self.assertEqual("visual imagery", first["effect"])
        self.assertEqual("visual imagery", second["effect"])
        self.assertEqual("entity; color-filled visions", first["detail"])
        self.assertEqual("eyes closed; color-filled visions", second["detail"])
        self.assertNotEqual(first, second)

    def test_unsafe_redirect_is_byte_value_identical_and_reported(self):
        original = make_tag(
            "mystical quality",
            domain="intentionally-wrong",
            detail="unity and sacredness",
        )
        transformed, result = transform_tag(original)

        self.assertEqual(original, transformed)
        self.assertEqual("unsafe_redirect", result.outcome)
        self.assertEqual("spiritual experience", result.canonical_effect)

    def test_nonredirect_alias_is_not_used_for_historical_migration(self):
        original = make_tag("self esteem boost")
        self.assertIn(
            "self esteem boost",
            ONTOLOGY.EFFECT_ALIASES,
        )
        self.assertNotIn(
            "self esteem boost",
            ONTOLOGY.DEPRECATED_EFFECT_REDIRECTS,
        )

        transformed, result = transform_tag(original)
        self.assertEqual(original, transformed)
        self.assertEqual("unsupported_effect", result.outcome)

    def test_missing_key_or_invalid_detail_fails_closed(self):
        missing = make_tag("closed-eye visuals")
        missing.pop("detail")
        transformed_missing, missing_result = transform_tag(missing)
        self.assertEqual(missing, transformed_missing)
        self.assertEqual("missing_ontology_keys", missing_result.outcome)

        invalid = make_tag("closed-eye visuals", detail={"not": "a string"})
        transformed_invalid, invalid_result = transform_tag(invalid)
        self.assertEqual(invalid, transformed_invalid)
        self.assertEqual("invalid_detail", invalid_result.outcome)

    def test_transform_never_calls_extraction_sanitizer(self):
        with patch.object(
            ONTOLOGY,
            "sanitize_extraction_payload",
            side_effect=AssertionError("must not be called"),
        ):
            transformed, result = transform_tag(make_tag("closed-eye visuals"))
        self.assertEqual("visual imagery", transformed["effect"])
        self.assertEqual("changed", result.outcome)


class RepairOverlayTests(unittest.TestCase):
    def make_v1_pair(self, **overrides):
        values = {"text_detail": "Everything looked unusually clear"}
        values.update(overrides)
        backup = make_tag("visual clarity", **values)
        current = migration.historical_v1_transform_tag(
            backup,
            ontology=ONTOLOGY,
        )
        return backup, current

    def test_exact_v1_ancestry_restores_unsafe_tag_only(self):
        backup, current = self.make_v1_pair()
        repaired, result = migration.repair_tag_from_backup(
            current,
            backup,
            tag_index=0,
            ontology=ONTOLOGY,
        )

        self.assertEqual("repaired", result.outcome)
        self.assertEqual("visual acuity enhancement", result.historical_effect)
        self.assertEqual("visual clarity", result.desired_effect)
        self.assertEqual(backup, repaired)
        self.assertEqual(current["confidence"], repaired["confidence"])
        self.assertEqual(current["attribution"], repaired["attribution"])
        self.assertEqual(
            {"domain", "effect", "subjective_effect", "parent_effect"},
            set(result.changed_fields),
        )

    def test_repair_is_idempotent(self):
        backup, current = self.make_v1_pair()
        repaired, _ = migration.repair_tag_from_backup(
            current,
            backup,
            tag_index=0,
            ontology=ONTOLOGY,
        )
        repeated, result = migration.repair_tag_from_backup(
            repaired,
            backup,
            tag_index=0,
            ontology=ONTOLOGY,
        )

        self.assertEqual(repaired, repeated)
        self.assertEqual("already_desired", result.outcome)
        self.assertEqual((), result.changed_fields)

    def test_nonontology_drift_is_a_lineage_conflict(self):
        backup, current = self.make_v1_pair()
        current["confidence"] = 0.42
        repaired, result = migration.repair_tag_from_backup(
            current,
            backup,
            tag_index=0,
            ontology=ONTOLOGY,
        )

        self.assertEqual(current, repaired)
        self.assertEqual("lineage_conflict", result.outcome)
        self.assertIn("neither the exact", result.reason)

    def test_document_conflict_blocks_sibling_repairs(self):
        first_backup, first_current = self.make_v1_pair(
            text_detail="first exact evidence"
        )
        second_backup, second_current = self.make_v1_pair(
            text_detail="second exact evidence"
        )
        first_current["confidence"] = 0.42
        backup_document = make_document([first_backup, second_backup])
        current_document = make_document([first_current, second_current])

        repaired, results = migration.repair_document_from_backup(
            current_document,
            backup_document,
            ontology=ONTOLOGY,
        )

        self.assertEqual(current_document, repaired)
        self.assertEqual("lineage_conflict", results[0].outcome)
        self.assertEqual("blocked_by_document_conflict", results[1].outcome)
        self.assertEqual((), results[1].changed_fields)

    def test_current_only_document_is_unchanged(self):
        current = make_document([make_tag("joy")])
        repaired, results = migration.repair_document_from_backup(
            current,
            None,
            ontology=ONTOLOGY,
        )
        self.assertEqual(current, repaired)
        self.assertEqual([], results)
        self.assertEqual(
            [],
            migration.verify_repair_document_pair(
                current,
                repaired,
                None,
                ontology=ONTOLOGY,
            ),
        )

    def test_frozen_v1_hierarchy_fingerprint_rejects_drift(self):
        backup, _ = self.make_v1_pair()
        with patch.dict(
            migration.HISTORICAL_V1_TARGET_HIERARCHY,
            {"visual acuity enhancement": ("visual", "changed-parent")},
        ):
            with self.assertRaisesRegex(
                migration.MigrationSafetyError,
                "migration specification changed",
            ):
                migration.historical_v1_transform_tag(
                    backup,
                    ontology=ONTOLOGY,
                )

    def test_frozen_v1_detail_fingerprint_rejects_drift(self):
        backup, _ = self.make_v1_pair()
        with patch.dict(
            migration.HISTORICAL_V1_DEPRECATED_EFFECT_DETAILS,
            {"open-eye visuals": "changed historical detail"},
        ):
            with self.assertRaisesRegex(
                migration.MigrationSafetyError,
                "migration specification changed",
            ):
                migration.historical_v1_transform_tag(
                    backup,
                    ontology=ONTOLOGY,
                )

    def test_frozen_v1_transform_ignores_current_normalizer_drift(self):
        backup = make_tag("  VISUAL_CLARITY  ")
        with patch.object(
            ONTOLOGY,
            "normalize_raw_effect_label",
            return_value="unrelated-current-normalization",
        ):
            historical = migration.historical_v1_transform_tag(
                backup,
                ontology=ONTOLOGY,
            )

        self.assertEqual("visual acuity enhancement", historical["effect"])
        self.assertEqual("visual", historical["domain"])
        self.assertEqual("visual distortions", historical["parent_effect"])

    def test_expected_backup_hash_is_mandatory_and_exact(self):
        snapshot = migration.CollectionSnapshot(
            collection="backup",
            collection_uuid="uuid",
            document_count=1,
            content_sha256="a" * 64,
            options_sha256="b" * 64,
            indexes_sha256="c" * 64,
            status_counts={"complete": 1},
            model_counts={"model": 1},
            tag_count=1,
            options={},
            indexes=[],
        )
        migration.assert_expected_repair_backup(snapshot, "a" * 64)
        with self.assertRaisesRegex(
            migration.MigrationSafetyError,
            "content hash mismatch",
        ):
            migration.assert_expected_repair_backup(snapshot, "d" * 64)


class DocumentPreservationTests(unittest.TestCase):
    def test_document_preserves_order_counts_keysets_and_nonontology_values(self):
        tags = [
            make_tag("closed-eye visuals", text_detail="first evidence"),
            make_tag("joy", text_detail="second evidence"),
            make_tag("mystical quality", text_detail="third evidence"),
        ]
        document = make_document(tags)
        transformed, _ = transform_document(document)

        transformed_tags = transformed["subjective_effect_tags"]
        self.assertEqual(len(tags), len(transformed_tags))
        self.assertEqual(
            [tag["text_detail"] for tag in tags],
            [tag["text_detail"] for tag in transformed_tags],
        )
        self.assertEqual(
            [set(tag) for tag in tags],
            [set(tag) for tag in transformed_tags],
        )
        self.assertEqual(
            document["subjective_effect_extraction"],
            transformed["subjective_effect_extraction"],
        )
        self.assertEqual([], verify_document_pair(document, transformed))

    def test_noncomplete_document_is_unchanged(self):
        document = make_document([make_tag("closed-eye visuals")], status="error")
        transformed, results = transform_document(document)
        self.assertEqual(document, transformed)
        self.assertEqual([], results)

    def test_metrics_report_unsafe_redirect_without_a_change(self):
        document = make_document([make_tag("mystical quality")])
        transformed, results = transform_document(document)
        metrics = migration.TransformMetrics()
        metrics.observe_document(
            document,
            transformed,
            results,
            max_examples=5,
        )

        self.assertEqual(0, metrics.documents_changed)
        self.assertEqual(0, metrics.tags_changed)
        self.assertEqual(
            1,
            metrics.unsafe_redirects[
                "mystical quality -> spiritual experience"
            ],
        )
        self.assertEqual("unsafe_redirect", metrics.examples[0]["outcome"])

    def test_verifier_rejects_forbidden_field_and_keyset_changes(self):
        document = make_document([make_tag("closed-eye visuals")])
        transformed, _ = transform_document(document)

        forbidden = copy.deepcopy(transformed)
        forbidden["subjective_effect_tags"][0]["confidence"] = 0.01
        forbidden_issues = verify_document_pair(document, forbidden)
        self.assertTrue(
            any("forbidden field changed: confidence" in issue for issue in forbidden_issues)
        )

        added = copy.deepcopy(transformed)
        added["subjective_effect_tags"][0]["new_field"] = True
        added_issues = verify_document_pair(document, added)
        self.assertTrue(any("key set changed" in issue for issue in added_issues))

    def test_sequence_verifier_rejects_source_id_reordering(self):
        first = make_document([make_tag("joy")])
        second = make_document([make_tag("joy")])
        first["_id"] = "a"
        second["_id"] = "b"
        first_after, _ = transform_document(first)
        second_after, _ = transform_document(second)

        issues = verify_document_sequences(
            [first, second],
            [second_after, first_after],
        )
        self.assertTrue(any("source _id/order changed" in issue for issue in issues))

    @unittest.skipIf(migration.bson_dumps is None, "host Python lacks PyMongo/BSON")
    def test_canonical_hash_ignores_mapping_key_order_but_not_array_order(self):
        left = {"a": 1, "b": [1, 2]}
        reordered_keys = {"b": [1, 2], "a": 1}
        reordered_array = {"a": 1, "b": [2, 1]}
        self.assertEqual(
            migration.sha256_value(left),
            migration.sha256_value(reordered_keys),
        )
        self.assertNotEqual(
            migration.sha256_value(left),
            migration.sha256_value(reordered_array),
        )


if __name__ == "__main__":
    unittest.main()
