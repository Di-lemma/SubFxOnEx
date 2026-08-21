import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import effect_extractor as extractor
import export_ontology as ontology_export


class OntologyExportTests(unittest.TestCase):
    def test_release_is_complete_and_self_validating(self):
        release = ontology_export.build_release()
        ontology_export.validate_release(release)

        expected_concepts = sum(
            len(effects)
            for effects in extractor.CONTROLLED_EFFECT_ONTOLOGY.values()
        )
        self.assertEqual(expected_concepts, release["counts"]["concepts"])
        self.assertEqual(485, release["counts"]["atomic_concepts"])
        self.assertEqual(21, release["counts"]["rollup_concepts"])
        self.assertEqual(extractor.ONTOLOGY_HASH, release["ontology_hash"])
        self.assertRegex(release["release_hash"], r"^[0-9a-f]{64}$")

        concept_ids = {concept["id"] for concept in release["concepts"]}
        self.assertEqual(expected_concepts, len(concept_ids))
        for concept_id in concept_ids:
            UUID(concept_id)

    def test_release_is_deterministic(self):
        first = ontology_export.build_release()
        second = ontology_export.build_release()
        self.assertEqual(
            ontology_export.canonical_json(first),
            ontology_export.canonical_json(second),
        )

    def test_prior_ids_and_redirected_renames_are_preserved(self):
        canonical_name = next(iter(ontology_export.build_release()["concepts"]))["name"]
        prior_id = "00000000-0000-5000-8000-000000000001"
        allocated = ontology_export.allocate_concept_ids(
            [canonical_name], {ontology_export.normalize_label(canonical_name): prior_id}
        )
        self.assertEqual(prior_id, allocated[canonical_name])

        retired, target = next(
            iter(extractor.SAFE_DEPRECATED_EFFECT_REDIRECTS.items())
        )
        allocated = ontology_export.allocate_concept_ids(
            [target], {ontology_export.normalize_label(retired): prior_id}
        )
        self.assertEqual(prior_id, allocated[target])

        unsafe_retired, unsafe_target = next(
            iter(extractor.UNSAFE_DEPRECATED_EFFECT_REDIRECTS.items())
        )
        allocated = ontology_export.allocate_concept_ids(
            [unsafe_target],
            {ontology_export.normalize_label(unsafe_retired): prior_id},
        )
        self.assertNotEqual(prior_id, allocated[unsafe_target])

        with self.assertRaisesRegex(ValueError, "canonical UUIDv5"):
            ontology_export.allocate_concept_ids(
                [canonical_name],
                {
                    ontology_export.normalize_label(canonical_name):
                    "00000000-0000-4000-8000-000000000001"
                },
            )

    def test_unsafe_and_ambiguous_names_are_not_automatic_aliases(self):
        release = ontology_export.build_release()
        automatic_aliases = {
            alias["normalized_label"] for alias in release["aliases"]
        }
        manual_redirects = {
            redirect["normalized_label"]
            for redirect in release["redirects"]
            if redirect["resolution"] == "manual_review"
        }
        self.assertTrue(
            set(extractor.UNSAFE_DEPRECATED_EFFECT_REDIRECTS) <= manual_redirects
        )
        self.assertTrue(
            set(release["ambiguous_labels"]).isdisjoint(automatic_aliases)
        )

    def test_immutable_release_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            release_directory = Path(directory)
            release = ontology_export.build_release()
            path = ontology_export.write_release(release, release_directory)
            self.assertEqual(release, json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(path, ontology_export.write_release(release, release_directory))
            self.assertEqual(
                {
                    ontology_export.normalize_label(concept["name"]): concept["id"]
                    for concept in release["concepts"]
                },
                ontology_export.read_previous_concept_ids(release_directory),
            )

    def test_prior_release_hash_filename_and_uniqueness_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            release_directory = Path(directory)
            release = ontology_export.build_release()
            path = ontology_export.write_release(release, release_directory)

            tampered = copy.deepcopy(release)
            tampered["counts"]["concepts"] += 1
            path.write_text(
                json.dumps(tampered, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Invalid prior ontology release"):
                ontology_export.read_previous_concept_ids(release_directory)

            path.write_text(
                json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            wrong_name = release_directory / "subjective-effects-wrong.json"
            path.rename(wrong_name)
            with self.assertRaisesRegex(ValueError, "filename does not match"):
                ontology_export.read_previous_concept_ids(release_directory)

    def test_competing_release_for_one_ontology_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            release_directory = Path(directory)
            release = ontology_export.build_release()
            ontology_export.write_release(release, release_directory)

            competing = copy.deepcopy(release)
            competing["aliases"][0]["detail"] = "competing release"
            body = {
                key: competing[key]
                for key in ontology_export.RELEASE_BODY_KEYS
            }
            competing["release_hash"] = ontology_export.stable_hash(body)
            competing_path = ontology_export.release_path(
                release_directory,
                competing["release_hash"],
            )
            competing_path.write_text(
                json.dumps(competing, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Competing schema-v1 releases"):
                ontology_export.read_previous_concept_ids(release_directory)

    def test_prior_release_schema_and_ontology_tampering_fail_closed(self):
        mutations = (
            ("schema_version", 99, "Unsupported prior ontology release schema"),
            ("ontology", "different-ontology", "Unexpected ontology name"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                release_directory = Path(directory)
                release = ontology_export.build_release()
                release[field] = value
                body = {
                    key: release[key]
                    for key in ontology_export.RELEASE_BODY_KEYS
                }
                release["release_hash"] = ontology_export.stable_hash(body)
                path = ontology_export.release_path(
                    release_directory,
                    release["release_hash"],
                )
                path.write_text(
                    json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, message):
                    ontology_export.read_previous_concept_ids(release_directory)

    def test_structural_validator_enforces_rollups_and_name_collisions(self):
        release = ontology_export.build_release()

        extra_rollup = copy.deepcopy(release)
        atomic = next(
            concept
            for concept in extra_rollup["concepts"]
            if concept["kind"] == "atomic"
        )
        atomic["kind"] = "rollup"
        atomic["parent_id"] = None
        atomic["parent_name"] = None
        extra_rollup["counts"]["atomic_concepts"] -= 1
        extra_rollup["counts"]["rollup_concepts"] += 1
        body = {
            key: extra_rollup[key]
            for key in ontology_export.RELEASE_BODY_KEYS
        }
        extra_rollup["release_hash"] = ontology_export.stable_hash(body)
        with self.assertRaisesRegex(ValueError, "exactly one rollup"):
            ontology_export.validate_release(
                extra_rollup,
                require_current_alignment=False,
            )

        canonical_collision = copy.deepcopy(release)
        alias = canonical_collision["aliases"][0]
        other_concept = next(
            concept
            for concept in canonical_collision["concepts"]
            if concept["id"] != alias["effect_id"]
            and concept["normalized_name"]
            not in {
                record["normalized_label"]
                for record in canonical_collision["aliases"][1:]
            }
        )
        alias["label"] = other_concept["name"]
        alias["normalized_label"] = other_concept["normalized_name"]
        canonical_collision["aliases"].sort(key=lambda record: record["label"])
        body = {
            key: canonical_collision[key]
            for key in ontology_export.RELEASE_BODY_KEYS
        }
        canonical_collision["release_hash"] = ontology_export.stable_hash(body)
        with self.assertRaisesRegex(ValueError, "different canonical concept"):
            ontology_export.validate_release(
                canonical_collision,
                require_current_alignment=False,
            )

        ambiguous_collision = copy.deepcopy(release)
        canonical_label = next(
            concept["normalized_name"]
            for concept in ambiguous_collision["concepts"]
            if concept["normalized_name"]
            not in ambiguous_collision["ambiguous_labels"]
        )
        ambiguous_collision["ambiguous_labels"].append(canonical_label)
        ambiguous_collision["ambiguous_labels"].sort()
        ambiguous_collision["counts"]["ambiguous_labels"] += 1
        body = {
            key: ambiguous_collision[key]
            for key in ontology_export.RELEASE_BODY_KEYS
        }
        ambiguous_collision["release_hash"] = ontology_export.stable_hash(body)
        with self.assertRaisesRegex(ValueError, "collide with canonical"):
            ontology_export.validate_release(
                ambiguous_collision,
                require_current_alignment=False,
            )

    def test_validator_rejects_rehashed_internal_drift(self):
        release = ontology_export.build_release()
        mutations = {
            "stale normalized name": lambda value: value["concepts"][0].__setitem__(
                "normalized_name", "stale"
            ),
            "incorrect count": lambda value: value["counts"].__setitem__(
                "aliases", value["counts"]["aliases"] + 1
            ),
            "mismatched target name": lambda value: value["aliases"][0].__setitem__(
                "effect_name", value["concepts"][0]["name"]
            ),
            "mismatched parent name": lambda value: next(
                concept for concept in value["concepts"] if concept["kind"] == "atomic"
            ).__setitem__("parent_name", "wrong parent"),
            "wrong redirect safety": lambda value: value["redirects"][0].__setitem__(
                "resolution",
                "manual_review"
                if value["redirects"][0]["resolution"] == "automatic"
                else "automatic",
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(release)
                mutate(changed)
                body = {
                    key: changed[key]
                    for key in ontology_export.RELEASE_BODY_KEYS
                }
                changed["release_hash"] = ontology_export.stable_hash(body)
                with self.assertRaises(ValueError):
                    ontology_export.validate_release(changed)

    def test_main_respects_an_explicit_empty_argument_list(self):
        with patch.object(sys, "argv", ["export_ontology.py", "--invalid"]), patch(
            "sys.stdout", new=io.StringIO()
        ):
            self.assertEqual(0, ontology_export.main([]))


if __name__ == "__main__":
    unittest.main()
