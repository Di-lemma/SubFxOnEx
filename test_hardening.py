import os
import unittest
from unittest.mock import patch

import effect_extractor as extractor


def raw_tag(
    effect,
    evidence,
    *,
    confidence=0.9,
    detail=None,
    attribution_type="unknown",
    dose_refs=None,
):
    return {
        "effect": effect,
        "detail": detail,
        "text_detail": evidence,
        "confidence": confidence,
        "attribution": {
            "attribution_type": attribution_type,
            "dose_refs": dose_refs or [],
            "attribution_note": None,
        },
    }


class ChunkingHardeningTests(unittest.TestCase):
    def test_early_paragraph_break_cannot_cause_one_character_progress(self):
        text = "a" * 100 + "\n\n" + "b" * 5000
        chunks = extractor.split_text_into_chunks(text, chunk_size=4000, overlap=600)

        self.assertLessEqual(len(chunks), 3)
        self.assertEqual(0, chunks[0].start)
        self.assertEqual(len(text), chunks[-1].end)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertGreater(current.start, previous.start)
            self.assertLessEqual(current.start, previous.end)
        for chunk in chunks:
            self.assertEqual(text[chunk.start : chunk.end], chunk.text)

    def test_pathological_corpus_shape_is_bounded(self):
        text = "header\n\n" + "x" * 12000
        chunks = extractor.split_text_into_chunks(text, chunk_size=4000, overlap=600)
        self.assertLessEqual(len(chunks), 4)

    def test_extreme_valid_overlap_still_covers_source_with_progress(self):
        text = "abcdefghij"
        chunks = extractor.split_text_into_chunks(text, chunk_size=4, overlap=3)
        self.assertEqual(0, chunks[0].start)
        self.assertEqual(len(text), chunks[-1].end)
        self.assertTrue(
            all(current.start > previous.start for previous, current in zip(chunks, chunks[1:]))
        )
        self.assertTrue(all(chunk.text == text[chunk.start : chunk.end] for chunk in chunks))


class EvidenceAndAttributionTests(unittest.TestCase):
    def test_grounded_evidence_is_replaced_with_exact_source_slice(self):
        report = "Before. I felt Warm, and calm. After."
        result = extractor.sanitize_extraction_payload(
            {"tags": [raw_tag("warmth", "i felt warm and calm")]},
            [],
            report_text=report,
        )

        self.assertEqual(1, len(result["tags"]))
        self.assertEqual("I felt Warm, and calm", result["tags"][0]["text_detail"])
        self.assertEqual(report.index("I felt"), result["tags"][0]["evidence_start"])

    def test_grounding_requires_word_boundaries(self):
        result = extractor.sanitize_extraction_payload(
            {"tags": [raw_tag("warmth", "warm")]},
            [],
            report_text="Swarming insects appeared.",
        )
        self.assertEqual([], result["tags"])

        accepted = extractor.sanitize_extraction_payload(
            {"tags": [raw_tag("warmth", "warm")]},
            [],
            report_text="I felt warm.",
        )
        self.assertEqual("warm", accepted["tags"][0]["text_detail"])

    def test_repeated_evidence_uses_successive_source_occurrences(self):
        report = "I felt joy. Later, I felt joy."
        result = extractor.sanitize_extraction_payload(
            {
                "tags": [
                    raw_tag("joy", "I felt joy", detail="early"),
                    raw_tag("joy", "I felt joy", detail="later"),
                ]
            },
            [],
            report_text=report,
        )
        self.assertEqual(2, len(result["tags"]))
        self.assertNotEqual(
            result["tags"][0]["evidence_start"],
            result["tags"][1]["evidence_start"],
        )
        for tag in result["tags"]:
            self.assertEqual(
                tag["text_detail"],
                report[tag["evidence_start"] : tag["evidence_end"]],
            )

    def test_paraphrased_evidence_is_rejected(self):
        result = extractor.sanitize_extraction_payload(
            {"tags": [raw_tag("joy", "I was delighted")]},
            [],
            report_text="I was in a noticeably positive mood.",
        )

        self.assertEqual([], result["tags"])
        self.assertIn("exact contiguous source excerpt", result["notes"])

    def test_source_dose_metadata_overrides_model_copy(self):
        dose_table = [
            {
                "dose_id": "d1",
                "substance": "Canonical substance",
                "dose": "10 mg",
                "route": "oral",
            }
        ]
        model_ref = {
            "dose_id": "d1",
            "substance": "Fabricated substance",
            "dose": "999 mg",
            "route": "invented",
        }
        report = "My fingers tingled."
        result = extractor.sanitize_extraction_payload(
            {
                "tags": [
                    raw_tag(
                        "tingling",
                        "My fingers tingled",
                        attribution_type="single_substance",
                        dose_refs=[model_ref],
                    )
                ]
            },
            dose_table,
            report_text=report,
        )

        reference = result["tags"][0]["attribution"]["dose_refs"][0]
        self.assertEqual("Canonical substance", reference["substance"])
        self.assertEqual("10 mg", reference["dose"])
        self.assertEqual("oral", reference["route"])

    def test_flat_dose_ids_are_derived_into_legacy_attribution_shape(self):
        dose_table = [
            {"dose_id": "d1", "substance": "A", "dose": "10 mg", "route": "oral"}
        ]
        raw = raw_tag("joy", "I felt joy")
        raw.pop("attribution")
        raw["dose_ids"] = ["d1"]
        raw["attribution_note"] = "linked to the dose"
        result = extractor.sanitize_extraction_payload(
            {"tags": [raw]}, dose_table, report_text="I felt joy."
        )
        attribution = result["tags"][0]["attribution"]
        self.assertEqual("single_substance", attribution["attribution_type"])
        self.assertEqual("d1", attribution["dose_refs"][0]["dose_id"])
        self.assertEqual("linked to the dose", attribution["attribution_note"])

    def test_duplicate_source_dose_ids_fail_attribution_closed(self):
        dose_table = [
            {"dose_id": "d1", "substance": "A"},
            {"dose_id": "d1", "substance": "B"},
        ]
        raw = raw_tag("joy", "joy")
        raw["dose_ids"] = ["d1"]
        result = extractor.sanitize_extraction_payload(
            {"tags": [raw]}, dose_table, report_text="joy"
        )
        self.assertEqual([], result["tags"][0]["attribution"]["dose_refs"])
        self.assertEqual(
            "unknown", result["tags"][0]["attribution"]["attribution_type"]
        )

    def test_any_invalid_dose_id_fails_attribution_closed(self):
        dose_table = [
            {"dose_id": "d1", "substance": "A", "dose": None, "route": None}
        ]
        result = extractor.sanitize_extraction_payload(
            {
                "tags": [
                    raw_tag(
                        "tingling",
                        "tingling",
                        attribution_type="combination",
                        dose_refs=[{"dose_id": "d1"}, {"dose_id": "missing"}],
                    )
                ]
            },
            dose_table,
            report_text="tingling",
        )

        attribution = result["tags"][0]["attribution"]
        self.assertEqual("unknown", attribution["attribution_type"])
        self.assertEqual([], attribution["dose_refs"])

    def test_empty_dose_table_rejects_fabricated_references(self):
        result = extractor.sanitize_extraction_payload(
            {
                "tags": [
                    raw_tag(
                        "joy",
                        "joy",
                        attribution_type="single_substance",
                        dose_refs=[{"dose_id": "d1", "substance": "invented"}],
                    )
                ]
            },
            [],
            report_text="joy",
        )
        attribution = result["tags"][0]["attribution"]
        self.assertEqual("unknown", attribution["attribution_type"])
        self.assertEqual([], attribution["dose_refs"])

    def test_compact_input_keeps_existing_persisted_tag_shape(self):
        result = extractor.ExtractionResult.model_validate(
            extractor.sanitize_extraction_payload(
                {"tags": [raw_tag("joy", "I felt joy")]},
                [],
                report_text="I felt joy.",
            )
        )
        self.assertEqual(
            {
                "domain",
                "effect",
                "subjective_effect",
                "parent_effect",
                "detail",
                "attribution",
                "text_detail",
                "confidence",
            },
            set(result.tags[0].model_dump()),
        )

    def test_obvious_polarity_errors_are_rejected(self):
        cases = [
            ("craving", "I don't have the desire to take any more"),
            ("difficulty falling asleep", "Falling asleep was easier"),
            ("time dilation", "the come up took around 3 hours"),
        ]
        for effect, evidence in cases:
            with self.subTest(effect=effect):
                result = extractor.sanitize_extraction_payload(
                    {"tags": [raw_tag(effect, evidence)]},
                    [],
                    report_text=evidence,
                )
                self.assertEqual([], result["tags"])
                self.assertIn("semantic guards", result["notes"])

    def test_valid_non_whitelisted_semantic_phrasing_survives(self):
        cases = [
            ("difficulty falling asleep", "I lay awake until dawn"),
            ("time dilation", "every second seemed endless"),
            ("time contraction", "the night disappeared in an instant"),
            ("craving", "I did not expect to want more, but later desperately craved another dose"),
        ]
        for effect, evidence in cases:
            with self.subTest(effect=effect):
                result = extractor.sanitize_extraction_payload(
                    {"tags": [raw_tag(effect, evidence)]},
                    [],
                    report_text=evidence,
                )
                self.assertEqual(1, len(result["tags"]))

    def test_confidence_filter_is_opt_in(self):
        payload = {"tags": [raw_tag("joy", "joy", confidence=0.3)]}
        self.assertEqual(
            1,
            len(
                extractor.sanitize_extraction_payload(
                    payload, [], report_text="joy"
                )["tags"]
            ),
        )
        with patch.dict(os.environ, {"MIN_TAG_CONFIDENCE": "0.5"}, clear=False):
            self.assertEqual(
                [],
                extractor.sanitize_extraction_payload(
                    payload, [], report_text="joy"
                )["tags"],
            )


class MergeAndRetryTests(unittest.TestCase):
    def build_tag(self, *, detail, start, end, dose_id=None, substance="A"):
        attribution = {"attribution_type": "unknown", "dose_refs": []}
        if dose_id is not None:
            attribution = {
                "attribution_type": "single_substance",
                "dose_refs": [
                    {
                        "dose_id": dose_id,
                        "substance": substance,
                        "dose": None,
                        "route": None,
                    }
                ],
            }
        return extractor.SubjectiveEffectTag(
            domain="emotional",
            effect="joy",
            subjective_effect="emotional change",
            parent_effect="emotional change",
            detail=detail,
            attribution=attribution,
            text_detail="I felt joy",
            confidence=0.9,
            evidence_start=start,
            evidence_end=end,
        )

    def test_overlapping_same_effect_spans_merge_despite_different_detail(self):
        result = extractor.merge_extraction_results(
            [
                extractor.ExtractionResult(
                    tags=[self.build_tag(detail="onset", start=100, end=120)]
                ),
                extractor.ExtractionResult(
                    tags=[self.build_tag(detail="come-up", start=102, end=120)]
                ),
            ]
        )
        self.assertEqual(1, len(result.tags))

    def test_identical_wording_at_disjoint_spans_is_preserved(self):
        result = extractor.merge_extraction_results(
            [
                extractor.ExtractionResult(
                    tags=[self.build_tag(detail="first", start=100, end=120)]
                ),
                extractor.ExtractionResult(
                    tags=[self.build_tag(detail="later", start=500, end=520)]
                ),
            ]
        )
        self.assertEqual(2, len(result.tags))

    def test_same_evidence_with_disjoint_explicit_attribution_is_preserved(self):
        result = extractor.merge_extraction_results(
            [
                extractor.ExtractionResult(
                    tags=[
                        self.build_tag(
                            detail="first substance",
                            start=10,
                            end=20,
                            dose_id="d1",
                            substance="A",
                        )
                    ]
                ),
                extractor.ExtractionResult(
                    tags=[
                        self.build_tag(
                            detail="second substance",
                            start=10,
                            end=20,
                            dose_id="d2",
                            substance="B",
                        )
                    ]
                ),
            ]
        )
        self.assertEqual(2, len(result.tags))

    def test_unknown_duplicate_merges_into_explicit_attribution(self):
        result = extractor.merge_extraction_results(
            [
                extractor.ExtractionResult(
                    tags=[self.build_tag(detail="unknown", start=10, end=20)]
                ),
                extractor.ExtractionResult(
                    tags=[
                        self.build_tag(
                            detail="known",
                            start=10,
                            end=20,
                            dose_id="d1",
                        )
                    ]
                ),
            ]
        )
        self.assertEqual(1, len(result.tags))
        self.assertEqual("d1", result.tags[0].attribution.dose_refs[0].dose_id)

    def test_transient_api_error_retries_then_succeeds(self):
        class RateLimitError(Exception):
            status_code = 429

        class Completions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RateLimitError("rate limited")
                return {"ok": True}

        completions = Completions()
        client = type("Client", (), {})()
        client.chat = type("Chat", (), {})()
        client.chat.completions = completions

        with patch.dict(
            os.environ,
            {
                "API_MAX_RETRIES": "2",
                "API_RETRY_BASE_SECONDS": "0",
                "API_RETRY_MAX_SECONDS": "0",
            },
            clear=False,
        ), patch.object(extractor.time, "sleep"):
            self.assertEqual({"ok": True}, extractor.call_zai_with_retry(client))
        self.assertEqual(2, completions.calls)

    def test_content_filter_is_terminal_and_unknown_errors_are_not_retried(self):
        self.assertEqual(
            ("content_filter", False),
            extractor.classify_extraction_error(
                RuntimeError("content filter rejected the report")
            ),
        )

        class Completions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                raise RuntimeError("local programming failure")

        completions = Completions()
        client = type("Client", (), {})()
        client.chat = type("Chat", (), {})()
        client.chat.completions = completions
        with self.assertRaises(RuntimeError):
            extractor.call_zai_with_retry(client)
        self.assertEqual(1, completions.calls)

    def test_retry_after_is_not_jittered_below_provider_delay(self):
        class RateLimitError(Exception):
            status_code = 429

            def __init__(self):
                super().__init__("rate limited")
                self.response = type(
                    "Response", (), {"headers": {"Retry-After": "7"}}
                )()

        class Completions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RateLimitError()
                return {"ok": True}

        completions = Completions()
        client = type("Client", (), {})()
        client.chat = type("Chat", (), {})()
        client.chat.completions = completions
        heartbeat_calls = []
        with patch.dict(
            os.environ,
            {
                "API_MAX_RETRIES": "1",
                "API_RETRY_BASE_SECONDS": "1",
                "API_RETRY_MAX_SECONDS": "10",
            },
            clear=False,
        ), patch.object(extractor.time, "sleep") as sleep_mock, patch.object(
            extractor.random, "uniform", return_value=0.5
        ):
            extractor.call_zai_with_retry(
                client, lease_heartbeat=lambda: heartbeat_calls.append(True)
            )
        sleep_mock.assert_called_once_with(7.0)
        self.assertEqual(3, len(heartbeat_calls))


class EnvironmentValidationTests(unittest.TestCase):
    def test_boolean_environment_values_are_explicit_and_fail_closed(self):
        truthy = ("1", "true", "YES", "On")
        falsy = ("0", "false", "NO", "off")
        for value in truthy:
            with self.subTest(value=value), patch.dict(
                os.environ, {"TEST_BOOLEAN": value}, clear=False
            ):
                self.assertTrue(extractor.env_bool("TEST_BOOLEAN"))
        for value in falsy:
            with self.subTest(value=value), patch.dict(
                os.environ, {"TEST_BOOLEAN": value}, clear=False
            ):
                self.assertFalse(extractor.env_bool("TEST_BOOLEAN", True))
        for value in ("", "treu", "2", "enabled"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"TEST_BOOLEAN": value}, clear=False
            ):
                with self.assertRaisesRegex(ValueError, "explicit boolean"):
                    extractor.env_bool("TEST_BOOLEAN")

    def test_float_environment_values_must_be_finite(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"TEST_FLOAT": value}, clear=False
            ):
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    extractor.env_float("TEST_FLOAT", 0.0)

    def test_live_run_rejects_identical_source_and_target_before_mongo(self):
        with patch.dict(
            os.environ,
            {
                "ZAI_API_KEY": "test-key",
                "DRY_RUN": "false",
                "MONGO_SOURCE_COLLECTION": "same-collection",
                "MONGO_TARGET_COLLECTION": "same-collection",
            },
            clear=True,
        ), patch.object(extractor, "MongoClient") as mongo_client:
            with self.assertRaisesRegex(
                ValueError,
                "MONGO_SOURCE_COLLECTION and MONGO_TARGET_COLLECTION must differ",
            ):
                extractor.run_extraction()

        mongo_client.assert_not_called()


class FingerprintTests(unittest.TestCase):
    def test_source_hash_is_stable_and_sensitive_to_effective_input(self):
        first = {
            "exp_id": 1,
            "title": "ignored",
            "report_text": "same report",
            "dose_table": [],
        }
        second = dict(first)
        second["title"] = "also ignored"
        self.assertEqual(
            extractor.build_source_hash(first), extractor.build_source_hash(second)
        )
        second["report_text"] = "changed report"
        self.assertNotEqual(
            extractor.build_source_hash(first), extractor.build_source_hash(second)
        )

    def test_run_fingerprint_is_stable(self):
        self.assertEqual(
            extractor.build_run_fingerprint("test-model"),
            extractor.build_run_fingerprint("test-model"),
        )


class StalePolicyTests(unittest.TestCase):
    def setUp(self):
        self.doc = {"exp_id": 1, "report_text": "report", "dose_table": []}
        self.fingerprint = extractor.build_run_fingerprint("test-model")
        self.source_hash = extractor.build_source_hash(self.doc)

    def reason(self, metadata, policy, reprocess=False):
        return extractor.target_document_eligibility_reason(
            self.doc,
            {"subjective_effect_extraction": metadata},
            self.fingerprint,
            stale_policy=policy,
            reprocess_unversioned=reprocess,
        )

    def test_each_policy_handles_partial_version_metadata_independently(self):
        self.assertEqual(
            "skip_current",
            self.reason(
                {"status": "complete", "source_hash": self.source_hash},
                "source",
            ),
        )
        self.assertEqual(
            "skip_current",
            self.reason(
                {
                    "status": "complete",
                    "pipeline_fingerprint": self.fingerprint["pipeline_fingerprint"],
                },
                "pipeline",
            ),
        )
        self.assertEqual(
            "skip_unversioned",
            self.reason(
                {"status": "complete", "source_hash": self.source_hash},
                "any",
            ),
        )
        self.assertEqual(
            "eligible_unversioned",
            self.reason(
                {"status": "complete", "source_hash": self.source_hash},
                "any",
                reprocess=True,
            ),
        )

    def test_cooldown_and_terminal_state_are_scoped_to_current_inputs(self):
        future = extractor.datetime.now(extractor.timezone.utc) + extractor.timedelta(hours=1)
        stale_error = {
            "status": "error",
            "next_retry_at": future,
            "last_error": {
                "terminal": True,
                "source_hash": "old",
                "pipeline_fingerprint": "old",
            },
        }
        self.assertEqual("eligible_incomplete", self.reason(stale_error, "none"))

        matching_error = {
            "status": "error",
            "next_retry_at": future,
            "last_error": {
                "terminal": True,
                "source_hash": self.source_hash,
                "pipeline_fingerprint": self.fingerprint["pipeline_fingerprint"],
            },
        }
        self.assertEqual("skip_terminal", self.reason(matching_error, "none"))


class RuntimeLifecycleTests(unittest.TestCase):
    def test_sigterm_uses_interrupted_exit_path(self):
        def terminate():
            os.kill(os.getpid(), extractor.signal.SIGTERM)

        with patch.object(extractor, "run_extraction", side_effect=terminate):
            self.assertEqual(130, extractor.main())


if __name__ == "__main__":
    unittest.main()
