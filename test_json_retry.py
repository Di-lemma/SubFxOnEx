import json
import os
import unittest
from unittest.mock import patch

import effect_extractor as extractor


class FakeCompletions:
    def __init__(self, contents):
        self._contents = iter(contents)
        self.payloads = []
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        user_content = kwargs["messages"][1]["content"]
        payload_json = user_content.split("Document:\n", 1)[1]
        self.payloads.append(json.loads(payload_json))
        return {
            "choices": [
                {
                    "message": {
                        "content": next(self._contents),
                    }
                }
            ]
        }


class FakeClient:
    def __init__(self, contents):
        self.chat = type("FakeChat", (), {})()
        self.chat.completions = FakeCompletions(contents)


class InvalidJsonRetryTests(unittest.TestCase):
    def test_glm_53_sends_enabled_thinking_and_low_reasoning_effort(self):
        client = FakeClient(['{"tags":[],"notes":null}'])
        payload = {"exp_id": 1, "dose_table": [], "report_text": "test"}

        with patch.dict(
            os.environ,
            {"ZAI_THINKING": "enabled", "ZAI_REASONING_EFFORT": "low"},
            clear=False,
        ):
            extractor.extract_effects_for_payload(client, "glm-5.3", payload)

        request = client.chat.completions.requests[0]
        self.assertEqual({"type": "enabled"}, request["thinking"])
        self.assertEqual(
            {"reasoning_effort": "low"}, request["extra_body"]
        )

    def test_response_requires_an_object_with_a_tags_array(self):
        malformed_contents = (
            "[]",
            "{}",
            '{"tags":"not-an-array"}',
        )
        for content in malformed_contents:
            with self.subTest(content=content):
                response = {
                    "choices": [{"message": {"content": content}}]
                }
                with self.assertRaises(extractor.InvalidModelResponseError):
                    extractor.extract_response_json(response)

        valid = extractor.extract_response_json(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"tags":[],"notes":null}'
                        }
                    }
                ]
            }
        )
        self.assertEqual([], valid["tags"])
        self.assertEqual(
            ("response_validation", True),
            extractor.classify_extraction_error(
                extractor.InvalidModelResponseError("missing tags array")
            ),
        )

    def test_truncated_json_retries_a_4000_character_payload_in_three_chunks(self):
        truncated_response = (
            '{"tags":[{"effect":"joy","attribution":'
            '{"attribution_type":"combination","dose_refs":['
            '{"dose_id":"d1","substance":"'
        )
        valid_response = '{"tags":[],"notes":null}'
        client = FakeClient(
            [truncated_response, valid_response, valid_response, valid_response]
        )
        payload = {
            "exp_id": 106074,
            "dose_table": [],
            "report_text": "x" * 4000,
        }

        with patch.dict(
            os.environ,
            {
                "MIN_RETRY_CHUNK_SIZE_CHARS": "1200",
                "REPORT_CHUNK_OVERLAP_CHARS": "600",
            },
            clear=False,
        ):
            result = extractor.extract_effects_for_payload_with_json_retry(
                client,
                "test-model",
                payload,
            )

        attempted_payloads = client.chat.completions.payloads
        self.assertEqual(4, len(attempted_payloads))
        self.assertEqual(4000, len(attempted_payloads[0]["report_text"]))
        self.assertEqual(
            [2000, 2000, 1000],
            [len(item["report_text"]) for item in attempted_payloads[1:]],
        )
        self.assertEqual([], result.tags)
        self.assertIn("Retried in 3 smaller chunks", result.notes)

    def test_invalid_json_at_retry_floor_is_not_retried_forever(self):
        truncated_response = '{"tags":[{"text_detail":"unterminated'
        client = FakeClient([truncated_response])
        payload = {
            "exp_id": 106074,
            "dose_table": [],
            "report_text": "x" * 1200,
        }

        with patch.dict(
            os.environ,
            {"MIN_RETRY_CHUNK_SIZE_CHARS": "1200"},
            clear=False,
        ):
            with self.assertRaises(extractor.InvalidModelJSONError):
                extractor.extract_effects_for_payload_with_json_retry(
                    client,
                    "test-model",
                    payload,
                )

        self.assertEqual(1, len(client.chat.completions.payloads))

    def test_nested_retry_preserves_global_evidence_offsets(self):
        evidence = "I felt unmistakable joy"
        report = (
            "x" * 898
            + ". "
            + evidence
            + " "
            + "y" * (4000 - 901 - len(evidence))
        )
        invalid = '{"tags":[{"effect":"joy","text_detail":"unterminated'
        empty = '{"tags":[],"notes":null}'
        tagged = json.dumps(
            {
                "tags": [
                    {
                        "effect": "joy",
                        "detail": "nested retry",
                        "dose_ids": [],
                        "attribution_note": None,
                        "text_detail": evidence,
                        "confidence": 0.9,
                    }
                ],
                "notes": None,
            }
        )
        client = FakeClient(
            [
                invalid,
                invalid,
                empty,
                tagged,
                empty,
                empty,
                empty,
            ]
        )
        payload = {"exp_id": 77, "dose_table": [], "report_text": report}

        with patch.dict(
            os.environ,
            {
                "MIN_RETRY_CHUNK_SIZE_CHARS": "1000",
                "REPORT_CHUNK_OVERLAP_CHARS": "600",
                "API_MAX_RETRIES": "0",
            },
            clear=False,
        ):
            result = extractor.extract_effects_for_payload_with_json_retry(
                client, "test-model", payload
            )

        self.assertEqual(1, len(result.tags))
        tag = result.tags[0]
        self.assertEqual(900, tag.evidence_start)
        self.assertEqual(evidence, report[tag.evidence_start : tag.evidence_end])
        self.assertEqual(evidence, tag.text_detail)


if __name__ == "__main__":
    unittest.main()
