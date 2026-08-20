import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

import effect_extractor as extractor


@unittest.skipUnless(
    os.getenv("RUN_MONGO_INTEGRATION") == "1",
    "set RUN_MONGO_INTEGRATION=1 to use a disposable MongoDB collection",
)
class QueueIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = MongoClient(
            os.getenv("MONGO_URI", "mongodb://host.docker.internal:27017"),
            serverSelectionTimeoutMS=5000,
            tz_aware=True,
        )
        cls.client.admin.command("ping")
        cls.db = cls.client[os.getenv("MONGO_DB", "tripindex")]
        cls.collection_name = f"_erowid_extractor_test_{uuid.uuid4().hex}"
        cls.collection = cls.db[cls.collection_name]
        extractor.ensure_target_indexes(cls.collection)
        cls.fingerprint = extractor.build_run_fingerprint("integration-test-model")

    @classmethod
    def tearDownClass(cls):
        if cls.collection_name.startswith("_erowid_extractor_test_"):
            cls.db.drop_collection(cls.collection_name)
        cls.client.close()

    def setUp(self):
        self.collection.delete_many({})

    def result(self):
        return extractor.ExtractionResult.model_validate(
            {
                "tags": [
                    {
                        "domain": "emotional",
                        "effect": "joy",
                        "subjective_effect": "emotional change",
                        "parent_effect": "emotional change",
                        "detail": None,
                        "attribution": {
                            "attribution_type": "unknown",
                            "dose_refs": [],
                            "attribution_note": None,
                        },
                        "text_detail": "I felt joy",
                        "confidence": 0.9,
                    }
                ],
                "notes": None,
            }
        )

    def test_unique_exp_id_index_rejects_duplicate(self):
        self.collection.insert_one({"exp_id": "duplicate"})
        with self.assertRaises(DuplicateKeyError):
            self.collection.insert_one({"exp_id": "duplicate"})

    def test_claim_rechecks_newly_completed_document(self):
        doc = {"_id": "source-1", "exp_id": "current", "report_text": "same", "dose_table": []}
        self.collection.insert_one(
            {
                "exp_id": doc["exp_id"],
                "subjective_effect_extraction": {
                    "status": "complete",
                    "source_hash": extractor.build_source_hash(doc),
                    "pipeline_fingerprint": self.fingerprint["pipeline_fingerprint"],
                    "state_revision": 4,
                },
            }
        )
        with patch.dict(os.environ, {"STALE_POLICY": "any"}, clear=False):
            self.assertIsNone(
                extractor.claim_document(
                    self.collection,
                    doc,
                    "source",
                    self.fingerprint,
                )
            )

    def test_finalization_revision_blocks_stale_claim(self):
        doc = {"_id": "source-2", "exp_id": "revision", "report_text": "I felt joy", "dose_table": []}
        expired = datetime.now(timezone.utc) - timedelta(seconds=1)
        self.collection.insert_one(
            {
                "exp_id": doc["exp_id"],
                "subjective_effect_extraction": {
                    "status": "error",
                    "lease": {"token": "worker-a", "expires_at": expired},
                    "state_revision": 1,
                    "attempt_count": 1,
                },
            }
        )
        extractor.persist_result(
            self.collection,
            doc,
            self.result(),
            "integration-test-model",
            "source",
            run_fingerprint=self.fingerprint,
            lease_token="worker-a",
        )
        finalized = self.collection.find_one({"exp_id": doc["exp_id"]})
        self.assertEqual(2, finalized["subjective_effect_extraction"]["state_revision"])
        stale_update = self.collection.update_one(
            {
                "_id": finalized["_id"],
                "subjective_effect_extraction.state_revision": 1,
                "subjective_effect_extraction.lease": {"$exists": False},
            },
            {"$set": {"subjective_effect_extraction.lease.token": "worker-b"}},
        )
        self.assertEqual(0, stale_update.matched_count)

    def test_failed_stale_refresh_preserves_successful_result(self):
        old_tags = [{"sentinel": "old-result"}]
        doc = {
            "_id": "new-source-id",
            "exp_id": "refresh",
            "title": "new title",
            "substance": "new substance",
            "report_text": "new report",
            "dose_table": [],
        }
        self.collection.insert_one(
            {
                "exp_id": doc["exp_id"],
                "source_doc_id": "old-source-id",
                "title": "old title",
                "substance": "old substance",
                "subjective_effect_tags": old_tags,
                "subjective_effect_extraction": {
                    "status": "complete",
                    "source_hash": "old-source-hash",
                    "pipeline_fingerprint": "old-pipeline",
                    "state_revision": 10,
                    "attempt_count": 2,
                },
            }
        )
        with patch.dict(os.environ, {"STALE_POLICY": "source"}, clear=False):
            token = extractor.claim_document(
                self.collection, doc, "source", self.fingerprint
            )
        self.assertIsNotNone(token)
        claimed = self.collection.find_one({"exp_id": doc["exp_id"]})
        self.assertEqual("old title", claimed["title"])
        state = extractor.mark_error(
            self.collection,
            doc,
            "integration-test-model",
            RuntimeError("content filter rejected the report"),
            "source",
            run_fingerprint=self.fingerprint,
            lease_token=token,
        )
        self.assertTrue(state["terminal"])
        failed = self.collection.find_one({"exp_id": doc["exp_id"]})
        self.assertEqual(old_tags, failed["subjective_effect_tags"])
        self.assertEqual("complete", failed["subjective_effect_extraction"]["status"])
        self.assertEqual("old-source-hash", failed["subjective_effect_extraction"]["source_hash"])
        self.assertEqual("old-pipeline", failed["subjective_effect_extraction"]["pipeline_fingerprint"])
        self.assertEqual("old title", failed["title"])
        self.assertEqual("old substance", failed["substance"])
        self.assertEqual("old-source-id", failed["source_doc_id"])

    def test_rate_limit_never_becomes_terminal_from_lifetime_attempts(self):
        class RateLimitError(Exception):
            status_code = 429

        doc = {"_id": "source-4", "exp_id": "rate", "report_text": "text", "dose_table": []}
        self.collection.insert_one(
            {
                "exp_id": doc["exp_id"],
                "subjective_effect_extraction": {
                    "lease": {"token": "rate-token", "expires_at": datetime.now(timezone.utc) + timedelta(hours=1)},
                    "state_revision": 1,
                    "attempt_count": 100,
                },
            }
        )
        state = extractor.mark_error(
            self.collection,
            doc,
            "integration-test-model",
            RateLimitError("rate limited"),
            "source",
            run_fingerprint=self.fingerprint,
            lease_token="rate-token",
        )
        self.assertFalse(state["terminal"])
        self.assertEqual(100, state["attempt_count"])
        self.assertEqual(1, state["consecutive_error_count"])

    def test_malformed_lease_is_reclaimable_and_renewal_is_fenced(self):
        doc = {"_id": "source-5", "exp_id": "malformed", "report_text": "text", "dose_table": []}
        self.collection.insert_one(
            {
                "exp_id": doc["exp_id"],
                "subjective_effect_extraction": {
                    "status": "error",
                    "lease": {"token": "broken", "expires_at": "not-a-date"},
                    "state_revision": 1,
                    "attempt_count": 1,
                },
            }
        )
        token = extractor.claim_document(
            self.collection, doc, "source", self.fingerprint
        )
        self.assertIsNotNone(token)
        extractor.renew_claim(self.collection, doc["exp_id"], token)
        with self.assertRaises(extractor.LeaseLostError):
            extractor.renew_claim(self.collection, doc["exp_id"], "wrong-token")


if __name__ == "__main__":
    unittest.main()
