import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from effect_ontology.release import (
    CURRENT_MANIFEST_FILENAME,
    OntologyResolver,
    load_pinned_release,
)


ROOT = Path(__file__).resolve().parent
RELEASE_DIRECTORY = ROOT / "ontology_releases"
MANIFEST_PATH = RELEASE_DIRECTORY / CURRENT_MANIFEST_FILENAME


class OntologyConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.release = load_pinned_release(MANIFEST_PATH)

    def test_pinned_release_exposes_complete_v3_contract(self):
        release = self.release
        self.assertEqual(3, release["schema_version"])
        self.assertEqual(
            {
                "concepts": 506,
                "atomic_concepts": 485,
                "rollup_concepts": 21,
                "aliases": 1178,
                "redirects": 70,
                "ambiguous_labels": 108,
            },
            release["counts"],
        )
        self.assertEqual(506, len({item["id"] for item in release["concepts"]}))
        self.assertEqual(506, len({item["definition"] for item in release["concepts"]}))
        self.assertTrue(all(item["definition"] for item in release["concepts"]))
        self.assertEqual(
            {"defined"}, {item["review_status"] for item in release["concepts"]}
        )
        self.assertEqual(
            40,
            sum(item["resolution"] == "automatic" for item in release["redirects"]),
        )
        self.assertEqual(
            30,
            sum(item["resolution"] == "manual_review" for item in release["redirects"]),
        )

    def test_ids_are_stable_from_schema_v2_to_v3(self):
        v2_path = RELEASE_DIRECTORY / (
            "subjective-effects-"
            "8adcb2f4ea4ac6bf4ae50dcc114c03a4f897a33fa99638b120ca048c8d78c013.json"
        )
        v2 = json.loads(v2_path.read_text(encoding="utf-8"))
        self.assertEqual(
            {item["name"]: item["id"] for item in v2["concepts"]},
            {item["name"]: item["id"] for item in self.release["concepts"]},
        )

    def test_resolver_keeps_automatic_and_manual_redirects_distinct(self):
        resolver = OntologyResolver(self.release)
        automatic = next(
            item for item in self.release["redirects"]
            if item["resolution"] == "automatic"
        )
        resolved = resolver.resolve_label(automatic["label"])
        self.assertEqual("automatic_redirect", resolved.mode)
        self.assertEqual(automatic["effect_id"], resolved.concept_id)
        self.assertIsNone(resolved.candidate_concept_id)

        manual = next(
            item for item in self.release["redirects"]
            if item["resolution"] == "manual_review"
        )
        unresolved = resolver.resolve_label(manual["label"])
        self.assertEqual("manual_review", unresolved.mode)
        self.assertIsNone(unresolved.concept_id)
        self.assertIsNone(unresolved.concept_name)
        self.assertEqual(manual["candidate_effect_id"], unresolved.candidate_concept_id)
        self.assertEqual(
            manual["candidate_effect_name"], unresolved.candidate_concept_name
        )
        self.assertIsNone(manual["effect_id"])
        self.assertIsNone(manual["effect_name"])

    def test_ambiguous_and_unknown_labels_remain_unresolved(self):
        resolver = OntologyResolver(self.release)
        redirect_labels = {
            item["normalized_label"] for item in self.release["redirects"]
        }
        ambiguous = next(
            label
            for label in self.release["ambiguous_labels"]
            if label not in redirect_labels
        )
        ambiguous_result = resolver.resolve_label(ambiguous)
        self.assertEqual("ambiguous", ambiguous_result.mode)
        self.assertIsNone(ambiguous_result.concept_id)
        unknown_result = resolver.resolve_label("not a subfxonex concept")
        self.assertEqual("unknown", unknown_result.mode)
        self.assertIsNone(unknown_result.concept_id)

    def test_manifest_pins_exact_artifact_and_all_release_hashes(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            f"subjective-effects-{self.release['release_hash']}.json",
            manifest["artifact"],
        )
        for field in (
            "ontology",
            "schema_version",
            "normalization_hash",
            "semantic_hash",
            "release_hash",
        ):
            self.assertEqual(self.release[field], manifest[field])

    def test_manifest_artifact_hash_tampering_fails_closed(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        artifact = RELEASE_DIRECTORY / manifest["artifact"]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / artifact.name).write_bytes(artifact.read_bytes())
            manifest["artifact_sha256"] = "0" * 64
            (target / CURRENT_MANIFEST_FILENAME).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_pinned_release(target / CURRENT_MANIFEST_FILENAME)

    def test_artifact_loads_without_extractor_or_runtime_dependencies(self):
        command = (
            "from effect_ontology import load_pinned_release; "
            "r=load_pinned_release(); "
            "assert r['schema_version']==3; "
            "print(r['release_hash'])"
        )
        completed = subprocess.run(
            [sys.executable, "-S", "-c", command],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual(self.release["release_hash"], completed.stdout.strip())


if __name__ == "__main__":
    unittest.main()
