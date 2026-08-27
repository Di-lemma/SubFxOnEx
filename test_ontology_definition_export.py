import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import effect_extractor as extractor
import export_ontology as ontology_export


def rehash_current_release(release):
    release["normalization_hash"] = ontology_export.compute_normalization_hash(release)
    release["semantic_hash"] = ontology_export.compute_semantic_hash(release)
    release["release_hash"] = ontology_export.compute_release_hash(release)


class OntologyDefinitionExportTests(unittest.TestCase):
    def test_schema_v3_exports_every_definition_with_truthful_status(self):
        release = ontology_export.build_release()
        self.assertEqual(3, release["schema_version"])
        self.assertEqual(
            extractor.EFFECT_DEFINITIONS,
            {
                concept["name"]: concept["definition"]
                for concept in release["concepts"]
            },
        )
        self.assertEqual(
            {"defined"},
            {concept["review_status"] for concept in release["concepts"]},
        )
        ontology_export.validate_release(release)

    def test_existing_v1_and_v2_releases_remain_valid(self):
        release_directory = Path(__file__).resolve().parent / "ontology_releases"
        for path in sorted(release_directory.glob("subjective-effects-*.json")):
            with self.subTest(path=path.name):
                release = json.loads(path.read_text(encoding="utf-8"))
                if release["schema_version"] in {1, 2}:
                    ontology_export.validate_release(
                        release, require_current_alignment=False
                    )

    def test_schema_v1_release_remains_stable_id_history(self):
        with tempfile.TemporaryDirectory() as directory:
            release_directory = Path(directory)
            source_path = Path(__file__).resolve().parent / "ontology_releases" / (
                "subjective-effects-"
                "4f0fd4edc4a82621e643e4ffec16c716c3cc44b7b2fe97dcbfcd93ff86806904.json"
            )
            legacy_release = json.loads(source_path.read_text(encoding="utf-8"))
            (release_directory / source_path.name).write_text(
                source_path.read_text(encoding="utf-8"), encoding="utf-8"
            )

            ontology_export.validate_release(
                legacy_release,
                require_current_alignment=False,
            )
            previous_ids = ontology_export.read_previous_concept_ids(
                release_directory
            )
            current_release = ontology_export.build_release(previous_ids)
            self.assertEqual(
                {
                    concept["name"]: concept["id"]
                    for concept in legacy_release["concepts"]
                },
                {
                    concept["name"]: concept["id"]
                    for concept in current_release["concepts"]
                },
            )

    def test_definition_only_edit_changes_semantic_and_release_hashes_only(self):
        before = ontology_export.build_release()
        with tempfile.TemporaryDirectory() as directory:
            release_directory = Path(directory)
            ontology_export.write_release(before, release_directory)
            previous_ids = ontology_export.read_previous_concept_ids(release_directory)
            original = extractor.EFFECT_DEFINITIONS["visual distortions"]
            with patch.dict(
                extractor.EFFECT_DEFINITIONS,
                {
                    "visual distortions": original
                    + " This definition-only test changes meaning."
                },
            ):
                after = ontology_export.build_release(previous_ids)
                ontology_export.validate_release(after)
                ontology_export.write_release(after, release_directory)
            self.assertEqual(
                2,
                len(list(release_directory.glob("subjective-effects-*.json"))),
            )
            self.assertEqual(
                previous_ids,
                ontology_export.read_previous_concept_ids(release_directory),
            )

        self.assertEqual(before["normalization_hash"], after["normalization_hash"])
        self.assertNotEqual(before["semantic_hash"], after["semantic_hash"])
        self.assertNotEqual(before["release_hash"], after["release_hash"])
        self.assertEqual(
            {concept["name"]: concept["id"] for concept in before["concepts"]},
            {concept["name"]: concept["id"] for concept in after["concepts"]},
        )

    def test_definition_tampering_fails_validation(self):
        invalid = ontology_export.build_release()
        invalid["concepts"][0]["definition"] = ""
        rehash_current_release(invalid)
        with self.assertRaisesRegex(ValueError, "invalid definition"):
            ontology_export.validate_release(
                invalid,
                require_current_alignment=False,
            )

        drifted = ontology_export.build_release()
        drifted["concepts"][0]["definition"] += " Material semantic drift."
        rehash_current_release(drifted)
        with self.assertRaisesRegex(ValueError, "definitions do not match"):
            ontology_export.validate_release(drifted)


if __name__ == "__main__":
    unittest.main()
