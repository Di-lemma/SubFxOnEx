import copy
import json
import tempfile
import unittest
from pathlib import Path

import effect_extractor as extractor
import export_ontology as ontology_export


class OntologyDefinitionExportTests(unittest.TestCase):
    def test_schema_v2_exports_every_definition(self):
        release = ontology_export.build_release()
        self.assertEqual(2, release["schema_version"])
        self.assertEqual(
            extractor.EFFECT_DEFINITIONS,
            {
                concept["name"]: concept["definition"]
                for concept in release["concepts"]
            },
        )
        ontology_export.validate_release(release)

    def test_schema_v1_release_remains_valid_stable_id_history(self):
        with tempfile.TemporaryDirectory() as directory:
            release_directory = Path(directory)
            legacy_release = copy.deepcopy(ontology_export.build_release())
            legacy_release["schema_version"] = 1
            for concept in legacy_release["concepts"]:
                concept.pop("definition")
            legacy_body = {
                key: legacy_release[key]
                for key in ontology_export.RELEASE_BODY_KEYS
            }
            legacy_release["release_hash"] = ontology_export.stable_hash(legacy_body)
            legacy_path = ontology_export.release_path(
                release_directory,
                legacy_release["release_hash"],
            )
            legacy_path.write_text(
                json.dumps(
                    legacy_release,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
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

    def test_definition_tampering_fails_validation(self):
        invalid = ontology_export.build_release()
        invalid["concepts"][0]["definition"] = "Too short."
        body = {key: invalid[key] for key in ontology_export.RELEASE_BODY_KEYS}
        invalid["release_hash"] = ontology_export.stable_hash(body)
        with self.assertRaisesRegex(ValueError, "invalid definition"):
            ontology_export.validate_release(
                invalid,
                require_current_alignment=False,
            )

        drifted = ontology_export.build_release()
        drifted["concepts"][0]["definition"] += " Material semantic drift."
        body = {key: drifted[key] for key in ontology_export.RELEASE_BODY_KEYS}
        drifted["release_hash"] = ontology_export.stable_hash(body)
        with self.assertRaisesRegex(ValueError, "definitions do not match"):
            ontology_export.validate_release(drifted)


if __name__ == "__main__":
    unittest.main()
