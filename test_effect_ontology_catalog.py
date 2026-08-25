import re
import unittest

import effect_extractor as extractor
import effect_ontology as ontology


class EffectOntologyCatalogTests(unittest.TestCase):
    def setUp(self):
        self.effects = {
            effect
            for domain_effects in extractor.CONTROLLED_EFFECT_ONTOLOGY.values()
            for effect in domain_effects
        }

    def test_every_canonical_concept_has_an_elaborate_definition(self):
        self.assertEqual(506, len(self.effects))
        self.assertEqual(self.effects, set(ontology.EFFECT_DEFINITIONS))
        self.assertEqual(self.effects, set(extractor.EFFECT_DEFINITIONS))
        self.assertIs(
            ontology.CONTROLLED_EFFECT_ONTOLOGY,
            extractor.CONTROLLED_EFFECT_ONTOLOGY,
        )
        self.assertIs(ontology.EFFECT_DEFINITIONS, extractor.EFFECT_DEFINITIONS)
        self.assertTrue(
            all(
                len(re.findall(r"\b[\w’-]+\b", definition)) >= 12
                for definition in ontology.EFFECT_DEFINITIONS.values()
            )
        )

    def test_definition_api_uses_exact_canonical_labels(self):
        definition = ontology.get_effect_definition("heautoscopy")
        self.assertIn("bodily double", definition)
        self.assertIn("self-location", definition)
        with self.assertRaises(KeyError):
            ontology.get_effect_definition("doppelganger")

    def test_typed_concept_iteration_preserves_canonical_order(self):
        concepts = list(ontology.iter_effect_concepts())
        expected_names = [
            effect
            for effects in extractor.CONTROLLED_EFFECT_ONTOLOGY.values()
            for effect in effects
        ]
        self.assertEqual(expected_names, [concept.name for concept in concepts])
        self.assertEqual(21, sum(concept.is_rollup for concept in concepts))
        self.assertEqual(
            ontology.EFFECT_DEFINITIONS["visual distortions"],
            concepts[0].definition,
        )

    def test_missing_definition_fails_runtime_validation(self):
        definitions = list(ontology.EFFECT_DEFINITIONS.items())
        ontology.EFFECT_DEFINITIONS.pop("warmth")
        try:
            with self.assertRaisesRegex(ValueError, "lack definitions"):
                extractor.validate_effect_ontology()
        finally:
            ontology.EFFECT_DEFINITIONS.clear()
            ontology.EFFECT_DEFINITIONS.update(definitions)
        extractor.validate_effect_ontology()


if __name__ == "__main__":
    unittest.main()
