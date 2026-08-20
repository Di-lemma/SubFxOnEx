import re
import unittest

import effect_extractor as extractor


class OntologyInvariantTests(unittest.TestCase):
    def setUp(self):
        self.effects = {
            effect
            for domain_effects in extractor.CONTROLLED_EFFECT_ONTOLOGY.values()
            for effect in domain_effects
        }

    def test_runtime_validator_accepts_current_ontology(self):
        extractor.validate_effect_ontology()

    def test_high_value_independent_gaps_are_canonical(self):
        expected = {
            "action automaticity",
            "atmospheric portent",
            "attentional absorption",
            "auditory pareidolia",
            "communitas",
            "double bookkeeping",
            "dyspnea",
            "ego inflation",
            "embarrassment",
            "guilt",
            "heautoscopy",
            "hyperfamiliarity",
            "ineffability",
            "inner speech externalization",
            "joy",
            "perceptual meaning loss",
            "pure awareness",
            "spacelessness",
            "temporal simultaneity",
            "thought ownership loss",
            "uncanniness",
        }
        self.assertEqual(set(), expected - self.effects)

    def test_retired_non_atomic_terms_are_redirected_not_canonical(self):
        retired = set(extractor.DEPRECATED_EFFECT_REDIRECTS)
        self.assertTrue(retired.isdisjoint(self.effects))
        self.assertTrue(
            set(extractor.DEPRECATED_EFFECT_REDIRECTS.values()) <= self.effects
        )

    def test_redirect_safety_partition_is_complete(self):
        safe = set(extractor.SAFE_DEPRECATED_EFFECT_REDIRECTS)
        unsafe = set(extractor.UNSAFE_DEPRECATED_EFFECT_REDIRECTS)
        complete = extractor.DEPRECATED_EFFECT_REDIRECTS
        expected = {
            **extractor.SAFE_DEPRECATED_EFFECT_REDIRECTS,
            **extractor.UNSAFE_DEPRECATED_EFFECT_REDIRECTS,
        }

        self.assertEqual(40, len(safe))
        self.assertEqual(30, len(unsafe))
        self.assertTrue(safe.isdisjoint(unsafe))
        self.assertEqual(set(complete), safe | unsafe)
        self.assertEqual(expected, complete)
        self.assertTrue(set(extractor.DEPRECATED_EFFECT_DETAILS) <= safe)

    def test_redirect_value_drift_invalidates_hash_and_validator(self):
        original_hash = extractor.build_ontology_hash()
        original_target = extractor.DEPRECATED_EFFECT_REDIRECTS["visual clarity"]
        extractor.DEPRECATED_EFFECT_REDIRECTS["visual clarity"] = (
            "brightness enhancement"
        )
        try:
            self.assertNotEqual(original_hash, extractor.build_ontology_hash())
            with self.assertRaisesRegex(ValueError, "mapping union"):
                extractor.validate_effect_ontology()
        finally:
            extractor.DEPRECATED_EFFECT_REDIRECTS["visual clarity"] = original_target
        extractor.validate_effect_ontology()
        self.assertEqual(original_hash, extractor.build_ontology_hash())

    def test_safe_redirects_match_runtime_aliases(self):
        for retired, target in extractor.SAFE_DEPRECATED_EFFECT_REDIRECTS.items():
            self.assertEqual(target, extractor.EFFECT_ALIASES[retired])

        original_target = extractor.EFFECT_ALIASES["entity imagery"]
        extractor.EFFECT_ALIASES["entity imagery"] = "brightness enhancement"
        try:
            with self.assertRaisesRegex(ValueError, "runtime aliases"):
                extractor.validate_effect_ontology()
        finally:
            extractor.EFFECT_ALIASES["entity imagery"] = original_target
        extractor.validate_effect_ontology()

    def test_each_domain_has_one_direct_rollup(self):
        for domain, effects in extractor.CONTROLLED_EFFECT_ONTOLOGY.items():
            rollups = {
                effect
                for effect, parent_effect in effects.items()
                if effect == parent_effect
            }
            self.assertEqual(1, len(rollups), domain)
            rollup = next(iter(rollups))
            self.assertEqual({rollup}, set(effects.values()), domain)

    def test_aliases_cannot_shadow_canonical_terms(self):
        collisions = {
            effect: extractor.EFFECT_ALIASES[effect]
            for effect in self.effects & set(extractor.EFFECT_ALIASES)
            if extractor.EFFECT_ALIASES[effect] != effect
        }
        self.assertEqual({}, collisions)
        self.assertEqual("warmth", extractor.normalize_effect_label("warmth"))
        self.assertEqual("guilt", extractor.normalize_effect_label("guilt"))
        self.assertEqual(
            "embarrassment",
            extractor.normalize_effect_label("embarrassment"),
        )
        self.assertEqual(
            "self-esteem elevation",
            extractor.normalize_effect_label("self esteem boost"),
        )
        for alias in (
            "heart pounding",
            "racing heart",
            "heart palpitations",
            "heart skipping",
            "forceful heartbeat",
        ):
            self.assertEqual("palpitations", extractor.normalize_effect_label(alias))
        for alias in (
            "shortness of breath",
            "hard to breathe",
            "air hunger",
            "labored breathing",
        ):
            self.assertEqual("dyspnea", extractor.normalize_effect_label(alias))
        self.assertEqual(
            "cardiac awareness",
            extractor.normalize_effect_label("aware of my heartbeat"),
        )
        self.assertEqual(
            "respiratory awareness",
            extractor.normalize_effect_label("conscious breathing"),
        )
        self.assertEqual(
            "brightness enhancement",
            extractor.normalize_effect_label("brighter colors"),
        )
        self.assertEqual(
            "color saturation enhancement",
            extractor.normalize_effect_label("vivid colours"),
        )
        self.assertEqual(
            "bodily pleasure",
            extractor.normalize_effect_label("touch felt amazing"),
        )
        self.assertEqual(
            "physical discomfort",
            extractor.normalize_effect_label("touch felt awful"),
        )

    def test_ambiguous_shorthand_is_not_coerced(self):
        self.assertTrue(
            set(extractor.AMBIGUOUS_EFFECT_ALIASES).isdisjoint(self.effects)
        )
        for label in extractor.AMBIGUOUS_EFFECT_ALIASES:
            self.assertNotIn(label, extractor.EFFECT_ALIASES)

    def test_unsafe_redirects_fail_closed_at_runtime(self):
        raw_result = {
            "tags": [
                {
                    "effect": label,
                    "detail": None,
                    "text_detail": f"reported {label}",
                    "confidence": 0.9,
                    "attribution": {
                        "attribution_type": "unknown",
                        "dose_refs": [],
                    },
                }
                for label in extractor.UNSAFE_DEPRECATED_EFFECT_REDIRECTS
            ]
        }

        result = extractor.sanitize_extraction_payload(
            raw_result, [], require_evidence_grounding=False
        )
        self.assertEqual([], result["tags"])
        self.assertIn("Rejected 30 unsupported effect tag proposals", result["notes"])
        for label in extractor.UNSAFE_DEPRECATED_EFFECT_REDIRECTS:
            self.assertNotIn(label, extractor.EFFECT_ALIASES)
            self.assertEqual(label, extractor.normalize_effect_label(label))
        for label in extractor.UNSAFE_EFFECT_ALIAS_LABELS:
            self.assertNotIn(label, extractor.EFFECT_ALIASES)
            self.assertEqual(label, extractor.normalize_effect_label(label))

    def test_ontology_hash_includes_final_alias_semantics(self):
        original_hash = extractor.build_ontology_hash()
        extractor.EFFECT_ALIASES["ontology hash test alias"] = "warmth"
        try:
            self.assertNotEqual(original_hash, extractor.build_ontology_hash())
        finally:
            extractor.EFFECT_ALIASES.pop("ontology hash test alias")
        self.assertEqual(original_hash, extractor.build_ontology_hash())

    def test_retired_context_is_preserved_as_detail(self):
        cases = [
            (
                "entity imagery",
                None,
                "visual imagery",
                "entity",
            ),
            (
                "status salience",
                None,
                "salience enhancement",
                "social status",
            ),
            (
                "auditory-visual synesthesia",
                None,
                "synesthesia",
                "auditory inducer; visual concurrent",
            ),
            (
                "pleasant touch amplification",
                "model supplied",
                "tactile amplification",
                "pleasant touch; model supplied",
            ),
            (
                "lattice patterns",
                None,
                "geometric imagery",
                "lattice",
            ),
            (
                "heavy limbs",
                None,
                "somatic heaviness",
                "limbs",
            ),
            (
                "shadow imagery",
                None,
                "visual imagery",
                "shadow figure",
            ),
            (
                "touch felt amazing",
                None,
                "bodily pleasure",
                "touch",
            ),
            (
                "touch felt awful",
                None,
                "physical discomfort",
                "touch",
            ),
            (
                "surface flowing",
                None,
                "visual liquefaction",
                "surface",
            ),
            (
                "low fps vision",
                None,
                "visual motion discontinuity",
                "frame-rate suppression",
            ),
            (
                "phantom ringing",
                None,
                "auditory hallucination",
                "ringing",
            ),
            (
                "phasing sound",
                None,
                "timbre distortion",
                "phasing",
            ),
            (
                "pleasure waves",
                None,
                "bodily pleasure",
                "waves",
            ),
            (
                "feeling colors",
                None,
                "synesthesia",
                "visual inducer; tactile concurrent",
            ),
            (
                "grapheme color",
                None,
                "synesthesia",
                "grapheme inducer; color concurrent",
            ),
            (
                "existential anxiety",
                None,
                "anxiety",
                "existential",
            ),
        ]
        raw_result = {
            "tags": [
                {
                    "effect": source,
                    "detail": supplied_detail,
                    "text_detail": f"supported {source}",
                    "confidence": 0.9,
                    "attribution": {
                        "attribution_type": "unknown",
                        "dose_refs": [],
                    },
                }
                for source, supplied_detail, _, _ in cases
            ]
        }

        result = extractor.sanitize_extraction_payload(
            raw_result, [], require_evidence_grounding=False
        )
        actual = [(tag["effect"], tag["detail"]) for tag in result["tags"]]
        expected = [(effect, detail) for _, _, effect, detail in cases]
        self.assertEqual(expected, actual)

    def test_unsafe_compound_redirect_is_rejected(self):
        result = extractor.sanitize_extraction_payload(
            {
                "tags": [
                    {
                        "effect": "manic mood",
                        "detail": None,
                        "text_detail": "the report merely called the state manic",
                        "confidence": 0.9,
                        "attribution": {
                            "attribution_type": "unknown",
                            "dose_refs": [],
                        },
                    }
                ]
            },
            [],
            require_evidence_grounding=False,
        )
        self.assertEqual([], result["tags"])
        self.assertIn("unsupported effect tag", result["notes"])
        self.assertIn("manic mood", result["notes"])

    def test_generated_prompt_uses_atomic_vocabulary(self):
        prompt = extractor.build_system_prompt(40, 180, 180, False)
        for required in (
            "Canonical effects are atomic",
            "world-experience: uncanniness",
            "spatial: spatial scale distortion",
            "thought ownership loss",
            "ineffability",
        ):
            self.assertIn(required, prompt)

        vocabulary = extractor.build_controlled_vocabulary_text()
        for retired in (
            "closed-eye visuals",
            "social euphoria",
            "novelty salience",
            "manic mood",
        ):
            self.assertNotIn(retired, vocabulary)

    def test_boundary_references_resolve(self):
        quote = chr(96)
        references = set(
            re.findall(
                quote + "([^" + quote + "]+)" + quote,
                extractor.ONTOLOGY_BOUNDARY_RULES,
            )
        )
        self.assertEqual(set(), references - self.effects - {"detail"})

    def test_validator_rejects_attached_course_modifier(self):
        emotional = extractor.CONTROLLED_EFFECT_ONTOLOGY["emotional"]
        emotional["transient dread"] = "emotional change"
        try:
            with self.assertRaisesRegex(ValueError, "qualifier"):
                extractor.validate_effect_ontology()
        finally:
            emotional.pop("transient dread")


if __name__ == "__main__":
    unittest.main()
