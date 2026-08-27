# Subjective Effect Ontology

This is a controlled vocabulary of directly reportable components of psychedelic
and other psychotropic experiences. It is developed independently from Josie
Kins's Subjective Effect Index; that index is not an ontology source for this
project.

## Representation

Each extracted observation has three semantic layers:

- `effect`: one atomic canonical phenomenon.
- `parent_effect`: a broad rollup, not an additional co-occurring effect.
- `detail`: content, direction, modality pair, location, phase, severity,
  duration, context, or another report-specific qualification.

For example, `geometric imagery` is canonical while “blue rotating mandalas
behind closed eyes” is detail. `salience enhancement` is canonical while
“mortality felt unusually important” places mortality in detail. `synesthesia`
is canonical while auditory-to-visual is a modality-pair detail.

Canonical hierarchy and ordering live in `effect_ontology/effects.py`, with labels
and definitions under `effect_ontology/definitions/`, which ontology tools can use
without importing MongoDB, Pydantic, or the model client.

The catalog validator requires exact definition coverage, rejects terse, malformed,
and duplicate prose, so every canonical node, including rollups, stays documented.

Severity and course words such as mild, intense, transient, or persistent must
never be baked into canonical labels. The runtime validator rejects those forms,
slash-joined alternatives, `-like` qualifiers, alias targets that do not exist,
duplicate terms across domains, and aliases that shadow canonical meanings.

## Orthogonality test

Admit a term only when all of the following hold:

1. It identifies a phenomenological structure, sensation, affect, cognitive
   operation, or self/world relation that a reporter can distinguish.
2. It is not merely the content, object, setting, valence, severity, duration,
   phase, or bodily location of an existing effect.
3. It can be supported by a different local evidence claim from its nearest
   neighbors. If the same clause would always support both labels, merge them or
   make one a detail.
4. It is not a composite syndrome. Extract the independently supported
   components of “mania,” “insanity,” or a “mystical experience.”
5. Its boundary can be stated positively and contrasted with the nearest
   confusable terms.
6. It plausibly occurs as a direct subjective effect. Behavior alone,
   pharmacological inference, diagnosis, and later interpretation are
   insufficient.

Opposites such as amplification and attenuation may both be canonical when they
are independently reportable directions rather than intensity modifiers.
Content-qualified variants such as music appreciation, nature appreciation, and
status salience are not separate effects.

## Sources and discovery

Candidate gaps come from two independent evidence streams:

- language actually present in the configured Erowid report corpus; and
- feature-level constructs in psychometric, neuropsychological, and
  phenomenological research.

Research sources used for coverage and boundary checking include:

- Schmidt & Berkemeyer, *The Altered States Database* (2018), summarizing the
  5D/11D-ASC, PCI, HRS, and MEQ factor structures:
  <https://doi.org/10.3389/fpsyg.2018.01028>
- Barrett et al., validation of the revised MEQ, including unity, sacredness,
  noetic quality, time/space transcendence, and ineffability:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC5203697/>
- Parnas et al., *EASE: Examination of Anomalous Self-Experience*:
  <https://doi.org/10.1159/000088441>
- Sass et al., *EAWE: Examination of Anomalous World Experience*:
  <https://doi.org/10.1159/000454928>
- Taves et al., the feature-first *Inventory of Nonordinary Experiences*:
  <https://doi.org/10.1371/journal.pone.0287780>
- Nour et al., validation of ego dissolution and orthogonal ego inflation:
  <https://pubmed.ncbi.nlm.nih.gov/27378878/>
- Blanke-style phenomenological distinctions among autoscopy, heautoscopy,
  out-of-body experience, and sensed presence:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC3032659/>
- Blom's systematic review of Alice in Wonderland syndrome phenomena:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC4909520/>

These sources are used to discover distinctions and test boundaries, not to copy
a questionnaire wholesale. Scale factors that bundle several experiences are
decomposed, and diagnostic or interpretive wording is translated into neutral
phenomenology where possible.

## Compatibility

`DEPRECATED_EFFECT_REDIRECTS` is the complete audit registry for older
canonical-looking outputs, while the runtime behavior is deliberately split:

- `SAFE_DEPRECATED_EFFECT_REDIRECTS` contains only labels whose replacement is
  semantically entailed. These may normalize fresh model output and historical
  stored data.
- `UNSAFE_DEPRECATED_EFFECT_REDIRECTS` records content-dependent legacy
  mappings for provenance and corrective migration. These labels are not
  runtime aliases and fail closed, so the extractor must return locally
  supported components instead.

Compatibility details exist only for safe redirects. A historical correction
may restore an unsafe old label only from a verified retained backup and only
when the current tag still matches the exact earlier migration transform.

Ambiguous shorthand such as `happy`, `detached`, or `thought i was dead` is
intentionally not coerced into one canonical effect; it is rejected so the
evidence can be decomposed correctly.

## Machine-readable releases

`export_ontology.py` publishes the modular ontology as immutable,
content-addressed JSON under `ontology_releases/`. Concept UUIDv5 IDs remain
stable across releases by reading and validating every prior artifact. The
stable `ontology_releases/current.json` manifest pins one exact current release;
consumers must not select an artifact by filename order.

Immutable schema v1 and v2 artifacts remain valid stable-ID history and are
verified by the same reader. Schema v3 adds:

- `normalization_hash` over the declared normalization behavior, canonical
  labels/slugs, aliases, redirects, resolution modes, and ambiguity blocklist;
- `semantic_hash` over concepts, definitions, hierarchy, redirect relations,
  and review metadata;
- `release_hash` over the complete canonical release body except its own hash;
  and
- a raw serialized `artifact_sha256` in the current manifest.

A definition-only edit therefore changes `semantic_hash` and `release_hash`
while leaving `normalization_hash` unchanged. Multiple schema-v3 releases may
share a normalization hash; their semantic hashes distinguish them.

Every current concept has `review_status: "defined"`. This controlled status
means exactly that the concept has one nonempty definition. It does not mean
that an editor, subject-matter expert, clinician, or formal review board has
reviewed it. The other reserved states are `draft`, `editorial_reviewed`,
`expert_reviewed`, and `deprecated`; use them only when the corresponding
process actually occurred.

The exported `normalized_name` and `normalized_label` values are Unicode search
keys, not permission to resolve an ambiguous phrase. Automatic aliases and
safe redirects remain distinct from `manual_review` redirects and the explicit
ambiguous-label list. Automatic redirects carry `effect_id` as resolvable
identity. Manual-review redirects instead carry a `candidate_effect_id` for
editorial context; the standard consumer resolver always leaves `concept_id`
unset for them. Ambiguous labels likewise remain unresolved.

For a canonical rename:

1. Ensure the prior ontology release is already published and immutable.
2. Rename the canonical concept and add the old label to
   `SAFE_DEPRECATED_EFFECT_REDIRECTS` only when the replacement is semantically
   entailed.
3. Export the new release and verify that the renamed concept inherited its
   prior ID.

Unsafe redirects never transfer stable identity. If label normalization or the
release structure changes, increment the export schema version; do not create
an incompatible body under an old schema. Any schema-version change must also
teach the prior-release reader how to validate older artifacts and carry their
concept IDs forward.

## Review checklist

When adding or renaming a term:

1. Search canonical labels, aliases, boundary notes, and redirects.
2. Compare it with every nearest neighbor, including terms in other domains.
3. Add an elaborate positive and differential definition in the relevant
   `effect_ontology/definitions/` module; add a prompt boundary rule when
   the extractor must distinguish the concept from a confusable neighbor.
4. Add only unambiguous aliases; do not map distinct emotions or symptoms merely
   because everyday speech sometimes conflates them.
5. Put contextual variants in `detail`.
6. Classify every redirect explicitly as safe or unsafe for stored-data
   migration; do not infer safety from whether its target is broad or narrow.
7. Import the module to run `validate_effect_ontology()`, compile it, and test
   representative normalization and sanitization paths.
