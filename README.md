# Erowid Effect Extractor

This project runs a Dockerized Z.ai GLM extractor that converts Erowid report
text into grounded, controlled subjective-effect tags stored in MongoDB.

Each extracted tag is stored with a controlled ontology shape:
`domain`, `effect`, `parent_effect`, optional `detail`, attribution metadata, `text_detail`, and `confidence`.
The legacy `subjective_effect` field is still written for compatibility, but now mirrors the broader `parent_effect` rollup rather than the more specific `effect`.
Broad rollup labels such as `visual distortions` or `body load` are fallback-only; the extractor now prefers more specific canonical tags when supported by the report text.
By default, broad rollup labels are no longer accepted as `effect` values at all; they are kept only in `parent_effect` for rollups.
Validation may append notes about rejected unsupported tag proposals so ontology gaps are visible during review. The Mongo tag schema is unchanged: ontology hierarchy, source dose metadata, exact evidence grounding, and attribution type are reconstructed deterministically before persistence.

The embedded vocabulary spans 21 domains and 506 ontology nodes: 485 atomic
effects and 21 broad parent rollups.

The extractor does not treat model output as ground truth. It rejects invented ontology labels, non-contiguous or ungrounded evidence, ambiguous dose IDs, malformed dose references, broad fallback labels, and a small set of explicit polarity contradictions. `MIN_TAG_CONFIDENCE` defaults to `0` because model self-confidence is not calibrated without a labelled evaluation set; it remains available as an opt-in filter.

See [ONTOLOGY.md](ONTOLOGY.md) for the atomicity rules, independent research
provenance, compatibility policy, and term-admission checklist.

Ontology identity is deterministic. `ONTOLOGY_HASH` covers the canonical
hierarchy, the final resolved alias map, safe and unsafe deprecated redirects,
compatibility details, and the ambiguous-alias blocklist. Changes to any of
those normalization semantics therefore change the pipeline fingerprint used
by the stale-result policy.

## Setup

1. Create a private environment file with
   `install -m 600 .env.example .env`.
2. Fill in `ZAI_API_KEY` and adjust the Mongo settings if needed. By default the extractor reads from `tripindex.erowid-clean` and writes extracted effects to `tripindex.erowid-effects-1`.
3. Build the image:

```bash
docker compose build
```

The container runs as the non-root `extractor` user with a read-only root
filesystem, a disposable `/tmp` tmpfs, all Linux capabilities dropped, and
`no-new-privileges` enabled.

## Run the extractor

Run one batch and watch progress in your terminal:

```bash
docker compose run --rm effect-extractor
```

That command prints each `exp_id` as it is processed and ends with a machine-readable `RUN_SUMMARY` JSON object. Successful results retain the existing `subjective_effect_tags` schema. Queue/version metadata is additive under `subjective_effect_extraction`.

Exit codes are:

- `0`: all attempted documents completed successfully.
- `1`: one or more document-level extraction failures occurred.
- `2`: configuration, MongoDB, index, authentication, or unexpected runtime failure.
- `75`: temporary provider failure exhausted retries; stop and retry later.
- `130`: interrupted.

## Optional dry run

```bash
docker compose run --rm -e DRY_RUN=true effect-extractor
```

## Tuning output truncation

If the model returns malformed, structurally invalid, or cut-off JSON, the
extractor automatically retries that payload in smaller overlapping chunks and
merges the valid results. These
environment variables tune the initial request and the retry floor:

- `MAX_COMPLETION_TOKENS` controls the response token budget. Default: `12000`.
- `MAX_TAGS_PER_PAYLOAD` caps the number of tags requested and retained per model call. Default: `40`.
- `ZAI_THINKING` controls GLM thinking mode. Default: `disabled` for reliable JSON output.
- `MAX_REPORT_TEXT_CHARS` controls when a report is chunked before sending. Default: `4000`.
- `REPORT_CHUNK_SIZE_CHARS` controls chunk size for long reports. Default: `4000`.
- `REPORT_CHUNK_OVERLAP_CHARS` controls overlap between chunks. Default: `600`.
- `MIN_RETRY_CHUNK_SIZE_CHARS` controls the smallest automatic retry chunk when Z.ai returns invalid JSON. Default: `1200`.
- `MAX_TEXT_DETAIL_CHARS` keeps evidence excerpts compact. Default: `180`.
- `MAX_ATTRIBUTION_NOTE_CHARS` keeps attribution notes compact. Default: `180`.
- `MIN_TAG_CONFIDENCE` optionally filters model confidence. Default: `0` (disabled because it is uncalibrated).
- `REQUIRE_GROUNDED_EVIDENCE` requires every retained excerpt to map to an exact contiguous source slice. Default: `true`.
- `ENABLE_SEMANTIC_GUARDS` rejects only narrowly detected explicit contradictions. Default: `true`.
- `ALLOW_BROAD_FALLBACK_EFFECTS` allows generic tags such as `body load` or `visual distortions` as `effect` values when set to `true`. Default: `false`.

Example:

```bash
docker compose run --rm \
  -e MAX_COMPLETION_TOKENS=8000 \
  -e MAX_REPORT_TEXT_CHARS=3000 \
  -e REPORT_CHUNK_SIZE_CHARS=3000 \
  -e MAX_TAGS_PER_PAYLOAD=10 \
  effect-extractor
```

## Retries, leases, and stale results

The SDK's hidden retries are disabled. The extractor owns bounded retry behavior, honors `Retry-After` within the configured cap, and classifies content filtering separately from authentication or configuration failures.

- `API_MAX_RETRIES`, `API_RETRY_BASE_SECONDS`, `API_RETRY_MAX_SECONDS`, and `API_TIMEOUT_SECONDS` control inline provider retries.
- `ERROR_MAX_ATTEMPTS`, `ERROR_RETRY_BASE_SECONDS`, and `ERROR_RETRY_MAX_SECONDS` control per-document cooldowns. Provider outages never become permanently terminal from lifetime attempt counts.
- `PROCESSING_LEASE_SECONDS` controls the fenced MongoDB lease and must exceed the API timeout plus maximum retry delay and a 60-second margin. Leases are renewed around every model call and retry sleep.
- `STALE_POLICY=none|source|pipeline|any` controls deliberate re-extraction of completed results. The safe default is `none`.
- `REPROCESS_UNVERSIONED=false` keeps legacy completed results without fingerprints untouched unless explicitly enabled.

A stale refresh never removes the previous complete tags before replacement
succeeds. Claim acquisition is revision-fenced; renewals, finalization, error
writes, and releases require the matching lease token. A unique partial index
enforces one target document per `exp_id`.

## Tests

Run deterministic tests against the source tree:

```bash
docker compose build
docker compose run --rm \
  -v "$PWD:/tests:ro" -w /tests --entrypoint python \
  effect-extractor -m unittest discover -v
```

Queue race tests use a uniquely named disposable MongoDB collection and remove it afterward:

```bash
docker compose run --rm -e RUN_MONGO_INTEGRATION=1 \
  -v "$PWD:/tests:ro" -w /tests --entrypoint python \
  effect-extractor -m unittest -v test_queue_integration.py
```

## Existing-data ontology migration

`migrate_existing_ontology.py` is read-only by default and never calls the
extraction sanitizer. In normal normalization and provenance-repair modes, its
transformations can modify only the existing `effect`, `domain`,
`parent_effect`, `subjective_effect`, and `detail` fields on complete documents.
Document/tag counts, `_id` order, tag keysets, evidence, attribution,
confidence, and all non-ontology fields are verified unchanged. Explicit
whole-collection rollback is the separate operation described below.

Extraction-time compatibility and stored-data migration intentionally use
different safety thresholds. Only redirects explicitly approved as lossless
for historical migration are rewritten. Content-dependent redirects are
counted and reported while the stored tag remains byte-value-identical.

Run a projection first:

```bash
docker compose run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace --entrypoint python \
  effect-extractor migrate_existing_ontology.py \
  --manifest /workspace/migration_manifests/preapply.json
```

Before `--apply`, stop every extractor, take an external `mongodump`, and inspect the projection manifest. Apply creates and verifies a retained exact backup plus a transformed shadow, re-hashes the source immediately before cutover, atomically renames the shadow, and then requires an idempotent second projection:

```bash
docker compose run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace --entrypoint python \
  effect-extractor migrate_existing_ontology.py --apply \
  --manifest /workspace/migration_manifests/apply.json
```

Rollback is explicit and also retains the current target before replacement:

```bash
docker compose run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace --entrypoint python \
  effect-extractor migrate_existing_ontology.py --apply \
  --rollback-backup 'retained_backup_collection_name'
```

Whole-collection rollback replaces the target and is appropriate only when the
entire retained snapshot is the intended destination. Do not use it to correct
the July ontology migration after newer documents have been written; that
would discard those newer documents. Use the provenance-checked repair overlay
described below instead.

### Correcting ontology v1 without losing newer documents

`--repair-from-backup` uses a retained pre-v1 collection only as tag-level
provenance. It begins from the current target, so documents and indexes created
after v1 remain intact. A tag is repairable only when the current document has
the same `_id` and tag position and the tag is still the exact deterministic v1
transform of its backup counterpart. Any lineage conflict fails closed, and
the expected backup content hash is mandatory.

Use the retained collection name and `backup_snapshot.content_sha256` from the
local v1 apply manifest in place of the placeholders below.

Project the overlay first without database writes:

```bash
docker compose run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace --entrypoint python \
  effect-extractor migrate_existing_ontology.py \
  --repair-from-backup 'retained_pre_v1_backup_collection' \
  --expected-repair-backup-sha256 'verified_backup_content_sha256' \
  --manifest /workspace/migration_manifests/repair-preapply.json
```

After stopping every writer, taking a fresh external dump, and reviewing a
conflict-free projection, apply through a verified current-target backup and
shadow cutover:

```bash
docker compose run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace --entrypoint python \
  effect-extractor migrate_existing_ontology.py --apply \
  --repair-from-backup 'retained_pre_v1_backup_collection' \
  --expected-repair-backup-sha256 'verified_backup_content_sha256' \
  --manifest /workspace/migration_manifests/repair-apply.json
```

Apply re-hashes both source collections before cutover, verifies that only the
five ontology fields changed, and requires an idempotent second repair
projection. The repair is not performed automatically by normal extraction.

### Historical migration record

The local July 17, 2026 pre-apply and apply manifests record a verified
migration of 3,681 documents and 60,824 tags. They prove count and structure
preservation, hash-checked shadow cutover, and an idempotent second projection.
They do not prove that every historical redirect preserved meaning, and they do
not contain rollback data.

The manifests contain environment-specific database identifiers, so they are
excluded from this public repository together with the corresponding database
dump and retained MongoDB backup. Those local artifacts remain required for
disaster recovery. An
`extractor_sha256` inside a manifest identifies the extractor used for that
operation and is not expected to match later source revisions.

## Stop the container

```bash
docker compose down
```

## License

GNU Lesser General Public License v2.1; see [LICENSE](LICENSE).
