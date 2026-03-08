# Subjective Effect Ontology Extractor for Trip Reports

Extract **high‑precision** subjective drug effects from narrative experience reports (Erowid, etc.) using Mistral AI, map them to a controlled ontology, attribute them to specific doses, and store the results in MongoDB – ready for knowledge graph construction.

> This pipeline is built for recall‑constrained information extraction. It prefers to omit a tag rather than guess, and it validates every output against a strict controlled vocabulary and dose table.

---

## Features

- **Controlled effect ontology** – 220+ canonical effect tags across 15 domains (visual, cognitive, somatic, emotional, etc.)
- **Dose attribution** – Links each effect to the specific dose(s) from the report’s `dose_table` (single substance, combination, or unknown)
- **Alias resolution** – Maps free‑text descriptions to canonical tags using a comprehensive alias dictionary (e.g., *“walls breathing”* → `surface breathing`)
- **Intelligent chunking** – Splits long reports into overlapping chunks while preserving context, then merges and deduplicates results
- **MongoDB persistence** – Stores extracted tags, metadata, and processing status; supports incremental batch processing
- **Dry‑run mode** – Preview extractions without writing to the database
- **Environment‑configurable** – All parameters (chunk size, batch size, model name, etc.) can be set via environment variables

---

## How It Works

1. **Load pending documents** from a source MongoDB collection (e.g., `erowid`) that haven’t been processed yet.
2. **Build a payload** containing the report text, dose table, and metadata.
3. **Send to Mistral** with a strict JSON‑schema response format and a system prompt that enforces the controlled vocabulary and attribution rules.
4. **Validate & canonicalize** the model’s output – reject out‑of‑vocabulary effects, fix dose references, and apply alias mapping.
5. **Handle long reports** – if the report exceeds a threshold, it’s split into overlapping chunks, each processed separately, then merged.
6. **Persist** the final `ExtractionResult` (tags + notes) into the target MongoDB collection (e.g., `erowid_effects`), or print it in dry‑run mode.

---

## Ontology Features

| report range | new effects |
| ------------ | ----------- |
| 1–100        | **99**      |
| 101–200      | **100**     |
| 201–300      | **27**      |

The ontology converges instead of exploding, as additional reports are tagged.

### Phase 1: Ontology Bootstrapping

Almost every report introduces something new.

This is expected because the model is still encountering new categories like:

- social confidence
- emotional warmth
- visual distortions
- patterning
- dissociation
- nausea

etc.

### Phase 2: Saturation

```
report 201–300 → 27 new effects
```

which is a **3–4× drop**.

Now reports mostly reuse existing labels instead of inventing new ones.

---

## Requirements

- Python **3.10+**
- MongoDB (local or remote)
- [Mistral AI API key](https://console.mistral.ai/)

---

## 🚀 Installation

```bash
git clone https://github.com/your-org/subjective-effect-extractor.git
cd subjective-effect-extractor
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

**`requirements.txt`** (minimal):

```
mistralai>=0.4.0
pymongo>=4.5
pydantic>=2.0
```

---

## ⚙️ Configuration

All settings are controlled via environment variables. Create a `.env` file or export them in your shell.

| Variable | Default | Description |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | **required** | Your Mistral API key |
| `MISTRAL_MODEL` | `mistral-large-2512` | Model name (must support JSON schema) |
| `MONGO_URI` | `mongodb://host.docker.internal:27017` | MongoDB connection string |
| `MONGO_DB` | `tripindex` | Database name |
| `MONGO_SOURCE_COLLECTION` | `erowid` | Collection containing raw reports |
| `MONGO_TARGET_COLLECTION` | `erowid_effects` | Collection where results will be stored |
| `BATCH_SIZE` | `10` | Number of documents to process in one run |
| `SOURCE_SCAN_BATCH_SIZE` | `BATCH_SIZE * 5` | Internal batch size for scanning source documents |
| `MAX_REPORT_TEXT_CHARS` | `8000` | If report exceeds this, it will be chunked |
| `REPORT_CHUNK_SIZE_CHARS` | `8000` | Character size of each chunk |
| `REPORT_CHUNK_OVERLAP_CHARS` | `1000` | Overlap between consecutive chunks |
| `MAX_COMPLETION_TOKENS` | `4000` | Max tokens for Mistral completion |
| `DRY_RUN` | `false` | If `true`, print results instead of writing to MongoDB |

---

## Usage

Run the extractor with:

```bash
python extractor.py
```

### Example dry‑run

```bash
export DRY_RUN=true
python extractor.py
```

You’ll see JSON output for each processed document, similar to:

```json
{
  "exp_id": "12345",
  "tags": [
    {
      "domain": "visual",
      "effect": "surface breathing",
      "parent_effect": "visual distortions",
      "detail": null,
      "attribution": {
        "attribution_type": "single_substance",
        "dose_refs": [
          {
            "dose_id": "d1",
            "substance": "psilocybin mushrooms",
            "dose": "2.5 g",
            "route": "oral"
          }
        ],
        "attribution_note": null
      },
      "text_detail": "walls started breathing",
      "confidence": 0.95
    }
  ],
  "notes": null
}
```

---

## Output Format

The target MongoDB collection contains documents with the following structure:

```json
{
  "exp_id": "12345",
  "source_doc_id": ObjectId("..."),
  "source_collection": "erowid",
  "title": "My Amazing Trip Report",
  "substance": "psilocybin mushrooms",
  "subjective_effect_tags": [ ... ],   // array of tags as shown above
  "subjective_effect_extraction": {
    "model_provider": "mistral",
    "model_name": "mistral-large-2512",
    "notes": "Processed in 2 chunks because report_text exceeded 8000 characters.",
    "tag_count": 12,
    "extracted_at": "2025-04-03T12:34:56Z",
    "status": "complete"
  }
}
```

On error, the document will contain a status `"error"` and an `error` field with the message.

---

## Controlled Vocabulary & Aliases

The system uses a **hand‑curated ontology** of subjective effects. Each effect belongs to a domain and has a canonical name and a broader parent effect (for roll‑ups).

- **Domains:** visual, auditory, somatic, motor, gastrointestinal, emotional, cognitive, temporal, selfhood, spiritual, social, tactile, sexual, thermal, sleep.
- **Parent effects** allow hierarchical grouping (e.g., all visual distortions roll up to `visual distortions`).

The alias dictionary maps hundreds of common descriptive phrases to canonical tags.  
For example:
- `"walls rippling"` → `texture rippling`
- `"time slowed down"` → `time dilation`
- `"jaw clenching"` → `jaw tension`

If an extracted effect doesn’t match any canonical tag or alias, it is **rejected** and recorded in the `notes` field.

---

## 💊 Dose Attribution

Each effect must be attributed to one or more entries from the report’s `dose_table`. The `dose_table` is an array of objects, each containing at least a substance name and optionally dose amount/route.

Attribution can be:

- **`single_substance`** – effect clearly belongs to one dose entry.
- **`combination`** – effect arises from the interaction or combined experience of multiple doses (e.g., a cannabis edible + LSD).
- **`unknown`** – dose table missing, ambiguous, or the effect cannot be confidently linked.

The model returns `dose_refs` – each referencing a `dose_id` from the original table. During validation, these references are enriched with the actual substance, dose phrase, and route from the source dose table.

---

## Chunking Logic

Long reports are split into overlapping chunks to stay within the model’s context window while preserving continuity.

- Chunks are split on paragraph breaks, newlines, or sentence boundaries to avoid cutting in the middle of a thought.
- Overlap (default 1000 characters) ensures that effects described near a chunk boundary are not missed.
- After all chunks are processed, results are merged and deduplicated based on `(domain, effect, parent_effect, attribution_type, dose_ids)`.

The final `notes` field indicates that chunking was used.

---

## Error Handling & Dry Run

- **Validation** rejects any tag that doesn’t conform to the controlled vocabulary, contains malformed dose references, or lacks supporting `text_detail`.
- **Rejected tags** are summarized in the `notes` field.
- **Dry‑run mode** (`DRY_RUN=true`) prints the final `ExtractionResult` as JSON instead of writing to MongoDB. Useful for testing and debugging.
- **Persistence errors** are logged, and the document is marked with `status: "error"` in the target collection.

---

## Incremental Processing

The script processes documents in batches and remembers which ones have already been completed (status `"complete"`). It never re‑processes a finished document unless you manually reset its status.

It uses a **cursor‑friendly scan** that avoids large `$nin` queries, making it suitable for large collections.

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.  
Areas for contribution:
- Expanding the controlled vocabulary and aliases
- Improving chunking heuristics
- Adding support for other LLM providers
- Better test coverage
