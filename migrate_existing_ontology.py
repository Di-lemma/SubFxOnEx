#!/usr/bin/env python3
"""Safely normalize stored effect ontology fields through a verified shadow copy.

The default mode is read-only.  It profiles the configured target collection,
computes the exact deterministic transformation, and prints a JSON manifest.

``--apply`` is deliberately explicit.  It requires a stable source snapshot,
creates and verifies both a retained backup collection and a transformed shadow
collection, rechecks the source snapshot, and finally atomically renames the
shadow over the configured target.  Existing writers must still be stopped by
the operator: repeated full-collection hashes detect writes during the run, but
cannot prove that an idle writer will not resume immediately after a check.

Rollback is also explicit::

    python migrate_existing_ontology.py --apply \
        --rollback-backup erowid-effects-1__ontology_backup__RUN_ID

Rollback first retains another exact copy of the current target, restores the
selected backup through a verified shadow, and atomically replaces the target.

An ontology-v1 correction is a separate, read-only-by-default overlay.  It
uses an exact retained pre-v1 backup only as provenance for individual tags,
copies the *current* target so newer documents and indexes survive, and changes
only tags whose current value is proven to be the deterministic v1 transform
of the corresponding backup tag::

    python migrate_existing_ontology.py \
        --repair-from-backup erowid-effects-1__ontology_backup__RUN_ID \
        --expected-repair-backup-sha256 CONTENT_SHA256

Add ``--apply`` only after the repair projection has no ancestry conflicts.

This module never calls ``sanitize_extraction_payload``.  Historical tags keep
their order, key sets, count, evidence, confidence, and attribution.  Complete
documents may change only these existing tag fields:

* effect
* domain
* parent_effect
* subjective_effect
* detail
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

try:
    from bson.json_util import CANONICAL_JSON_OPTIONS, dumps as bson_dumps
except ModuleNotFoundError as exc:  # Pure transforms/tests do not need MongoDB.
    CANONICAL_JSON_OPTIONS = None
    bson_dumps = None
    _BSON_IMPORT_ERROR: Optional[Exception] = exc
else:
    _BSON_IMPORT_ERROR = None

try:
    from pymongo import ASCENDING, IndexModel, MongoClient
    from pymongo.collection import Collection
    from pymongo.database import Database
    from pymongo.write_concern import WriteConcern
except ModuleNotFoundError as exc:  # Keep pure transforms dependency-light.
    ASCENDING = 1
    IndexModel = None
    MongoClient = None
    Collection = Any
    Database = Any
    WriteConcern = None
    _PYMONGO_IMPORT_ERROR: Optional[Exception] = exc
else:
    _PYMONGO_IMPORT_ERROR = None

try:
    import effect_extractor as extractor
except ModuleNotFoundError as exc:  # Tests can inject a minimal ontology object.
    extractor = None
    _EXTRACTOR_IMPORT_ERROR: Optional[Exception] = exc
else:
    _EXTRACTOR_IMPORT_ERROR = None


ALLOWED_TAG_FIELDS = (
    "effect",
    "domain",
    "parent_effect",
    "subjective_effect",
    "detail",
)
ALLOWED_TAG_FIELD_SET = frozenset(ALLOWED_TAG_FIELDS)
DEFAULT_BATCH_SIZE = 200
DEFAULT_QUIESCENCE_SECONDS = 5.0
MAX_VERIFICATION_ISSUES = 25
MANIFEST_VERSION = "ontology-migration-v3"

# The v1 migration used this exact detail map while applying every redirect
# whose target was not a broad rollup.  Keep the snapshot local and immutable:
# current ontology detail improvements must not change the ancestry proof for
# a tag that was written by v1.
HISTORICAL_V1_ALLOWED_TAG_FIELDS = (
    "effect",
    "domain",
    "parent_effect",
    "subjective_effect",
    "detail",
)
HISTORICAL_V1_ALLOWED_TAG_FIELD_SET = frozenset(
    HISTORICAL_V1_ALLOWED_TAG_FIELDS
)
HISTORICAL_V1_DEPRECATED_EFFECT_DETAILS = {
    "lattice imagery": "lattice",
    "tessellation": "tessellation",
    "mandala imagery": "mandala",
    "closed-eye visuals": "eyes closed",
    "open-eye visuals": "eyes open",
    "entity imagery": "entity",
    "shadow imagery": "shadow figure",
    "peripheral imagery": "peripheral visual field",
    "symbolic imagery": "symbolic content",
    "double vision": "two images",
    "synesthetic visuals": "visual concurrent",
    "auditory-visual synesthesia": "auditory inducer; visual concurrent",
    "tactile-visual synesthesia": "tactile inducer; visual concurrent",
    "conceptual synesthesia": "conceptual inducer",
    "music appreciation enhancement": "music",
    "enhanced appreciation of nature": "nature",
    "threat salience": "threat",
    "responsibility salience": "responsibility",
    "mortality salience": "mortality",
    "moral salience": "morality",
    "rule salience": "rules",
    "status salience": "social status",
    "social euphoria": "social context",
    "pleasant touch amplification": "pleasant touch",
    "unpleasant touch amplification": "unpleasant touch",
}
HISTORICAL_V1_UNSAFE_BROAD_REDIRECTS = frozenset(
    {"manic mood", "mystical quality"}
)
HISTORICAL_V1_REDIRECTS = {
    "melting/flowing": "visual liquefaction",
    "lattice imagery": "geometric imagery",
    "tessellation": "geometric imagery",
    "mandala imagery": "geometric imagery",
    "closed-eye visuals": "visual imagery",
    "open-eye visuals": "visual imagery",
    "entity imagery": "complex visual hallucination",
    "shadow imagery": "complex visual hallucination",
    "peripheral imagery": "visual imagery",
    "enhanced colors": "color saturation enhancement",
    "visual clarity": "visual acuity enhancement",
    "symbolic imagery": "visual imagery",
    "double vision": "visual multiplicity",
    "diffraction": "light haloing",
    "frame rate suppression": "visual motion discontinuity",
    "scenery slicing": "visual fragmentation",
    "delirious hallucination": "complex visual hallucination",
    "synesthetic visuals": "synesthesia",
    "auditory warping": "timbre distortion",
    "auditory stretching": "sound duration distortion",
    "music appreciation enhancement": "aesthetic appreciation",
    "music immersion": "attentional absorption",
    "internal music": "auditory imagery",
    "voices": "auditory hallucination",
    "ringing": "auditory hallucination",
    "humming": "auditory hallucination",
    "buzzing": "auditory hallucination",
    "externalized sounds": "auditory hallucination",
    "phantom auditory events": "auditory hallucination",
    "flanging": "timbre distortion",
    "auditory-visual synesthesia": "synesthesia",
    "tactile-visual synesthesia": "synesthesia",
    "conceptual synesthesia": "synesthesia",
    "perceived acceleration": "illusory acceleration",
    "perceived levitation": "illusory levitation",
    "sense of falling": "illusory falling",
    "heaviness in limbs": "somatic heaviness",
    "somatic euphoria": "bodily pleasure",
    "clumsiness": "incoordination",
    "akathisia-like movement": "akathisia",
    "shaking": "tremor",
    "enhanced appreciation of nature": "aesthetic appreciation",
    "manic mood": "emotional change",
    "anxiety suppression": "anxiety relief",
    "fear suppression": "fear relief",
    "threat salience": "salience enhancement",
    "existential dread": "dread",
    "aesthetic appreciation enhancement": "aesthetic appreciation",
    "responsibility salience": "salience enhancement",
    "cognitive euphoria": "ideational pleasure",
    "mortality salience": "salience enhancement",
    "moral salience": "salience enhancement",
    "novelty salience": "perceptual freshness",
    "rule salience": "salience enhancement",
    "present-moment absorption": "attentional absorption",
    "perceived theriomorphosis": "theriomorphosis",
    "agency disturbance": "agency loss",
    "perceived inanimate transformation": "inanimate self-transformation",
    "perceived death experience": "felt death",
    "mystical quality": "spiritual experience",
    "contact-with-presence": "sensed presence",
    "social euphoria": "euphoria",
    "status salience": "salience enhancement",
    "enhanced touch": "tactile amplification",
    "distorted touch": "tactile distortion",
    "pleasant touch amplification": "tactile amplification",
    "unpleasant touch amplification": "tactile amplification",
    "texture recognition suppression": "tactile recognition impairment",
    "tactile sensual enhancement": "tactile amplification",
    "dream enhancement": "vivid dreams",
}
HISTORICAL_V1_TARGET_INDEX_SHA256 = (
    "55797e13322664c01cf89ab99571a66662d49421ab85b1a9f3559c366a3cfde8"
)
HISTORICAL_V1_SPEC_SHA256 = (
    "780bc5efa7352859f198b1fb26c40b41d1a1f1cbffb71e569d20cce95077fd60"
)
HISTORICAL_V1_TARGET_HIERARCHY = {
    "aesthetic appreciation": ("emotional", "emotional change"),
    "agency loss": ("selfhood", "selfhood change"),
    "akathisia": ("motor", "motor change"),
    "anxiety relief": ("emotional", "emotional change"),
    "attentional absorption": ("cognitive", "cognitive change"),
    "auditory hallucination": ("auditory", "auditory distortions"),
    "auditory imagery": ("auditory", "auditory distortions"),
    "bodily pleasure": ("somatic", "body load"),
    "color saturation enhancement": ("visual", "visual distortions"),
    "complex visual hallucination": ("visual", "visual distortions"),
    "dread": ("emotional", "emotional change"),
    "emotional change": ("emotional", "emotional change"),
    "euphoria": ("emotional", "emotional change"),
    "fear relief": ("emotional", "emotional change"),
    "felt death": ("selfhood", "selfhood change"),
    "geometric imagery": ("visual", "visual distortions"),
    "ideational pleasure": ("cognitive", "cognitive change"),
    "illusory acceleration": ("vestibular", "vestibular change"),
    "illusory falling": ("vestibular", "vestibular change"),
    "illusory levitation": ("vestibular", "vestibular change"),
    "inanimate self-transformation": ("selfhood", "selfhood change"),
    "incoordination": ("motor", "motor change"),
    "light haloing": ("visual", "visual distortions"),
    "perceptual freshness": ("world-experience", "world-experience change"),
    "salience enhancement": ("cognitive", "cognitive change"),
    "sensed presence": ("spiritual", "spiritual experience"),
    "somatic heaviness": ("somatic", "body load"),
    "sound duration distortion": ("auditory", "auditory distortions"),
    "spiritual experience": ("spiritual", "spiritual experience"),
    "synesthesia": ("synesthetic", "synesthetic change"),
    "tactile amplification": ("tactile", "tactile change"),
    "tactile distortion": ("tactile", "tactile change"),
    "tactile recognition impairment": ("tactile", "tactile change"),
    "theriomorphosis": ("selfhood", "selfhood change"),
    "timbre distortion": ("auditory", "auditory distortions"),
    "tremor": ("somatic", "body load"),
    "visual acuity enhancement": ("visual", "visual distortions"),
    "visual fragmentation": ("visual", "visual distortions"),
    "visual imagery": ("visual", "visual distortions"),
    "visual liquefaction": ("visual", "visual distortions"),
    "visual motion discontinuity": ("visual", "visual distortions"),
    "visual multiplicity": ("visual", "visual distortions"),
    "vivid dreams": ("sleep", "sleep disturbance"),
}


class MigrationSafetyError(RuntimeError):
    """Raised when a safety invariant prevents migration or rollback."""


def require_extractor(ontology: Any = None) -> Any:
    if ontology is not None:
        return ontology
    if extractor is None:
        raise MigrationSafetyError(
            "effect_extractor and its project dependencies are required for this operation"
        ) from _EXTRACTOR_IMPORT_ERROR
    return extractor


def require_database_dependencies() -> None:
    if bson_dumps is None:
        raise MigrationSafetyError(
            "PyMongo/BSON is required for collection hashing and database operations"
        ) from _BSON_IMPORT_ERROR
    if MongoClient is None or IndexModel is None or WriteConcern is None:
        raise MigrationSafetyError(
            "PyMongo is required for database operations"
        ) from _PYMONGO_IMPORT_ERROR


@dataclass(frozen=True)
class TagTransformResult:
    outcome: str
    raw_effect: Optional[str] = None
    canonical_effect: Optional[str] = None
    changed_fields: tuple[str, ...] = ()
    detail_action: Optional[str] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class RepairTagResult:
    tag_index: int
    outcome: str
    raw_effect: Optional[str] = None
    historical_effect: Optional[str] = None
    desired_effect: Optional[str] = None
    changed_fields: tuple[str, ...] = ()
    reason: Optional[str] = None


@dataclass
class TransformMetrics:
    documents_seen: int = 0
    complete_documents: int = 0
    skipped_noncomplete_documents: int = 0
    documents_changed: int = 0
    tags_seen: int = 0
    tags_changed: int = 0
    tag_outcomes: Counter[str] = field(default_factory=Counter)
    field_changes: Counter[str] = field(default_factory=Counter)
    safe_redirects: Counter[str] = field(default_factory=Counter)
    unsafe_redirects: Counter[str] = field(default_factory=Counter)
    unsupported_effects: Counter[str] = field(default_factory=Counter)
    detail_actions: Counter[str] = field(default_factory=Counter)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def observe_document(
        self,
        before: Any,
        after: Any,
        tag_results: Sequence[TagTransformResult],
        *,
        max_examples: int,
    ) -> None:
        self.documents_seen += 1
        extraction = (
            before.get("subjective_effect_extraction")
            if isinstance(before, dict)
            else None
        )
        status = extraction.get("status") if isinstance(extraction, dict) else None
        if status != "complete":
            self.skipped_noncomplete_documents += 1
            return

        self.complete_documents += 1
        if before != after:
            self.documents_changed += 1

        tags = before.get("subjective_effect_tags") if isinstance(before, dict) else None
        if isinstance(tags, list):
            self.tags_seen += len(tags)

        for tag_index, result in enumerate(tag_results):
            self.tag_outcomes[result.outcome] += 1
            if result.changed_fields:
                self.tags_changed += 1
                self.field_changes.update(result.changed_fields)
            if result.detail_action:
                self.detail_actions[result.detail_action] += 1
            if result.outcome == "changed" and result.raw_effect != result.canonical_effect:
                self.safe_redirects[
                    f"{result.raw_effect} -> {result.canonical_effect}"
                ] += 1
            elif result.outcome == "unsafe_redirect":
                self.unsafe_redirects[
                    f"{result.raw_effect} -> {result.canonical_effect}"
                ] += 1
            elif result.outcome in {
                "unsupported_effect",
                "invalid_effect",
                "missing_ontology_keys",
                "invalid_detail",
                "invalid_tag",
            }:
                self.unsupported_effects[result.raw_effect or "<missing>"] += 1

            if (
                result.outcome
                in {
                    "unsafe_redirect",
                    "unsupported_effect",
                    "invalid_effect",
                    "missing_ontology_keys",
                    "invalid_detail",
                    "invalid_tag",
                }
                and len(self.examples) < max_examples
            ):
                self.examples.append(
                    {
                        "source_id": stringify_id(
                            before.get("_id") if isinstance(before, dict) else None
                        ),
                        "exp_id": (
                            before.get("exp_id") if isinstance(before, dict) else None
                        ),
                        "tag_index": tag_index,
                        "outcome": result.outcome,
                        "effect": result.raw_effect,
                        "target": result.canonical_effect,
                        "reason": result.reason,
                    }
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents_seen": self.documents_seen,
            "complete_documents": self.complete_documents,
            "skipped_noncomplete_documents": self.skipped_noncomplete_documents,
            "documents_changed": self.documents_changed,
            "tags_seen": self.tags_seen,
            "tags_changed": self.tags_changed,
            "tag_outcomes": sorted_counter(self.tag_outcomes),
            "field_changes": sorted_counter(self.field_changes),
            "safe_redirects": sorted_counter(self.safe_redirects),
            "unsafe_redirects": sorted_counter(self.unsafe_redirects),
            "unsupported_effects": sorted_counter(self.unsupported_effects),
            "detail_actions": sorted_counter(self.detail_actions),
            "examples": self.examples,
        }


@dataclass(frozen=True)
class CollectionSnapshot:
    collection: str
    collection_uuid: Optional[str]
    document_count: int
    content_sha256: str
    options_sha256: str
    indexes_sha256: str
    status_counts: dict[str, int]
    model_counts: dict[str, int]
    tag_count: int
    options: dict[str, Any]
    indexes: list[dict[str, Any]]

    def stable_identity(self) -> dict[str, Any]:
        """Fields that must remain identical for a quiescent collection."""

        return {
            "collection_uuid": self.collection_uuid,
            "document_count": self.document_count,
            "content_sha256": self.content_sha256,
            "options_sha256": self.options_sha256,
            "indexes_sha256": self.indexes_sha256,
            "status_counts": self.status_counts,
            "model_counts": self.model_counts,
            "tag_count": self.tag_count,
        }

    def clone_identity(self) -> dict[str, Any]:
        """Fields expected to survive a copy despite a new name and UUID."""

        identity = self.stable_identity()
        identity.pop("collection_uuid", None)
        return identity

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "collection_uuid": self.collection_uuid,
            "document_count": self.document_count,
            "content_sha256": self.content_sha256,
            "options_sha256": self.options_sha256,
            "indexes_sha256": self.indexes_sha256,
            "status_counts": self.status_counts,
            "model_counts": self.model_counts,
            "tag_count": self.tag_count,
            "options": self.options,
            "indexes": self.indexes,
        }


@dataclass(frozen=True)
class ProjectionResult:
    source_sha256: str
    projected_sha256: str
    document_count: int
    tag_count: int
    metrics: TransformMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "projected_sha256": self.projected_sha256,
            "document_count": self.document_count,
            "tag_count": self.tag_count,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class RepairMetrics:
    documents_seen: int = 0
    backup_documents_matched: int = 0
    current_only_documents: int = 0
    documents_changed: int = 0
    tags_seen: int = 0
    tags_repaired: int = 0
    tag_outcomes: Counter[str] = field(default_factory=Counter)
    field_changes: Counter[str] = field(default_factory=Counter)
    repaired_redirects: Counter[str] = field(default_factory=Counter)
    conflicts: Counter[str] = field(default_factory=Counter)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def observe_document(
        self,
        before: Any,
        after: Any,
        tag_results: Sequence[RepairTagResult],
        *,
        backup_matched: bool,
        max_examples: int,
    ) -> None:
        self.documents_seen += 1
        if backup_matched:
            self.backup_documents_matched += 1
        else:
            self.current_only_documents += 1
        if before != after:
            self.documents_changed += 1

        tags = before.get("subjective_effect_tags") if isinstance(before, dict) else None
        if isinstance(tags, list):
            self.tags_seen += len(tags)

        for result in tag_results:
            self.tag_outcomes[result.outcome] += 1
            if result.changed_fields:
                self.tags_repaired += 1
                self.field_changes.update(result.changed_fields)
                self.repaired_redirects[
                    f"{result.raw_effect}: "
                    f"{result.historical_effect} -> {result.desired_effect}"
                ] += 1
            if result.outcome == "lineage_conflict":
                self.conflicts[result.raw_effect or "<missing>"] += 1
            if (
                result.outcome == "lineage_conflict"
                and len(self.examples) < max_examples
            ):
                self.examples.append(
                    {
                        "source_id": stringify_id(
                            before.get("_id") if isinstance(before, dict) else None
                        ),
                        "exp_id": (
                            before.get("exp_id") if isinstance(before, dict) else None
                        ),
                        "tag_index": result.tag_index,
                        "outcome": result.outcome,
                        "effect": result.raw_effect,
                        "historical_effect": result.historical_effect,
                        "desired_effect": result.desired_effect,
                        "reason": result.reason,
                    }
                )

    @property
    def conflict_count(self) -> int:
        return sum(self.conflicts.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents_seen": self.documents_seen,
            "backup_documents_matched": self.backup_documents_matched,
            "current_only_documents": self.current_only_documents,
            "documents_changed": self.documents_changed,
            "tags_seen": self.tags_seen,
            "tags_repaired": self.tags_repaired,
            "tag_outcomes": sorted_counter(self.tag_outcomes),
            "field_changes": sorted_counter(self.field_changes),
            "repaired_redirects": sorted_counter(self.repaired_redirects),
            "conflicts": sorted_counter(self.conflicts),
            "conflict_count": self.conflict_count,
            "examples": self.examples,
        }


@dataclass(frozen=True)
class RepairProjectionResult:
    source_sha256: str
    projected_sha256: str
    document_count: int
    tag_count: int
    metrics: RepairMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "projected_sha256": self.projected_sha256,
            "document_count": self.document_count,
            "tag_count": self.tag_count,
            "metrics": self.metrics.to_dict(),
        }


def sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def stringify_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_timestamp(value: Optional[datetime] = None) -> str:
    value = value or utc_now()
    return value.strftime("%Y%m%dT%H%M%SZ")


def make_run_id() -> str:
    return f"{utc_timestamp()}-{uuid.uuid4().hex[:8]}"


def canonical_bson_bytes(value: Any) -> bytes:
    require_database_dependencies()
    serialized = bson_dumps(
        value,
        json_options=CANONICAL_JSON_OPTIONS,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return serialized.encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bson_bytes(value)).hexdigest()


def update_stream_hash(digest: "hashlib._Hash", value: Any) -> None:
    payload = canonical_bson_bytes(value)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def file_sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def normalized_words(value: str) -> tuple[str, ...]:
    words: list[str] = []
    current: list[str] = []
    for char in value.casefold():
        if char.isalnum():
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return tuple(words)


def contains_word_phrase(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(
        haystack[index : index + width] == needle
        for index in range(len(haystack) - width + 1)
    )


def detail_contains_compatibility(detail: str, compatibility: str) -> bool:
    haystack = normalized_words(detail)
    clauses = [
        normalized_words(clause)
        for clause in compatibility.split(";")
        if normalized_words(clause)
    ]
    return bool(clauses) and all(
        contains_word_phrase(haystack, clause) for clause in clauses
    )


def merge_compatibility_detail(
    existing_detail: Optional[str], compatibility: str
) -> tuple[str, str]:
    """Return an idempotent compatibility-detail merge and its action."""

    compatibility = compatibility.strip()
    if not compatibility:
        raise ValueError("compatibility detail must not be blank")

    if not isinstance(existing_detail, str) or not existing_detail.strip():
        return compatibility, "filled"

    if detail_contains_compatibility(existing_detail, compatibility):
        return existing_detail, "already_present"
    return f"{compatibility}; {existing_detail}", "merged"


def historical_v1_normalize_raw_effect_label(value: Any) -> Optional[str]:
    """Reproduce the exact raw-label normalization used by migration v1."""

    if not isinstance(value, str):
        return None
    normalized = " ".join(value.strip().lower().replace("_", " ").split())
    return normalized or None


def historical_v1_normalized_words(value: str) -> tuple[str, ...]:
    """Tokenize detail text exactly as migration v1 did."""

    words: list[str] = []
    current: list[str] = []
    for char in value.casefold():
        if char.isalnum():
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return tuple(words)


def historical_v1_detail_contains_compatibility(
    detail: str,
    compatibility: str,
) -> bool:
    haystack = historical_v1_normalized_words(detail)
    clauses = [
        historical_v1_normalized_words(clause)
        for clause in compatibility.split(";")
        if historical_v1_normalized_words(clause)
    ]
    return bool(clauses) and all(
        contains_word_phrase(haystack, clause) for clause in clauses
    )


def historical_v1_merge_compatibility_detail(
    existing_detail: Optional[str],
    compatibility: str,
) -> str:
    """Reproduce the exact compatibility-detail merge used by migration v1."""

    compatibility = compatibility.strip()
    if not compatibility:
        raise MigrationSafetyError(
            "frozen v1 compatibility detail must not be blank"
        )
    if not isinstance(existing_detail, str) or not existing_detail.strip():
        return compatibility
    if historical_v1_detail_contains_compatibility(
        existing_detail,
        compatibility,
    ):
        return existing_detail
    return f"{compatibility}; {existing_detail}"


def validate_historical_v1_spec() -> None:
    """Reject accidental edits to any frozen migration-v1 ontology input."""

    payload = {
        "allowed_tag_fields": list(HISTORICAL_V1_ALLOWED_TAG_FIELDS),
        "deprecated_effect_details": HISTORICAL_V1_DEPRECATED_EFFECT_DETAILS,
        "redirects": HISTORICAL_V1_REDIRECTS,
        "target_hierarchy": HISTORICAL_V1_TARGET_HIERARCHY,
        "unsafe_broad_redirects": sorted(
            HISTORICAL_V1_UNSAFE_BROAD_REDIRECTS
        ),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    actual_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if actual_hash != HISTORICAL_V1_SPEC_SHA256:
        raise MigrationSafetyError(
            "frozen ontology-v1 migration specification changed: "
            f"expected {HISTORICAL_V1_SPEC_SHA256}, got {actual_hash}"
        )


def effect_redirect_maps(
    ontology: Any,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Return and validate the ontology's explicit redirect partition."""

    safe = getattr(ontology, "SAFE_DEPRECATED_EFFECT_REDIRECTS", None)
    unsafe = getattr(ontology, "UNSAFE_DEPRECATED_EFFECT_REDIRECTS", None)
    declared_union = getattr(ontology, "DEPRECATED_EFFECT_REDIRECTS", None)
    if not isinstance(safe, dict) or not isinstance(unsafe, dict):
        raise MigrationSafetyError(
            "ontology must expose explicit SAFE_DEPRECATED_EFFECT_REDIRECTS "
            "and UNSAFE_DEPRECATED_EFFECT_REDIRECTS mappings"
        )
    overlap = set(safe) & set(unsafe)
    if overlap:
        raise MigrationSafetyError(
            "safe and unsafe redirect labels overlap: "
            + ", ".join(sorted(overlap))
        )
    combined = {**safe, **unsafe}
    if declared_union != combined:
        raise MigrationSafetyError(
            "DEPRECATED_EFFECT_REDIRECTS must equal the safe/unsafe redirect union"
        )
    return safe, unsafe, combined


def historical_v1_effect_index(ontology: Any) -> dict[str, dict[str, str]]:
    """Return the frozen v1 target hierarchy after checking its fingerprint."""

    del ontology  # The ancestry snapshot must never depend on current ontology state.
    validate_historical_v1_spec()
    targets = set(HISTORICAL_V1_REDIRECTS.values())
    if targets != set(HISTORICAL_V1_TARGET_HIERARCHY):
        missing = sorted(targets - set(HISTORICAL_V1_TARGET_HIERARCHY))
        extra = sorted(set(HISTORICAL_V1_TARGET_HIERARCHY) - targets)
        raise MigrationSafetyError(
            "frozen v1 redirect targets and hierarchy differ: "
            f"missing={missing}, extra={extra}"
        )
    target_index = {
        target: {
            "domain": domain,
            "effect": target,
            "parent_effect": parent_effect,
        }
        for target, (domain, parent_effect) in HISTORICAL_V1_TARGET_HIERARCHY.items()
    }
    serialized = json.dumps(
        target_index,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    actual_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if actual_hash != HISTORICAL_V1_TARGET_INDEX_SHA256:
        raise MigrationSafetyError(
            "frozen target hierarchy cannot reproduce migration v1: "
            f"expected {HISTORICAL_V1_TARGET_INDEX_SHA256}, got {actual_hash}"
        )
    return target_index


def transform_tag(
    tag: Any,
    *,
    ontology: Any = None,
) -> tuple[Any, TagTransformResult]:
    """Normalize one stored tag without adding, removing, or reordering keys."""

    ontology = require_extractor(ontology)

    if not isinstance(tag, dict):
        return copy.deepcopy(tag), TagTransformResult(
            outcome="invalid_tag",
            reason="tag is not an object",
        )

    raw_effect = ontology.normalize_raw_effect_label(tag.get("effect"))
    if raw_effect is None:
        return copy.deepcopy(tag), TagTransformResult(
            outcome="invalid_effect",
            reason="stored effect is missing or not a string",
        )

    safe_redirects, unsafe_redirects, _ = effect_redirect_maps(ontology)
    is_safe_redirect = raw_effect in safe_redirects
    is_unsafe_redirect = raw_effect in unsafe_redirects
    if raw_effect in ontology.EFFECT_INDEX:
        canonical_effect = raw_effect
    elif is_unsafe_redirect:
        canonical_effect = unsafe_redirects[raw_effect]
        if canonical_effect not in ontology.EFFECT_INDEX:
            raise MigrationSafetyError(
                f"unsafe redirect target is absent from the current ontology: "
                f"{raw_effect!r} -> {canonical_effect!r}"
            )
        return copy.deepcopy(tag), TagTransformResult(
            outcome="unsafe_redirect",
            raw_effect=raw_effect,
            canonical_effect=canonical_effect,
            reason=(
                "explicitly unsafe redirect cannot prove semantic equivalence "
                "or preserve the original meaning"
            ),
        )
    elif is_safe_redirect:
        canonical_effect = safe_redirects[raw_effect]
    else:
        return copy.deepcopy(tag), TagTransformResult(
            outcome="unsupported_effect",
            raw_effect=raw_effect,
            reason="stored effect is neither canonical nor a deprecated redirect",
        )

    if canonical_effect not in ontology.EFFECT_INDEX:
        return copy.deepcopy(tag), TagTransformResult(
            outcome="unsupported_effect",
            raw_effect=raw_effect,
            canonical_effect=canonical_effect,
            reason="redirect target is absent from the current ontology",
        )

    missing_keys = ALLOWED_TAG_FIELD_SET - set(tag)
    if missing_keys:
        return copy.deepcopy(tag), TagTransformResult(
            outcome="missing_ontology_keys",
            raw_effect=raw_effect,
            canonical_effect=canonical_effect,
            reason=f"missing keys: {', '.join(sorted(missing_keys))}",
        )

    compatibility_detail = (
        ontology.DEPRECATED_EFFECT_DETAILS.get(raw_effect)
        if is_safe_redirect
        else None
    )
    if compatibility_detail and tag.get("detail") is not None and not isinstance(
        tag.get("detail"), str
    ):
        return copy.deepcopy(tag), TagTransformResult(
            outcome="invalid_detail",
            raw_effect=raw_effect,
            canonical_effect=canonical_effect,
            reason="detail is neither null nor a string",
        )

    expected = ontology.EFFECT_INDEX[canonical_effect]
    transformed = copy.deepcopy(tag)
    replacements = {
        "effect": canonical_effect,
        "domain": expected["domain"],
        "parent_effect": expected["parent_effect"],
        "subjective_effect": expected["parent_effect"],
    }
    for field_name, value in replacements.items():
        transformed[field_name] = value

    detail_action = None
    if compatibility_detail:
        transformed["detail"], detail_action = merge_compatibility_detail(
            transformed.get("detail"), compatibility_detail
        )

    changed_fields = tuple(
        field_name
        for field_name in ALLOWED_TAG_FIELDS
        if transformed.get(field_name) != tag.get(field_name)
    )
    return transformed, TagTransformResult(
        outcome="changed" if changed_fields else "unchanged",
        raw_effect=raw_effect,
        canonical_effect=canonical_effect,
        changed_fields=changed_fields,
        detail_action=detail_action,
    )


def document_status(document: Any) -> Optional[str]:
    if not isinstance(document, dict):
        return None
    extraction = document.get("subjective_effect_extraction")
    return extraction.get("status") if isinstance(extraction, dict) else None


def transform_document(
    document: Any,
    *,
    ontology: Any = None,
) -> tuple[Any, list[TagTransformResult]]:
    """Transform a complete document while preserving all structural shape."""

    transformed = copy.deepcopy(document)
    if not isinstance(document, dict) or document_status(document) != "complete":
        return transformed, []

    tags = document.get("subjective_effect_tags")
    if not isinstance(tags, list):
        return transformed, []

    transformed_tags = []
    results = []
    for tag in tags:
        transformed_tag, result = transform_tag(tag, ontology=ontology)
        transformed_tags.append(transformed_tag)
        results.append(result)
    transformed["subjective_effect_tags"] = transformed_tags
    return transformed, results


def historical_v1_transform_tag(
    tag: Any,
    *,
    ontology: Any = None,
) -> Any:
    """Reproduce the exact redirect behavior used by ontology migration v1."""

    ontology = require_extractor(ontology)
    redirects = HISTORICAL_V1_REDIRECTS
    historical_index = historical_v1_effect_index(ontology)
    transformed = copy.deepcopy(tag)
    if not isinstance(tag, dict):
        return transformed

    raw_effect = historical_v1_normalize_raw_effect_label(tag.get("effect"))
    if raw_effect not in redirects:
        return transformed
    if raw_effect in HISTORICAL_V1_UNSAFE_BROAD_REDIRECTS:
        return transformed

    canonical_effect = redirects[raw_effect]
    if HISTORICAL_V1_ALLOWED_TAG_FIELD_SET - set(tag):
        return transformed

    compatibility_detail = HISTORICAL_V1_DEPRECATED_EFFECT_DETAILS.get(raw_effect)
    if (
        compatibility_detail
        and tag.get("detail") is not None
        and not isinstance(tag.get("detail"), str)
    ):
        return transformed

    expected = historical_index[canonical_effect]
    replacements = {
        "effect": canonical_effect,
        "domain": expected["domain"],
        "parent_effect": expected["parent_effect"],
        "subjective_effect": expected["parent_effect"],
    }
    for field_name, value in replacements.items():
        transformed[field_name] = value
    if compatibility_detail:
        transformed["detail"] = historical_v1_merge_compatibility_detail(
            transformed.get("detail"), compatibility_detail
        )
    return transformed


def repair_tag_from_backup(
    current_tag: Any,
    backup_tag: Any,
    *,
    tag_index: int,
    ontology: Any = None,
) -> tuple[Any, RepairTagResult]:
    """Repair one tag only when its exact ontology-v1 ancestry is proven."""

    ontology = require_extractor(ontology)
    redirects = HISTORICAL_V1_REDIRECTS
    raw_effect = (
        historical_v1_normalize_raw_effect_label(backup_tag.get("effect"))
        if isinstance(backup_tag, dict)
        else None
    )
    if raw_effect not in redirects:
        return copy.deepcopy(current_tag), RepairTagResult(
            tag_index=tag_index,
            outcome="not_applicable",
            raw_effect=raw_effect,
        )

    historical = historical_v1_transform_tag(backup_tag, ontology=ontology)
    desired, _ = transform_tag(backup_tag, ontology=ontology)
    historical_effect = (
        historical.get("effect") if isinstance(historical, dict) else None
    )
    desired_effect = desired.get("effect") if isinstance(desired, dict) else None

    if historical == desired:
        return copy.deepcopy(current_tag), RepairTagResult(
            tag_index=tag_index,
            outcome=(
                "already_desired"
                if current_tag == desired
                else "not_applicable_current_changed"
            ),
            raw_effect=raw_effect,
            historical_effect=historical_effect,
            desired_effect=desired_effect,
        )
    if current_tag == desired:
        return copy.deepcopy(current_tag), RepairTagResult(
            tag_index=tag_index,
            outcome="already_desired",
            raw_effect=raw_effect,
            historical_effect=historical_effect,
            desired_effect=desired_effect,
        )
    if current_tag != historical:
        return copy.deepcopy(current_tag), RepairTagResult(
            tag_index=tag_index,
            outcome="lineage_conflict",
            raw_effect=raw_effect,
            historical_effect=historical_effect,
            desired_effect=desired_effect,
            reason=(
                "current tag is neither the exact ontology-v1 transform nor "
                "the desired repaired value"
            ),
        )
    if not isinstance(current_tag, dict) or not isinstance(desired, dict):
        return copy.deepcopy(current_tag), RepairTagResult(
            tag_index=tag_index,
            outcome="lineage_conflict",
            raw_effect=raw_effect,
            historical_effect=historical_effect,
            desired_effect=desired_effect,
            reason="repair candidate is not an object",
        )
    if set(current_tag) != set(desired):
        return copy.deepcopy(current_tag), RepairTagResult(
            tag_index=tag_index,
            outcome="lineage_conflict",
            raw_effect=raw_effect,
            historical_effect=historical_effect,
            desired_effect=desired_effect,
            reason="repair would change the tag key set",
        )

    repaired = copy.deepcopy(current_tag)
    for field_name in HISTORICAL_V1_ALLOWED_TAG_FIELDS:
        repaired[field_name] = desired[field_name]
    changed_fields = tuple(
        field_name
        for field_name in ALLOWED_TAG_FIELDS
        if repaired.get(field_name) != current_tag.get(field_name)
    )
    return repaired, RepairTagResult(
        tag_index=tag_index,
        outcome="repaired",
        raw_effect=raw_effect,
        historical_effect=historical_effect,
        desired_effect=desired_effect,
        changed_fields=changed_fields,
    )


def repair_document_from_backup(
    current: Any,
    backup: Any,
    *,
    ontology: Any = None,
) -> tuple[Any, list[RepairTagResult]]:
    """Overlay deterministic v2 ontology fields onto a current document."""

    ontology = require_extractor(ontology)
    repaired = copy.deepcopy(current)
    if not isinstance(backup, dict) or document_status(backup) != "complete":
        return repaired, []
    if not isinstance(current, dict) or current.get("_id") != backup.get("_id"):
        return repaired, [
            RepairTagResult(
                tag_index=-1,
                outcome="lineage_conflict",
                reason="current and backup document identities do not match",
            )
        ]

    backup_tags = backup.get("subjective_effect_tags")
    current_tags = current.get("subjective_effect_tags")
    if not isinstance(backup_tags, list):
        return repaired, []

    candidate_indices = [
        index
        for index, tag in enumerate(backup_tags)
        if isinstance(tag, dict)
        and historical_v1_normalize_raw_effect_label(tag.get("effect"))
        in HISTORICAL_V1_REDIRECTS
        and historical_v1_transform_tag(tag, ontology=ontology)
        != transform_tag(tag, ontology=ontology)[0]
    ]
    if not candidate_indices:
        return repaired, []
    if document_status(current) != "complete":
        return repaired, [
            RepairTagResult(
                tag_index=index,
                outcome="lineage_conflict",
                raw_effect=historical_v1_normalize_raw_effect_label(
                    backup_tags[index].get("effect")
                ),
                reason="current document is no longer complete",
            )
            for index in candidate_indices
        ]
    if not isinstance(current_tags, list) or len(current_tags) != len(backup_tags):
        return repaired, [
            RepairTagResult(
                tag_index=index,
                outcome="lineage_conflict",
                raw_effect=historical_v1_normalize_raw_effect_label(
                    backup_tags[index].get("effect")
                ),
                reason="current and backup tag counts differ",
            )
            for index in candidate_indices
        ]

    repaired_tags = []
    results = []
    for index, (current_tag, backup_tag) in enumerate(
        zip(current_tags, backup_tags)
    ):
        repaired_tag, result = repair_tag_from_backup(
            current_tag,
            backup_tag,
            tag_index=index,
            ontology=ontology,
        )
        repaired_tags.append(repaired_tag)
        results.append(result)

    if any(result.outcome == "lineage_conflict" for result in results):
        blocked_results = [
            replace(
                result,
                outcome="blocked_by_document_conflict",
                changed_fields=(),
                reason="another repair candidate in this document has a lineage conflict",
            )
            if result.outcome == "repaired"
            else result
            for result in results
        ]
        return copy.deepcopy(current), blocked_results

    repaired["subjective_effect_tags"] = repaired_tags
    return repaired, results


def verify_repair_document_pair(
    before: Any,
    after: Any,
    backup: Any,
    *,
    ontology: Any = None,
) -> list[str]:
    """Return shape, field-scope, or deterministic repair violations."""

    issues: list[str] = []
    if not isinstance(before, dict) or not isinstance(after, dict):
        if before != after:
            issues.append("non-object document changed")
        return issues
    if set(before) != set(after):
        issues.append("document key set changed")
        return issues
    if before.get("_id") != after.get("_id"):
        issues.append("document _id changed")

    for key in before:
        if key != "subjective_effect_tags" and before[key] != after[key]:
            issues.append(f"non-tag document field changed: {key}")

    before_tags = before.get("subjective_effect_tags")
    after_tags = after.get("subjective_effect_tags")
    if not isinstance(before_tags, list) or not isinstance(after_tags, list):
        if before_tags != after_tags:
            issues.append("non-list subjective_effect_tags changed")
    else:
        if len(before_tags) != len(after_tags):
            issues.append("tag count changed")
        for index, (before_tag, after_tag) in enumerate(zip(before_tags, after_tags)):
            if not isinstance(before_tag, dict) or not isinstance(after_tag, dict):
                if before_tag != after_tag:
                    issues.append(f"tag {index}: non-object tag changed")
                continue
            if set(before_tag) != set(after_tag):
                issues.append(f"tag {index}: key set changed")
                continue
            for key in before_tag:
                if key not in ALLOWED_TAG_FIELD_SET and before_tag[key] != after_tag[key]:
                    issues.append(f"tag {index}: forbidden field changed: {key}")

    expected, _ = repair_document_from_backup(
        before,
        backup,
        ontology=ontology,
    )
    if expected != after:
        issues.append("document does not equal the deterministic backup repair")
    return issues


def verify_document_pair(
    before: Any,
    after: Any,
    *,
    ontology: Any = None,
) -> list[str]:
    """Return structural or deterministic-transform violations for one pair."""

    issues: list[str] = []
    if not isinstance(before, dict) or not isinstance(after, dict):
        if before != after:
            issues.append("non-object document changed")
        return issues

    if set(before) != set(after):
        issues.append("document key set changed")
        return issues
    if before.get("_id") != after.get("_id"):
        issues.append("document _id changed")

    if document_status(before) != "complete":
        if before != after:
            issues.append("non-complete document changed")
        return issues

    for key in before:
        if key != "subjective_effect_tags" and before[key] != after[key]:
            issues.append(f"non-tag document field changed: {key}")

    before_tags = before.get("subjective_effect_tags")
    after_tags = after.get("subjective_effect_tags")
    if not isinstance(before_tags, list) or not isinstance(after_tags, list):
        if before_tags != after_tags:
            issues.append("non-list subjective_effect_tags changed")
    else:
        if len(before_tags) != len(after_tags):
            issues.append("tag count changed")
        for index, (before_tag, after_tag) in enumerate(
            zip(before_tags, after_tags)
        ):
            if not isinstance(before_tag, dict) or not isinstance(after_tag, dict):
                if before_tag != after_tag:
                    issues.append(f"tag {index}: non-object tag changed")
                continue
            if set(before_tag) != set(after_tag):
                issues.append(f"tag {index}: key set changed")
                continue
            for key in before_tag:
                if key not in ALLOWED_TAG_FIELD_SET and before_tag[key] != after_tag[key]:
                    issues.append(f"tag {index}: forbidden field changed: {key}")

    expected, _ = transform_document(before, ontology=ontology)
    if expected != after:
        issues.append("document does not equal the deterministic transform")
    return issues


def verify_document_sequences(
    before_documents: Sequence[Any],
    after_documents: Sequence[Any],
    *,
    ontology: Any = None,
) -> list[str]:
    issues: list[str] = []
    if len(before_documents) != len(after_documents):
        issues.append("document count changed")
    for position, (before, after) in enumerate(
        zip(before_documents, after_documents)
    ):
        before_id = before.get("_id") if isinstance(before, dict) else None
        after_id = after.get("_id") if isinstance(after, dict) else None
        if before_id != after_id:
            issues.append(f"position {position}: source _id/order changed")
            continue
        issues.extend(
            f"_id={stringify_id(before_id)}: {issue}"
            for issue in verify_document_pair(before, after, ontology=ontology)
        )
    return issues


def collection_info(database: Database, collection_name: str) -> dict[str, Any]:
    info = next(
        database.list_collections(filter={"name": collection_name}),
        None,
    )
    if info is None:
        raise MigrationSafetyError(
            f"collection {database.name}.{collection_name} does not exist"
        )
    if info.get("type") != "collection":
        raise MigrationSafetyError(
            f"{database.name}.{collection_name} is not a normal collection"
        )
    return info


def collection_options(database: Database, collection_name: str) -> dict[str, Any]:
    return copy.deepcopy(collection_info(database, collection_name).get("options") or {})


def normalized_index_specs(collection: Collection) -> list[dict[str, Any]]:
    specs = []
    for raw_spec in collection.list_indexes():
        spec = dict(raw_spec)
        spec.pop("ns", None)
        spec["key"] = list(spec["key"].items())
        specs.append(spec)
    return sorted(specs, key=lambda item: item.get("name", ""))


def iter_documents(collection: Collection, batch_size: int) -> Iterator[dict]:
    return (
        collection.find({})
        .sort("_id", ASCENDING)
        .batch_size(batch_size)
    )


def snapshot_collection(
    collection: Collection,
    *,
    batch_size: int,
) -> CollectionSnapshot:
    digest = hashlib.sha256()
    document_count = 0
    tag_count = 0
    statuses: Counter[str] = Counter()
    models: Counter[str] = Counter()

    for document in iter_documents(collection, batch_size):
        update_stream_hash(digest, document)
        document_count += 1
        extraction = document.get("subjective_effect_extraction")
        status = extraction.get("status") if isinstance(extraction, dict) else None
        model = extraction.get("model_name") if isinstance(extraction, dict) else None
        statuses[str(status) if status is not None else "<missing>"] += 1
        models[str(model) if model is not None else "<missing>"] += 1
        tags = document.get("subjective_effect_tags")
        if isinstance(tags, list):
            tag_count += len(tags)

    info = collection_info(collection.database, collection.name)
    options = copy.deepcopy(info.get("options") or {})
    indexes = normalized_index_specs(collection)
    raw_uuid = (info.get("info") or {}).get("uuid")
    return CollectionSnapshot(
        collection=collection.name,
        collection_uuid=stringify_id(raw_uuid),
        document_count=document_count,
        content_sha256=digest.hexdigest(),
        options_sha256=sha256_value(options),
        indexes_sha256=sha256_value(indexes),
        status_counts=sorted_counter(statuses),
        model_counts=sorted_counter(models),
        tag_count=tag_count,
        options=options,
        indexes=indexes,
    )


def assert_same_snapshot(
    expected: CollectionSnapshot,
    actual: CollectionSnapshot,
    *,
    context: str,
    clone: bool = False,
) -> None:
    left = expected.clone_identity() if clone else expected.stable_identity()
    right = actual.clone_identity() if clone else actual.stable_identity()
    if left == right:
        return
    differences = {
        key: {"expected": left.get(key), "actual": right.get(key)}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }
    raise MigrationSafetyError(
        f"{context} snapshot mismatch: {json.dumps(differences, default=str, sort_keys=True)}"
    )


def assert_quiescent(
    collection: Collection,
    *,
    batch_size: int,
    quiescence_seconds: float,
) -> CollectionSnapshot:
    first = snapshot_collection(collection, batch_size=batch_size)
    if quiescence_seconds > 0:
        time.sleep(quiescence_seconds)
    second = snapshot_collection(collection, batch_size=batch_size)
    assert_same_snapshot(
        first,
        second,
        context=(
            f"{collection.full_name} changed during the quiescence preflight; "
            "stop every extractor/writer before continuing"
        ),
    )
    return second


def project_collection(
    collection: Collection,
    *,
    batch_size: int,
    max_examples: int,
) -> ProjectionResult:
    source_digest = hashlib.sha256()
    projected_digest = hashlib.sha256()
    metrics = TransformMetrics()
    document_count = 0
    tag_count = 0

    for document in iter_documents(collection, batch_size):
        transformed, tag_results = transform_document(document)
        issues = verify_document_pair(document, transformed)
        if issues:
            raise MigrationSafetyError(
                f"pure transform verification failed for _id={stringify_id(document.get('_id'))}: "
                + "; ".join(issues)
            )
        update_stream_hash(source_digest, document)
        update_stream_hash(projected_digest, transformed)
        metrics.observe_document(
            document,
            transformed,
            tag_results,
            max_examples=max_examples,
        )
        document_count += 1
        tags = document.get("subjective_effect_tags")
        if isinstance(tags, list):
            tag_count += len(tags)

    return ProjectionResult(
        source_sha256=source_digest.hexdigest(),
        projected_sha256=projected_digest.hexdigest(),
        document_count=document_count,
        tag_count=tag_count,
        metrics=metrics,
    )


def project_repair_collection(
    current: Collection,
    backup: Collection,
    *,
    batch_size: int,
    max_examples: int,
) -> RepairProjectionResult:
    """Project a provenance-checked repair without writing either collection."""

    source_digest = hashlib.sha256()
    projected_digest = hashlib.sha256()
    metrics = RepairMetrics()
    document_count = 0
    tag_count = 0

    for document in iter_documents(current, batch_size):
        backup_document = backup.find_one({"_id": document.get("_id")})
        repaired, tag_results = repair_document_from_backup(
            document,
            backup_document,
        )
        issues = verify_repair_document_pair(
            document,
            repaired,
            backup_document,
        )
        if issues:
            raise MigrationSafetyError(
                f"pure repair verification failed for "
                f"_id={stringify_id(document.get('_id'))}: "
                + "; ".join(issues)
            )
        update_stream_hash(source_digest, document)
        update_stream_hash(projected_digest, repaired)
        metrics.observe_document(
            document,
            repaired,
            tag_results,
            backup_matched=backup_document is not None,
            max_examples=max_examples,
        )
        document_count += 1
        tags = document.get("subjective_effect_tags")
        if isinstance(tags, list):
            tag_count += len(tags)

    return RepairProjectionResult(
        source_sha256=source_digest.hexdigest(),
        projected_sha256=projected_digest.hexdigest(),
        document_count=document_count,
        tag_count=tag_count,
        metrics=metrics,
    )


def assert_expected_repair_backup(
    snapshot: CollectionSnapshot,
    expected_sha256: str,
) -> None:
    if snapshot.content_sha256 != expected_sha256:
        raise MigrationSafetyError(
            "repair backup content hash mismatch: "
            f"expected {expected_sha256}, got {snapshot.content_sha256}"
        )


def recreate_indexes(source: Collection, destination: Collection) -> None:
    models = []
    for raw_spec in source.list_indexes():
        spec = dict(raw_spec)
        keys = list(spec.pop("key").items())
        spec.pop("ns", None)
        spec.pop("v", None)
        if spec.get("name") == "_id_":
            continue
        models.append(IndexModel(keys, **spec))
    if models:
        destination.create_indexes(models)


def clone_collection(
    source: Collection,
    destination_name: str,
    *,
    batch_size: int,
    transform: Optional[Callable[[dict], tuple[dict, list[TagTransformResult]]]] = None,
    max_examples: int,
) -> tuple[Collection, TransformMetrics]:
    database = source.database
    if destination_name in database.list_collection_names(
        filter={"name": destination_name}
    ):
        raise MigrationSafetyError(
            f"refusing to overwrite existing collection {database.name}.{destination_name}"
        )

    options = collection_options(database, source.name)
    database.create_collection(destination_name, **options)
    destination = database.get_collection(
        destination_name,
        write_concern=WriteConcern(w="majority", j=True),
    )
    metrics = TransformMetrics()
    pending: list[dict] = []

    for document in iter_documents(source, batch_size):
        if transform is None:
            cloned = copy.deepcopy(document)
            tag_results: list[TagTransformResult] = []
        else:
            cloned, tag_results = transform(document)
            issues = verify_document_pair(document, cloned)
            if issues:
                raise MigrationSafetyError(
                    f"refusing invalid shadow document _id={stringify_id(document.get('_id'))}: "
                    + "; ".join(issues)
                )
        metrics.observe_document(
            document,
            cloned,
            tag_results,
            max_examples=max_examples,
        )
        pending.append(cloned)
        if len(pending) >= batch_size:
            destination.insert_many(pending, ordered=True)
            pending = []

    if pending:
        destination.insert_many(pending, ordered=True)
    recreate_indexes(source, destination)
    return destination, metrics


def clone_repair_collection(
    current: Collection,
    backup: Collection,
    destination_name: str,
    *,
    batch_size: int,
    max_examples: int,
) -> tuple[Collection, RepairMetrics]:
    """Clone current state while overlaying only proven backup-derived repairs."""

    database = current.database
    if destination_name in database.list_collection_names(
        filter={"name": destination_name}
    ):
        raise MigrationSafetyError(
            f"refusing to overwrite existing collection "
            f"{database.name}.{destination_name}"
        )

    options = collection_options(database, current.name)
    database.create_collection(destination_name, **options)
    destination = database.get_collection(
        destination_name,
        write_concern=WriteConcern(w="majority", j=True),
    )
    metrics = RepairMetrics()
    pending: list[dict] = []

    for document in iter_documents(current, batch_size):
        backup_document = backup.find_one({"_id": document.get("_id")})
        repaired, tag_results = repair_document_from_backup(
            document,
            backup_document,
        )
        issues = verify_repair_document_pair(
            document,
            repaired,
            backup_document,
        )
        if issues:
            raise MigrationSafetyError(
                f"refusing invalid repair shadow document "
                f"_id={stringify_id(document.get('_id'))}: "
                + "; ".join(issues)
            )
        metrics.observe_document(
            document,
            repaired,
            tag_results,
            backup_matched=backup_document is not None,
            max_examples=max_examples,
        )
        pending.append(repaired)
        if len(pending) >= batch_size:
            destination.insert_many(pending, ordered=True)
            pending = []

    if pending:
        destination.insert_many(pending, ordered=True)
    recreate_indexes(current, destination)
    return destination, metrics


def verify_exact_collections(
    source: Collection,
    clone: Collection,
    *,
    batch_size: int,
) -> None:
    issues: list[str] = []
    sentinel = object()
    source_cursor = iter_documents(source, batch_size)
    clone_cursor = iter_documents(clone, batch_size)
    for position, (before, after) in enumerate(
        itertools.zip_longest(source_cursor, clone_cursor, fillvalue=sentinel)
    ):
        if before is sentinel or after is sentinel:
            issues.append(f"position {position}: document count differs")
            break
        if before != after:
            issues.append(
                f"position {position}: exact backup differs at _id={stringify_id(before.get('_id'))}"
            )
        if len(issues) >= MAX_VERIFICATION_ISSUES:
            break
    if issues:
        raise MigrationSafetyError("exact clone verification failed: " + "; ".join(issues))

    source_snapshot = snapshot_collection(source, batch_size=batch_size)
    clone_snapshot = snapshot_collection(clone, batch_size=batch_size)
    assert_same_snapshot(
        source_snapshot,
        clone_snapshot,
        context=f"exact clone {clone.full_name}",
        clone=True,
    )


def verify_transformed_shadow(
    source: Collection,
    shadow: Collection,
    *,
    batch_size: int,
) -> None:
    issues: list[str] = []
    sentinel = object()
    source_cursor = iter_documents(source, batch_size)
    shadow_cursor = iter_documents(shadow, batch_size)
    for position, (before, after) in enumerate(
        itertools.zip_longest(source_cursor, shadow_cursor, fillvalue=sentinel)
    ):
        if before is sentinel or after is sentinel:
            issues.append(f"position {position}: document count differs")
            break
        before_id = before.get("_id")
        after_id = after.get("_id")
        if before_id != after_id:
            issues.append(
                f"position {position}: source _id/order differs ({before_id!r} != {after_id!r})"
            )
            continue
        issues.extend(
            f"_id={stringify_id(before_id)}: {issue}"
            for issue in verify_document_pair(before, after)
        )
        if len(issues) >= MAX_VERIFICATION_ISSUES:
            break
    if issues:
        raise MigrationSafetyError(
            "transformed shadow verification failed: " + "; ".join(issues)
        )

    source_snapshot = snapshot_collection(source, batch_size=batch_size)
    shadow_snapshot = snapshot_collection(shadow, batch_size=batch_size)
    if source_snapshot.options_sha256 != shadow_snapshot.options_sha256:
        raise MigrationSafetyError("shadow collection options differ from source")
    if source_snapshot.indexes_sha256 != shadow_snapshot.indexes_sha256:
        raise MigrationSafetyError("shadow collection indexes differ from source")
    if source_snapshot.document_count != shadow_snapshot.document_count:
        raise MigrationSafetyError("shadow document count differs from source")
    if source_snapshot.tag_count != shadow_snapshot.tag_count:
        raise MigrationSafetyError("shadow tag count differs from source")


def verify_repair_shadow(
    current: Collection,
    backup: Collection,
    shadow: Collection,
    *,
    batch_size: int,
) -> None:
    issues: list[str] = []
    sentinel = object()
    current_cursor = iter_documents(current, batch_size)
    shadow_cursor = iter_documents(shadow, batch_size)
    for position, (before, after) in enumerate(
        itertools.zip_longest(current_cursor, shadow_cursor, fillvalue=sentinel)
    ):
        if before is sentinel or after is sentinel:
            issues.append(f"position {position}: document count differs")
            break
        before_id = before.get("_id")
        if before_id != after.get("_id"):
            issues.append(f"position {position}: current _id/order differs")
            continue
        backup_document = backup.find_one({"_id": before_id})
        issues.extend(
            f"_id={stringify_id(before_id)}: {issue}"
            for issue in verify_repair_document_pair(
                before,
                after,
                backup_document,
            )
        )
        if len(issues) >= MAX_VERIFICATION_ISSUES:
            break
    if issues:
        raise MigrationSafetyError(
            "repair shadow verification failed: " + "; ".join(issues)
        )

    current_snapshot = snapshot_collection(current, batch_size=batch_size)
    shadow_snapshot = snapshot_collection(shadow, batch_size=batch_size)
    if current_snapshot.options_sha256 != shadow_snapshot.options_sha256:
        raise MigrationSafetyError("repair shadow options differ from current target")
    if current_snapshot.indexes_sha256 != shadow_snapshot.indexes_sha256:
        raise MigrationSafetyError("repair shadow indexes differ from current target")
    if current_snapshot.document_count != shadow_snapshot.document_count:
        raise MigrationSafetyError(
            "repair shadow document count differs from current target"
        )
    if current_snapshot.tag_count != shadow_snapshot.tag_count:
        raise MigrationSafetyError("repair shadow tag count differs from current target")


def atomic_replace_collection(
    client: MongoClient,
    *,
    database_name: str,
    replacement_name: str,
    target_name: str,
) -> None:
    client.admin.command(
        {
            "renameCollection": f"{database_name}.{replacement_name}",
            "to": f"{database_name}.{target_name}",
            "dropTarget": True,
            "writeConcern": {"w": "majority", "j": True},
        }
    )


def observe_cutover_collection(
    database: Database,
    collection_name: str,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Return a non-throwing collection observation for cutover recovery."""

    observation: dict[str, Any] = {"collection": collection_name}
    try:
        names = database.list_collection_names(
            filter={"name": collection_name}
        )
        if collection_name not in names:
            observation["exists"] = False
            return observation
        snapshot = snapshot_collection(
            database[collection_name],
            batch_size=batch_size,
        )
    except Exception as exc:
        observation.update(
            {
                "exists": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return observation

    observation.update(
        {
            "exists": True,
            "snapshot": snapshot.to_dict(),
        }
    )
    return observation


def observation_matches_snapshot(
    observation: dict[str, Any],
    expected: CollectionSnapshot,
) -> bool:
    if observation.get("exists") is not True:
        return False
    actual = observation.get("snapshot")
    if not isinstance(actual, dict):
        return False
    return all(
        actual.get(key) == value
        for key, value in expected.stable_identity().items()
    )


def classify_cutover_outcome(
    observations: dict[str, dict[str, Any]],
    *,
    expected_target_before: CollectionSnapshot,
    expected_replacement: CollectionSnapshot,
    expected_retained_backup: CollectionSnapshot,
) -> tuple[str, dict[str, bool]]:
    """Classify an atomic rename only from exact names and snapshot identities."""

    target = observations["target"]
    replacement = observations["replacement"]
    retained_backup = observations["retained_backup"]
    matches = {
        "target_before": observation_matches_snapshot(
            target,
            expected_target_before,
        ),
        "target_after": observation_matches_snapshot(
            target,
            expected_replacement,
        ),
        "replacement_before": observation_matches_snapshot(
            replacement,
            expected_replacement,
        ),
        "replacement_absent": replacement.get("exists") is False,
        "retained_backup": observation_matches_snapshot(
            retained_backup,
            expected_retained_backup,
        ),
    }
    if (
        matches["target_after"]
        and matches["replacement_absent"]
    ):
        return "committed", matches
    if (
        matches["target_before"]
        and matches["replacement_before"]
    ):
        return "not_committed", matches
    return "indeterminate", matches


def reconcile_cutover(
    database: Database,
    *,
    target_name: str,
    replacement_name: str,
    retained_backup_name: str,
    expected_target_before: CollectionSnapshot,
    expected_replacement: CollectionSnapshot,
    expected_retained_backup: CollectionSnapshot,
    batch_size: int,
) -> dict[str, Any]:
    """Observe all recovery collections and classify the rename outcome."""

    observations = {
        "target": observe_cutover_collection(
            database,
            target_name,
            batch_size=batch_size,
        ),
        "replacement": observe_cutover_collection(
            database,
            replacement_name,
            batch_size=batch_size,
        ),
        "retained_backup": observe_cutover_collection(
            database,
            retained_backup_name,
            batch_size=batch_size,
        ),
    }
    outcome, matches = classify_cutover_outcome(
        observations,
        expected_target_before=expected_target_before,
        expected_replacement=expected_replacement,
        expected_retained_backup=expected_retained_backup,
    )
    retained_observation = observations["retained_backup"]
    if matches["retained_backup"]:
        retained_backup_status = "verified"
    elif retained_observation.get("exists") is None:
        retained_backup_status = "indeterminate"
    else:
        retained_backup_status = "degraded"
    return {
        "outcome": outcome,
        "retained_backup_status": retained_backup_status,
        "reconciled_at": utc_now().isoformat(),
        "matches": matches,
        "observations": observations,
    }


def persist_cutover_attempt(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    database_name: str,
    target_name: str,
    replacement_name: str,
    retained_backup_name: str,
    expected_target_before: CollectionSnapshot,
    expected_replacement: CollectionSnapshot,
    expected_retained_backup: CollectionSnapshot,
) -> None:
    """Durably record enough state to reconcile an interrupted rename."""

    names = {target_name, replacement_name, retained_backup_name}
    if len(names) != 3:
        raise MigrationSafetyError(
            "cutover target, replacement, and retained backup must be distinct"
        )
    manifest.update(
        {
            "status": "cutover_attempted",
            "cutover_attempted": True,
            "cutover_outcome": "indeterminate",
            "writes_performed": True,
            "cutover": {
                "attempted_at": utc_now().isoformat(),
                "database": database_name,
                "target_collection": target_name,
                "replacement_collection": replacement_name,
                "retained_backup_collection": retained_backup_name,
                "expected": {
                    "target_before": expected_target_before.stable_identity(),
                    "replacement_before_and_target_after": (
                        expected_replacement.stable_identity()
                    ),
                    "retained_backup": (
                        expected_retained_backup.stable_identity()
                    ),
                },
            },
        }
    )
    write_manifest(manifest_path, manifest)


def persist_cutover_reconciliation(
    database: Database,
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    trigger: str,
    target_name: str,
    replacement_name: str,
    retained_backup_name: str,
    expected_target_before: CollectionSnapshot,
    expected_replacement: CollectionSnapshot,
    expected_retained_backup: CollectionSnapshot,
    batch_size: int,
    failure: Optional[BaseException] = None,
) -> dict[str, Any]:
    reconciliation = reconcile_cutover(
        database,
        target_name=target_name,
        replacement_name=replacement_name,
        retained_backup_name=retained_backup_name,
        expected_target_before=expected_target_before,
        expected_replacement=expected_replacement,
        expected_retained_backup=expected_retained_backup,
        batch_size=batch_size,
    )
    reconciliation["trigger"] = trigger
    if failure is not None:
        reconciliation["failure"] = {
            "error_type": type(failure).__name__,
            "error": str(failure),
        }
    outcome = reconciliation["outcome"]
    manifest.setdefault("cutover", {})["reconciliation"] = reconciliation
    manifest.update(
        {
            "status": f"cutover_{outcome}",
            "cutover_outcome": outcome,
            "writes_performed": True,
        }
    )
    write_manifest(manifest_path, manifest)
    return reconciliation


def execute_verified_cutover(
    client: MongoClient,
    database: Database,
    *,
    target_name: str,
    replacement_name: str,
    retained_backup_name: str,
    expected_target_before: CollectionSnapshot,
    expected_replacement: CollectionSnapshot,
    expected_retained_backup: CollectionSnapshot,
    batch_size: int,
    manifest: dict[str, Any],
    manifest_path: Path,
    verify_after: Callable[[], Any],
) -> Any:
    """Persist intent, rename, verify, and reconcile every observable outcome."""

    persist_cutover_attempt(
        manifest,
        manifest_path,
        database_name=database.name,
        target_name=target_name,
        replacement_name=replacement_name,
        retained_backup_name=retained_backup_name,
        expected_target_before=expected_target_before,
        expected_replacement=expected_replacement,
        expected_retained_backup=expected_retained_backup,
    )
    phase = "rename_command_exception"
    try:
        atomic_replace_collection(
            client,
            database_name=database.name,
            replacement_name=replacement_name,
            target_name=target_name,
        )
        phase = "command_acknowledgement_manifest_failure"
        manifest.setdefault("cutover", {})[
            "command_acknowledged_at"
        ] = utc_now().isoformat()
        manifest.update(
            {
                "status": "cutover_committed_verification_pending",
                "cutover_outcome": "committed",
                "writes_performed": True,
            }
        )
        write_manifest(manifest_path, manifest)
        phase = "post_cutover_verification_failure"
        verified = verify_after()
    except BaseException as exc:
        try:
            persist_cutover_reconciliation(
                database,
                manifest,
                manifest_path,
                trigger=phase,
                target_name=target_name,
                replacement_name=replacement_name,
                retained_backup_name=retained_backup_name,
                expected_target_before=expected_target_before,
                expected_replacement=expected_replacement,
                expected_retained_backup=expected_retained_backup,
                batch_size=batch_size,
                failure=exc,
            )
        except Exception as reconciliation_error:
            manifest["cutover_reconciliation_error"] = {
                "error_type": type(reconciliation_error).__name__,
                "error": str(reconciliation_error),
            }
        raise

    reconciliation = persist_cutover_reconciliation(
        database,
        manifest,
        manifest_path,
        trigger="post_cutover_verification_complete",
        target_name=target_name,
        replacement_name=replacement_name,
        retained_backup_name=retained_backup_name,
        expected_target_before=expected_target_before,
        expected_replacement=expected_replacement,
        expected_retained_backup=expected_retained_backup,
        batch_size=batch_size,
    )
    if reconciliation["outcome"] != "committed":
        raise MigrationSafetyError(
            "post-cutover reconciliation did not prove a committed rename: "
            f"{reconciliation['outcome']}"
        )
    return verified


def safe_collection_component(value: str, limit: int = 52) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value
    )
    return cleaned[:limit] or "collection"


def generated_collection_name(target_name: str, role: str, run_id: str) -> str:
    return f"{safe_collection_component(target_name)}__{role}__{run_id}"


def manifest_base(
    *,
    mode: str,
    run_id: str,
    database_name: str,
    target_name: str,
) -> dict[str, Any]:
    ontology = require_extractor()
    script_path = Path(__file__).resolve()
    extractor_path = Path(ontology.__file__).resolve()
    return {
        "manifest_version": MANIFEST_VERSION,
        "mode": mode,
        "run_id": run_id,
        "started_at": utc_now().isoformat(),
        "database": database_name,
        "target_collection": target_name,
        "status": "started",
        "allowed_tag_fields": list(ALLOWED_TAG_FIELDS),
        "ontology_hash": getattr(ontology, "ONTOLOGY_HASH", None),
        "script_sha256": file_sha256(script_path),
        "extractor_sha256": file_sha256(extractor_path),
    }


def default_manifest_path(mode: str, run_id: str) -> Path:
    directory = Path(__file__).resolve().parent / "migration_manifests"
    return directory / f"{mode}-{run_id}.json"


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            manifest,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def emit_manifest(manifest: dict[str, Any]) -> None:
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def persist_database_write_phase(
    manifest: dict[str, Any],
    manifest_path: Path,
    phase: str,
) -> None:
    manifest.setdefault("database_writes_started_at", utc_now().isoformat())
    manifest.update(
        {
            "status": phase,
            "last_database_write_phase": phase,
            "writes_performed": True,
        }
    )
    write_manifest(manifest_path, manifest)


def verify_projection_against_snapshot(
    projection: ProjectionResult | RepairProjectionResult,
    snapshot: CollectionSnapshot,
    *,
    context: str,
) -> None:
    if projection.source_sha256 != snapshot.content_sha256:
        raise MigrationSafetyError(
            f"{context}: source changed between snapshot and projection"
        )
    if projection.document_count != snapshot.document_count:
        raise MigrationSafetyError(f"{context}: projected document count changed")
    if projection.tag_count != snapshot.tag_count:
        raise MigrationSafetyError(f"{context}: projected tag count changed")


def run_dry_run(
    collection: Collection,
    *,
    batch_size: int,
    quiescence_seconds: float,
    max_examples: int,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    progress("Running read-only quiescence preflight ...")
    stable_snapshot = assert_quiescent(
        collection,
        batch_size=batch_size,
        quiescence_seconds=quiescence_seconds,
    )
    progress("Computing deterministic migration projection ...")
    projection = project_collection(
        collection,
        batch_size=batch_size,
        max_examples=max_examples,
    )
    verify_projection_against_snapshot(
        projection,
        stable_snapshot,
        context="dry-run projection",
    )
    final_snapshot = snapshot_collection(collection, batch_size=batch_size)
    assert_same_snapshot(
        stable_snapshot,
        final_snapshot,
        context="source changed during dry-run projection",
    )

    manifest.update(
        {
            "status": "dry_run_complete",
            "completed_at": utc_now().isoformat(),
            "source_snapshot": stable_snapshot.to_dict(),
            "projection": projection.to_dict(),
            "writes_performed": False,
            "operator_notice": (
                "Dry-run hashes cannot prove that an idle writer will remain stopped; "
                "stop every extractor before --apply."
            ),
        }
    )
    return manifest


def run_repair_dry_run(
    target: Collection,
    backup: Collection,
    *,
    expected_backup_sha256: str,
    batch_size: int,
    quiescence_seconds: float,
    max_examples: int,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    progress("Running read-only repair quiescence preflight ...")
    target_snapshot = assert_quiescent(
        target,
        batch_size=batch_size,
        quiescence_seconds=quiescence_seconds,
    )
    backup_snapshot = assert_quiescent(
        backup,
        batch_size=batch_size,
        quiescence_seconds=quiescence_seconds,
    )
    assert_expected_repair_backup(backup_snapshot, expected_backup_sha256)

    progress("Computing provenance-checked repair projection ...")
    projection = project_repair_collection(
        target,
        backup,
        batch_size=batch_size,
        max_examples=max_examples,
    )
    verify_projection_against_snapshot(
        projection,
        target_snapshot,
        context="repair dry-run projection",
    )

    final_target_snapshot = snapshot_collection(target, batch_size=batch_size)
    final_backup_snapshot = snapshot_collection(backup, batch_size=batch_size)
    assert_same_snapshot(
        target_snapshot,
        final_target_snapshot,
        context="target changed during repair dry-run projection",
    )
    assert_same_snapshot(
        backup_snapshot,
        final_backup_snapshot,
        context="repair backup changed during projection",
    )

    has_conflicts = projection.metrics.conflict_count != 0
    manifest.update(
        {
            "status": (
                "repair_dry_run_conflicts"
                if has_conflicts
                else "repair_dry_run_complete"
            ),
            "completed_at": utc_now().isoformat(),
            "target_snapshot": target_snapshot.to_dict(),
            "repair_source_snapshot": backup_snapshot.to_dict(),
            "expected_repair_backup_sha256": expected_backup_sha256,
            "repair_projection": projection.to_dict(),
            "writes_performed": False,
            "operator_notice": (
                "Repair apply is blocked unless conflict_count is zero. "
                "Stop every extractor/writer before --apply."
            ),
        }
    )
    return manifest


def run_repair_apply(
    client: MongoClient,
    target: Collection,
    backup: Collection,
    *,
    expected_backup_sha256: str,
    batch_size: int,
    quiescence_seconds: float,
    max_examples: int,
    run_id: str,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    database = target.database
    pre_repair_backup_name = generated_collection_name(
        target.name, "pre_repair_backup", run_id
    )
    repair_shadow_name = generated_collection_name(
        target.name, "repair_shadow", run_id
    )
    manifest.update(
        {
            "repair_source_collection": backup.name,
            "expected_repair_backup_sha256": expected_backup_sha256,
            "pre_repair_backup_collection": pre_repair_backup_name,
            "repair_shadow_collection": repair_shadow_name,
            "operator_notice": (
                "--apply asserts that all extractor/writer processes were stopped."
            ),
        }
    )
    write_manifest(manifest_path, manifest)

    progress("Running repair apply quiescence preflight ...")
    target_snapshot = assert_quiescent(
        target,
        batch_size=batch_size,
        quiescence_seconds=quiescence_seconds,
    )
    backup_snapshot = assert_quiescent(
        backup,
        batch_size=batch_size,
        quiescence_seconds=quiescence_seconds,
    )
    assert_expected_repair_backup(backup_snapshot, expected_backup_sha256)
    projection = project_repair_collection(
        target,
        backup,
        batch_size=batch_size,
        max_examples=max_examples,
    )
    verify_projection_against_snapshot(
        projection,
        target_snapshot,
        context="repair apply projection",
    )
    manifest.update(
        {
            "target_snapshot": target_snapshot.to_dict(),
            "repair_source_snapshot": backup_snapshot.to_dict(),
            "repair_projection": projection.to_dict(),
            "status": "repair_preflight_complete",
        }
    )
    write_manifest(manifest_path, manifest)
    if projection.metrics.conflict_count:
        raise MigrationSafetyError(
            "repair projection has "
            f"{projection.metrics.conflict_count} ancestry conflict(s); "
            "no repair collections were written"
        )

    persist_database_write_phase(
        manifest,
        manifest_path,
        "pre_repair_backup_creation_started",
    )
    progress(
        f"Retaining current target as "
        f"{database.name}.{pre_repair_backup_name} ..."
    )
    pre_repair_backup, _ = clone_collection(
        target,
        pre_repair_backup_name,
        batch_size=batch_size,
        transform=None,
        max_examples=max_examples,
    )
    verify_exact_collections(target, pre_repair_backup, batch_size=batch_size)
    pre_repair_backup_snapshot = snapshot_collection(
        pre_repair_backup,
        batch_size=batch_size,
    )
    assert_same_snapshot(
        target_snapshot,
        pre_repair_backup_snapshot,
        context="pre-repair retained backup",
        clone=True,
    )
    manifest.update(
        {
            "pre_repair_backup_snapshot": pre_repair_backup_snapshot.to_dict(),
            "status": "pre_repair_backup_verified",
        }
    )
    write_manifest(manifest_path, manifest)

    persist_database_write_phase(
        manifest,
        manifest_path,
        "repair_shadow_creation_started",
    )
    progress(f"Creating repair shadow {database.name}.{repair_shadow_name} ...")
    shadow, shadow_metrics = clone_repair_collection(
        target,
        backup,
        repair_shadow_name,
        batch_size=batch_size,
        max_examples=max_examples,
    )
    if shadow_metrics.to_dict() != projection.metrics.to_dict():
        raise MigrationSafetyError(
            "repair shadow metrics differ from the dry projection"
        )
    verify_repair_shadow(target, backup, shadow, batch_size=batch_size)
    shadow_snapshot = snapshot_collection(shadow, batch_size=batch_size)
    if shadow_snapshot.content_sha256 != projection.projected_sha256:
        raise MigrationSafetyError(
            "repair shadow content hash differs from projected content hash"
        )
    manifest.update(
        {
            "repair_shadow_snapshot": shadow_snapshot.to_dict(),
            "status": "repair_shadow_verified",
        }
    )
    write_manifest(manifest_path, manifest)

    progress("Rechecking target and repair source immediately before cutover ...")
    final_target_snapshot = snapshot_collection(target, batch_size=batch_size)
    final_backup_snapshot = snapshot_collection(backup, batch_size=batch_size)
    assert_same_snapshot(
        target_snapshot,
        final_target_snapshot,
        context="target changed while repair copies were being built",
    )
    assert_same_snapshot(
        backup_snapshot,
        final_backup_snapshot,
        context="repair source changed while repair shadow was being built",
    )

    def verify_repaired_target() -> tuple[
        CollectionSnapshot,
        RepairProjectionResult,
    ]:
        repaired_target = database[target.name]
        repaired_snapshot = snapshot_collection(
            repaired_target,
            batch_size=batch_size,
        )
        if repaired_snapshot.content_sha256 != projection.projected_sha256:
            raise MigrationSafetyError(
                "post-cutover target hash differs from the verified repair "
                "shadow; retained pre-repair backup is "
                f"{database.name}.{pre_repair_backup_name}"
            )
        if repaired_snapshot.options_sha256 != target_snapshot.options_sha256:
            raise MigrationSafetyError("post-repair collection options changed")
        if repaired_snapshot.indexes_sha256 != target_snapshot.indexes_sha256:
            raise MigrationSafetyError("post-repair collection indexes changed")
        if repaired_snapshot.document_count != target_snapshot.document_count:
            raise MigrationSafetyError("post-repair document count changed")
        if repaired_snapshot.tag_count != target_snapshot.tag_count:
            raise MigrationSafetyError("post-repair tag count changed")

        second_projection = project_repair_collection(
            repaired_target,
            backup,
            batch_size=batch_size,
            max_examples=max_examples,
        )
        if second_projection.metrics.documents_changed != 0:
            raise MigrationSafetyError(
                "post-repair idempotence projection still proposes changes"
            )
        if second_projection.metrics.conflict_count != 0:
            raise MigrationSafetyError(
                "post-repair idempotence projection reports ancestry conflicts"
            )
        return repaired_snapshot, second_projection

    progress("Atomically replacing target with the verified repair shadow ...")
    repaired_snapshot, second_projection = execute_verified_cutover(
        client,
        database,
        target_name=target.name,
        replacement_name=repair_shadow_name,
        retained_backup_name=pre_repair_backup_name,
        expected_target_before=target_snapshot,
        expected_replacement=shadow_snapshot,
        expected_retained_backup=pre_repair_backup_snapshot,
        batch_size=batch_size,
        manifest=manifest,
        manifest_path=manifest_path,
        verify_after=verify_repaired_target,
    )
    manifest.update(
        {
            "status": "repaired_and_verified",
            "completed_at": utc_now().isoformat(),
            "target_snapshot_after": repaired_snapshot.to_dict(),
            "idempotence_projection": second_projection.to_dict(),
            "retained_pre_repair_backup": pre_repair_backup_name,
            "writes_performed": True,
        }
    )
    write_manifest(manifest_path, manifest)
    return manifest


def run_apply(
    client: MongoClient,
    collection: Collection,
    *,
    batch_size: int,
    quiescence_seconds: float,
    max_examples: int,
    run_id: str,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    database = collection.database
    backup_name = generated_collection_name(
        collection.name, "ontology_backup", run_id
    )
    shadow_name = generated_collection_name(
        collection.name, "ontology_shadow", run_id
    )
    manifest.update(
        {
            "backup_collection": backup_name,
            "shadow_collection": shadow_name,
            "operator_notice": (
                "--apply asserts that all extractor/writer processes were stopped."
            ),
        }
    )
    write_manifest(manifest_path, manifest)

    progress("Running apply quiescence preflight ...")
    stable_snapshot = assert_quiescent(
        collection,
        batch_size=batch_size,
        quiescence_seconds=quiescence_seconds,
    )
    projection = project_collection(
        collection,
        batch_size=batch_size,
        max_examples=max_examples,
    )
    verify_projection_against_snapshot(
        projection,
        stable_snapshot,
        context="apply projection",
    )
    manifest.update(
        {
            "source_snapshot": stable_snapshot.to_dict(),
            "projection": projection.to_dict(),
            "status": "preflight_complete",
        }
    )
    write_manifest(manifest_path, manifest)

    persist_database_write_phase(
        manifest,
        manifest_path,
        "backup_creation_started",
    )
    progress(f"Creating retained exact backup {database.name}.{backup_name} ...")
    backup, _ = clone_collection(
        collection,
        backup_name,
        batch_size=batch_size,
        transform=None,
        max_examples=max_examples,
    )
    verify_exact_collections(collection, backup, batch_size=batch_size)
    backup_snapshot = snapshot_collection(backup, batch_size=batch_size)
    assert_same_snapshot(
        stable_snapshot,
        backup_snapshot,
        context="retained backup",
        clone=True,
    )
    manifest.update(
        {
            "backup_snapshot": backup_snapshot.to_dict(),
            "status": "backup_verified",
        }
    )
    write_manifest(manifest_path, manifest)

    persist_database_write_phase(
        manifest,
        manifest_path,
        "shadow_creation_started",
    )
    progress(f"Creating transformed shadow {database.name}.{shadow_name} ...")
    shadow, shadow_metrics = clone_collection(
        collection,
        shadow_name,
        batch_size=batch_size,
        transform=transform_document,
        max_examples=max_examples,
    )
    if shadow_metrics.to_dict() != projection.metrics.to_dict():
        raise MigrationSafetyError(
            "shadow transform metrics differ from the dry projection"
        )
    verify_transformed_shadow(collection, shadow, batch_size=batch_size)
    shadow_snapshot = snapshot_collection(shadow, batch_size=batch_size)
    if shadow_snapshot.content_sha256 != projection.projected_sha256:
        raise MigrationSafetyError(
            "shadow content hash differs from the projected content hash"
        )
    manifest.update(
        {
            "shadow_snapshot": shadow_snapshot.to_dict(),
            "status": "shadow_verified",
        }
    )
    write_manifest(manifest_path, manifest)

    progress("Rechecking source immediately before atomic cutover ...")
    final_source_snapshot = snapshot_collection(collection, batch_size=batch_size)
    assert_same_snapshot(
        stable_snapshot,
        final_source_snapshot,
        context="source changed while backup/shadow were being built",
    )

    def verify_migrated_target() -> tuple[
        CollectionSnapshot,
        ProjectionResult,
    ]:
        migrated = database[collection.name]
        migrated_snapshot = snapshot_collection(
            migrated,
            batch_size=batch_size,
        )
        if migrated_snapshot.content_sha256 != projection.projected_sha256:
            raise MigrationSafetyError(
                "post-cutover target hash differs from the verified shadow "
                f"hash; retained backup is {database.name}.{backup_name}"
            )
        if migrated_snapshot.options_sha256 != stable_snapshot.options_sha256:
            raise MigrationSafetyError("post-cutover collection options changed")
        if migrated_snapshot.indexes_sha256 != stable_snapshot.indexes_sha256:
            raise MigrationSafetyError("post-cutover collection indexes changed")
        if migrated_snapshot.document_count != stable_snapshot.document_count:
            raise MigrationSafetyError("post-cutover document count changed")
        if migrated_snapshot.tag_count != stable_snapshot.tag_count:
            raise MigrationSafetyError("post-cutover tag count changed")

        second_projection = project_collection(
            migrated,
            batch_size=batch_size,
            max_examples=max_examples,
        )
        if second_projection.metrics.documents_changed != 0:
            raise MigrationSafetyError(
                "post-cutover idempotence check still proposes document changes"
            )
        return migrated_snapshot, second_projection

    progress("Atomically replacing the target with the verified shadow ...")
    migrated_snapshot, second_projection = execute_verified_cutover(
        client,
        database,
        target_name=collection.name,
        replacement_name=shadow_name,
        retained_backup_name=backup_name,
        expected_target_before=stable_snapshot,
        expected_replacement=shadow_snapshot,
        expected_retained_backup=backup_snapshot,
        batch_size=batch_size,
        manifest=manifest,
        manifest_path=manifest_path,
        verify_after=verify_migrated_target,
    )
    manifest.update(
        {
            "status": "applied_and_verified",
            "completed_at": utc_now().isoformat(),
            "target_snapshot_after": migrated_snapshot.to_dict(),
            "idempotence_projection": second_projection.to_dict(),
            "retained_backup_collection": backup_name,
            "writes_performed": True,
        }
    )
    write_manifest(manifest_path, manifest)
    return manifest


def run_rollback(
    client: MongoClient,
    target: Collection,
    *,
    backup_name: str,
    batch_size: int,
    quiescence_seconds: float,
    max_examples: int,
    run_id: str,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    database = target.database
    if "." in backup_name or "\x00" in backup_name:
        raise MigrationSafetyError("rollback backup must be a collection name, not a namespace")
    if backup_name == target.name:
        raise MigrationSafetyError("rollback backup must not be the live target collection")
    backup = database[backup_name]
    collection_info(database, backup_name)

    current_backup_name = generated_collection_name(
        target.name, "pre_rollback_backup", run_id
    )
    restore_shadow_name = generated_collection_name(
        target.name, "rollback_shadow", run_id
    )
    manifest.update(
        {
            "rollback_source_collection": backup_name,
            "pre_rollback_backup_collection": current_backup_name,
            "restore_shadow_collection": restore_shadow_name,
        }
    )
    write_manifest(manifest_path, manifest)

    progress("Running rollback quiescence preflight on current target ...")
    current_snapshot = assert_quiescent(
        target,
        batch_size=batch_size,
        quiescence_seconds=quiescence_seconds,
    )
    backup_snapshot = assert_quiescent(
        backup,
        batch_size=batch_size,
        quiescence_seconds=quiescence_seconds,
    )
    manifest.update(
        {
            "target_snapshot_before": current_snapshot.to_dict(),
            "rollback_source_snapshot": backup_snapshot.to_dict(),
            "status": "rollback_preflight_complete",
        }
    )
    write_manifest(manifest_path, manifest)

    persist_database_write_phase(
        manifest,
        manifest_path,
        "pre_rollback_backup_creation_started",
    )
    progress(
        f"Retaining current target as {database.name}.{current_backup_name} ..."
    )
    current_backup, _ = clone_collection(
        target,
        current_backup_name,
        batch_size=batch_size,
        transform=None,
        max_examples=max_examples,
    )
    verify_exact_collections(target, current_backup, batch_size=batch_size)
    current_backup_snapshot = snapshot_collection(
        current_backup,
        batch_size=batch_size,
    )
    assert_same_snapshot(
        current_snapshot,
        current_backup_snapshot,
        context="pre-rollback retained backup",
        clone=True,
    )
    manifest.update(
        {
            "pre_rollback_backup_snapshot": current_backup_snapshot.to_dict(),
            "status": "pre_rollback_backup_verified",
        }
    )
    write_manifest(manifest_path, manifest)

    persist_database_write_phase(
        manifest,
        manifest_path,
        "rollback_shadow_creation_started",
    )
    progress(
        f"Creating rollback shadow from {database.name}.{backup_name} ..."
    )
    restore_shadow, _ = clone_collection(
        backup,
        restore_shadow_name,
        batch_size=batch_size,
        transform=None,
        max_examples=max_examples,
    )
    verify_exact_collections(backup, restore_shadow, batch_size=batch_size)
    restore_snapshot = snapshot_collection(restore_shadow, batch_size=batch_size)
    assert_same_snapshot(
        backup_snapshot,
        restore_snapshot,
        context="rollback shadow",
        clone=True,
    )
    manifest.update(
        {
            "restore_shadow_snapshot": restore_snapshot.to_dict(),
            "status": "rollback_shadow_verified",
        }
    )
    write_manifest(manifest_path, manifest)

    final_current_snapshot = snapshot_collection(target, batch_size=batch_size)
    assert_same_snapshot(
        current_snapshot,
        final_current_snapshot,
        context="target changed while rollback copies were being built",
    )
    def verify_restored_target() -> CollectionSnapshot:
        restored = database[target.name]
        restored_snapshot = snapshot_collection(
            restored,
            batch_size=batch_size,
        )
        assert_same_snapshot(
            backup_snapshot,
            restored_snapshot,
            context="post-rollback target",
            clone=True,
        )
        return restored_snapshot

    progress("Atomically replacing target with the verified rollback shadow ...")
    restored_snapshot = execute_verified_cutover(
        client,
        database,
        target_name=target.name,
        replacement_name=restore_shadow_name,
        retained_backup_name=current_backup_name,
        expected_target_before=current_snapshot,
        expected_replacement=restore_snapshot,
        expected_retained_backup=current_backup_snapshot,
        batch_size=batch_size,
        manifest=manifest,
        manifest_path=manifest_path,
        verify_after=verify_restored_target,
    )

    manifest.update(
        {
            "status": "rolled_back_and_verified",
            "completed_at": utc_now().isoformat(),
            "target_snapshot_before": current_snapshot.to_dict(),
            "rollback_source_snapshot": backup_snapshot.to_dict(),
            "target_snapshot_after": restored_snapshot.to_dict(),
            "retained_pre_rollback_backup": current_backup_name,
            "writes_performed": True,
        }
    )
    write_manifest(manifest_path, manifest)
    return manifest


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def nonblank_selector(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise argparse.ArgumentTypeError("must not be blank")
    return normalized


def sha256_hex(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError("must be a 64-character SHA-256 hex digest")
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize existing effect ontology fields through a verified shadow. "
            "Default and repair modes are read-only unless --apply is supplied. "
            "Stop every extractor before --apply."
        )
    )
    parser.add_argument(
        "--mongo-uri",
        default=os.getenv("MONGO_URI", "mongodb://host.docker.internal:27017"),
        help="MongoDB URI (default: MONGO_URI; never written to the manifest)",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("MONGO_DB", "tripindex"),
        help="database name (default: MONGO_DB or tripindex)",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("MONGO_TARGET_COLLECTION", "erowid-effects-1"),
        help="target collection (default: MONGO_TARGET_COLLECTION)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform verified backup/shadow writes and atomic cutover",
    )
    parser.add_argument(
        "--rollback-backup",
        type=nonblank_selector,
        help="with --apply, atomically restore this retained backup collection",
    )
    parser.add_argument(
        "--repair-from-backup",
        type=nonblank_selector,
        help=(
            "project or apply a tag-level ontology-v1 correction using this "
            "retained backup collection as provenance"
        ),
    )
    parser.add_argument(
        "--expected-repair-backup-sha256",
        type=sha256_hex,
        help=(
            "required with --repair-from-backup; exact expected content hash "
            "of the retained provenance collection"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "manifest output path; dry-run writes only when this is supplied, "
            "while apply defaults under migration_manifests/"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--quiescence-seconds",
        type=nonnegative_float,
        default=DEFAULT_QUIESCENCE_SECONDS,
        help="seconds between full snapshot hashes (default: 5)",
    )
    parser.add_argument(
        "--max-examples",
        type=positive_int,
        default=20,
        help="maximum unsafe/unsupported examples retained in the manifest",
    )
    return parser


def failure_status_for_manifest(manifest: dict[str, Any]) -> str:
    cutover_outcome = manifest.get("cutover_outcome")
    if cutover_outcome == "committed":
        return "failed_after_cutover_committed"
    if cutover_outcome == "not_committed":
        return "failed_cutover_not_committed"
    if cutover_outcome == "indeterminate" and manifest.get(
        "cutover_attempted"
    ):
        return "failed_cutover_indeterminate"
    return "failed"


def completion_exit_code(result: dict[str, Any]) -> int:
    if result.get("status") == "repair_dry_run_conflicts":
        return 3
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rollback_requested = args.rollback_backup is not None
    repair_requested = args.repair_from_backup is not None
    repair_hash_supplied = args.expected_repair_backup_sha256 is not None
    if rollback_requested and not args.apply:
        parser.error("--rollback-backup requires --apply")
    if rollback_requested and repair_requested:
        parser.error("--rollback-backup and --repair-from-backup are mutually exclusive")
    if repair_requested and not repair_hash_supplied:
        parser.error(
            "--repair-from-backup requires --expected-repair-backup-sha256"
        )
    if repair_hash_supplied and not repair_requested:
        parser.error(
            "--expected-repair-backup-sha256 requires --repair-from-backup"
        )

    if rollback_requested:
        mode = "rollback"
    elif repair_requested:
        mode = "repair_apply" if args.apply else "repair_dry_run"
    else:
        mode = "apply" if args.apply else "dry_run"
    run_id = make_run_id()
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "mode": mode,
        "run_id": run_id,
        "database": args.db,
        "target_collection": args.collection,
        "status": "starting",
    }
    manifest_path = args.manifest
    if args.apply and manifest_path is None:
        manifest_path = default_manifest_path(mode, run_id)

    client: Optional[MongoClient] = None
    try:
        require_database_dependencies()
        require_extractor()
        manifest = manifest_base(
            mode=mode,
            run_id=run_id,
            database_name=args.db,
            target_name=args.collection,
        )
        client = MongoClient(
            args.mongo_uri,
            serverSelectionTimeoutMS=5000,
            tz_aware=True,
        )
        client.admin.command("ping")
        database = client[args.db]
        target = database[args.collection]
        collection_info(database, args.collection)

        if rollback_requested:
            result = run_rollback(
                client,
                target,
                backup_name=args.rollback_backup,
                batch_size=args.batch_size,
                quiescence_seconds=args.quiescence_seconds,
                max_examples=args.max_examples,
                run_id=run_id,
                manifest=manifest,
                manifest_path=manifest_path,
            )
        elif repair_requested:
            if "." in args.repair_from_backup or "\x00" in args.repair_from_backup:
                raise MigrationSafetyError(
                    "repair backup must be a collection name, not a namespace"
                )
            if args.repair_from_backup == args.collection:
                raise MigrationSafetyError(
                    "repair backup must not be the live target collection"
                )
            collection_info(database, args.repair_from_backup)
            repair_backup = database[args.repair_from_backup]
            if args.apply:
                result = run_repair_apply(
                    client,
                    target,
                    repair_backup,
                    expected_backup_sha256=args.expected_repair_backup_sha256,
                    batch_size=args.batch_size,
                    quiescence_seconds=args.quiescence_seconds,
                    max_examples=args.max_examples,
                    run_id=run_id,
                    manifest=manifest,
                    manifest_path=manifest_path,
                )
            else:
                result = run_repair_dry_run(
                    target,
                    repair_backup,
                    expected_backup_sha256=args.expected_repair_backup_sha256,
                    batch_size=args.batch_size,
                    quiescence_seconds=args.quiescence_seconds,
                    max_examples=args.max_examples,
                    manifest=manifest,
                )
                if manifest_path is not None:
                    write_manifest(manifest_path, result)
        elif args.apply:
            result = run_apply(
                client,
                target,
                batch_size=args.batch_size,
                quiescence_seconds=args.quiescence_seconds,
                max_examples=args.max_examples,
                run_id=run_id,
                manifest=manifest,
                manifest_path=manifest_path,
            )
        else:
            result = run_dry_run(
                target,
                batch_size=args.batch_size,
                quiescence_seconds=args.quiescence_seconds,
                max_examples=args.max_examples,
                manifest=manifest,
            )
            if manifest_path is not None:
                write_manifest(manifest_path, result)

        if manifest_path is not None:
            result["manifest_path"] = str(manifest_path.resolve())
            write_manifest(manifest_path, result)
        emit_manifest(result)
        return completion_exit_code(result)
    except Exception as exc:
        manifest.update(
            {
                "status": failure_status_for_manifest(manifest),
                "failed_at": utc_now().isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        if manifest_path is not None:
            try:
                manifest["manifest_path"] = str(manifest_path.resolve())
                write_manifest(manifest_path, manifest)
            except Exception as manifest_error:
                manifest["manifest_write_error"] = str(manifest_error)
        emit_manifest(manifest)
        return 2
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
