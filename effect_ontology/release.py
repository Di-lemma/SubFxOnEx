"""Pure-data loading, validation, hashing, and resolution for ontology releases.

This module intentionally imports only the Python standard library.  Consumers
can therefore load a pinned SubFxOnEx artifact without installing the extractor,
MongoDB client, Pydantic models, or provider SDK.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID


CURRENT_SCHEMA_VERSION = 3
ONTOLOGY_NAME = "erowid-subjective-effects"
ONTOLOGY_NAMESPACE = UUID("2f0d5b0c-8ee5-5c33-bbcb-850e1461c63e")
RELEASE_PREFIX = "subjective-effects-"
CURRENT_MANIFEST_FILENAME = "current.json"

REVIEW_STATUSES = frozenset(
    {"draft", "defined", "editorial_reviewed", "expert_reviewed", "deprecated"}
)
DEFAULT_REVIEW_STATUS = "defined"

# This structured profile is part of normalization_hash.  Any implementation
# change to label or slug normalization must update the profile at the same time.
NORMALIZATION_PROFILE = {
    "id": "subfxonex-normalization-v1",
    "label": {
        "case": "lower",
        "dash_characters_to_ascii": ["‐", "‑", "‒", "–", "—", "―", "−"],
        "strip_combining_marks": True,
        "symbol_replacements": {"Δ": "delta", "α": "alpha", "β": "beta"},
        "underscore": "space",
        "unicode_normalization": "NFKD",
        "whitespace": "collapse",
    },
    "slug": {
        "allowed": "ascii-lowercase-alphanumeric",
        "separator": "-",
        "strip_combining_marks": True,
        "symbol_replacements": {"Δ": "delta", "α": "alpha", "β": "beta"},
        "unicode_normalization": "NFKD",
    },
}

V3_CONCEPT_KEYS = {
    "id",
    "name",
    "normalized_name",
    "slug",
    "definition",
    "domain",
    "kind",
    "parent_id",
    "parent_name",
    "position",
    "review_status",
}
V3_ALIAS_KEYS = {
    "label",
    "normalized_label",
    "effect_id",
    "effect_name",
    "detail",
}
V3_REDIRECT_KEYS = {
    "label",
    "normalized_label",
    "effect_id",
    "effect_name",
    "candidate_effect_id",
    "candidate_effect_name",
    "resolution",
    "detail",
}
V3_COUNT_KEYS = {
    "concepts",
    "atomic_concepts",
    "rollup_concepts",
    "aliases",
    "redirects",
    "ambiguous_labels",
}
V3_RELEASE_BODY_KEYS = {
    "schema_version",
    "ontology",
    "id_namespace",
    "normalization",
    "normalization_hash",
    "semantic_hash",
    "concepts",
    "aliases",
    "redirects",
    "ambiguous_labels",
    "counts",
}
V3_RELEASE_KEYS = V3_RELEASE_BODY_KEYS | {"release_hash"}
MANIFEST_KEYS = {
    "artifact",
    "artifact_sha256",
    "ontology",
    "schema_version",
    "normalization_hash",
    "semantic_hash",
    "release_hash",
}

_HASH_RE = re.compile(r"[0-9a-f]{64}")


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for ontology identity hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    normalized = (
        normalized.replace("‐", "-")
        .replace("‑", "-")
        .replace("‒", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("―", "-")
        .replace("−", "-")
        .replace("α", "alpha")
        .replace("β", "beta")
        .replace("Δ", "delta")
    )
    return " ".join(normalized.strip().lower().replace("_", " ").split())


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    normalized = (
        normalized.replace("α", "alpha")
        .replace("β", "beta")
        .replace("Δ", "delta")
    )
    return re.sub(r"(^-|-$)", "", re.sub(r"[^a-z0-9]+", "-", normalized.lower()))


def normalization_payload(release: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields that control label normalization and resolution."""

    return {
        "ontology": release["ontology"],
        "normalization": release["normalization"],
        "labels": [
            {
                "id": concept["id"],
                "name": concept["name"],
                "normalized_name": concept["normalized_name"],
                "slug": concept["slug"],
            }
            for concept in release["concepts"]
        ],
        "aliases": release["aliases"],
        "redirects": release["redirects"],
        "ambiguous_labels": release["ambiguous_labels"],
    }


def semantic_payload(release: Mapping[str, Any]) -> dict[str, Any]:
    """Return concept meaning, hierarchy, review state, and redirect relations."""

    return {
        "ontology": release["ontology"],
        "concepts": release["concepts"],
        "redirect_relations": release["redirects"],
    }


def compute_normalization_hash(release: Mapping[str, Any]) -> str:
    return stable_hash(normalization_payload(release))


def compute_semantic_hash(release: Mapping[str, Any]) -> str:
    return stable_hash(semantic_payload(release))


def release_body(release: Mapping[str, Any]) -> dict[str, Any]:
    return {key: release[key] for key in V3_RELEASE_BODY_KEYS}


def compute_release_hash(release: Mapping[str, Any]) -> str:
    """Hash the complete canonical release body, excluding its self-hash field."""

    return stable_hash(release_body(release))


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hex characters")
    return value


def validate_consumer_release(release: dict[str, Any]) -> None:
    """Validate a schema-v3 consumer artifact without extractor dependencies."""

    if not isinstance(release, dict) or set(release) != V3_RELEASE_KEYS:
        raise ValueError("Ontology release has an unexpected schema-v3 shape")
    if release.get("schema_version") != CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported consumer ontology schema: {release.get('schema_version')!r}"
        )
    if release.get("ontology") != ONTOLOGY_NAME:
        raise ValueError(f"Unexpected ontology name: {release.get('ontology')!r}")
    if release.get("id_namespace") != str(ONTOLOGY_NAMESPACE):
        raise ValueError("Ontology release uses an unexpected stable-ID namespace")
    if release.get("normalization") != NORMALIZATION_PROFILE:
        raise ValueError("Ontology release uses an unsupported normalization profile")

    normalization_hash = _require_hash(
        release.get("normalization_hash"), "Normalization hash"
    )
    semantic_hash = _require_hash(release.get("semantic_hash"), "Semantic hash")
    release_hash = _require_hash(release.get("release_hash"), "Release hash")
    if normalization_hash != compute_normalization_hash(release):
        raise ValueError("Ontology normalization hash does not match its content")
    if semantic_hash != compute_semantic_hash(release):
        raise ValueError("Ontology semantic hash does not match its content")
    if release_hash != compute_release_hash(release):
        raise ValueError("Ontology release hash does not match its canonical body")

    concepts = release.get("concepts")
    if not isinstance(concepts, list) or not concepts:
        raise ValueError("Ontology release has no concepts")
    concepts_by_id: dict[str, dict[str, Any]] = {}
    concepts_by_name: dict[str, dict[str, Any]] = {}
    slugs: set[str] = set()
    for position, concept in enumerate(concepts):
        if not isinstance(concept, dict) or set(concept) != V3_CONCEPT_KEYS:
            raise ValueError(f"Concept at position {position} has an invalid shape")
        concept_id = concept.get("id")
        try:
            parsed_id = UUID(concept_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"Concept at position {position} has an invalid ID") from exc
        if str(parsed_id) != concept_id or parsed_id.version != 5:
            raise ValueError(f"Concept at position {position} must use a UUIDv5")
        name = concept.get("name")
        domain = concept.get("domain")
        definition = concept.get("definition")
        if not isinstance(name, str) or not name or not isinstance(domain, str) or not domain:
            raise ValueError(f"Concept at position {position} has an invalid name or domain")
        if (
            not isinstance(definition, str)
            or not definition
            or definition != definition.strip()
        ):
            raise ValueError(f"Concept {name!r} has an invalid definition")
        if concept.get("normalized_name") != normalize_label(name):
            raise ValueError(f"Concept {name!r} has a stale normalized name")
        slug = concept.get("slug")
        if not isinstance(slug, str) or not slug or slug != slugify(name):
            raise ValueError(f"Concept {name!r} has a stale slug")
        if concept.get("kind") not in {"atomic", "rollup"}:
            raise ValueError(f"Concept {name!r} has an invalid kind")
        if concept.get("review_status") not in REVIEW_STATUSES:
            raise ValueError(f"Concept {name!r} has an invalid review status")
        if concept.get("position") != position:
            raise ValueError(f"Concept {name!r} has a non-canonical position")
        if concept_id in concepts_by_id or name in concepts_by_name or slug in slugs:
            raise ValueError("Ontology concept IDs, names, and slugs must be unique")
        concepts_by_id[concept_id] = concept
        concepts_by_name[name] = concept
        slugs.add(slug)

    for concept in concepts:
        parent_id = concept.get("parent_id")
        parent_name = concept.get("parent_name")
        if concept["kind"] == "rollup":
            if parent_id is not None or parent_name is not None:
                raise ValueError(f"Rollup concept {concept['name']!r} must not have a parent")
            continue
        parent = concepts_by_id.get(parent_id)
        if (
            parent is None
            or parent.get("kind") != "rollup"
            or parent.get("name") != parent_name
            or parent.get("domain") != concept.get("domain")
        ):
            raise ValueError(f"Atomic concept {concept['name']!r} has an invalid rollup")
    domains = {concept["domain"] for concept in concepts}
    if any(
        sum(
            concept["kind"] == "rollup" and concept["domain"] == domain
            for concept in concepts
        )
        != 1
        for domain in domains
    ):
        raise ValueError("Every ontology domain must contain exactly one rollup")

    aliases = release.get("aliases")
    if not isinstance(aliases, list):
        raise ValueError("Ontology aliases must be an array")
    alias_labels: set[str] = set()
    for alias in aliases:
        if not isinstance(alias, dict) or set(alias) != V3_ALIAS_KEYS:
            raise ValueError("Ontology aliases contain an invalid record")
        label = alias.get("label")
        normalized = alias.get("normalized_label")
        target = concepts_by_id.get(alias.get("effect_id"))
        if (
            not isinstance(label, str)
            or not label
            or normalized != normalize_label(label)
            or normalized in alias_labels
            or target is None
            or target["name"] != alias.get("effect_name")
        ):
            raise ValueError("Ontology aliases contain an invalid mapping")
        if alias.get("detail") is not None and not isinstance(alias["detail"], str):
            raise ValueError("Ontology alias detail must be a string or null")
        alias_labels.add(normalized)
    if [alias["label"] for alias in aliases] != sorted(alias["label"] for alias in aliases):
        raise ValueError("Ontology aliases must be sorted")

    redirects = release.get("redirects")
    if not isinstance(redirects, list):
        raise ValueError("Ontology redirects must be an array")
    redirect_labels: set[str] = set()
    for redirect in redirects:
        if not isinstance(redirect, dict) or set(redirect) != V3_REDIRECT_KEYS:
            raise ValueError("Ontology redirects contain an invalid record")
        label = redirect.get("label")
        normalized = redirect.get("normalized_label")
        resolution = redirect.get("resolution")
        if (
            not isinstance(label, str)
            or not label
            or normalized != normalize_label(label)
            or normalized in redirect_labels
            or resolution not in {"automatic", "manual_review"}
        ):
            raise ValueError("Ontology redirects contain an invalid record")
        if resolution == "automatic":
            target = concepts_by_id.get(redirect.get("effect_id"))
            if (
                target is None
                or target["name"] != redirect.get("effect_name")
                or redirect.get("candidate_effect_id") is not None
                or redirect.get("candidate_effect_name") is not None
            ):
                raise ValueError("Automatic redirect has an invalid target")
        else:
            candidate = concepts_by_id.get(redirect.get("candidate_effect_id"))
            if (
                redirect.get("effect_id") is not None
                or redirect.get("effect_name") is not None
                or candidate is None
                or candidate["name"] != redirect.get("candidate_effect_name")
            ):
                raise ValueError("Manual-review redirect has an invalid candidate")
        if redirect.get("detail") is not None and not isinstance(redirect["detail"], str):
            raise ValueError("Ontology redirect detail must be a string or null")
        redirect_labels.add(normalized)
    if [item["label"] for item in redirects] != sorted(item["label"] for item in redirects):
        raise ValueError("Ontology redirects must be sorted")
    if alias_labels & redirect_labels:
        raise ValueError("Aliases and redirects must be disjoint")

    canonical_by_label = {
        concept["normalized_name"]: concept for concept in concepts
    }
    for alias in aliases:
        canonical = canonical_by_label.get(alias["normalized_label"])
        if canonical is not None and canonical["id"] != alias["effect_id"]:
            raise ValueError("Alias collides with a different canonical concept")
    for redirect in redirects:
        canonical = canonical_by_label.get(redirect["normalized_label"])
        if canonical is None:
            continue
        if redirect["resolution"] == "manual_review":
            raise ValueError("Manual-review redirect collides with a canonical concept")
        if canonical["id"] != redirect["effect_id"]:
            raise ValueError("Redirect collides with a different canonical concept")
    canonical_labels = set(canonical_by_label)
    ambiguous_labels = release.get("ambiguous_labels")
    if (
        not isinstance(ambiguous_labels, list)
        or ambiguous_labels != sorted(set(ambiguous_labels))
        or any(
            not isinstance(label, str) or not label or normalize_label(label) != label
            for label in ambiguous_labels
        )
    ):
        raise ValueError("Ambiguous labels must be unique, normalized, and sorted")
    ambiguous = set(ambiguous_labels)
    if ambiguous & (alias_labels | canonical_labels):
        raise ValueError("Ambiguous labels must not resolve automatically")
    automatic_redirect_labels = {
        redirect["normalized_label"]
        for redirect in redirects
        if redirect["resolution"] == "automatic"
    }
    if ambiguous & automatic_redirect_labels:
        raise ValueError("Ambiguous labels must not be automatic redirects")

    counts = release.get("counts")
    expected_counts = {
        "concepts": len(concepts),
        "atomic_concepts": sum(item["kind"] == "atomic" for item in concepts),
        "rollup_concepts": sum(item["kind"] == "rollup" for item in concepts),
        "aliases": len(aliases),
        "redirects": len(redirects),
        "ambiguous_labels": len(ambiguous_labels),
    }
    if not isinstance(counts, dict) or set(counts) != V3_COUNT_KEYS or counts != expected_counts:
        raise ValueError("Ontology release counts do not match its content")


def build_release_manifest(
    release: dict[str, Any], artifact_name: str, artifact_bytes: bytes
) -> dict[str, Any]:
    validate_consumer_release(release)
    if artifact_name != f"{RELEASE_PREFIX}{release['release_hash']}.json":
        raise ValueError("Ontology artifact filename does not match its release hash")
    return {
        "artifact": artifact_name,
        "artifact_sha256": sha256(artifact_bytes).hexdigest(),
        "ontology": release["ontology"],
        "schema_version": release["schema_version"],
        "normalization_hash": release["normalization_hash"],
        "semantic_hash": release["semantic_hash"],
        "release_hash": release["release_hash"],
    }


def validate_release_manifest(
    manifest: dict[str, Any], release: dict[str, Any], artifact_bytes: bytes
) -> None:
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise ValueError("Ontology release manifest has an unexpected shape")
    artifact = manifest.get("artifact")
    if (
        not isinstance(artifact, str)
        or Path(artifact).name != artifact
        or artifact != f"{RELEASE_PREFIX}{release['release_hash']}.json"
    ):
        raise ValueError("Ontology release manifest has an invalid artifact path")
    if manifest.get("artifact_sha256") != sha256(artifact_bytes).hexdigest():
        raise ValueError("Ontology artifact SHA-256 does not match the manifest")
    for key in (
        "ontology",
        "schema_version",
        "normalization_hash",
        "semantic_hash",
        "release_hash",
    ):
        if manifest.get(key) != release.get(key):
            raise ValueError(f"Ontology manifest {key} does not match the release")


def load_release(path: str | Path) -> dict[str, Any]:
    release_path = Path(path)
    try:
        release = json.loads(release_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load ontology release {release_path}: {exc}") from exc
    validate_consumer_release(release)
    return release


def load_pinned_release(manifest_path: str | Path | None = None) -> dict[str, Any]:
    if manifest_path is None:
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "ontology_releases"
            / CURRENT_MANIFEST_FILENAME
        )
    manifest_path = Path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load ontology manifest {manifest_path}: {exc}") from exc
    artifact = manifest.get("artifact") if isinstance(manifest, dict) else None
    if not isinstance(artifact, str) or Path(artifact).name != artifact:
        raise ValueError("Ontology release manifest has an invalid artifact path")
    artifact_path = manifest_path.parent / artifact
    try:
        artifact_bytes = artifact_path.read_bytes()
        release = json.loads(artifact_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load pinned ontology artifact {artifact_path}: {exc}") from exc
    validate_consumer_release(release)
    validate_release_manifest(manifest, release, artifact_bytes)
    return release


@dataclass(frozen=True)
class LabelResolution:
    """A fail-closed label lookup result from a validated release."""

    mode: str
    input_label: str
    normalized_label: str
    concept_id: str | None = None
    concept_name: str | None = None
    candidate_concept_id: str | None = None
    candidate_concept_name: str | None = None
    detail: str | None = None


class OntologyResolver:
    """Indexed resolver that never assigns manual-review candidates as identity."""

    def __init__(self, release: dict[str, Any]):
        validate_consumer_release(release)
        self.release = release
        self._canonical = {
            concept["normalized_name"]: concept for concept in release["concepts"]
        }
        self._aliases = {
            alias["normalized_label"]: alias for alias in release["aliases"]
        }
        self._redirects = {
            redirect["normalized_label"]: redirect
            for redirect in release["redirects"]
        }
        self._ambiguous = frozenset(release["ambiguous_labels"])

    def resolve_label(self, label: str) -> LabelResolution:
        if not isinstance(label, str):
            raise TypeError("Ontology labels must be strings")
        normalized = normalize_label(label)
        concept = self._canonical.get(normalized)
        if concept is not None:
            return LabelResolution(
                "canonical", label, normalized, concept["id"], concept["name"]
            )
        alias = self._aliases.get(normalized)
        if alias is not None:
            return LabelResolution(
                "alias",
                label,
                normalized,
                alias["effect_id"],
                alias["effect_name"],
                detail=alias["detail"],
            )
        redirect = self._redirects.get(normalized)
        if redirect is not None and redirect["resolution"] == "automatic":
            return LabelResolution(
                "automatic_redirect",
                label,
                normalized,
                redirect["effect_id"],
                redirect["effect_name"],
                detail=redirect["detail"],
            )
        if redirect is not None:
            return LabelResolution(
                "manual_review",
                label,
                normalized,
                candidate_concept_id=redirect["candidate_effect_id"],
                candidate_concept_name=redirect["candidate_effect_name"],
                detail=redirect["detail"],
            )
        if normalized in self._ambiguous:
            return LabelResolution("ambiguous", label, normalized)
        return LabelResolution("unknown", label, normalized)


def resolve_label(release: dict[str, Any], label: str) -> LabelResolution:
    return OntologyResolver(release).resolve_label(label)


__all__ = [
    "CURRENT_MANIFEST_FILENAME",
    "CURRENT_SCHEMA_VERSION",
    "DEFAULT_REVIEW_STATUS",
    "LabelResolution",
    "NORMALIZATION_PROFILE",
    "ONTOLOGY_NAME",
    "ONTOLOGY_NAMESPACE",
    "OntologyResolver",
    "RELEASE_PREFIX",
    "REVIEW_STATUSES",
    "V3_ALIAS_KEYS",
    "V3_CONCEPT_KEYS",
    "V3_COUNT_KEYS",
    "V3_REDIRECT_KEYS",
    "V3_RELEASE_BODY_KEYS",
    "V3_RELEASE_KEYS",
    "build_release_manifest",
    "canonical_json",
    "compute_normalization_hash",
    "compute_release_hash",
    "compute_semantic_hash",
    "load_pinned_release",
    "load_release",
    "normalize_label",
    "release_body",
    "resolve_label",
    "slugify",
    "stable_hash",
    "validate_consumer_release",
    "validate_release_manifest",
]
