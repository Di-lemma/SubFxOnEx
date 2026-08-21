#!/usr/bin/env python3
"""Export the embedded subjective-effect ontology as an immutable release."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID, uuid4, uuid5

import effect_extractor as extractor


SCHEMA_VERSION = 1
ONTOLOGY_NAME = "erowid-subjective-effects"
ONTOLOGY_NAMESPACE = UUID("2f0d5b0c-8ee5-5c33-bbcb-850e1461c63e")
RELEASE_PREFIX = "subjective-effects-"
RELEASE_BODY_KEYS = {
    "schema_version",
    "ontology",
    "ontology_hash",
    "id_namespace",
    "concepts",
    "aliases",
    "redirects",
    "ambiguous_labels",
    "counts",
}
RELEASE_KEYS = RELEASE_BODY_KEYS | {"release_hash"}
CONCEPT_KEYS = {
    "id",
    "name",
    "normalized_name",
    "slug",
    "domain",
    "kind",
    "parent_id",
    "parent_name",
    "position",
}
ALIAS_KEYS = {
    "label",
    "normalized_label",
    "effect_id",
    "effect_name",
    "detail",
}
REDIRECT_KEYS = ALIAS_KEYS | {"resolution"}
COUNT_KEYS = {
    "concepts",
    "atomic_concepts",
    "rollup_concepts",
    "aliases",
    "redirects",
    "ambiguous_labels",
}


def canonical_json(value: Any) -> str:
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


def read_previous_concept_ids(release_directory: Path) -> dict[str, str]:
    """Read immutable prior releases as the stable-ID history."""

    concept_ids: dict[str, str] = {}
    release_hashes_by_ontology: dict[str, str] = {}
    for path in sorted(release_directory.glob(f"{RELEASE_PREFIX}*.json")):
        try:
            release = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read prior ontology release {path}: {exc}") from exc
        if release.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported prior ontology release schema in {path}: "
                f"{release.get('schema_version')!r}"
            )
        if release.get("ontology") != ONTOLOGY_NAME:
            raise ValueError(
                f"Unexpected ontology name in prior release {path}: "
                f"{release.get('ontology')!r}"
            )
        try:
            validate_release(release, require_current_alignment=False)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid prior ontology release {path}: {exc}") from exc
        expected_name = f"{RELEASE_PREFIX}{release['release_hash']}.json"
        if path.name != expected_name:
            raise ValueError(
                f"Prior ontology release filename does not match its hash: {path}"
            )
        ontology_hash = release["ontology_hash"]
        existing_release_hash = release_hashes_by_ontology.get(ontology_hash)
        if (
            existing_release_hash is not None
            and existing_release_hash != release["release_hash"]
        ):
            raise ValueError(
                "Competing schema-v1 releases exist for ontology hash "
                f"{ontology_hash}: {existing_release_hash} and "
                f"{release['release_hash']}"
            )
        release_hashes_by_ontology[ontology_hash] = release["release_hash"]
        for concept in release.get("concepts", []):
            name = normalize_label(str(concept.get("name", "")))
            concept_id = str(concept.get("id", ""))
            if not name or not concept_id:
                raise ValueError(f"Invalid concept identity in prior release {path}")
            existing = concept_ids.get(name)
            if existing is not None and existing != concept_id:
                raise ValueError(
                    f"Conflicting stable IDs for {name!r}: {existing} and {concept_id}"
                )
            concept_ids[name] = concept_id
    return concept_ids


def allocate_concept_ids(
    canonical_names: Iterable[str],
    previous_ids: dict[str, str] | None = None,
) -> dict[str, str]:
    """Preserve released IDs, including explicit canonical-label renames."""

    previous = previous_ids or {}
    result: dict[str, str] = {}
    claimed_ids: dict[str, str] = {}
    redirects = extractor.SAFE_DEPRECATED_EFFECT_REDIRECTS

    for name in canonical_names:
        normalized_name = normalize_label(name)
        candidates = {previous[normalized_name]} if normalized_name in previous else set()
        candidates.update(
            previous[normalize_label(retired)]
            for retired, target in redirects.items()
            if target == name and normalize_label(retired) in previous
        )
        if len(candidates) > 1:
            raise ValueError(
                f"Canonical effect {name!r} inherits conflicting prior IDs: "
                f"{sorted(candidates)!r}"
            )
        concept_id = next(iter(candidates), str(uuid5(ONTOLOGY_NAMESPACE, f"effect:{name}")))
        parsed_id = UUID(concept_id)
        if str(parsed_id) != concept_id or parsed_id.version != 5:
            raise ValueError(
                f"Stable effect ID for {name!r} must be a canonical UUIDv5"
            )
        prior_name = claimed_ids.get(concept_id)
        if prior_name is not None and prior_name != name:
            raise ValueError(
                f"Stable effect ID {concept_id} is claimed by {prior_name!r} and {name!r}"
            )
        claimed_ids[concept_id] = name
        result[name] = concept_id
    return result


def build_release(previous_ids: dict[str, str] | None = None) -> dict[str, Any]:
    extractor.validate_effect_ontology()
    canonical_names = [
        effect
        for effects in extractor.CONTROLLED_EFFECT_ONTOLOGY.values()
        for effect in effects
    ]
    concept_ids = allocate_concept_ids(canonical_names, previous_ids)

    concepts: list[dict[str, Any]] = []
    for position, (domain, effects) in enumerate(
        (
            (domain, (effect, parent_effect))
            for domain, domain_effects in extractor.CONTROLLED_EFFECT_ONTOLOGY.items()
            for effect, parent_effect in domain_effects.items()
        )
    ):
        effect, parent_effect = effects
        is_rollup = effect == parent_effect
        concepts.append(
            {
                "id": concept_ids[effect],
                "name": effect,
                "normalized_name": normalize_label(effect),
                "slug": slugify(effect),
                "domain": domain,
                "kind": "rollup" if is_rollup else "atomic",
                "parent_id": None if is_rollup else concept_ids[parent_effect],
                "parent_name": None if is_rollup else parent_effect,
                "position": position,
            }
        )

    safe_redirects = extractor.SAFE_DEPRECATED_EFFECT_REDIRECTS
    unsafe_redirects = extractor.UNSAFE_DEPRECATED_EFFECT_REDIRECTS
    deprecated_labels = set(extractor.DEPRECATED_EFFECT_REDIRECTS)
    aliases = []
    for alias, target in sorted(extractor.EFFECT_ALIASES.items()):
        if alias == target or alias in deprecated_labels:
            continue
        aliases.append(
            {
                "label": alias,
                "normalized_label": normalize_label(alias),
                "effect_id": concept_ids[target],
                "effect_name": target,
                "detail": extractor.EFFECT_COMPATIBILITY_DETAILS.get(alias),
            }
        )

    redirects = []
    for label, target in sorted(extractor.DEPRECATED_EFFECT_REDIRECTS.items()):
        safe = label in safe_redirects
        if not safe and label not in unsafe_redirects:
            raise ValueError(f"Deprecated redirect {label!r} has no safety classification")
        redirects.append(
            {
                "label": label,
                "normalized_label": normalize_label(label),
                "effect_id": concept_ids[target],
                "effect_name": target,
                "resolution": "automatic" if safe else "manual_review",
                "detail": extractor.DEPRECATED_EFFECT_DETAILS.get(label),
            }
        )

    ambiguous_labels = sorted(
        {
            normalize_label(label)
            for label in (
                set(extractor.AMBIGUOUS_EFFECT_ALIASES)
                | set(extractor.UNSAFE_EFFECT_ALIAS_LABELS)
            )
        }
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "ontology": ONTOLOGY_NAME,
        "ontology_hash": extractor.ONTOLOGY_HASH,
        "id_namespace": str(ONTOLOGY_NAMESPACE),
        "concepts": concepts,
        "aliases": aliases,
        "redirects": redirects,
        "ambiguous_labels": ambiguous_labels,
        "counts": {
            "concepts": len(concepts),
            "atomic_concepts": sum(concept["kind"] == "atomic" for concept in concepts),
            "rollup_concepts": sum(concept["kind"] == "rollup" for concept in concepts),
            "aliases": len(aliases),
            "redirects": len(redirects),
            "ambiguous_labels": len(ambiguous_labels),
        },
    }
    return {**body, "release_hash": stable_hash(body)}


def validate_release(
    release: dict[str, Any],
    *,
    require_current_alignment: bool = True,
) -> None:
    """Validate the release body independently and, by default, against source."""

    if not isinstance(release, dict) or set(release) != RELEASE_KEYS:
        raise ValueError("Ontology release has an unexpected top-level shape")
    release_hash = release.get("release_hash")
    if not isinstance(release_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", release_hash):
        raise ValueError("Ontology release hash must be 64 lowercase hex characters")
    body = {key: release[key] for key in RELEASE_BODY_KEYS}
    if release_hash != stable_hash(body):
        raise ValueError("Ontology release hash does not match its canonical content")
    if release.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported ontology release schema: {release.get('schema_version')!r}"
        )
    if release.get("ontology") != ONTOLOGY_NAME:
        raise ValueError(f"Unexpected ontology name: {release.get('ontology')!r}")
    ontology_hash = release.get("ontology_hash")
    if not isinstance(ontology_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", ontology_hash
    ):
        raise ValueError("Ontology hash must be 64 lowercase hex characters")
    if release.get("id_namespace") != str(ONTOLOGY_NAMESPACE):
        raise ValueError("Ontology release uses an unexpected stable-ID namespace")

    concepts = release.get("concepts")
    if not isinstance(concepts, list) or not concepts:
        raise ValueError("Ontology release has no concepts")
    concepts_by_id: dict[str, dict[str, Any]] = {}
    normalized_names: set[str] = set()
    slugs: set[str] = set()
    for position, concept in enumerate(concepts):
        if not isinstance(concept, dict) or set(concept) != CONCEPT_KEYS:
            raise ValueError(f"Concept at position {position} has an invalid shape")
        concept_id = concept.get("id")
        try:
            parsed_id = UUID(concept_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"Concept at position {position} has an invalid ID") from exc
        if str(parsed_id) != concept_id or parsed_id.version != 5:
            raise ValueError(f"Concept at position {position} must use a canonical UUIDv5")
        if concept_id in concepts_by_id:
            raise ValueError(f"Duplicate concept ID: {concept_id}")
        name = concept.get("name")
        domain = concept.get("domain")
        if not isinstance(name, str) or not name or not isinstance(domain, str) or not domain:
            raise ValueError(f"Concept at position {position} has an invalid name or domain")
        normalized_name = concept.get("normalized_name")
        if normalized_name != normalize_label(name):
            raise ValueError(f"Concept {name!r} has a stale normalized name")
        slug = concept.get("slug")
        if not slug or slug != slugify(name):
            raise ValueError(f"Concept {name!r} has a stale slug")
        if normalized_name in normalized_names or slug in slugs:
            raise ValueError("Ontology release concept names and slugs must be unique")
        if concept.get("kind") not in {"atomic", "rollup"}:
            raise ValueError(f"Invalid concept kind for {name!r}: {concept.get('kind')!r}")
        if concept.get("position") != position:
            raise ValueError(f"Concept {name!r} has a non-canonical position")
        concepts_by_id[concept_id] = concept
        normalized_names.add(normalized_name)
        slugs.add(slug)

    for concept in concepts:
        parent_id = concept.get("parent_id")
        parent_name = concept.get("parent_name")
        if concept["kind"] == "rollup":
            if parent_id is not None or parent_name is not None:
                raise ValueError(f"Rollup concept {concept['name']!r} must not have a parent")
            continue
        if not isinstance(parent_id, str) or parent_id not in concepts_by_id:
            raise ValueError(f"Atomic concept {concept['name']!r} has a missing parent")
        parent = concepts_by_id[parent_id]
        if (
            parent.get("kind") != "rollup"
            or parent.get("name") != parent_name
            or parent.get("domain") != concept.get("domain")
        ):
            raise ValueError(f"Atomic concept {concept['name']!r} has an invalid parent")
    domains = {concept["domain"] for concept in concepts}
    rollup_counts = {
        domain: sum(
            concept["kind"] == "rollup" and concept["domain"] == domain
            for concept in concepts
        )
        for domain in domains
    }
    if any(count != 1 for count in rollup_counts.values()):
        raise ValueError("Every ontology domain must contain exactly one rollup")

    def validate_name_records(
        key: str,
        required_keys: set[str],
    ) -> tuple[list[dict[str, Any]], set[str]]:
        records = release.get(key)
        if not isinstance(records, list):
            raise ValueError(f"Ontology release {key} must be an array")
        if any(
            not isinstance(record, dict) or set(record) != required_keys
            for record in records
        ):
            raise ValueError(f"Ontology release {key} contains an invalid record")
        labels = [record["label"] for record in records]
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError(f"Ontology release {key} contains a blank label")
        if labels != sorted(labels) or len(labels) != len(set(labels)):
            raise ValueError(f"Ontology release {key} must have unique sorted labels")
        normalized_labels: set[str] = set()
        for record in records:
            label = record.get("label")
            normalized_label = record.get("normalized_label")
            if normalized_label != normalize_label(label):
                raise ValueError(f"Ontology release {key} contains a stale normalized label")
            if normalized_label in normalized_labels:
                raise ValueError(f"Ontology release {key} contains a normalized collision")
            normalized_labels.add(normalized_label)
            effect_id = record.get("effect_id")
            target = concepts_by_id.get(effect_id)
            if target is None or target.get("name") != record.get("effect_name"):
                raise ValueError(f"Ontology release {key} targets an inconsistent concept")
            detail = record.get("detail")
            if detail is not None and not isinstance(detail, str):
                raise ValueError(f"Ontology release {key} has an invalid detail")
        return records, normalized_labels

    aliases, alias_labels = validate_name_records("aliases", ALIAS_KEYS)
    redirects, redirect_labels = validate_name_records("redirects", REDIRECT_KEYS)
    if alias_labels & redirect_labels:
        raise ValueError("Automatic aliases and deprecated redirects must be disjoint")
    concepts_by_normalized_name = {
        concept["normalized_name"]: concept for concept in concepts
    }
    for record in [*aliases, *redirects]:
        colliding_concept = concepts_by_normalized_name.get(
            record["normalized_label"]
        )
        if (
            colliding_concept is not None
            and colliding_concept["id"] != record["effect_id"]
        ):
            raise ValueError(
                "Normalized name record collides with a different canonical concept"
            )
    for redirect in redirects:
        if redirect.get("resolution") not in {"automatic", "manual_review"}:
            raise ValueError("Deprecated redirect has an invalid resolution")

    ambiguous_labels = release.get("ambiguous_labels")
    if (
        not isinstance(ambiguous_labels, list)
        or any(
            not isinstance(label, str)
            or not label
            for label in ambiguous_labels
        )
    ):
        raise ValueError("Ambiguous labels must be nonblank strings")
    if (
        ambiguous_labels != sorted(set(ambiguous_labels))
        or any(normalize_label(label) != label for label in ambiguous_labels)
    ):
        raise ValueError("Ambiguous labels must be unique, normalized, and sorted")
    if alias_labels & set(ambiguous_labels):
        raise ValueError("Ambiguous labels must not be automatic aliases")
    if set(ambiguous_labels) & set(concepts_by_normalized_name):
        raise ValueError("Ambiguous labels must not collide with canonical concepts")
    redirects_by_label = {
        redirect["normalized_label"]: redirect for redirect in redirects
    }
    for label in set(ambiguous_labels) & redirect_labels:
        if redirects_by_label[label]["resolution"] != "manual_review":
            raise ValueError("Ambiguous redirects must require manual review")

    counts = release.get("counts")
    expected_counts = {
        "concepts": len(concepts),
        "atomic_concepts": sum(concept["kind"] == "atomic" for concept in concepts),
        "rollup_concepts": sum(concept["kind"] == "rollup" for concept in concepts),
        "aliases": len(aliases),
        "redirects": len(redirects),
        "ambiguous_labels": len(ambiguous_labels),
    }
    if not isinstance(counts, dict) or set(counts) != COUNT_KEYS or counts != expected_counts:
        raise ValueError("Ontology release counts do not match its content")

    if not require_current_alignment:
        return
    if ontology_hash != extractor.ONTOLOGY_HASH:
        raise ValueError("Ontology release does not match the current ontology hash")
    expected_concepts = {
        effect: (domain, parent_effect)
        for domain, effects in extractor.CONTROLLED_EFFECT_ONTOLOGY.items()
        for effect, parent_effect in effects.items()
    }
    actual_concepts = {
        concept["name"]: (concept["domain"], concept["parent_name"] or concept["name"])
        for concept in concepts
    }
    if actual_concepts != expected_concepts:
        raise ValueError("Ontology release concepts do not match the current hierarchy")
    expected_aliases = {
        alias: (
            target,
            extractor.EFFECT_COMPATIBILITY_DETAILS.get(alias),
        )
        for alias, target in extractor.EFFECT_ALIASES.items()
        if alias != target and alias not in extractor.DEPRECATED_EFFECT_REDIRECTS
    }
    actual_aliases = {
        record["label"]: (record["effect_name"], record["detail"])
        for record in aliases
    }
    if actual_aliases != expected_aliases:
        raise ValueError("Ontology release aliases do not match the current runtime")
    expected_redirects = {
        label: (
            target,
            "automatic"
            if label in extractor.SAFE_DEPRECATED_EFFECT_REDIRECTS
            else "manual_review",
            extractor.DEPRECATED_EFFECT_DETAILS.get(label),
        )
        for label, target in extractor.DEPRECATED_EFFECT_REDIRECTS.items()
    }
    actual_redirects = {
        record["label"]: (
            record["effect_name"],
            record["resolution"],
            record["detail"],
        )
        for record in redirects
    }
    if actual_redirects != expected_redirects:
        raise ValueError("Ontology release redirects do not match current safety policy")
    expected_ambiguous = sorted(
        {
            normalize_label(label)
            for label in (
                set(extractor.AMBIGUOUS_EFFECT_ALIASES)
                | set(extractor.UNSAFE_EFFECT_ALIAS_LABELS)
            )
        }
    )
    if ambiguous_labels != expected_ambiguous:
        raise ValueError("Ontology release ambiguity policy does not match the runtime")


def release_path(release_directory: Path, release_hash: str) -> Path:
    return release_directory / f"{RELEASE_PREFIX}{release_hash}.json"


def write_release(release: dict[str, Any], release_directory: Path) -> Path:
    validate_release(release)
    release_directory.mkdir(parents=True, exist_ok=True)
    path = release_path(release_directory, release["release_hash"])
    serialized = json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"Refusing to overwrite changed immutable release {path}")
        return path
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != serialized:
                raise ValueError(f"Refusing to overwrite changed immutable release {path}")
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-directory",
        type=Path,
        default=Path(__file__).resolve().parent / "ontology_releases",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write the immutable release file")
    mode.add_argument("--check", action="store_true", help="require the current release file to exist exactly")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    previous_ids = read_previous_concept_ids(args.release_directory)
    release = build_release(previous_ids)
    validate_release(release)
    path = release_path(args.release_directory, release["release_hash"])

    if args.write:
        print(write_release(release, args.release_directory))
        return 0
    if args.check:
        expected = json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            print(
                f"Current ontology release is missing or stale: {path}",
                file=sys.stderr,
            )
            return 1
        print(path)
        return 0

    print(json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
