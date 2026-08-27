"""Data-first API for the Erowid subjective-effect ontology.

The extraction runtime consumes the legacy ``CONTROLLED_EFFECT_ONTOLOGY``
mapping exported here. Ontology tooling can use ``EFFECT_DEFINITIONS`` or the
typed concept records without importing the extractor or its runtime clients.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from .definitions import EFFECT_DEFINITIONS
from .effects import CONTROLLED_EFFECT_ONTOLOGY
from .release import (
    CURRENT_MANIFEST_FILENAME,
    CURRENT_SCHEMA_VERSION,
    DEFAULT_REVIEW_STATUS,
    LabelResolution,
    NORMALIZATION_PROFILE,
    OntologyResolver,
    REVIEW_STATUSES,
    load_pinned_release,
    load_release,
    resolve_label,
    validate_consumer_release,
)


@dataclass(frozen=True)
class EffectConcept:
    """One canonical ontology concept and its direct rollup relationship."""

    domain: str
    name: str
    parent_effect: str
    definition: str

    @property
    def is_rollup(self) -> bool:
        return self.name == self.parent_effect


def iter_effect_concepts() -> Iterator[EffectConcept]:
    """Yield every canonical concept in stable ontology order."""

    for domain, effects in CONTROLLED_EFFECT_ONTOLOGY.items():
        for name, parent_effect in effects.items():
            yield EffectConcept(
                domain=domain,
                name=name,
                parent_effect=parent_effect,
                definition=EFFECT_DEFINITIONS[name],
            )


def get_effect_definition(effect: str) -> str:
    """Return the definition for an exact canonical label.

    Alias resolution remains the extractor's responsibility so ambiguous or
    unsafe historical labels cannot be silently treated as canonical here.
    """

    return EFFECT_DEFINITIONS[effect]


def validate_effect_catalog() -> None:
    """Fail fast if hierarchy and definition sources drift apart."""

    errors: list[str] = []
    canonical_names: set[str] = set()
    for domain, effects in CONTROLLED_EFFECT_ONTOLOGY.items():
        for name in effects:
            if name in canonical_names:
                errors.append(f"canonical effect appears in multiple domains: {name!r}")
            canonical_names.add(name)

    definition_names = set(EFFECT_DEFINITIONS)
    missing_definitions = canonical_names - definition_names
    extra_definitions = definition_names - canonical_names
    if missing_definitions:
        errors.append(f"canonical effects lack definitions: {sorted(missing_definitions)!r}")
    if extra_definitions:
        errors.append(f"definitions target non-canonical effects: {sorted(extra_definitions)!r}")

    definitions_seen: dict[str, str] = {}
    for name, definition in EFFECT_DEFINITIONS.items():
        if not isinstance(definition, str) or definition != definition.strip():
            errors.append(f"definition is not a stripped string: {name!r}")
            continue
        if "\n" in definition:
            errors.append(f"definition must be one normalized paragraph: {name!r}")
        if len(re.findall(r"\b[\w’-]+\b", definition, flags=re.UNICODE)) < 12:
            errors.append(f"definition is too terse to establish a boundary: {name!r}")
        if definition[-1:] not in {".", "?", "!"}:
            errors.append(f"definition lacks terminal punctuation: {name!r}")
        duplicate_name = definitions_seen.get(definition)
        if duplicate_name is not None:
            errors.append(
                f"effects {duplicate_name!r} and {name!r} have identical definitions"
            )
        definitions_seen[definition] = name

    if errors:
        preview = "\n- ".join(errors[:25])
        remainder = len(errors) - 25
        if remainder > 0:
            preview += f"\n- ... and {remainder} more"
        raise ValueError(f"Invalid effect catalog:\n- {preview}")


validate_effect_catalog()


__all__ = [
    "CONTROLLED_EFFECT_ONTOLOGY",
    "CURRENT_MANIFEST_FILENAME",
    "CURRENT_SCHEMA_VERSION",
    "DEFAULT_REVIEW_STATUS",
    "EFFECT_DEFINITIONS",
    "EffectConcept",
    "LabelResolution",
    "NORMALIZATION_PROFILE",
    "OntologyResolver",
    "REVIEW_STATUSES",
    "get_effect_definition",
    "iter_effect_concepts",
    "load_pinned_release",
    "load_release",
    "resolve_label",
    "validate_consumer_release",
    "validate_effect_catalog",
]
