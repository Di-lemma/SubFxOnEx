"""Complete prose definition registry for the subjective-effect ontology."""

from .bodily import BODILY_EFFECT_DEFINITIONS
from .cognitive import COGNITIVE_EFFECT_DEFINITIONS
from .emotional import EMOTIONAL_EFFECT_DEFINITIONS
from .perceptual import PERCEPTUAL_EFFECT_DEFINITIONS
from .selfhood import SELFHOOD_EFFECT_DEFINITIONS
from .social import SOCIAL_EFFECT_DEFINITIONS
from .spiritual import SPIRITUAL_EFFECT_DEFINITIONS
from .temporal import TEMPORAL_EFFECT_DEFINITIONS


DEFINITION_GROUPS = (
    PERCEPTUAL_EFFECT_DEFINITIONS,
    BODILY_EFFECT_DEFINITIONS,
    EMOTIONAL_EFFECT_DEFINITIONS,
    COGNITIVE_EFFECT_DEFINITIONS,
    TEMPORAL_EFFECT_DEFINITIONS,
    SELFHOOD_EFFECT_DEFINITIONS,
    SPIRITUAL_EFFECT_DEFINITIONS,
    SOCIAL_EFFECT_DEFINITIONS,
)

EFFECT_DEFINITIONS: dict[str, str] = {}
for definition_group in DEFINITION_GROUPS:
    duplicate_names = set(EFFECT_DEFINITIONS) & set(definition_group)
    if duplicate_names:
        raise ValueError(
            "Effect definitions are duplicated across modules: "
            f"{sorted(duplicate_names)!r}"
        )
    EFFECT_DEFINITIONS.update(definition_group)


__all__ = ["DEFINITION_GROUPS", "EFFECT_DEFINITIONS"]
