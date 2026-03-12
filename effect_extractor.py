import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from json import JSONDecodeError
from typing import List, Literal, Optional

from mistralai import Mistral
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import CursorNotFound


class DoseReference(BaseModel):
    dose_id: str = Field(
        description="Stable ID for a dose entry from dose_table, e.g. d1"
    )
    substance: str = Field(
        description="Normalized substance name from the referenced dose_table entry"
    )
    dose: Optional[str] = Field(
        default=None,
        description="Dose phrase from the referenced dose_table entry, if present",
    )
    route: Optional[str] = Field(
        default=None,
        description="Route from the referenced dose_table entry, if present",
    )


class EffectAttribution(BaseModel):
    attribution_type: Literal["single_substance", "combination", "unknown"] = Field(
        description="Whether the effect pertains to one drug, multiple drugs together, or is unclear"
    )
    dose_refs: List[DoseReference] = Field(
        default_factory=list,
        description="All dose_table entries this effect is attributed to",
    )
    attribution_note: Optional[str] = Field(
        default=None,
        description="Short explanation if attribution is ambiguous, inferred, or depends on timing/context",
    )


class SubjectiveEffectTag(BaseModel):
    domain: str = Field(
        description="Broad family for the effect, e.g. visual, somatic, emotional, cognitive"
    )
    effect: str = Field(
        description="Canonical effect tag from the controlled vocabulary, e.g. texture rippling, nausea, time dilation"
    )
    subjective_effect: Optional[str] = Field(
        default=None,
        description="Deprecated alias of effect retained for backward compatibility with older downstream consumers",
    )
    parent_effect: str = Field(
        description="Broader fallback effect family for rollups, e.g. visual distortions, body load, emotional change"
    )
    detail: Optional[str] = Field(
        default=None,
        description="Optional short normalized subtype or nuance when the source is more specific than the canonical effect tag",
    )
    attribution: EffectAttribution
    text_detail: str = Field(
        description="Short quote or paraphrase from the report supporting the extraction"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence from 0 to 1",
    )


class ExtractionResult(BaseModel):
    tags: List[SubjectiveEffectTag]
    notes: Optional[str] = Field(
        default=None,
        description="Any limitations, ambiguity, or missing dose information",
    )


@dataclass
class TextChunk:
    text: str
    start: int
    end: int
    index: int
    count: int


CONTROLLED_EFFECT_ONTOLOGY = {
    "visual": {
        "visual distortions": "visual distortions",
        "patterning": "visual distortions",
        "texture rippling": "visual distortions",
        "surface breathing": "visual distortions",
        "melting/flowing": "visual distortions",
        
        "warping": "visual distortions",
         # Reported as objects, surfaces, faces, text, or parts of the visual field appearing bent, curved, stretched, twisted, or otherwise deformed from their expected shape.

        "morphing": "visual distortions",
         # Reported as objects, faces, patterns, or visual forms appearing to transform continuously into other shapes, identities, structures, or appearances over time.

        "drifting": "visual distortions",
         # Reported as stationary objects, patterns, surfaces, or parts of the visual field appearing to slide, float, sway, or slowly move despite remaining physically still.

        "shimmering": "visual distortions",
         # Reported as surfaces, edges, textures, light, or the visual field appearing to flicker, glint, ripple, or display a subtle unsteady luminous instability.

        "visual vibration": "visual distortions",
         # Reported as objects, edges, patterns, or the visual field appearing to tremble, oscillate, buzz, or rapidly quiver in place.

        "fractal imagery": "visual distortions",
         # Reported as self-similar, recursively repeating visual forms appearing at multiple scales, often branching, proliferating, or unfolding in increasingly detailed patterns.

        "geometric imagery": "visual distortions",
         # Reported as grids, lattices, tessellations, mandala-like forms, angular structures, or other abstract geometric patterns appearing in the visual field.
        "lattice imagery": "visual distortions",
        "tessellation": "visual distortions",
        "mandala imagery": "visual distortions",
        "closed-eye visuals": "visual distortions",
        "open-eye visuals": "visual distortions",
        "visual trails": "visual distortions",
        "afterimages": "visual distortions",
        "motion exaggeration": "visual distortions",
        "enhanced colors": "visual distortions",
        "brightness enhancement": "visual distortions",
        "contrast enhancement": "visual distortions",
        "visual sharpening": "visual distortions",
        "visual clarity": "visual distortions",
        "visual snow": "visual distortions",
        "pattern recognition enhancement": "visual distortions",
        "pareidolia": "visual distortions",
        "entity imagery": "visual distortions",
        "shadow imagery": "visual distortions",
        
        "peripheral imagery": "visual distortions",
         # Reported as the appearance of visual forms, motion, figures, patterns, or fleeting events in the peripheral field of vision that are not seen directly when looked at head-on.

        "size distortion": "visual distortions",
         # Reported as objects, body parts, people, or features of the environment appearing unusually enlarged, reduced, inflated, shrunken, or otherwise altered in apparent size.

        "distance distortion": "visual distortions",
         # Reported as objects, people, or parts of the environment appearing closer, farther away, or shifted in apparent distance relative to their expected position.

        "depth distortion": "visual distortions",
         # Reported as an altered sense of spatial depth, with scenes or objects appearing unusually flattened, layered, recessed, expanded, or otherwise changed in three-dimensional structure.

        "foreshortening distortion": "visual distortions",
         # Reported as body parts, objects, or spatial forms appearing unnaturally compressed, elongated, or altered along the line of view, as though perspective shortening is exaggerated or disrupted.

        "perspective distortion": "visual distortions",
         # Reported as changes in the perceived geometric relations of a scene, with angles, proportions, orientation, or lines of sight appearing warped, skewed, tilted, or otherwise visually misaligned.

        "scene replacement": "visual distortions",
         # Reported as part or all of the perceived visual scene appearing substituted with a different place, setting, or constructed environment, while the underlying physical surroundings may remain unchanged.

        "environmental transfiguration": "visual distortions",
         # Reported as the surrounding environment appearing transformed in character, style, material, atmosphere, or identity, so that ordinary settings take on a distinctly altered or reimagined appearance.

        "symbolic imagery": "visual distortions",
         # Reported as visual patterns, forms, objects, or scenes being perceived as containing symbolic, archetypal, mythic, sacred, or meaningful imagery beyond their ordinary appearance.
        "zooming": "visual distortions",
        "visual fragmentation": "visual distortions",
        "double vision": "visual distortions",
        "blurred vision": "visual distortions",
        "field narrowing": "visual distortions",
        "field widening": "visual distortions",
        "synesthetic visuals": "visual distortions",
    },
    "auditory": {
        "auditory distortions": "auditory distortions",
        "auditory enhancement": "auditory distortions",
        "sound clarity enhancement": "auditory distortions",
        "sound amplification": "auditory distortions",
        "sound dampening": "auditory distortions",
        "echoing": "auditory distortions",
        "reverberation": "auditory distortions",
        "pitch distortion": "auditory distortions",
        "tempo distortion": "auditory distortions",
        "auditory warping": "auditory distortions",
        "auditory stretching": "auditory distortions",
        "music appreciation enhancement": "auditory distortions",
        "music immersion": "auditory distortions",
        "internal music": "auditory distortions",
        "auditory imagery": "auditory distortions",
        "voices": "auditory distortions",
        "ringing": "auditory distortions",
        "humming": "auditory distortions",
        "buzzing": "auditory distortions",
        "externalized sounds": "auditory distortions",
        "sound localization distortion": "auditory distortions",
    },
    "somatic": {
        "body load": "body load",
        "somatic heaviness": "body load",
        "somatic lightness": "body load",
        "stimulation": "body load",
        "sedation": "body load",
        "physical energy": "body load",
        "fatigue": "body load",
        "restlessness": "body load",
        "agitation": "body load",
        "tremor": "body load",
        "muscle tension": "body load",
        "muscle relaxation": "body load",
        "jaw tension": "body load",
        "bruxism": "body load",
        "tingling": "body load",
        "pins and needles": "body load",
        "numbness": "body load",
        "warmth": "body load",
        "coldness": "body load",
        "flushing": "body load",
        "chills": "body load",
        "goosebumps": "body load",
        "pressure sensation": "body load",
        "tightness": "body load",
        "lightheadedness": "body load",
        "dizziness": "body load",
        "weakness": "body load",
        "heaviness in limbs": "body load",
        "bodily buzzing": "body load",
        "electric sensations": "body load",
        "physical comfort": "body load",
        "physical discomfort": "body load",
        "pain relief": "body load",
        "pain amplification": "body load",
        "itching": "body load",
    },
    "motor": {
        "motor change": "motor change",
        "incoordination": "motor change",
        "impaired balance": "motor change",
        "impaired fine motor control": "motor change",
        "clumsiness": "motor change",
        "akathisia-like movement": "motor change",
        "compulsive movement": "motor change",
        "stillness": "motor change",
        "immobility": "motor change",
        "muscle twitching": "motor change",
        "jerking": "motor change",
        "shaking": "motor change",
    },
    "gastrointestinal": {
        "gastrointestinal effects": "gastrointestinal effects",
        "nausea": "gastrointestinal effects",
        "vomiting": "gastrointestinal effects",
        "stomach discomfort": "gastrointestinal effects",
        "cramping": "gastrointestinal effects",
        "bloating": "gastrointestinal effects",
        "loss of appetite": "gastrointestinal effects",
        "increased appetite": "gastrointestinal effects",
        "dry mouth": "gastrointestinal effects",
        "salivation": "gastrointestinal effects",
        "difficulty swallowing": "gastrointestinal effects",
    },
    "emotional": {
        "emotional change": "emotional change",
        "euphoria": "emotional change",
        "contentment": "emotional change",
        "emotional warmth": "emotional change",
        "affection": "emotional change",
        "compassion": "emotional change",
        "gratitude": "emotional change",
        "relief": "emotional change",
        "calmness": "emotional change",
        "inner peace": "emotional change",
        "anxiety": "emotional change",
        "apprehension": "emotional change",
        "fear": "emotional change",
        "panic": "emotional change",
        "impending doom": "emotional change",
        "paranoia": "emotional change",
        "irritability": "emotional change",
        "anger": "emotional change",
        "frustration": "emotional change",
        "sadness": "emotional change",
        "grief": "emotional change",
        "emotional lability": "emotional change",
        "emotional blunting": "emotional change",
        "emotional sensitivity": "emotional change",
        "awe": "emotional change",
        "wonder": "emotional change",
        "surprise": "emotional change",
        "giddiness": "emotional change",
        "manic mood": "emotional change",
        "emotional catharsis": "emotional change",
        "helplessness": "emotional change",
        "suspiciousness": "emotional change",
    },
    "cognitive": {
        "cognitive change": "cognitive change",
        "confusion": "cognitive change",
        "mental clarity": "cognitive change",
        "clear-headedness": "cognitive change",
        "foggy thinking": "cognitive change",
        "racing thoughts": "cognitive change",
        "thought slowing": "cognitive change",
        "thought looping": "cognitive change",
        "distractibility": "cognitive change",
        "rumination": "cognitive change",
        "obsessive thinking": "cognitive change",
        "memory impairment": "cognitive change",
        "memory enhancement": "cognitive change",
        "language impairment": "cognitive change",
        "speech fluency enhancement": "cognitive change",
        "analysis enhancement": "cognitive change",
        "introspection enhancement": "cognitive change",
        "novel associations": "cognitive change",
        "conceptual thinking": "cognitive change",
        "suggestibility": "cognitive change",
        "reduced inhibition": "cognitive change",
        "increased focus": "cognitive change",
        "decreased focus": "cognitive change",
        "thought disorganization": "cognitive change",
        "mind blanking": "cognitive change",
        "delusional thinking": "cognitive change",
        "ideas of reference": "cognitive change",
        "grandiosity": "cognitive change",
        "hypervigilance": "cognitive change",
        "compulsive meaning-making": "cognitive change",
        "decision paralysis": "cognitive change",
        "internal cognitive split": "cognitive change",
        # Reported experience of the mind as divided into conflicting parts, streams, or modes of thought.
        # -> “part of me knew it was nonsense, but I still believed it"
        "meta-awareness of irrationality": "cognitive change",
        # Reported awareness that one’s thoughts, beliefs, fears, or interpretations are irrational, exaggerated, or unfounded while still experiencing them.
        # -> “it felt like two parts of my mind were fighting"
    },
    "temporal": {
        "time distortion": "time distortion",
        "time dilation": "time distortion",
        "time contraction": "time distortion",
        "timelessness": "time distortion",
        "time fragmentation": "time distortion",
        "looping sense of time": "time distortion",
    },
    "selfhood": {
        "selfhood change": "selfhood change",
        "dissociation": "selfhood change",
        "depersonalization": "selfhood change",
        "derealization": "selfhood change",
        "ego softening": "selfhood change",
        "ego dissolution": "selfhood change",
        "loss of self-other boundary": "selfhood change",
        "identity confusion": "selfhood change",
        "observer perspective": "selfhood change",
        "disembodiment": "selfhood change",
        "unity experience": "selfhood change",
        "perceived theriomorphosis": "selfhood change",
        "agency disturbance": "selfhood change",
        "identity fluidity": "selfhood change",
        "self-multiplication": "selfhood change",
        "perceived inanimate transformation": "selfhood change",
    },
    "spiritual": {
        "spiritual experience": "spiritual experience",
        "mystical quality": "spiritual experience",
        # Reported sense that the experience possesses an ineffable, profound, or otherworldly character that feels beyond ordinary explanation or language.
        "sacredness": "spiritual experience",
        # Reported feeling that people, objects, places, or the experience itself possess a holy, reverent, or spiritually significant quality.
        "revelatory insight": "spiritual experience",
        # Reported experience of receiving sudden or powerful insight that feels like a revealed truth, message, or realization about reality, life, or oneself.
        "existential insight": "spiritual experience",
        # Reported realization or reflection concerning fundamental aspects of existence, such as meaning, identity, mortality, purpose, or the nature of being.
        "cosmic significance": "spiritual experience",
        # Reported sense that events, thoughts, or perceptions carry immense universal importance or are connected to a larger cosmic order or purpose.
        "oneness": "spiritual experience",
        # Reported experience of unity or dissolution of boundaries between self and other entities, the environment, or the universe.
        "contact-with-presence": "spiritual experience",
        # Reported sensation of encountering, communicating with, or being accompanied by an unseen entity, intelligence, or presence.
    },
    "social": {
        "social change": "social change",
        "sociability enhancement": "social change",
        "talkativeness": "social change",
        "empathy enhancement": "social change",
        "social openness": "social change",
        "social confidence": "social change",
        "social anxiety": "social change",
        "withdrawal": "social change",
        "feeling understood": "social change",
        "feeling connected": "social change",
        "feeling alienated": "social change",
    },
    "tactile": {
        "tactile change": "tactile change",
        "enhanced touch": "tactile change",
        "distorted touch": "tactile change",
        "skin sensitivity": "tactile change",
        "pleasant touch amplification": "tactile change",
        "unpleasant touch amplification": "tactile change",
        "texture recognition suppression": "tactile change"
    },
    "sexual": {
        "sexual change": "sexual change",
        "increased libido": "sexual change",
        "decreased libido": "sexual change",
        "increased sensuality": "sexual change",
        "tactile sensual enhancement": "sexual change",
        "sexual dysfunction": "sexual change",
        "orgasm enhancement": "sexual change",
    },
    "thermal": {
        "temperature change": "temperature change",
        "feeling hot": "temperature change",
        "feeling cold": "temperature change",
        "temperature fluctuation": "temperature change",
    },
    "sleep": {
        "sleep disturbance": "sleep disturbance",
        "insomnia": "sleep disturbance",
        "difficulty falling asleep": "sleep disturbance",
        "sleep fragmentation": "sleep disturbance",
        "vivid dreams": "sleep disturbance",
        "lucid dreams": "sleep disturbance",
        "dream enhancement": "sleep disturbance",
        "drowsiness": "sleep disturbance",
    },
}

EFFECT_ALIASES = EFFECT_ALIASES = {
    # visual: broad / umbrella
    "visual distortion": "visual distortions",
    "visual distortions": "visual distortions",
    "visual warping": "warping",
    "visual morphing": "morphing",
    "visual drifting": "drifting",
    "visual shimmering": "shimmering",
    "visual vibration": "visual vibration",
    "visual anomalies": "visual distortions",
    "visual effects": "visual distortions",
    # visual: pattern / movement / surfaces
    "patterns": "patterning",
    "patterns everywhere": "patterning",
    "pattern recognition": "pattern recognition enhancement",
    "enhanced pattern recognition": "pattern recognition enhancement",
    "rippling": "texture rippling",
    "texture distortion": "texture rippling",
    "textural rippling": "texture rippling",
    "surface rippling": "texture rippling",
    "walls rippling": "texture rippling",
    "floor rippling": "texture rippling",
    "breathing surfaces": "surface breathing",
    "breathing walls": "surface breathing",
    "walls breathing": "surface breathing",
    "floor breathing": "surface breathing",
    "pattern breathing": "surface breathing",
    "surface flowing": "melting/flowing",
    "flowing": "melting/flowing",
    "melting": "melting/flowing",
    "flowing surfaces": "melting/flowing",
    "walls melting": "melting/flowing",
    "objects melting": "melting/flowing",
    "warped visuals": "warping",
    "warped vision": "warping",
    "morphing visuals": "morphing",
    "drifty visuals": "drifting",
    "shimmering visuals": "shimmering",
    # visual: geometry / imagery
    "closed eye visuals": "closed-eye visuals",
    "closed-eye visuals": "closed-eye visuals",
    "cev": "closed-eye visuals",
    "cevs": "closed-eye visuals",
    "open eye visuals": "open-eye visuals",
    "open-eye visuals": "open-eye visuals",
    "oev": "open-eye visuals",
    "oevs": "open-eye visuals",
    "closed-eye geometry": "geometric imagery",
    "geometric patterns": "geometric imagery",
    "geometry": "geometric imagery",
    "geometric visuals": "geometric imagery",
    "fractal visuals": "fractal imagery",
    "fractals": "fractal imagery",
    "fractal patterns": "fractal imagery",
    "lattice patterns": "lattice imagery",
    "grid patterns": "lattice imagery",
    "tessellations": "tessellation",
    "mandalas": "mandala imagery",
    "entity visions": "entity imagery",
    "entities": "entity imagery",
    "beings": "entity imagery",
    "shadow people": "shadow imagery",
    "shadow figures": "shadow imagery",
    "peripheral figures": "peripheral imagery",
    "things in the corner of my eye": "peripheral imagery",
    # visual: trails / afterimages / motion
    "tracers": "visual trails",
    "trails": "visual trails",
    "motion trails": "visual trails",
    "light trails": "visual trails",
    "after images": "afterimages",
    "after-images": "afterimages",
    "motion exaggeration": "motion exaggeration",
    # visual: color / brightness / clarity
    "brighter colors": "enhanced colors",
    "enhanced colour": "enhanced colors",
    "enhanced colours": "enhanced colors",
    "color enhancement": "enhanced colors",
    "colors enhanced": "enhanced colors",
    "colour enhancement": "enhanced colors",
    "colours enhanced": "enhanced colors",
    "more vivid colors": "enhanced colors",
    "more vivid colours": "enhanced colors",
    "vivid colors": "enhanced colors",
    "vivid colours": "enhanced colors",
    "brightness": "brightness enhancement",
    "increased brightness": "brightness enhancement",
    "everything looked brighter": "brightness enhancement",
    "contrast enhancement": "contrast enhancement",
    "higher contrast": "contrast enhancement",
    "sharper vision": "visual sharpening",
    "visual sharpening": "visual sharpening",
    "clearer vision": "visual clarity",
    "enhanced clarity": "visual clarity",
    "crystal clear vision": "visual clarity",
    "visual static": "visual snow",
    # visual: perception / space
    "distance distortions": "distance distortion",
    "distance distortion": "distance distortion",
    "size distortions": "size distortion",
    "micropsia": "size distortion",
    "macropsia": "size distortion",
    "depth distortion": "depth distortion",
    "perspective shift": "perspective distortion",
    "perspective distortion": "perspective distortion",
    "zooming in": "zooming",
    "zooming out": "zooming",
    "visual fragmentation": "visual fragmentation",
    "double vision": "double vision",
    "blurred vision": "blurred vision",
    "tunnel vision": "field narrowing",
    "expanded visual field": "field widening",
    "synaesthetic visuals": "synesthetic visuals",
    "synesthetic visuals": "synesthetic visuals",
    "pareidolia": "pareidolia",
    "seeing faces in things": "pareidolia",
    "seeing patterns in things": "pattern recognition enhancement",
    # auditory
    "auditory distortion": "auditory distortions",
    "sound distortion": "auditory distortions",
    "audio distortion": "auditory distortions",
    "enhanced hearing": "auditory enhancement",
    "heightened hearing": "auditory enhancement",
    "clearer sound": "sound clarity enhancement",
    "crisper sound": "sound clarity enhancement",
    "louder sounds": "sound amplification",
    "sound amplification": "sound amplification",
    "muted sounds": "sound dampening",
    "sounds muffled": "sound dampening",
    "echoes": "echoing",
    "echoing sounds": "echoing",
    "reverb": "reverberation",
    "reverberating sounds": "reverberation",
    "pitch shifts": "pitch distortion",
    "pitch distortion": "pitch distortion",
    "tempo changes": "tempo distortion",
    "slowed music": "tempo distortion",
    "warped sound": "auditory warping",
    "stretched sound": "auditory stretching",
    "music enhancement": "music appreciation enhancement",
    "music appreciation": "music appreciation enhancement",
    "music sounded amazing": "music appreciation enhancement",
    "immersive music": "music immersion",
    "music immersion": "music immersion",
    "hearing music in my head": "internal music",
    "internal music": "internal music",
    "auditory imagery": "auditory imagery",
    "hearing voices": "voices",
    "voices": "voices",
    "ringing ears": "ringing",
    "tinnitus": "ringing",
    "humming sound": "humming",
    "buzzing sound": "buzzing",
    # somatic
    "body heaviness": "somatic heaviness",
    "heavy body": "somatic heaviness",
    "body load/heaviness": "somatic heaviness",
    "heavy limbs": "heaviness in limbs",
    "limb heaviness": "heaviness in limbs",
    "body lightness": "somatic lightness",
    "felt light": "somatic lightness",
    "stimulated": "stimulation",
    "energized": "physical energy",
    "energetic": "physical energy",
    "physical stimulation": "stimulation",
    "sedated": "sedation",
    "sleepy": "sedation",
    "fatigued": "fatigue",
    "tired": "fatigue",
    "restless": "restlessness",
    "physically restless": "restlessness",
    "agitated": "agitation",
    "shaky": "tremor",
    "trembling": "tremor",
    "muscle tension": "muscle tension",
    "tight muscles": "muscle tension",
    "relaxed muscles": "muscle relaxation",
    "muscle relaxation": "muscle relaxation",
    "jaw clenching": "jaw tension",
    "jaw tightness": "jaw tension",
    "teeth grinding": "bruxism",
    "grinding teeth": "bruxism",
    "tingles": "tingling",
    "tingly": "tingling",
    "pins and needles": "pins and needles",
    "numb": "numbness",
    "numbness": "numbness",
    "body warmth": "warmth",
    "warm feeling": "warmth",
    "body coldness": "coldness",
    "cold feeling": "coldness",
    "flushed": "flushing",
    "hot flush": "flushing",
    "chilly": "chills",
    "goose bumps": "goosebumps",
    "gooseflesh": "goosebumps",
    "pressure": "pressure sensation",
    "bodily pressure": "pressure sensation",
    "tightness": "tightness",
    "light headedness": "lightheadedness",
    "light-headedness": "lightheadedness",
    "dizzy": "dizziness",
    "weak": "weakness",
    "buzzing body": "bodily buzzing",
    "body buzzing": "bodily buzzing",
    "electric body sensations": "electric sensations",
    "electrical sensations": "electric sensations",
    "physically comfortable": "physical comfort",
    "body comfort": "physical comfort",
    "physically uncomfortable": "physical discomfort",
    "body discomfort": "physical discomfort",
    "pain reduction": "pain relief",
    "analgesia": "pain relief",
    "increased pain": "pain amplification",
    "itchiness": "itching",
    # motor
    "loss of coordination": "incoordination",
    "poor coordination": "incoordination",
    "uncoordinated": "incoordination",
    "balance problems": "impaired balance",
    "loss of balance": "impaired balance",
    "fine motor impairment": "impaired fine motor control",
    "bad fine motor control": "impaired fine motor control",
    "clumsy": "clumsiness",
    "akathisia": "akathisia-like movement",
    "compulsive movement": "compulsive movement",
    "could not stop moving": "compulsive movement",
    "stillness": "stillness",
    "frozen": "immobility",
    "couldn't move": "immobility",
    "muscle twitching": "muscle twitching",
    "twitching": "muscle twitching",
    "jerks": "jerking",
    "jerking": "jerking",
    "shaking": "shaking",
    # gastrointestinal
    "gi effects": "gastrointestinal effects",
    "gastrointestinal effect": "gastrointestinal effects",
    "queasiness": "nausea",
    "queasy": "nausea",
    "sick to my stomach": "nausea",
    "vomited": "vomiting",
    "throwing up": "vomiting",
    "threw up": "vomiting",
    "stomach ache": "stomach discomfort",
    "upset stomach": "stomach discomfort",
    "stomach discomfort": "stomach discomfort",
    "cramps": "cramping",
    "abdominal cramping": "cramping",
    "bloating": "bloating",
    "no appetite": "loss of appetite",
    "reduced appetite": "loss of appetite",
    "hungry": "increased appetite",
    "increased hunger": "increased appetite",
    "cotton mouth": "dry mouth",
    "dry mouth": "dry mouth",
    "excess salivation": "salivation",
    "drooling": "salivation",
    "trouble swallowing": "difficulty swallowing",
    # emotional: positive / neutral / negative
    "happy": "euphoria",
    "euphoric": "euphoria",
    "content": "contentment",
    "emotionally warm": "emotional warmth",
    "warmth": "emotional warmth",
    "loving": "affection",
    "affectionate": "affection",
    "compassionate": "compassion",
    "grateful": "gratitude",
    "thankful": "gratitude",
    "relieved": "relief",
    "calm": "calmness",
    "peaceful": "inner peace",
    "inner calm": "inner peace",
    "anxious": "anxiety",
    "nervous": "apprehension",
    "nervousness": "apprehension",
    "uneasy": "apprehension",
    "scared": "fear",
    "terrified": "fear",
    "panic attack": "panic",
    "panicky": "panic",
    "sense of doom": "impending doom",
    "impending doom": "impending doom",
    "paranoid": "paranoia",
    "irritable": "irritability",
    "angry": "anger",
    "frustrated": "frustration",
    "depressed": "sadness",
    "sad": "sadness",
    "grieving": "grief",
    "emotionally labile": "emotional lability",
    "mood swings": "emotional lability",
    "emotionally numb": "emotional blunting",
    "blunted affect": "emotional blunting",
    "emotionally sensitive": "emotional sensitivity",
    "awe": "awe",
    "wonder": "wonder",
    "giddy": "giddiness",
    # cognitive
    "mental confusion": "confusion",
    "clear mind": "mental clarity",
    "mental clarity": "mental clarity",
    "clear headed": "clear-headedness",
    "clear-headed": "clear-headedness",
    "brain fog": "foggy thinking",
    "foggy": "foggy thinking",
    "racing mind": "racing thoughts",
    "racing thoughts": "racing thoughts",
    "slow thoughts": "thought slowing",
    "thought slowing": "thought slowing",
    "thought loops": "thought looping",
    "looping thoughts": "thought looping",
    "distracted": "distractibility",
    "ruminating": "rumination",
    "obsessive thoughts": "obsessive thinking",
    "memory problems": "memory impairment",
    "poor memory": "memory impairment",
    "better memory": "memory enhancement",
    "enhanced memory": "memory enhancement",
    "language problems": "language impairment",
    "word-finding difficulty": "language impairment",
    "speech better": "speech fluency enhancement",
    "analyzing everything": "analysis enhancement",
    "analytical thinking": "analysis enhancement",
    "introspective": "introspection enhancement",
    "deep introspection": "introspection enhancement",
    "novel connections": "novel associations",
    "new associations": "novel associations",
    "conceptual thinking": "conceptual thinking",
    "suggestible": "suggestibility",
    "disinhibited": "reduced inhibition",
    "focused": "increased focus",
    "more focused": "increased focus",
    "unfocused": "decreased focus",
    "can't focus": "decreased focus",
    # temporal
    "time distortion": "time distortion",
    "time slowed": "time dilation",
    "time slowing": "time dilation",
    "slowed time": "time dilation",
    "time stretched": "time dilation",
    "time sped up": "time contraction",
    "time speeding up": "time contraction",
    "compressed time": "time contraction",
    "timelessness": "timelessness",
    "time fragments": "time fragmentation",
    "fragmented time": "time fragmentation",
    "time loop": "looping sense of time",
    "time looping": "looping sense of time",
    # selfhood
    "self change": "selfhood change",
    "dissociated": "dissociation",
    "detached": "dissociation",
    "depersonalized": "depersonalization",
    "derealized": "derealization",
    "ego softening": "ego softening",
    "ego loss": "ego dissolution",
    "ego death": "ego dissolution",
    "loss of ego": "ego dissolution",
    "loss of self-other boundary": "loss of self-other boundary",
    "boundary dissolution": "loss of self-other boundary",
    "identity confusion": "identity confusion",
    "observer state": "observer perspective",
    "third-person perspective": "observer perspective",
    "out of body": "disembodiment",
    "out-of-body": "disembodiment",
    "unity": "unity experience",
    "oneness with everything": "unity experience",
    # spiritual
    "spiritual experiences": "spiritual experience",
    "spiritual experience": "spiritual experience",
    "mystical experience": "mystical quality",
    "mystical state": "mystical quality",
    "sacred feeling": "sacredness",
    "holy feeling": "sacredness",
    "revelation": "revelatory insight",
    "epiphany": "revelatory insight",
    "existential insight": "existential insight",
    "cosmic importance": "cosmic significance",
    "cosmic significance": "cosmic significance",
    "oneness": "oneness",
    "presence": "contact-with-presence",
    "presence felt": "contact-with-presence",
    # social
    "social change": "social change",
    "more social": "sociability enhancement",
    "sociable": "sociability enhancement",
    "talkative": "talkativeness",
    "chatty": "talkativeness",
    "empathic": "empathy enhancement",
    "empathetic": "empathy enhancement",
    "more open socially": "social openness",
    "social openness": "social openness",
    "social confidence": "social confidence",
    "confident socially": "social confidence",
    "socially anxious": "social anxiety",
    "withdrawn": "withdrawal",
    "felt understood": "feeling understood",
    "felt connected": "feeling connected",
    "connectedness": "feeling connected",
    "felt alienated": "feeling alienated",
    "alienation": "feeling alienated",
    # tactile
    "tactile effects": "tactile change",
    "enhanced touch": "enhanced touch",
    "touch enhancement": "enhanced touch",
    "distorted touch": "distorted touch",
    "skin sensitivity": "skin sensitivity",
    "pleasant touch enhanced": "pleasant touch amplification",
    "touch felt amazing": "pleasant touch amplification",
    "touch felt awful": "unpleasant touch amplification",
    # sexual
    "sexual effects": "sexual change",
    "higher libido": "increased libido",
    "horny": "increased libido",
    "lower libido": "decreased libido",
    "increased sensuality": "increased sensuality",
    "more sensual": "increased sensuality",
    "sexual touch enhancement": "tactile sensual enhancement",
    "sexual dysfunction": "sexual dysfunction",
    # thermal
    "temperature change": "temperature change",
    "felt hot": "feeling hot",
    "hot": "feeling hot",
    "felt cold": "feeling cold",
    "cold": "feeling cold",
    "temperature swings": "temperature fluctuation",
    # sleep
    "sleep issues": "sleep disturbance",
    "insomnia": "insomnia",
    "couldn't sleep": "insomnia",
    "difficulty sleeping": "difficulty falling asleep",
    "trouble sleeping": "difficulty falling asleep",
    "waking up repeatedly": "sleep fragmentation",
    "fragmented sleep": "sleep fragmentation",
    "vivid dreaming": "vivid dreams",
    "lucid dreaming": "lucid dreams",
    "enhanced dreams": "dream enhancement",
    "dream enhancement": "dream enhancement",
    "drowsy": "drowsiness",
}


def build_controlled_vocabulary_text() -> str:
    lines = []
    for domain, effects in CONTROLLED_EFFECT_ONTOLOGY.items():
        canonical_effects = ", ".join(effects.keys())
        lines.append(f"- {domain}: {canonical_effects}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""
You are a strict information extraction system. You will be extracting subjective effects from a trip report on either a single substance, or a substance combination.

Task:
Extract ONLY subjective effects that are explicitly supported by the report text and attributable to the listed dose_table entries.

Non-negotiable constraints:
- Use ONLY the report text as evidence.
- Do NOT use background knowledge about pharmacology, drug classes, common effects, or likely implications.
- When uncertain, omit the tag.
- Prefer omission over inference.

Attribution constraints:
- If dose_table is non-empty, extract ONLY effects attributable to one or more listed dose_table entries.
- If an effect is attributed in the text to a non-listed substance, omit it unless the same effect is also separately and explicitly attributed to a listed dose_table entry.
- Do NOT fabricate placeholder dose_ids or synthetic exposures.
- Do NOT include withdrawal, comedown, rebound, or aftermath effects unless they are explicitly described as direct effects of the listed dose(s).

Interpret attribution_type strictly as follows:
- single_substance:
  Use when all referenced dose_refs belong to the same substance, even if multiple listed dose events are involved.
  This includes redosing, cumulative exposure, or repeated administrations of the same substance.
- combination:
  Use only when the effect is attributed to multiple different substances together, or when the report explicitly describes an interactional combined state involving multiple substances.
  Do NOT use combination merely because multiple dose events of the same substance are referenced.
- unknown:
  Use only when the report clearly describes an effect and the effect is still plausibly tied to the listed dose_table entries, but attribution to specific listed entry/entries cannot be resolved from the text.
  Do NOT use unknown for effects attributed to non-listed substances, withdrawal states, aftermath states, or unsupported speculation.

Dose reference rules:
- Prefer the narrowest supported attribution.
- If the text supports only one listed dose event for an effect, include only that dose_ref.
- Use multiple same-substance dose_refs only when the text clearly indicates cumulative, repeated, carryover, or post-redose attribution across those dose events.
- Include all and only the dose_table entries actually supported by the text for that effect.
- If the effect is tied to one specific listed dose event, include only that dose_ref.
- If the effect is tied to cumulative or repeated exposure to the same substance across multiple listed dose events, include those same-substance dose_refs and still use attribution_type="single_substance".
- If the effect is tied to multiple different listed substances, include those dose_refs and use attribution_type="combination".
- Do not include extra dose_refs just because they appear elsewhere in the report.

Evidence constraints:
- Every extracted tag must be supported by a short, local quote or minimally trimmed excerpt from the text.
- text_detail must stay close to the exact wording and must not introduce interpretation.
- If no short supporting excerpt exists, omit the tag.

De-duplication constraints:
- Do not output duplicates.
- Do not output multiple overlapping tags for the same evidence passage unless the passage independently supports multiple distinct experiences.
- When several tags could fit one passage, choose exactly one: the most specific directly supported tag. If none is directly supported, omit.

Ontology constraints:
- Map only to canonical tags from the controlled vocabulary.
- Do not invent tags.
- Use broad fallback tags only when a more specific canonical tag is not directly supported.

Examples:
- If d1 and d2 are both MDMA and the effect is described after a redose or as cumulative across both doses, use attribution_type="single_substance" and include [d1, d2].
- If d1 is MDMA and d2 is cannabis and the effect is described as arising from taking them together, use attribution_type="combination" and include [d1, d2].
- If only d2 is clearly linked to the effect, include only [d2], not [d1, d2].

Controlled vocabulary:
{build_controlled_vocabulary_text()}
"""

USER_TEMPLATE = """
Extract subjective effects from this report.

Document:
{doc_json}
"""

DEFAULT_REPORT_CHUNK_SIZE_CHARS = 8000
DEFAULT_REPORT_CHUNK_OVERLAP_CHARS = 1000
DEFAULT_MAX_COMPLETION_TOKENS = 4000
SAFER_MAX_REPORT_TEXT_CHARS = 8000


def normalize_substance_name(value) -> Optional[str]:
    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def build_dose_phrase(entry: dict) -> Optional[str]:
    amount = entry.get("amount")
    form = entry.get("form")

    parts = []
    if isinstance(amount, str) and amount.strip():
        parts.append(amount.strip())
    if isinstance(form, str) and form.strip():
        parts.append(form.strip())

    if parts:
        return " ".join(parts)
    return None


def normalize_dose_entry(entry: dict, index: int) -> dict:
    normalized_entry = dict(entry)
    normalized_entry["dose_id"] = entry.get("dose_id") or f"d{index}"
    normalized_entry["substance"] = normalize_substance_name(entry.get("substance"))
    normalized_entry["dose"] = entry.get("dose") or build_dose_phrase(entry)
    normalized_entry["route"] = entry.get("route") or entry.get("method")
    return normalized_entry


def build_doc_payload(doc: dict) -> dict:
    raw_dose_table = doc.get("dose_table", []) or []
    dose_table = []
    for index, entry in enumerate(raw_dose_table, start=1):
        dose_table.append(normalize_dose_entry(entry, index))

    return {
        "exp_id": doc.get("exp_id"),
        "title": doc.get("title"),
        "substance": doc.get("substance"),
        "body_weight": doc.get("body_weight"),
        "dose_table": dose_table,
        "report_text": doc.get("report_text", ""),
        "footdata": {
            "exp_year": (doc.get("footdata") or {}).get("exp_year"),
            "gender": (doc.get("footdata") or {}).get("gender"),
            "age_at_time_of_experience": (doc.get("footdata") or {}).get(
                "age_at_time_of_experience"
            ),
            "published": (doc.get("footdata") or {}).get("published"),
        },
    }


def build_response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "subjective_effect_extraction",
            "schema": ExtractionResult.model_json_schema(),
        },
    }


def normalize_effect_label(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None

    normalized = " ".join(value.strip().lower().replace("_", " ").split())
    if not normalized:
        return None
    return EFFECT_ALIASES.get(normalized, normalized)


def build_effect_index() -> dict[str, dict[str, str]]:
    effect_index = {}
    for domain, effects in CONTROLLED_EFFECT_ONTOLOGY.items():
        for effect, parent_effect in effects.items():
            effect_index[effect] = {
                "domain": domain,
                "effect": effect,
                "parent_effect": parent_effect,
            }
    return effect_index


EFFECT_INDEX = build_effect_index()


def extract_response_json(response) -> dict:
    if hasattr(response, "outputs"):
        outputs = response.outputs
    elif isinstance(response, dict):
        outputs = response.get("outputs", [])
    else:
        outputs = []

    for output in outputs:
        content = getattr(output, "content", None)
        if content is None and isinstance(output, dict):
            content = output.get("content")

        if isinstance(content, dict):
            return content

        if isinstance(content, str):
            return parse_response_json(content)

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        return parse_response_json(text)

                    if item.get("content") and isinstance(item["content"], str):
                        return parse_response_json(item["content"])

                else:
                    text = getattr(item, "text", None)
                    if text:
                        return parse_response_json(text)

                    nested_content = getattr(item, "content", None)
                    if isinstance(nested_content, str):
                        return parse_response_json(nested_content)

    raise ValueError("Mistral response did not contain a JSON payload in outputs")


def parse_response_json(content: str) -> dict:
    try:
        return json.loads(content)
    except JSONDecodeError as exc:
        content_preview = content[max(0, exc.pos - 120) : exc.pos + 120]
        raise ValueError(
            "Mistral returned invalid JSON. This is often caused by output truncation; "
            "try increasing MAX_COMPLETION_TOKENS or decreasing MAX_REPORT_TEXT_CHARS / "
            f"REPORT_CHUNK_SIZE_CHARS. Parse error: {exc}. Nearby content: {content_preview!r}"
        ) from exc


def split_text_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[TextChunk]:
    if not text:
        return [TextChunk(text="", start=0, end=0, index=1, count=1)]

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if len(text) <= chunk_size:
        return [TextChunk(text=text, start=0, end=len(text), index=1, count=1)]

    separators = ("\n\n", "\n", ". ")
    raw_chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        target_end = min(start + chunk_size, text_length)
        end = target_end

        if target_end < text_length:
            for separator in separators:
                split_at = text.rfind(separator, start, target_end)
                if split_at > start:
                    end = split_at + len(separator)
                    break

        if end <= start:
            end = target_end

        chunk_text = text[start:end].strip()
        if chunk_text:
            raw_chunks.append((chunk_text, start, end))

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    count = len(raw_chunks)
    return [
        TextChunk(text=chunk_text, start=start, end=end, index=i, count=count)
        for i, (chunk_text, start, end) in enumerate(raw_chunks, start=1)
    ]


def normalize_evidence_text(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[\"'`]+", "", value)
    value = re.sub(r"[^a-z0-9\s]", "", value)
    return value.strip()


def evidence_texts_equivalent(a: str, b: str) -> bool:
    na = normalize_evidence_text(a)
    nb = normalize_evidence_text(b)

    if not na or not nb:
        return False
    if na == nb:
        return True

    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)

    if len(shorter) >= 30 and shorter in longer:
        return True

    return SequenceMatcher(None, na, nb).ratio() >= 0.90


def build_effect_group_key(
    tag: SubjectiveEffectTag,
) -> tuple[str, str, str, Optional[str]]:
    return (
        tag.domain,
        tag.effect,
        tag.parent_effect,
        (
            tag.detail.strip().lower()
            if isinstance(tag.detail, str) and tag.detail.strip()
            else None
        ),
    )


def dose_id_set(tag: SubjectiveEffectTag) -> set[str]:
    return {
        ref.dose_id.strip()
        for ref in tag.attribution.dose_refs
        if isinstance(ref.dose_id, str) and ref.dose_id.strip()
    }


def is_same_substance_dose_set(tag: SubjectiveEffectTag) -> bool:
    substances = {
        ref.substance.strip().lower()
        for ref in tag.attribution.dose_refs
        if isinstance(ref.substance, str) and ref.substance.strip()
    }
    return len(substances) == 1


def attribution_text(tag: SubjectiveEffectTag) -> str:
    note = tag.attribution.attribution_note or ""
    return f"{tag.text_detail} {note}".lower()


def has_explicit_combination_support(tag: SubjectiveEffectTag) -> bool:
    if tag.attribution.attribution_type != "combination":
        return False

    text = attribution_text(tag)
    substances = [
        ref.substance.strip().lower()
        for ref in tag.attribution.dose_refs
        if isinstance(ref.substance, str) and ref.substance.strip()
    ]

    named_substances = sum(1 for s in set(substances) if s in text)
    combo_cues = (
        "together",
        "combined",
        "combination",
        "mix",
        "mixed",
        "with ",
        "after adding",
        "adding more",
        "both",
    )

    return named_substances >= 2 or any(cue in text for cue in combo_cues)


def attribution_rank(tag: SubjectiveEffectTag) -> int:
    tag_type = tag.attribution.attribution_type
    refs = dose_id_set(tag)

    if tag_type == "unknown":
        return 0

    if tag_type == "combination":
        if has_explicit_combination_support(tag):
            return 4
        return 1

    if len(refs) == 1:
        return 3
    if is_same_substance_dose_set(tag):
        return 2
    return 1


def interpretive_note_penalty(tag: SubjectiveEffectTag) -> int:
    note = (tag.attribution.attribution_note or "").lower()
    penalties = (
        "suggesting",
        "consistent with",
        "indicating",
        "which is consistent with",
        "implying",
    )
    return 5 if any(p in note for p in penalties) else 0


def tag_score(tag: SubjectiveEffectTag) -> float:
    score = 0.0
    score += tag.confidence * 100.0
    score += attribution_rank(tag) * 10.0
    score += min(len(normalize_evidence_text(tag.text_detail)), 160) / 40.0
    score -= interpretive_note_penalty(tag)
    return score


def choose_best_candidate(
    candidates: List[SubjectiveEffectTag],
) -> SubjectiveEffectTag:
    return max(candidates, key=tag_score)


def append_note(existing_note: Optional[str], extra_note: str) -> str:
    extra_note = extra_note.strip()
    if not extra_note:
        return existing_note or ""
    if not existing_note:
        return extra_note
    if extra_note in existing_note:
        return existing_note
    return f"{existing_note.strip()} {extra_note}"


def summarize_rejected_tags(
    rejected_tags: List[str], max_examples: int = 5
) -> Optional[str]:
    if not rejected_tags:
        return None

    unique_examples = []
    for tag in rejected_tags:
        if tag not in unique_examples:
            unique_examples.append(tag)

    example_text = ", ".join(unique_examples[:max_examples])
    note = f"Rejected {len(rejected_tags)} unsupported effect tag proposals during validation."
    if example_text:
        note = f"{note} Examples: {example_text}."
    return note


def canonicalize_effect_tag(raw_tag: dict) -> tuple[Optional[dict], Optional[str]]:
    effect_candidate = normalize_effect_label(raw_tag.get("effect"))
    parent_candidate = normalize_effect_label(raw_tag.get("parent_effect"))
    legacy_candidate = normalize_effect_label(raw_tag.get("subjective_effect"))

    if effect_candidate in EFFECT_INDEX:
        return dict(EFFECT_INDEX[effect_candidate]), None

    if effect_candidate:
        return None, effect_candidate

    if legacy_candidate in EFFECT_INDEX:
        return dict(EFFECT_INDEX[legacy_candidate]), None

    if legacy_candidate:
        return None, legacy_candidate

    if parent_candidate in EFFECT_INDEX:
        return dict(EFFECT_INDEX[parent_candidate]), None

    if parent_candidate:
        return None, parent_candidate

    return None, None


def sanitize_extraction_payload(raw_result: dict, dose_table: List[dict]) -> dict:
    dose_index = {
        entry["dose_id"]: entry
        for entry in dose_table
        if isinstance(entry.get("dose_id"), str) and entry["dose_id"].strip()
    }

    raw_tags = raw_result.get("tags")
    sanitized_tags = []
    rejected_tags = []
    for raw_tag in raw_tags if isinstance(raw_tags, list) else []:
        if not isinstance(raw_tag, dict):
            continue

        canonical_effect_tag, rejected_effect_label = canonicalize_effect_tag(raw_tag)
        detail = raw_tag.get("detail")
        text_detail = raw_tag.get("text_detail")
        if canonical_effect_tag is None:
            if rejected_effect_label:
                rejected_tags.append(rejected_effect_label)
            continue
        if not isinstance(text_detail, str) or not text_detail.strip():
            continue
        if not isinstance(detail, str) or not detail.strip():
            detail = None

        confidence = raw_tag.get("confidence")
        if isinstance(confidence, (int, float)):
            confidence_value = min(1.0, max(0.0, float(confidence)))
        else:
            confidence_value = 0.0

        raw_attribution = raw_tag.get("attribution")
        if not isinstance(raw_attribution, dict):
            raw_attribution = {}

        attribution_type = raw_attribution.get("attribution_type")
        if attribution_type not in {"single_substance", "combination", "unknown"}:
            attribution_type = "unknown"

        attribution_note = raw_attribution.get("attribution_note")
        if not isinstance(attribution_note, str):
            attribution_note = None

        sanitized_dose_refs = []
        raw_dose_refs = raw_attribution.get("dose_refs")
        invalid_dose_ref_found = False
        for raw_dose_ref in raw_dose_refs if isinstance(raw_dose_refs, list) else []:
            if not isinstance(raw_dose_ref, dict):
                invalid_dose_ref_found = True
                continue

            dose_id = raw_dose_ref.get("dose_id")
            if not isinstance(dose_id, str) or not dose_id.strip():
                invalid_dose_ref_found = True
                continue

            dose_id = dose_id.strip()
            source_entry = dose_index.get(dose_id, {})
            substance = raw_dose_ref.get("substance")
            if not isinstance(substance, str) or not substance.strip():
                substance = source_entry.get("substance")
            if not isinstance(substance, str) or not substance.strip():
                invalid_dose_ref_found = True
                continue

            dose = raw_dose_ref.get("dose")
            if not isinstance(dose, str) or not dose.strip():
                dose = source_entry.get("dose")
            if not isinstance(dose, str) or not dose.strip():
                dose = None

            route = raw_dose_ref.get("route")
            if not isinstance(route, str) or not route.strip():
                route = source_entry.get("route")
            if not isinstance(route, str) or not route.strip():
                route = None

            sanitized_dose_refs.append(
                {
                    "dose_id": dose_id,
                    "substance": substance.strip(),
                    "dose": dose.strip() if isinstance(dose, str) else None,
                    "route": route.strip() if isinstance(route, str) else None,
                }
            )

        if (
            invalid_dose_ref_found
            and attribution_type != "unknown"
            and not sanitized_dose_refs
        ):
            attribution_type = "unknown"
            attribution_note = append_note(
                attribution_note,
                "Malformed dose references were discarded during validation.",
            )

        sanitized_tags.append(
            {
                "domain": canonical_effect_tag["domain"],
                "effect": canonical_effect_tag["effect"],
                "subjective_effect": canonical_effect_tag["parent_effect"],
                "parent_effect": canonical_effect_tag["parent_effect"],
                "detail": detail.strip() if isinstance(detail, str) else None,
                "attribution": {
                    "attribution_type": attribution_type,
                    "dose_refs": sanitized_dose_refs,
                    "attribution_note": attribution_note,
                },
                "text_detail": text_detail.strip(),
                "confidence": confidence_value,
            }
        )

    notes = raw_result.get("notes")
    if not isinstance(notes, str):
        notes = None
    rejected_note = summarize_rejected_tags(rejected_tags)
    if rejected_note:
        notes = append_note(notes, rejected_note)

    return {
        "tags": sanitized_tags,
        "notes": notes,
    }


def mergeable_note_paragraphs(note: str) -> List[str]:
    kept = []
    for paragraph in [p.strip() for p in note.split("\n\n") if p.strip()]:
        if paragraph.startswith("Rejected "):
            kept.append(paragraph)
        elif "Malformed dose references were discarded during validation." in paragraph:
            kept.append(paragraph)
    return kept


def merge_extraction_results(results: List[ExtractionResult]) -> ExtractionResult:
    grouped: dict[
        tuple[str, str, str, Optional[str]],
        List[SubjectiveEffectTag],
    ] = {}
    merged_notes: List[str] = []

    for result in results:
        for tag in result.tags:
            group_key = build_effect_group_key(tag)
            grouped.setdefault(group_key, []).append(tag)
        if result.notes:
            for paragraph in mergeable_note_paragraphs(result.notes):
                if paragraph not in merged_notes:
                    merged_notes.append(paragraph)

    final_tags: List[SubjectiveEffectTag] = []

    for group_tags in grouped.values():
        evidence_clusters: List[List[SubjectiveEffectTag]] = []

        for tag in group_tags:
            placed = False
            for cluster in evidence_clusters:
                if any(
                    evidence_texts_equivalent(tag.text_detail, existing.text_detail)
                    for existing in cluster
                ):
                    cluster.append(tag)
                    placed = True
                    break
            if not placed:
                evidence_clusters.append([tag])

        for cluster in evidence_clusters:
            final_tags.append(choose_best_candidate(cluster))

    return ExtractionResult(
        tags=final_tags,
        notes="\n\n".join(merged_notes) if merged_notes else None,
    )


def enrich_result_with_dose_table(
    result: ExtractionResult, dose_table: List[dict]
) -> ExtractionResult:
    dose_index = {
        entry["dose_id"]: entry for entry in dose_table if entry.get("dose_id")
    }

    for tag in result.tags:
        for dose_ref in tag.attribution.dose_refs:
            source_entry = dose_index.get(dose_ref.dose_id)
            if not source_entry:
                continue

            if dose_ref.substance in (None, ""):
                dose_ref.substance = source_entry.get("substance") or dose_ref.substance
            if dose_ref.dose is None:
                dose_ref.dose = source_entry.get("dose")
            if dose_ref.route is None:
                dose_ref.route = source_entry.get("route")

    return result


def extract_effects_for_payload(
    client: Mistral, model: str, payload: dict
) -> ExtractionResult:
    max_completion_tokens = int(
        os.getenv("MAX_COMPLETION_TOKENS", str(DEFAULT_MAX_COMPLETION_TOKENS))
    )

    response = client.beta.conversations.start(
        model=model,
        inputs=[
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    doc_json=json.dumps(payload, ensure_ascii=False)
                ),
            },
        ],
        instructions=SYSTEM_PROMPT,
        completion_args={
            "temperature": 0,
            "max_tokens": max_completion_tokens,
            "response_format": build_response_format(),
        },
        tools=[],
    )

    raw_result = extract_response_json(response)
    result = ExtractionResult.model_validate(
        sanitize_extraction_payload(raw_result, payload["dose_table"])
    )
    return enrich_result_with_dose_table(result, payload["dose_table"])


def extract_effects(client: Mistral, model: str, doc: dict) -> ExtractionResult:
    payload = build_doc_payload(doc)
    report_text = payload.get("report_text", "") or ""

    max_report_text_chars = int(
        os.getenv("MAX_REPORT_TEXT_CHARS", str(SAFER_MAX_REPORT_TEXT_CHARS))
    )
    chunk_size = int(
        os.getenv("REPORT_CHUNK_SIZE_CHARS", str(DEFAULT_REPORT_CHUNK_SIZE_CHARS))
    )
    chunk_overlap = int(
        os.getenv("REPORT_CHUNK_OVERLAP_CHARS", str(DEFAULT_REPORT_CHUNK_OVERLAP_CHARS))
    )

    if len(report_text) <= max_report_text_chars:
        return extract_effects_for_payload(client, model, payload)

    chunks = split_text_into_chunks(
        report_text, chunk_size=chunk_size, overlap=chunk_overlap
    )
    chunk_results = []

    for chunk in chunks:
        chunk_payload = dict(payload)
        chunk_payload["report_text"] = chunk.text
        chunk_payload["report_chunk"] = {
            "index": chunk.index,
            "count": chunk.count,
            "strategy": "char_window_with_overlap",
            "start_char": chunk.start,
            "end_char": chunk.end,
        }
        chunk_results.append(extract_effects_for_payload(client, model, chunk_payload))

    merged_result = merge_extraction_results(chunk_results)
    if merged_result.notes:
        merged_result.notes = (
            f"Processed in {len(chunks)} chunks because report_text exceeded "
            f"{max_report_text_chars} characters.\n\n{merged_result.notes}"
        )
    else:
        merged_result.notes = (
            f"Processed in {len(chunks)} chunks because report_text exceeded "
            f"{max_report_text_chars} characters."
        )

    return merged_result


def persist_result(
    collection,
    doc: dict,
    result: ExtractionResult,
    model: str,
    source_collection_name: str,
):
    now = datetime.now(timezone.utc)
    exp_id = doc.get("exp_id")

    update_result = collection.update_one(
        {"exp_id": exp_id},
        {
            "$set": {
                "exp_id": exp_id,
                "source_doc_id": doc.get("_id"),
                "source_collection": source_collection_name,
                "title": doc.get("title"),
                "substance": doc.get("substance"),
                "subjective_effect_tags": [tag.model_dump() for tag in result.tags],
                "subjective_effect_extraction": {
                    "model_provider": "mistral",
                    "model_name": model,
                    "notes": result.notes,
                    "tag_count": len(result.tags),
                    "extracted_at": now,
                    "status": "complete",
                },
            }
        },
        upsert=True,
    )

    if update_result.matched_count == 0 and update_result.upserted_id is None:
        raise RuntimeError(f"MongoDB persist failed for exp_id={exp_id!r}")


def mark_error(
    collection, doc: dict, model: str, error_message: str, source_collection_name: str
):
    now = datetime.now(timezone.utc)
    exp_id = doc.get("exp_id")

    update_result = collection.update_one(
        {"exp_id": exp_id},
        {
            "$set": {
                "exp_id": exp_id,
                "source_doc_id": doc.get("_id"),
                "source_collection": source_collection_name,
                "title": doc.get("title"),
                "substance": doc.get("substance"),
                "subjective_effect_extraction": {
                    "model_provider": "mistral",
                    "model_name": model,
                    "status": "error",
                    "error": error_message[:2000],
                    "extracted_at": now,
                },
            }
        },
        upsert=True,
    )

    if update_result.matched_count == 0 and update_result.upserted_id is None:
        print(
            f"WARNING: failed to record error status for exp_id={exp_id!r}",
            file=sys.stderr,
            flush=True,
        )


def load_source_batch(source_collection, query: dict, batch_size: int) -> list[dict]:
    """Avoid holding a Mongo cursor open while each doc is processed downstream."""
    try:
        return list(
            source_collection.find(query, max_time_ms=30000)
            .sort("_id", 1)
            .limit(batch_size)
        )
    except CursorNotFound:
        print(
            "WARNING: source cursor expired while loading batch; retrying once.",
            file=sys.stderr,
            flush=True,
        )
        return list(
            source_collection.find(query, max_time_ms=30000)
            .sort("_id", 1)
            .limit(batch_size)
        )


def load_pending_batch(
    source_collection,
    target_collection,
    batch_size: int,
    source_scan_batch_size: int,
) -> list[dict]:
    """
    Incrementally scan source docs and filter out already-completed exp_ids.

    This avoids a full target-collection distinct followed by a large $nin query,
    which becomes unresponsive on larger datasets.
    """
    pending_docs: list[dict] = []
    last_seen_id = None
    scanned_docs = 0

    while len(pending_docs) < batch_size:
        query = {}
        if last_seen_id is not None:
            query["_id"] = {"$gt": last_seen_id}

        candidate_docs = load_source_batch(
            source_collection, query, source_scan_batch_size
        )
        if not candidate_docs:
            break

        scanned_docs += len(candidate_docs)
        last_seen_id = candidate_docs[-1]["_id"]

        candidate_exp_ids = [
            doc.get("exp_id") for doc in candidate_docs if doc.get("exp_id") is not None
        ]

        completed_exp_ids = set()
        if candidate_exp_ids:
            completed_exp_ids = set(
                target_collection.distinct(
                    "exp_id",
                    {
                        "subjective_effect_extraction.status": "complete",
                        "exp_id": {"$in": candidate_exp_ids},
                    },
                    maxTimeMS=30000,
                )
            )

        for doc in candidate_docs:
            if doc.get("exp_id") not in completed_exp_ids:
                pending_docs.append(doc)
                if len(pending_docs) >= batch_size:
                    break

    print(
        f"Selected {len(pending_docs)} pending documents after scanning {scanned_docs} source docs.",
        flush=True,
    )
    return pending_docs


def main():
    mistral_api_key = os.environ["MISTRAL_API_KEY"]
    mistral_model = os.getenv("MISTRAL_MODEL", "mistral-large-2512")
    mongo_uri = os.getenv("MONGO_URI", "mongodb://host.docker.internal:27017")
    mongo_db = os.getenv("MONGO_DB", "tripindex")
    mongo_source_collection = os.getenv("MONGO_SOURCE_COLLECTION", "erowid")
    mongo_target_collection = os.getenv("MONGO_TARGET_COLLECTION", "erowid_effects")
    batch_size = int(os.getenv("BATCH_SIZE", "10"))
    source_scan_batch_size = int(
        os.getenv("SOURCE_SCAN_BATCH_SIZE", str(batch_size * 5))
    )
    dry_run = os.getenv("DRY_RUN", "false").lower() == "true"

    mongo = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    mongo.admin.command("ping")
    db = mongo[mongo_db]
    source_collection = db[mongo_source_collection]
    target_collection = db[mongo_target_collection]

    mistral_client = Mistral(api_key=mistral_api_key)

    print(
        (
            f"Connected to MongoDB db={mongo_db} "
            f"source_collection={mongo_source_collection} "
            f"target_collection={mongo_target_collection} "
            f"dry_run={dry_run}"
        ),
        flush=True,
    )
    if dry_run:
        print("DRY_RUN is enabled; results will not be written to MongoDB.", flush=True)

    print(
        (
            f"Selecting up to {batch_size} pending documents "
            f"(source_scan_batch_size={source_scan_batch_size}) ..."
        ),
        flush=True,
    )
    docs = load_pending_batch(
        source_collection,
        target_collection,
        batch_size=batch_size,
        source_scan_batch_size=source_scan_batch_size,
    )

    processed = 0
    for doc in docs:
        exp_id = doc.get("exp_id")
        print(f"Processing exp_id={exp_id} ...", flush=True)

        try:
            result = extract_effects(mistral_client, mistral_model, doc)

            if dry_run:
                print(
                    json.dumps(
                        {
                            "exp_id": exp_id,
                            "tags": [t.model_dump() for t in result.tags],
                            "notes": result.notes,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                persist_result(
                    target_collection,
                    doc,
                    result,
                    mistral_model,
                    mongo_source_collection,
                )

            processed += 1
            time.sleep(0.5)

        except Exception as e:
            print(f"ERROR exp_id={exp_id}: {e}", file=sys.stderr, flush=True)
            if not dry_run:
                mark_error(
                    target_collection,
                    doc,
                    mistral_model,
                    str(e),
                    mongo_source_collection,
                )

    print(f"Done. Processed {processed} documents.", flush=True)


if __name__ == "__main__":
    main()
