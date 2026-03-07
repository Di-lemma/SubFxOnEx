import json
import os
import sys
import time
from datetime import datetime, timezone
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
        description="Deprecated alias of effect retained for backward compatibility with older downstream consumers"
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

CONTROLLED_EFFECT_ONTOLOGY = {
    "visual": {
        "visual distortions": "visual distortions",
        "patterning": "visual distortions",
        "texture rippling": "visual distortions",
        "surface breathing": "visual distortions",
        "melting/flowing": "visual distortions",
        "warping": "visual distortions",
        "morphing": "visual distortions",
        "drifting": "visual distortions",
        "shimmering": "visual distortions",
        "visual vibration": "visual distortions",
        "fractal imagery": "visual distortions",
        "geometric imagery": "visual distortions",
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
        "size distortion": "visual distortions",
        "distance distortion": "visual distortions",
        "depth distortion": "visual distortions",
        "foreshortening distortion": "visual distortions",
        "perspective distortion": "visual distortions",
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
    },

    "spiritual": {
        "spiritual experience": "spiritual experience",
        "mystical quality": "spiritual experience",
        "sacredness": "spiritual experience",
        "revelatory insight": "spiritual experience",
        "existential insight": "spiritual experience",
        "cosmic significance": "spiritual experience",
        "oneness": "spiritual experience",
        "contact-with-presence": "spiritual experience",
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
    },

    "sexual": {
        "sexual change": "sexual change",
        "increased libido": "sexual change",
        "decreased libido": "sexual change",
        "increased sensuality": "sexual change",
        "tactile sensual enhancement": "sexual change",
        "sexual dysfunction": "sexual change",
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
You are an expert biomedical and psychopharmacology information extraction engine.

Task:
Extract ONLY subjective effects that are directly supported by the report text and attributable to the dose or dose combination listed in the dose_table.

Primary objective:
Produce HIGH-PRECISION extractions suitable for later use in a knowledge graph.
When uncertain, OMIT the tag rather than guessing.

Critical rules:
1. Extract an effect ONLY when the report text explicitly states it, clearly describes it phenomenologically, or supports it with very strong local evidence.
2. Do NOT extract effects based on weak inference, speculation, interpretation, background knowledge, or likely-but-unstated implications.
3. If a tag would require guessing beyond the text, omit it.
4. Prefer UNDER-extraction to over-extraction.

Dose attribution rules:
5. Extract effects ONLY if they pertain to the dose or dose combination listed in the dose_table.
6. Use dose_table entries as the source of truth for exposure attribution. When possible, reference the relevant entries by their dose_id in attribution.dose_refs.
7. If an effect pertains to one dose entry, set attribution_type to single_substance and include only that entry.
8. If an effect clearly arises from an interaction, combined state, or sequential combination of multiple listed dose entries, set attribution_type to combination and include all relevant entries.
9. Do NOT assign combination attribution merely because multiple substances or doses appear in the report overall. Use combination only when the text supports a combined or interactional attribution.
10. If the dose_table is empty or insufficient, still extract effects from the report, but set attribution_type to unknown, use an empty dose_refs list when needed, and explain ambiguity in attribution_note or notes.
11. If a non-listed substance is mentioned as contributing to an effect, do NOT attribute the effect to that non-listed substance. Either omit the effect or extract it only if the listed dose_table entries still clearly support attribution.

Effect selection rules:
12. Extract subjective effects only. Do NOT extract logistics, social facts, advice, moral commentary, scene description, or metadata.
13. Do NOT diagnose the author.
14. Do NOT convert consequences, aftermath details, bodily damage, or practical problems into subjective effects unless the text directly describes them as experienced effects.
15. Do NOT convert generic enthusiasm or stylistic language into extra effect tags unless a specific effect is actually described.
16. Do NOT infer a more specific effect from a broader description unless the specific effect is clearly supported by the wording.
17. Do NOT infer bodily sensations from unrelated phrases. For example, dizziness does not imply tingling; nose irritation does not imply numbness; meaningful conversation does not imply analysis enhancement.
18. Do NOT extract the same local phenomenon multiple ways unless the text clearly supports multiple distinct effects.

Ontology mapping rules:
19. Map each extracted effect to the most specific matching canonical effect tag from the controlled vocabulary below.
20. Use broad fallback tags such as visual distortions, body load, emotional change, cognitive change, selfhood change, gastrointestinal effects, auditory distortions, or sleep disturbance ONLY when no more specific canonical tag is directly supported by the text.
21. Do NOT invent new canonical effect tags outside the controlled vocabulary.
22. If the text contains finer phenomenology than the canonical tag captures, preserve it in detail rather than inventing a new effect tag.

De-duplication and overlap rules:
23. Prefer the SMALLEST set of non-overlapping tags that faithfully captures the report.
24. Do NOT emit multiple near-synonymous, parent/child, or highly overlapping tags for the same evidence snippet unless each is independently supported.
25. If one specific tag fully captures the evidence, do not also emit a broader sibling or nearby interpretation from the same snippet.
26. For a single passage, prefer one well-supported canonical tag over several speculative neighboring tags.

Evidence rules:
27. Keep text_detail brief, grounded, and as close as possible to the exact supporting wording.
28. text_detail must point to the actual passage supporting the tag, not a paraphrase that introduces new interpretation.
29. If support is weak, omit the tag instead of compensating with an attribution_note.

Output quality rules:
30. Favor precision, specificity, and evidence quality over recall.
31. Return only tags that would be defensible as graph edges later.
32. Avoid duplicates.

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
            "age_at_time_of_experience": (doc.get("footdata") or {}).get("age_at_time_of_experience"),
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
        content_preview = content[max(0, exc.pos - 120):exc.pos + 120]
        raise ValueError(
            "Mistral returned invalid JSON. This is often caused by output truncation; "
            "try increasing MAX_COMPLETION_TOKENS or decreasing MAX_REPORT_TEXT_CHARS / "
            f"REPORT_CHUNK_SIZE_CHARS. Parse error: {exc}. Nearby content: {content_preview!r}"
        ) from exc


def split_text_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    if not text:
        return [""]

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if len(text) <= chunk_size:
        return [text]

    separators = ("\n\n", "\n", ". ")
    chunks: List[str] = []
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

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_tag_dedup_key(tag: SubjectiveEffectTag) -> str:
    dose_ids = sorted(
        dose_ref.dose_id.strip()
        for dose_ref in tag.attribution.dose_refs
        if isinstance(dose_ref.dose_id, str) and dose_ref.dose_id.strip()
    )
    return json.dumps(
        {
            "domain": tag.domain,
            "effect": tag.effect,
            "parent_effect": tag.parent_effect,
            "attribution_type": tag.attribution.attribution_type,
            "dose_ids": dose_ids,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def append_note(existing_note: Optional[str], extra_note: str) -> str:
    extra_note = extra_note.strip()
    if not extra_note:
        return existing_note or ""
    if not existing_note:
        return extra_note
    if extra_note in existing_note:
        return existing_note
    return f"{existing_note.strip()} {extra_note}"


def summarize_rejected_tags(rejected_tags: List[str], max_examples: int = 5) -> Optional[str]:
    if not rejected_tags:
        return None

    unique_examples = []
    for tag in rejected_tags:
        if tag not in unique_examples:
            unique_examples.append(tag)

    example_text = ", ".join(unique_examples[:max_examples])
    note = (
        f"Rejected {len(rejected_tags)} unsupported effect tag proposals during validation."
    )
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

        if invalid_dose_ref_found and attribution_type != "unknown" and not sanitized_dose_refs:
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


def merge_extraction_results(results: List[ExtractionResult]) -> ExtractionResult:
    deduped_tags = {}
    notes = []

    for result in results:
        for tag in result.tags:
            deduped_tags.setdefault(build_tag_dedup_key(tag), tag)
        if result.notes:
            note = result.notes.strip()
            if note and note not in notes:
                notes.append(note)

    return ExtractionResult(
        tags=list(deduped_tags.values()),
        notes="\n\n".join(notes) if notes else None,
    )


def enrich_result_with_dose_table(result: ExtractionResult, dose_table: List[dict]) -> ExtractionResult:
    dose_index = {
        entry["dose_id"]: entry
        for entry in dose_table
        if entry.get("dose_id")
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


def extract_effects_for_payload(client: Mistral, model: str, payload: dict) -> ExtractionResult:
    max_completion_tokens = int(
        os.getenv("MAX_COMPLETION_TOKENS", str(DEFAULT_MAX_COMPLETION_TOKENS))
    )

    response = client.beta.conversations.start(
        model=model,
        inputs=[
            {
                "role": "user",
                "content": USER_TEMPLATE.format(doc_json=json.dumps(payload, ensure_ascii=False)),
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

    chunks = split_text_into_chunks(report_text, chunk_size=chunk_size, overlap=chunk_overlap)
    chunk_results = []

    for chunk_index, chunk in enumerate(chunks, start=1):
        chunk_payload = dict(payload)
        chunk_payload["report_text"] = chunk
        chunk_payload["report_chunk"] = {
            "index": chunk_index,
            "count": len(chunks),
            "strategy": "char_window_with_overlap",
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


def persist_result(collection, doc: dict, result: ExtractionResult, model: str, source_collection_name: str):
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
        raise RuntimeError(
            f"MongoDB persist failed for exp_id={exp_id!r}"
        )


def mark_error(collection, doc: dict, model: str, error_message: str, source_collection_name: str):
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
                }
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
            source_collection.find(query, max_time_ms=30000).sort("_id", 1).limit(batch_size)
        )
    except CursorNotFound:
        print(
            "WARNING: source cursor expired while loading batch; retrying once.",
            file=sys.stderr,
            flush=True,
        )
        return list(
            source_collection.find(query, max_time_ms=30000).sort("_id", 1).limit(batch_size)
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

        candidate_docs = load_source_batch(source_collection, query, source_scan_batch_size)
        if not candidate_docs:
            break

        scanned_docs += len(candidate_docs)
        last_seen_id = candidate_docs[-1]["_id"]

        candidate_exp_ids = [
            doc.get("exp_id")
            for doc in candidate_docs
            if doc.get("exp_id") is not None
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
    source_scan_batch_size = int(os.getenv("SOURCE_SCAN_BATCH_SIZE", str(batch_size * 5)))
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
                    target_collection, doc, result, mistral_model, mongo_source_collection
                )

            processed += 1
            time.sleep(0.5)

        except Exception as e:
            print(f"ERROR exp_id={exp_id}: {e}", file=sys.stderr, flush=True)
            if not dry_run:
                mark_error(
                    target_collection, doc, mistral_model, str(e), mongo_source_collection
                )

    print(f"Done. Processed {processed} documents.", flush=True)


if __name__ == "__main__":
    main()
