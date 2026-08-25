"""Definitions for perceptual, spatial, and world-experience effects."""

PERCEPTUAL_EFFECT_DEFINITIONS = {
    "visual distortions": (
        "A broad fallback for a directly perceived alteration in vision whose specific form "
        "cannot be determined from the report; it is not a label for generic strangeness."
    ),
    "patterning": (
        "Organized, repeated, or decorative motifs become unusually prominent in or across "
        "the visual field, without requiring explicitly geometric or self-similar structure."
    ),
    "texture rippling": (
        "Fine visual texture appears to travel in small wave-like undulations across a surface, "
        "rather than the whole surface rhythmically expanding as in surface breathing."
    ),
    "surface breathing": (
        "A surface appears to swell and contract in a slow, coherent rhythm resembling "
        "breathing, distinct from directional drifting or fluid-like visual liquefaction."
    ),
    "visual liquefaction": (
        "Objects or surfaces appear to melt, flow, drip, or behave as a continuous fluid while "
        "remaining visual experiences rather than beliefs about physical transformation."
    ),
    "warping": (
        "Objects, surfaces, faces, text, or parts of the visual field appear bent, stretched, "
        "twisted, or otherwise deformed without continuously changing identity."
    ),
    "morphing": (
        "Objects, faces, patterns, or visual forms appear to transform continuously into other "
        "shapes, identities, structures, or forms rather than merely bending in place."
    ),
    "drifting": (
        "Stationary objects, patterns, surfaces, or portions of the visual field appear to slide, "
        "float, sway, or slowly move from their expected position."
    ),
    "shimmering": (
        "Surfaces, edges, textures, light, or the visual field appear to flicker, glint, ripple "
        "with light, or show an unsteady luminous instability."
    ),
    "visual vibration": (
        "Objects, edges, patterns, or the visual field appear to tremble, oscillate, buzz, or "
        "rapidly quiver in place without undergoing a larger displacement."
    ),
    "fractal imagery": (
        "Self-similar visual forms repeat at multiple apparent scales, often branching or "
        "unfolding into recursively detailed structure rather than simple repeated motifs."
    ),
    "geometric imagery": (
        "Grids, lattices, tessellations, mandala-like forms, angular structures, or other abstract "
        "geometric patterns are seen; their color and eye state belong in detail."
    ),
    "visual imagery": (
        "Internally generated image-like content is experienced and recognized as imagery, with "
        "its eye state and subject matter recorded as detail rather than separate effects."
    ),
    "simple visual hallucination": (
        "Unformed light, color, flashes, points, lines, or elementary shapes are perceived without "
        "an ordinary external source and as present in perceptual space."
    ),
    "complex visual hallucination": (
        "Fully formed objects, people, animals, entities, or scenes are perceived without an "
        "ordinary external source; the depicted content belongs in detail."
    ),
    "visual trails": (
        "Successive residual copies appear behind a moving object or viewpoint, tracing its recent "
        "path rather than leaving one stationary image after the object disappears."
    ),
    "afterimages": (
        "A visual impression persists or recurs after the original stimulus has moved or vanished, "
        "distinct from a chain of copies continuously following motion."
    ),
    "motion exaggeration": (
        "Real movement appears unusually large, fast, forceful, or visually emphasized even though "
        "its direction and continuity remain recognizable."
    ),
    "color saturation enhancement": (
        "Colors appear unusually vivid, pure, or intense in chromatic strength, distinct from a "
        "general increase in luminance or contrast."
    ),
    "brightness enhancement": (
        "The visual field, lights, or objects appear more luminous or brightly lit than expected, "
        "without necessarily becoming more colorful or sharply contrasted."
    ),
    "contrast enhancement": (
        "Differences between light and dark, colors, edges, or adjacent regions appear unusually "
        "pronounced, producing stronger visual separation."
    ),
    "visual acuity enhancement": (
        "Edges and fine spatial detail appear unusually crisp or resolvable, distinct from merely "
        "brighter, more saturated, or more meaningful vision."
    ),
    "visual snow": (
        "Numerous fine, rapidly changing dots resembling television static are seen across a "
        "substantial portion of the visual field."
    ),
    "pattern recognition enhancement": (
        "Existing visual regularities and organized forms are detected more readily than usual, "
        "without necessarily misidentifying ambiguous material as a specific object."
    ),
    "pareidolia": (
        "Ambiguous real visual material is perceived as a meaningful familiar form, such as a "
        "face or figure, rather than a formed image appearing without a stimulus."
    ),
    "size distortion": (
        "Objects, body parts, people, or environmental features appear unusually enlarged, reduced, "
        "inflated, or shrunken relative to their expected size."
    ),
    "distance distortion": (
        "Objects, people, or environmental features appear closer or farther away than their "
        "expected location, while their perceived size may remain independently unchanged."
    ),
    "depth distortion": (
        "The depth structure of a scene changes so that space appears flattened, layered, recessed, "
        "expanded, or otherwise altered in three-dimensional organization."
    ),
    "foreshortening distortion": (
        "Body parts, objects, or spatial forms appear unnaturally compressed or elongated along the "
        "viewer's line of sight."
    ),
    "perspective distortion": (
        "A scene's geometric relations appear skewed: angles, proportions, orientation, convergence, "
        "or lines of sight no longer fit the expected viewpoint."
    ),
    "scene replacement": (
        "Part or all of the visible scene appears replaced by a different place, setting, or "
        "constructed environment rather than merely altered in style."
    ),
    "environmental transfiguration": (
        "The existing environment remains recognizable yet appears transformed in character, style, "
        "material, atmosphere, or identity."
    ),
    "zooming": (
        "The visual scene or an attended region appears to enlarge or recede as though a camera lens "
        "were zooming, without ordinary movement toward or away from it."
    ),
    "visual fragmentation": (
        "Objects or the scene appear divided into separate slices, tiles, shards, or discontinuous "
        "parts instead of one spatially coherent image."
    ),
    "visual multiplicity": (
        "One object or image appears as two or more simultaneous copies; the number of copies and "
        "binocular or monocular context belong in detail."
    ),
    "blurred vision": (
        "Visual boundaries and fine details lose focus and spread into one another, resembling an "
        "out-of-focus image rather than a fog laid over the scene."
    ),
    "field narrowing": (
        "The usable or attended visual field appears constricted toward its center, with peripheral "
        "vision diminished or absent."
    ),
    "field widening": (
        "The visual field appears unusually broad or panoramic, with more peripheral extent or "
        "awareness than is ordinarily experienced."
    ),
    "light haloing": (
        "Luminous rings, glows, coronas, or radiating haze appear around lights or bright objects "
        "rather than uniformly across the whole scene."
    ),
    "color tinting": (
        "A color cast overlays objects or the visual field, shifting their apparent hue in a "
        "consistent or locally bounded way."
    ),
    "color suppression": (
        "Colors appear washed out, muted, gray, or reduced in chromatic intensity while visual "
        "brightness and clarity may remain separately intact."
    ),
    "visual motion discontinuity": (
        "Moving objects are seen as discrete jumps, freeze-frames, or missing intervals rather than "
        "as continuous motion."
    ),
    "object animation": (
        "An inanimate visible object appears to move with self-directed or lifelike articulation, "
        "beyond simple drifting, breathing, or geometric deformation."
    ),
    "photophobia": (
        "Ordinary light is experienced as visually aversive, painful, or intolerably intense, "
        "rather than simply appearing brighter."
    ),
    "visual haziness": (
        "A misty, smoky, veiled, or fog-like layer appears across vision while object contours may "
        "remain more focused than in blurred vision."
    ),
    "visual recursion": (
        "The visual field, an image, or a scene appears nested or repeated within itself, without "
        "requiring the scale-invariant geometry of fractal imagery."
    ),
    "textual instability": (
        "Letters, words, or lines appear to move, rearrange, detach, or lose stable visual organization "
        "despite still being experienced as text."
    ),
    "auditory distortions": (
        "A broad fallback for a directly experienced change in hearing whose specific acoustic form "
        "cannot be determined; it does not mean merely liking or noticing sound."
    ),
    "auditory enhancement": (
        "Sound is experienced as globally richer, more vivid, detailed, or immersive, without enough "
        "evidence to isolate clarity, loudness, pitch, or another narrower dimension."
    ),
    "sound clarity enhancement": (
        "Individual tones, voices, instruments, or acoustic details become unusually distinct and "
        "easy to resolve, independently of perceived loudness."
    ),
    "sound amplification": (
        "External sounds are perceived as louder or more forceful than expected, without necessarily "
        "becoming clearer, richer, or more emotionally valued."
    ),
    "sound dampening": (
        "External sounds seem quieter, muffled, distant, or reduced in impact despite no adequate "
        "change in the sound source."
    ),
    "echoing": (
        "A sound is followed by one or more distinct perceived repetitions, separable from the "
        "continuous decaying resonance characteristic of reverberation."
    ),
    "reverberation": (
        "Sounds appear to persist and decay through an enlarged or unusually resonant acoustic "
        "space rather than returning as discrete repeats."
    ),
    "pitch distortion": (
        "A sound's perceived frequency or melodic height shifts, bends, wobbles, or becomes unstable "
        "while the source remains otherwise identifiable."
    ),
    "tempo distortion": (
        "Music, speech rhythm, or repeated sound appears faster, slower, or rhythmically displaced "
        "without a corresponding source change."
    ),
    "sound duration distortion": (
        "Individual sounds appear unnaturally stretched, shortened, arrested, or temporally smeared, "
        "distinct from a change in a sequence's overall tempo."
    ),
    "timbre distortion": (
        "The tonal color or texture of a sound changes, becoming metallic, phased, synthetic, hollow, "
        "or otherwise qualitatively unlike its familiar source."
    ),
    "auditory imagery": (
        "Sound-like content is experienced internally and recognized as imagination or mental sound, "
        "rather than localized as an event in external auditory space."
    ),
    "auditory hallucination": (
        "A voice, tone, music, noise, or other sound is heard without an ordinary external source; "
        "the specific content belongs in detail."
    ),
    "sound localization distortion": (
        "A real or perceived sound seems to originate from the wrong direction, distance, location, "
        "or spatial extent."
    ),
    "auditory looping": (
        "A short sound, word, musical segment, or internally heard sequence repeats involuntarily "
        "with a recognizable cycle."
    ),
    "auditory pareidolia": (
        "Ambiguous real noise is heard as a voice, music, words, or another organized auditory object, "
        "rather than sound arising with no stimulus."
    ),
    "olfactory change": (
        "A broad fallback for a directly experienced alteration in smell whose specific direction "
        "or form cannot be established from the report."
    ),
    "olfactory enhancement": (
        "Existing odors seem unusually strong, vivid, differentiated, or easy to detect, without a "
        "new smell being perceived in their absence."
    ),
    "olfactory suppression": (
        "Existing odors become unusually faint, difficult to identify, or absent from awareness "
        "despite an expected source."
    ),
    "olfactory hallucination": (
        "A smell is perceived without an ordinary external odor source; its qualities, such as smoke "
        "or floral scent, belong in detail."
    ),
    "olfactory distortion": (
        "A present or familiar odor smells qualitatively altered, wrong, or unlike its usual character "
        "rather than merely stronger or weaker."
    ),
    "gustatory change": (
        "A broad fallback for a directly experienced alteration in taste whose specific direction "
        "or form cannot be established from the report."
    ),
    "taste enhancement": (
        "Flavor or basic taste qualities seem unusually vivid, intense, differentiated, or easy to "
        "notice while an ordinary taste source is present."
    ),
    "taste suppression": (
        "Flavor or basic taste qualities seem unusually faint, muted, or absent despite food, drink, "
        "or another expected source."
    ),
    "gustatory hallucination": (
        "A taste is experienced without an ordinary source in the mouth; its metallic, bitter, "
        "chemical, or other content belongs in detail."
    ),
    "taste distortion": (
        "Food, drink, or another present source tastes qualitatively wrong or different from its "
        "familiar character, beyond a simple gain or loss of intensity."
    ),
    "synesthetic change": (
        "A broad fallback for an unusual coupling between sensory or conceptual categories when the "
        "report does not establish a specific synesthetic relation."
    ),
    "synesthesia": (
        "A stimulus or concept in one modality or category automatically evokes a concurrent "
        "experience in another; inducer and concurrent modalities belong in detail."
    ),
    "vestibular change": (
        "A broad fallback for a directly experienced alteration in balance, gravity, acceleration, "
        "or bodily motion when no narrower vestibular effect is justified."
    ),
    "vertigo": (
        "The self or surroundings seem to rotate or spin despite no matching movement, distinct from "
        "non-rotational rocking, translation, or lightheadedness."
    ),
    "illusory acceleration": (
        "The body feels pushed, pulled, launched, or accelerated in a direction despite no matching "
        "change in physical velocity."
    ),
    "illusory levitation": (
        "The body feels lifted, suspended, or floating above its support despite remaining physically "
        "in place; this concerns vertical support rather than self-location."
    ),
    "illusory falling": (
        "A compelling bodily sensation of dropping or falling occurs without corresponding downward "
        "movement."
    ),
    "illusory self-motion": (
        "The self feels translated, rocked, swayed, or moved despite physical stillness; rotational "
        "self or world motion is classified as vertigo."
    ),
    "gravitational distortion": (
        "The direction or strength of gravity feels altered, tilted, reversed, absent, or multiplied, "
        "beyond a single sensation of falling or floating."
    ),
    "spatial change": (
        "A broad fallback for an alteration in the experienced organization or extent of space when "
        "no more specific spatial structure is supported."
    ),
    "spatial scale distortion": (
        "Experienced space itself feels globally enlarged, compressed, miniature, or vast, rather "
        "than one visible object's size or distance changing."
    ),
    "spatial boundlessness": (
        "Space is experienced as limitless or extending without edges, enclosure, or reachable "
        "boundary while spatial extension still remains present."
    ),
    "spacelessness": (
        "Spatial extension or dimensionality seems absent altogether, distinct from very large, "
        "boundaryless, flattened, or visually empty space."
    ),
    "perspectival dislocation": (
        "Experienced perspective loses a single determinate spatial viewpoint, seeming nowhere, "
        "everywhere, or multiply located while the self remains embodied; bodily separation is disembodiment."
    ),
    "world-experience change": (
        "A broad fallback for a directly felt change in how the surrounding world presents as a whole, "
        "when no narrower world-experience structure is established."
    ),
    "uncanniness": (
        "Familiar surroundings or situations feel subtly strange, eerie, or not-quite-right without "
        "a specific threat message, recognition failure, or sensory distortion."
    ),
    "atmospheric portent": (
        "The entire situation carries an impending, charged sense that something momentous is about "
        "to happen, without a definite event or belief supplying that meaning."
    ),
    "environmental vitality loss": (
        "The surrounding world appears experientially drained of life, animation, warmth, or expressive "
        "presence, rather than simply visually dim or emotionally disliked."
    ),
    "perceptual meaning loss": (
        "Objects remain perceptually recognizable yet seem stripped of ordinary significance, function, "
        "or implicit meaning, distinct from language or memory impairment."
    ),
    "perceptual freshness": (
        "Ordinary objects or surroundings present with a striking first-time freshness or novelty "
        "despite still being correctly recognized as familiar."
    ),
    "hyperfamiliarity": (
        "A person, place, object, or situation feels far more familiar than recognition evidence "
        "warrants, distinct from merely remembering it clearly."
    ),
}
