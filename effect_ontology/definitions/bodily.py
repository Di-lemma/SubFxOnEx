"""Definitions for interoceptive, somatic, motor, and related bodily effects."""

BODILY_EFFECT_DEFINITIONS = {
    "body awareness change": (
        "A broad fallback for a directly experienced alteration in bodily position or internal-state "
        "awareness when its specific interoceptive or proprioceptive form is unclear."
    ),
    "proprioceptive distortion": (
        "The felt position, shape, movement, or spatial arrangement of one's body or limbs differs "
        "from their actual configuration, independently of how they look."
    ),
    "interoceptive amplification": (
        "Ordinary signals from within the body, such as heartbeat, breathing, fullness, or visceral "
        "activity, enter awareness with unusually strong intensity or detail."
    ),
    "interoceptive attenuation": (
        "Ordinary signals from within the body become unusually faint, remote, or inaccessible to "
        "awareness without necessarily producing general numbness."
    ),
    "visceral distortion": (
        "Internal organs or bodily interiors feel displaced, reshaped, empty, enlarged, rearranged, "
        "or otherwise configured differently from their familiar felt organization."
    ),
    "cardiac awareness": (
        "Attention is unusually drawn to an otherwise ordinary-feeling heartbeat; forceful, racing, "
        "or irregular beats are instead palpitations."
    ),
    "respiratory awareness": (
        "Breathing becomes unusually prominent in attention while remaining physically easy; air "
        "hunger or felt breathing difficulty is instead dyspnea."
    ),
    "body load": (
        "A broad fallback for a directly felt bodily burden or alteration that the report does not "
        "describe precisely enough for a narrower somatic effect."
    ),
    "somatic heaviness": (
        "The body or a body region feels unusually heavy, weighted, dense, or difficult to lift, "
        "whether or not actual strength is reduced."
    ),
    "somatic lightness": (
        "The body or a body region feels unusually light, buoyant, hollow, or nearly weightless, "
        "without necessarily creating an illusion of levitation."
    ),
    "stimulation": (
        "The body feels physiologically activated, keyed up, or driven into heightened arousal; this "
        "is distinct from having comfortable strength or endurance."
    ),
    "sedation": (
        "The body and behavior feel pharmacologically slowed, subdued, or difficult to activate, "
        "beyond ordinary tiredness alone."
    ),
    "physical energy": (
        "A directly felt increase in bodily vigor, stamina, or capacity for exertion, without requiring "
        "tense arousal, restlessness, or compulsive activity."
    ),
    "fatigue": (
        "Bodily or mental resources feel depleted, exhausted, or unable to sustain effort, distinct "
        "from calm sedation or simple sleepiness."
    ),
    "restlessness": (
        "Remaining still feels difficult or uncomfortable and movement is desired, without the severe "
        "inner motor compulsion required for akathisia."
    ),
    "agitation": (
        "A distressing state of tense activation, irritability, or unsettled bodily arousal is felt, "
        "often but not necessarily accompanied by movement."
    ),
    "tremor": (
        "A body part shakes with relatively rhythmic, repeated oscillation that is experienced or "
        "observed, rather than isolated twitches or larger sudden jerks."
    ),
    "muscle tension": (
        "Skeletal muscles feel persistently contracted, rigid, clenched, or resistant to release, "
        "whether generalized or localized in detail."
    ),
    "muscle relaxation": (
        "Skeletal muscles feel unusually loose, released, or free of habitual contraction without "
        "necessarily causing weakness or sedation."
    ),
    "jaw tension": (
        "The jaw muscles feel clenched, tight, sore, or resistant to opening; actual repetitive tooth "
        "grinding or clenching is classified as bruxism."
    ),
    "bruxism": (
        "The teeth are involuntarily or compulsively ground or clenched, as an action rather than only "
        "a static sensation of jaw tightness."
    ),
    "tingling": (
        "A light prickling, sparkling, vibrating, or effervescent skin or bodily sensation is felt, "
        "without the sharper clustered quality of pins and needles."
    ),
    "pins and needles": (
        "Numerous small, sharp prickling sensations resembling a limb waking from compression are "
        "felt in a localized or spreading region."
    ),
    "numbness": (
        "Ordinary touch or bodily sensation is reduced or absent in a region, distinct from merely "
        "feeling emotionally detached from the body."
    ),
    "warmth": (
        "The body or a region feels warmer than expected without necessarily showing skin redness, "
        "sweating, or a changing temperature."
    ),
    "coldness": (
        "The body or a region feels colder than expected without necessarily involving shivering, "
        "goosebumps, or alternating temperature."
    ),
    "temperature fluctuation": (
        "Felt bodily temperature shifts back and forth, changes abruptly, or varies in waves rather "
        "than remaining consistently warm or cold."
    ),
    "flushing": (
        "A sudden spreading heat, redness, or hot rush is felt at the skin, commonly in the face or "
        "upper body, rather than diffuse steady warmth."
    ),
    "chills": (
        "Cold waves or shivering sensations pass through the body, whether or not ambient temperature "
        "is low or goosebumps are visible."
    ),
    "goosebumps": (
        "The skin's hairs seem to stand and small raised bumps are felt or observed, independently of "
        "whether fear, cold, music, or awe triggered them."
    ),
    "pressure sensation": (
        "A body area feels pushed upon, compressed, full, or under internal or external pressure "
        "without enough evidence for a narrower painful or visceral effect."
    ),
    "tightness": (
        "A body region feels constricted, bound, squeezed, or unable to expand normally, distinct from "
        "a clearly muscular contraction alone."
    ),
    "lightheadedness": (
        "The head feels faint, hollow, floating, or close to losing consciousness, without the false "
        "rotational motion that defines vertigo."
    ),
    "dizziness": (
        "A nonspecific sense of unsteadiness, disorientation, or disequilibrium is felt when the report "
        "does not establish spinning vertigo or near-faint lightheadedness."
    ),
    "weakness": (
        "The body or a muscle group feels unable to produce normal force or support, whether or not "
        "objective power was tested."
    ),
    "bodily buzzing": (
        "A sustained fine vibration or humming sensation seems to fill the body or a region, distinct "
        "from discrete shocks or surface-level prickling."
    ),
    "electric sensations": (
        "Sudden shocks, currents, zaps, or electrically charged waves are felt moving through or "
        "across the body."
    ),
    "physical comfort": (
        "The body feels unusually settled, easeful, well-positioned, or free of strain, distinct from "
        "an actively pleasurable bodily sensation."
    ),
    "physical discomfort": (
        "The body feels unpleasant, ill-at-ease, or physically wrong without a more specific localized "
        "symptom or clearly painful quality."
    ),
    "pain": (
        "An unpleasant bodily sensation with aching, burning, stabbing, throbbing, or other painful "
        "quality is directly experienced; location and quality belong in detail."
    ),
    "pain relief": (
        "Pre-existing or expected pain becomes noticeably reduced or absent, rather than attention "
        "simply moving away from an unchanged sensation."
    ),
    "pain amplification": (
        "Existing or provoked pain feels stronger, sharper, broader, or more intrusive than expected "
        "without a corresponding change in its source."
    ),
    "itching": (
        "A cutaneous irritation produces a direct urge or desire to scratch, whether localized, "
        "migrating, or generalized."
    ),
    "bodily pleasure": (
        "A directly felt pleasant bodily sensation or somatic euphoria occurs, distinct from positive "
        "mood, comfort alone, or appreciation of external touch."
    ),
    "palpitations": (
        "The heartbeat feels unusually forceful, fast, irregular, fluttering, skipping, or pounding, "
        "rather than merely receiving increased attention."
    ),
    "dyspnea": (
        "Breathing feels difficult, insufficient, effortful, tight, or air-hungry, distinct from simply "
        "becoming conscious of otherwise comfortable respiration."
    ),
    "malaise": (
        "A diffuse bodily sense of sickness, unwellness, or physical debility is felt without a narrower "
        "symptom adequately accounting for it."
    ),
    "thirst": (
        "A direct bodily desire or need to drink is unusually present, typically with felt dryness or "
        "a sense that fluid is needed."
    ),
    "urinary urgency": (
        "A sudden or unusually compelling need to urinate is felt, whether or not the bladder is full "
        "or urination follows."
    ),
    "urinary retention": (
        "Urination is difficult to initiate or complete despite a felt need or apparently full bladder, "
        "rather than the need itself being absent."
    ),
    "perspiration change": (
        "Sweating becomes unusually increased, reduced, localized, or otherwise altered; its direction "
        "and body area belong in detail."
    ),
    "formication": (
        "A tactile sensation resembling insects crawling on or beneath the skin is felt without a "
        "matching external cause."
    ),
    "motor change": (
        "A broad fallback for a directly experienced alteration in voluntary movement, posture, or "
        "motor control when no narrower motor effect is established."
    ),
    "incoordination": (
        "Movements fail to combine smoothly or accurately, producing clumsy, poorly timed, or mismatched "
        "actions beyond a problem limited to balance or fine dexterity."
    ),
    "impaired balance": (
        "Maintaining upright posture or a stable stance becomes difficult, with swaying or instability "
        "that is not merely an illusory sensation of motion."
    ),
    "impaired fine motor control": (
        "Precise small movements, such as typing, writing, fastening, or manipulating objects, become "
        "unusually inaccurate or difficult."
    ),
    "akathisia": (
        "A distressing inner motor restlessness creates a compelling need to move repeatedly and an "
        "inability to remain still, stronger than ordinary restlessness."
    ),
    "compulsive movement": (
        "A specific movement or sequence is repeatedly performed with a felt drive or difficulty "
        "resisting it, rather than from generalized restlessness."
    ),
    "stillness": (
        "Voluntary movement markedly decreases because remaining motionless feels natural, absorbing, "
        "or preferred, while the capacity to move is retained."
    ),
    "immobility": (
        "The person experiences being unable or nearly unable to initiate bodily movement despite "
        "wanting or attempting to move."
    ),
    "muscle twitching": (
        "Brief, localized, involuntary muscle contractions occur as small flicks or jumps, distinct "
        "from rhythmic tremor or larger whole-limb jerks."
    ),
    "jerking": (
        "Sudden, brief, involuntary movements displace a limb, body segment, or the whole body, rather "
        "than producing a regular oscillation."
    ),
    "nystagmus": (
        "The eyes make involuntary repetitive oscillating or jerking movements, experienced directly "
        "or clearly reported as eye movement rather than visual-field vibration alone."
    ),
    "dysarthria": (
        "Speech articulation becomes slurred, poorly coordinated, or mechanically difficult because "
        "the motor production of speech is impaired, not because words cannot be found."
    ),
    "catalepsy": (
        "The body or limbs remain rigidly fixed in an imposed or adopted posture with markedly reduced "
        "spontaneous movement, beyond chosen stillness."
    ),
    "gastrointestinal effects": (
        "A broad fallback for a directly experienced digestive, oral, or bowel change when the report "
        "does not justify a more specific gastrointestinal effect."
    ),
    "nausea": (
        "A sick, queasy sensation with an urge or possibility of vomiting is felt, whether or not "
        "vomiting actually occurs."
    ),
    "vomiting": (
        "Stomach contents are forcefully expelled through the mouth or the report clearly describes "
        "throwing up, rather than nausea alone."
    ),
    "stomach discomfort": (
        "The stomach or upper abdomen feels unsettled, sore, heavy, or unpleasant without clear nausea, "
        "cramping, bloating, or reflux."
    ),
    "cramping": (
        "Painful or uncomfortable involuntary contractions are felt in the abdomen or digestive tract, "
        "often arriving in waves rather than as diffuse discomfort."
    ),
    "bloating": (
        "The abdomen feels swollen, distended, pressurized, or unusually full, whether or not visible "
        "enlargement or gas is present."
    ),
    "loss of appetite": (
        "Desire or willingness to eat is reduced or absent despite an ordinary opportunity or prior "
        "need for food."
    ),
    "increased appetite": (
        "Hunger, interest in food, or the desire to eat becomes stronger or more persistent than "
        "expected."
    ),
    "dry mouth": (
        "The mouth feels unusually dry, sticky, or lacking saliva, independently of whether thirst is "
        "also present."
    ),
    "salivation": (
        "Saliva production or the felt accumulation of saliva becomes unusually prominent or excessive, "
        "rather than ordinary mouth moisture."
    ),
    "difficulty swallowing": (
        "Initiating or completing a swallow feels unusually effortful, obstructed, uncoordinated, or "
        "unsafe, distinct from simple throat dryness."
    ),
    "diarrhea": (
        "Bowel movements become abnormally loose, watery, or frequent rather than merely urgent or "
        "accompanied by cramping."
    ),
    "constipation": (
        "Passing stool becomes unusually infrequent, difficult, incomplete, or hard despite an apparent "
        "need or usual routine."
    ),
    "bowel urgency": (
        "A sudden, compelling need to defecate is felt and is difficult to defer, whether or not stool "
        "is loose or evacuation occurs."
    ),
    "acid reflux": (
        "Burning, acidic, or regurgitating stomach contents are felt rising into the chest, throat, or "
        "mouth, distinct from nonspecific stomach discomfort."
    ),
    "flatulence": (
        "Intestinal gas production, pressure, or passage becomes unusually prominent and is directly "
        "experienced or reported."
    ),
    "hiccups": (
        "Repeated involuntary diaphragm contractions produce abrupt inspirations and characteristic "
        "hiccup sounds or sensations."
    ),
    "tactile change": (
        "A broad fallback for an alteration in externally directed touch perception when no narrower "
        "change in intensity, quality, location, or recognition is established."
    ),
    "tactile amplification": (
        "Real touch, texture, pressure, or contact feels unusually intense, detailed, vivid, or salient; "
        "its pleasantness or unpleasantness belongs in detail or a separate effect."
    ),
    "tactile attenuation": (
        "Real touch, texture, pressure, or contact feels unusually faint, distant, muted, or difficult "
        "to detect without being wholly numb."
    ),
    "tactile distortion": (
        "A real touch stimulus feels qualitatively unlike its physical character, such as wet, rubbery, "
        "rough, or moving when it is not."
    ),
    "skin sensitivity": (
        "The skin reacts to ordinary contact, clothing, air, or temperature with unusual tenderness, "
        "irritation, or sensory reactivity rather than simple intensity gain."
    ),
    "tactile localization distortion": (
        "A touch is felt at the wrong bodily location, over an enlarged area, or displaced from where "
        "the stimulus actually contacts the body."
    ),
    "tactile recognition impairment": (
        "A touched object's identity, shape, or material becomes difficult to recognize by touch despite "
        "the contact itself still being sensed."
    ),
    "tactile hallucination": (
        "Touch, pressure, movement, contact, or another tactile event is felt without a matching external "
        "stimulus; its location and content belong in detail."
    ),
    "sexual change": (
        "A broad fallback for a directly experienced alteration in sexual desire, sensuality, arousal, "
        "function, or orgasm when no narrower effect is supported."
    ),
    "increased libido": (
        "Sexual desire, interest, or motivation becomes stronger or more frequent, independently of "
        "current genital arousal or general tactile pleasure."
    ),
    "decreased libido": (
        "Sexual desire, interest, or motivation becomes weaker or absent, independently of whether "
        "physiological arousal remains possible."
    ),
    "increased sensuality": (
        "Sensory and embodied experiences feel more sensuous, intimate, or erotically textured without "
        "necessarily producing explicit sexual desire or genital arousal."
    ),
    "sexual arousal enhancement": (
        "Subjective or physiological sexual arousal arises more readily or intensely, distinct from "
        "the motivational desire measured by libido."
    ),
    "sexual arousal suppression": (
        "Subjective or physiological sexual arousal is reduced or difficult to sustain despite context "
        "or desire that would ordinarily support it."
    ),
    "sexual dysfunction": (
        "A broad functional difficulty affects sexual response or activity but the report does not "
        "establish a narrower arousal or orgasm-related impairment."
    ),
    "orgasm enhancement": (
        "Orgasm is experienced as unusually intense, prolonged, full-bodied, or subjectively satisfying "
        "once it occurs."
    ),
    "orgasm delay": (
        "Reaching orgasm takes unusually long or requires substantially more stimulation, while orgasm "
        "remains possible."
    ),
    "anorgasmia": (
        "Orgasm cannot be reached despite sexual stimulation or arousal that would ordinarily be "
        "sufficient, rather than merely being delayed."
    ),
    "spontaneous orgasm": (
        "An orgasm occurs without deliberate sexual stimulation or an ordinary triggering activity; "
        "its context and bodily focus belong in detail."
    ),
    "sleep disturbance": (
        "A broad fallback for a directly experienced change in sleep, dreaming, or sleep-wake transition "
        "when no narrower phenomenon is established."
    ),
    "insomnia": (
        "Adequate sleep is persistently or substantially unavailable despite an opportunity to sleep, "
        "potentially involving onset, maintenance, or early awakening."
    ),
    "difficulty falling asleep": (
        "Transitioning from wakefulness into sleep takes unusually long or feels repeatedly interrupted, "
        "without necessarily implying broader insomnia across the night."
    ),
    "sleep fragmentation": (
        "Sleep is repeatedly broken by awakenings or brief returns to wakefulness, preventing a "
        "continuous sleep period despite successful sleep onset."
    ),
    "vivid dreams": (
        "Dreams contain unusually intense, detailed, lifelike, colorful, or emotionally immediate "
        "experience without requiring awareness that one is dreaming."
    ),
    "lucid dreams": (
        "Awareness that one is currently dreaming occurs within the dream, whether or not the dream's "
        "events can be deliberately controlled."
    ),
    "dream recall enhancement": (
        "Dream content is remembered after waking with greater frequency, duration, or detail than is "
        "usual for the reporter."
    ),
    "drowsiness": (
        "Wakefulness becomes difficult to maintain and sleep onset feels imminent, distinct from low "
        "energy without a direct tendency to fall asleep."
    ),
    "hypnagogia": (
        "Image-like, auditory, tactile, or other dreamlike experiences arise during the transition from "
        "wakefulness into sleep while some awareness is retained."
    ),
    "hypnopompia": (
        "Image-like, auditory, tactile, or other dreamlike experiences persist or arise during the "
        "transition from sleep into wakefulness."
    ),
    "sleep paralysis": (
        "Awareness returns at sleep onset or awakening while voluntary skeletal movement remains "
        "temporarily unavailable, sometimes alongside transitional imagery."
    ),
    "false awakening": (
        "The person dreams that they have awakened and begun ordinary waking activity, only later "
        "recognizing that the apparent awakening occurred within a dream."
    ),
    "dream-reality confusion": (
        "Dream events or memories are difficult to distinguish from waking events after or around sleep, "
        "rather than simply feeling that waking life is dreamlike."
    ),
    "nightmares": (
        "Dreams are intensely distressing, threatening, or frightening and commonly provoke awakening "
        "or lingering distress, beyond vividness alone."
    ),
    "hypersomnia": (
        "Sleep duration or the need to sleep becomes substantially greater than usual, beyond a brief "
        "episode of drowsiness or ordinary recovery sleep."
    ),
}
