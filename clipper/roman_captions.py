"""Transliteration for multilingual captions (Clipping Final Boss).

Ported from ~/workspace/gta6-live-clipper/transliterate.py (donor, read-only)
plus the auto-subs / word-retiming helpers from donor vod.py
(_natural_word_dur, _retime_words_natural, autosubs_to_transcript,
_parse_vtt, fetch_autosubs). Stdlib only; indic-transliteration is
optional — the vendored _ck_* engine is the fallback and works without it
(it is the live path in this venv: indic-transliteration is not installed).

Problem (sohaib, 2026-11-17): PeakCut's live clip is the quality/caption/
facecam target, BUT it only detects English (its Parakeet model is
English-only) while the streamer speaks Roman Urdu / Hindi. Captions must
be in ROMAN (Latin) script when the speech is Hindi/Urdu (Hinglish), and
in each other language's own script otherwise.

Pipeline:
  Whisper (faster-whisper, auto-detect) transcribes Hindi as Devanagari
  and Urdu as Nastaliq. This module converts those words to Roman:
  - Devanagari -> Roman: indic-transliteration (ITRANS) + a Hinglish
    normalizer (long vowels to lower case, anusvara -> n, word-final
    schwa dropped) + a small overrides dict for the common words where
    the mechanical output is visibly off (ajad -> azad, skul -> school).
  - Urdu (Nastaliq) -> Roman: a rule-based letter map + the same
    overrides dict (local libs have no usable Urdu scheme on this stack:
    indic-transliteration has no Urdu scheme; aksharamukha 2.3's database
    has no Urdu->Latin entry — KeyError 'Urdu' — so it cannot do it
    either). Approximate but readable; the Gemini path below gives
    natural Roman Urdu when a key is set.
  - Gemini path (optional, key from clipper.config.settings — CFB .env
    GEMINI_API_KEY, never hard-coded): natural Hinglish / Roman Urdu
    spelling. Used only when roman_mode="gemini" AND a key is present.

CFB adaptations vs the donor:
  - load_gemini_key() reads clipper.config.settings.gemini_api_key
    (CFB .env) instead of the donor's config.json; an explicit api_key
    argument wins over settings. A config_path argument is still
    accepted as a legacy fallback (donor config.json layout).
  - autosubs_to_transcript() returns the CFB transcript shape:
    {"language", "duration", "segments": [{start, end, text, words}],
    "full_text", "words", ...} so clipper.candidate_picker (which reads
    seg["words"]) works unchanged. The donor returned segments without
    words and no "language"/"full_text" keys; those are added here.
    It also accepts an optional gemini_key (donor always passed None).
  - fetch_autosubs() uses _ytdlp_cmd() — [sys.executable, "-m",
    "yt_dlp", "--no-check-certificates"] — no hardcoded paths, no
    frozen-exe branch (CFB is a source checkout, not a PyInstaller exe).

Honest limits (measured 2026-11-17, see BUILD-REPORT):
- Local Devanagari->Roman is ~good: proper nouns and common words come
  out right; internal schwa is NOT deleted (karana stays karana, natural
  is karna) except via the overrides dict; loanwords need overrides.
- Local Urdu->Roman is approximate (unwritten vowels are guessed);
  common words are fixed by overrides, rare words may read oddly.
- Gemini transliteration is natural (azad, karna, school, pakistan) but
  needs the user's own free Gemini key in Settings.
"""

import json
import os
import re

# ---------------------------------------------------------------------------
# script detection
# ---------------------------------------------------------------------------

_DEV_RE = re.compile(r"[\u0900-\u097F]")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def script_of(text):
    """'devanagari' | 'urdu' (Arabic script) | 'latin'."""
    if _DEV_RE.search(text):
        return "devanagari"
    if _ARABIC_RE.search(text):
        return "urdu"
    return "latin"


# ---------------------------------------------------------------------------
# Devanagari -> Roman (indic-transliteration ITRANS + Hinglish normalize)
# ---------------------------------------------------------------------------

# Mechanical ITRANS output -> natural Hinglish, for the words where the
# mechanical form is visibly wrong. Extend freely; keys are the normalized
# form, values the natural spelling.
ROMAN_OVERRIDES = {
    "ajad": "azad",
    "skul": "school",
    "karana": "karna",
    "bolana": "bolna",
    "dekhana": "dekhna",
    "sunana": "sunna",
    "milana": "milna",
    "khelana": "khelna",
    "chalana": "chalna",
    "karane": "karne",
    "bolane": "bolne",
    "dekhane": "dekhne",
    "sunane": "sunne",
    "milane": "milne",
    "jane": "jaane",
    "ane": "aane",
    "khane": "khana",
    "pine": "peene",
    "pakstan": "pakistan",
    "mjhe": "mujhe",
    "mire": "mere",
    "tmhara": "tumhara",
    "nhin": "nahin",
    "min": "mein",
    "kia": "kya",
    "krna": "karna",
    "pina": "peena",
    "jina": "jeena",
    "men": "mein",        # में — "men" reads as the English word
    "gae": "gaye",         # गए — natural Hinglish "gaye"
    # English loanwords — natural Roman spelling, not phonetic ITRANS
    "soshalist": "socialist",
    "ripablikan": "republican",
    "esosieshan": "association",
    "pulis": "police",
    "palis": "police",       # पलिस — small-model mishearing of पुलिस
    "pulees": "police",      # पूलीस — small-model mishearing of पुलिस
    "pulisko": "police ko",  # पूलिसको — police + ko joined by Whisper
    "chaik": "check",        # चैक
    "chait": "chat",         # चैट
    "drivar": "driver",      # द्रीवर
    "staar": "star",         # स्टार
    "taiksi": "taxi",        # टैक्सी
    "priviyasli": "previously",  # प्रीवियसली
    "rod": "road",           # रोड
    "lets": "let's",         # लेट्स
    "phayaring": "firing",
    "alphred": "alfred",
    "pharavari": "farvari",
    "echaesaarae": "HSRA",  # एचएसआरए — acronym, not a word
    "echaarae": "HRA",      # एचआरए — acronym, not a word
    "sinh": "singh",        # सिंह — natural Hinglish "singh", not "sinh"
    "kamared": "comrade",   # कॉमरेड — English loanword
    "aichsare": "HSRA",     # small-model mishearing of एचएसआरए
    "echshvare": "HSRA",    # small-model mishearing of एचएसआरए
    "echsare": "HSRA",      # small-model mishearing of एचएसआरए
    "netritv": "netritva",  # नेतृत्व — natural Hinglish keeps the final 'a'
    # natural Hinglish spellings (ITRANS phonetics read oddly)
    "unaki": "unki",
    "apane": "apne",
    "apli": "apni",          # अपली — small-model mishearing of अपनी
    "acha": "accha",         # आचा — small-model mishearing of अच्छा
    "idar": "idhar",         # इदर — small-model mishearing of इधर
    "konsi": "kaunsi",       # कोंसी
    "kunsi": "kaunsi",       # कुनसी
    "bahi": "bhai",          # बही — small-model mishearing of भाई
    "chute": "chhute",       # चुटे
    "chol": "chalo",         # चोल — small-model mishearing of चलो
    "gadi": "gaadi",         # गाडी — natural Hinglish keeps the double 'a'
    "tin": "teen",
    "nind": "neend",
    "donon": "dono",
    "charon": "charo",
    "taraph": "taraf",
    "akhirakar": "aakhirkar",
    "unhen": "unhein",
    "valon": "walon",
    "die": "diye",         # दिए — ITRANS drops the 'y'
    "chupi": "chhupi",     # छुपी — ITRANS drops the 'h'
    "naheen": "nahin",     # नहीं — step-4 long-vowel fold gives "naheen"
    # ClipKontrast-fallback spellings (the ck engine spells these
    # loanwords differently from ITRANS — both keys stay, each path
    # fires its own): skool/komred/soshlist/phaayring/faravri/azaad/
    # pakistaan are what _ck_dev_word produces.
    "skool": "school",
    "komred": "comrade",
    "soshlist": "socialist",
    "phaayring": "firing",
    "faravri": "farvari",
    "azaad": "azad",
    "pakistaan": "pakistan",
}

# Urdu-only overrides (applied only when the source word is Nastaliq, so
# "he" -> "hai" never touches an English or Hindi word).
URDU_OVERRIDES = {
    "mjhe": "mujhe",
    "mire": "mere",
    "tmhara": "tumhara",
    "tmhari": "tumhari",
    "nhin": "nahin",
    "he": "hai",
    "pakstan": "pakistan",
    "dost": "dost",
    "kya": "kya",
    "min": "mein",
    "kia": "kya",
    "krna": "karna",
    "mlna": "milna",
    "sanna": "sunna",
    "kha": "kaha",
    "dikhna": "dekhna",
    "hndostan": "hindustan",
    "bnane": "banane",
    "nbhai": "nibhai",
}


_VOWEL_CHARS = set("aeiouAEIOU")


def _drop_medial_schwa(w):
    """Drop a medial short 'a' per ClipKontrast's right-to-left rule
    (roman-string form, applied BEFORE the A/I/U fold so long vowels
    are still visible as capitals).

    Only a lowercase short 'a' is a candidate (long 'A' is a real
    vowel); the word's first vowel never drops; the 'a' drops only
    when the previous letter is a consonant, the next is a consonant,
    and the letter after that is a vowel. karana -> karna, samajhana
    -> samajhna — but kahan stays kahan (the final-drop above already
    removed its last vowel, so the 'a' after 'h' sees no vowel after
    the 'n') and hamara stays hamara (its vowels are long). Right to
    left, so a drop on the right does not license a wrong drop on
    the left (samajhana: the 'a' after 'h' drops, the 'a' after 'j'
    then sees a consonant and stays)."""
    chars = list(w)
    first_vowel = next((i for i, c in enumerate(chars)
                        if c in _VOWEL_CHARS), None)
    if first_vowel is None:
        return w
    for i in range(len(chars) - 1, first_vowel, -1):
        if chars[i] != "a":
            continue
        prev = chars[i - 1] if i > 0 else ""
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        after = chars[i + 2] if i + 2 < len(chars) else ""
        if (prev.isalpha() and prev not in _VOWEL_CHARS
                and nxt.isalpha() and nxt not in _VOWEL_CHARS
                and after in _VOWEL_CHARS):
            chars[i] = ""
    return "".join(chars)


def _shorten_long_vowels(w):
    """'aa'->'a', 'ee'->'i', 'oo'->'u' when the next consonant carries a
    vowel or at word end (ClipKontrast's rule): baat, kaam, aaj and
    teen keep the double; hamara, pata, ki and tu go single."""
    out = []
    i = 0
    short = {"aa": "a", "ee": "i", "oo": "u"}
    while i < len(w):
        two = w[i:i + 2]
        if two in short:
            j = i + 2
            nxt = w[j] if j < len(w) else ""
            after = w[j + 1] if j + 1 < len(w) else ""
            # shorten at word end, or when the following consonant
            # carries a vowel (after is a real vowel letter — note
            # "" in "aeiou" is True in Python, so test it explicitly).
            if nxt == "" or (after != "" and after in "aeiou"):
                out.append(short[two])
                i += 2
                continue
            out.append(two)
            i += 2
            continue
        out.append(w[i])
        i += 1
    return "".join(out)


def _hinglish_normalize(itrans_word):
    """ITRANS -> natural Hinglish spelling (lowercase, no tech capitals)."""
    w = itrans_word
    # two-letter sequences first (Th/Dh/Sh/Ng/JN are single sounds)
    for src, dst in (("Th", "th"), ("Dh", "dh"), ("Sh", "sh"), ("Ch", "ch"),
                     ("Ng", "ng"), ("JN", "gy"), ("RRi", "ri"), ("RR", "ri"),
                     (".D", "D"), (".d", "d"), (".T", "T"), (".t", "t")):
        w = w.replace(src, dst)
    # a word-final LONG 'A' (आ) is a real vowel — keep it (kaha, jana).
    # Only a word-final SHORT 'a' (अ, the inherent schwa) is dropped.
    ends_long_a = w.endswith("A")
    w = w.replace("M", "n").replace("~", "n")   # anusvara / chandrabindu -> n
    # word-final schwa: pulisa -> pulis, milane -> milne, parka -> park
    if not ends_long_a and len(w) > 2 and w.endswith("a"):
        w = w[:-1]
    # medial schwa (see _drop_medial_schwa) — before the fold below,
    # so long vowels are still capitals and visible to the rule.
    w = _drop_medial_schwa(w)
    # long vowels doubled first (A->aa, I->ee, U->oo), then shortened
    # where people type them single (see _shorten_long_vowels).
    w = (w.replace("A", "aa").replace("I", "ee").replace("U", "oo")
          .replace("R", "ri").replace("L", "l")
          .replace("T", "t").replace("D", "d").replace("N", "n")
          .replace("G", "gh").replace("K", "q").replace("Z", "z")
          .replace("F", "f").replace("X", "ksh").replace("Y", "y"))
    return _shorten_long_vowels(w)


def devanagari_to_roman(text):
    """Devanagari string -> Roman (Latin) string, word by word.

    Common words come from the ClipKontrast table first (correct schwa
    + long vowels, no library needed); the rest go through
    indic-transliteration (ITRANS) when installed, else the vendored
    ClipKontrast engine (_ck_dev_word, stdlib only) — so a machine
    without indic-transliteration NO LONGER falls back to Devanagari
    captions (measured 2026-11-18: the lib is installed nowhere on
    the build VM). ROMAN_OVERRIDES still win last (loanwords, Whisper
    mishearings, acronyms)."""
    text = text.replace("।", ".").replace("॥", ".")
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        have_itrans = True
    except ImportError:
        have_itrans = False
    out = []
    for tok in re.split(r"(\s+)", text):
        if not tok.strip():
            out.append(tok)
            continue
        m = re.match(r"^([^\w\u0900-\u097F]*)([\w\u0900-\u097F]+)([^\w\u0900-\u097F]*)$",
                     tok, re.UNICODE)
        if not m:
            out.append(tok)
            continue
        pre, core, post = m.groups()
        if _DEV_RE.search(core):
            if core in _CK_DEV_WORDS:
                rom = _CK_DEV_WORDS[core]
            elif have_itrans:
                it = transliterate(core, sanscript.DEVANAGARI, sanscript.ITRANS)
                rom = _hinglish_normalize(it)
                # defensive: no Devanagari glyph may survive into a Roman caption
                # (ITRANS leaves the ऑ vowel sign ॉ unconverted in कॉमरेड) —
                # strip leftovers BEFORE the override lookup, or keys never match.
                rom = re.sub(r"[\u0900-\u097F]", "", rom)
            else:
                rom = _ck_dev_word(core)
            rom = ROMAN_OVERRIDES.get(rom, rom)
            out.append(pre + rom + post)
        else:
            out.append(tok)
    return "".join(out)


# ---------------------------------------------------------------------------
# Urdu (Nastaliq) -> Roman, rule-based (no usable local lib on this stack)
# ---------------------------------------------------------------------------

def _urdu_word_to_roman(core):
    # ClipKontrast's engine (vendored below): a 95-word table for the
    # words whose unwritten vowels no rule can guess, then letter rules
    # with the correct a-insertion (after the first letter, or before
    # the consonant at i==2 when w[1] is do-chashmi he: ghar, not gahr).
    # Measured 2026-11-18: ours was wrong on ~30/57 common words, ck
    # right on all but one (gaye, fixed in the table below).
    rom = _ck_urdu_word(core)
    rom = URDU_OVERRIDES.get(rom, ROMAN_OVERRIDES.get(rom, rom))
    return rom


def urdu_to_roman(text):
    """Nastaliq string -> Roman (Latin) string, word by word. Approximate:
    unwritten short vowels are guessed; common words are fixed by the
    overrides dict above. Gemini (with the user's key) is the accurate
    path — see gemini_transliterate()."""
    text = (text.replace("،", ",").replace("۔", ".")
                .replace("؟", "?").replace("؛", ";"))
    out = []
    for tok in re.split(r"(\s+)", text):
        if not tok.strip():
            out.append(tok)
            continue
        m = re.match(r"^([^\w\u0600-\u06FF]*)([\w\u0600-\u06FF]+)([^\w\u0600-\u06FF]*)$",
                     tok, re.UNICODE)
        if not m:
            out.append(tok)
            continue
        pre, core, post = m.groups()
        if _ARABIC_RE.search(core):
            out.append(pre + _urdu_word_to_roman(core) + post)
        else:
            out.append(tok)
    return "".join(out)


# ---------------------------------------------------------------------------
# ClipKontrast roman engine (vendored, stdlib only)
# ---------------------------------------------------------------------------
# From anistroz/ClipKontrast studio/roman.py (MIT licence), renamed with
# a _ck_ prefix. Used as (a) the word-table + fallback for Devanagari
# when indic-transliteration is missing, (b) the Urdu engine, and
# (c) the romanizer for scripts the caption fonts cannot draw
# (Cyrillic, Greek, Korean). Two corrections vs upstream, measured
# 2026-11-18: "मैं" -> "mein" (not "main"), "नहीं" -> "nahin" (not
# "nahi"), "میں" -> "mein" (not "main"), and "گئے" -> "gaye"
# (upstream gives the wrong "geee").

_CK_DEV_WORDS = {"में": "mein", "मैं": "mein", "हैं": "hain", "है": "hai", "नहीं": "nahin", "नही": "nahi", "हूँ": "hoon",
                 "हूं": "hoon", "क्या": "kya", "क्यों": "kyun", "क्यूँ": "kyun", "यह": "yeh", "वह": "woh", "और": "aur",
                 "भी": "bhi", "ही": "hi", "की": "ki", "जी": "ji", "तो": "to", "हाँ": "haan", "हां": "haan", "यहाँ": "yahan",
                 "यहां": "yahan", "वहाँ": "wahan", "वहां": "wahan", "कहाँ": "kahan", "कहां": "kahan", "थी": "thi",
                 "रही": "rahi", "कभी": "kabhi", "अभी": "abhi", "सभी": "sabhi", "भाई": "bhai", "मुझे": "mujhe",
                 "तुम": "tum", "हम": "hum", "आप": "aap", "बहुत": "bahut", "अच्छा": "accha", "ठीक": "theek", "लिए": "liye"}
_CK_VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e", "ऐ": "ai",
              "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e"}
_CK_SIGNS = {"ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o",
             "ौ": "au", "ॉ": "o", "ॅ": "e"}
_CK_CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh",
            "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh",
            "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v",
            "श": "sh", "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
            "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "फ़": "f", "ड़": "r", "ढ़": "rh", "य़": "y"}
_CK_NUKTA = {"क": "q", "ख": "kh", "ग": "gh", "ज": "z", "फ": "f", "ड": "r", "ढ": "rh", "य": "y"}
_CK_VIRAMA, _CK_NUKTA_SIGN = "्", "़"
_CK_NASAL = {"ं": "n", "ँ": "n"}
_CK_OTHER = {"ः": "h", "।": ".", "॥": ".", "ॐ": "om", **{chr(0x0966 + i): str(i) for i in range(10)}}


def _ck_dev_word(w: str) -> str:
    if w in _CK_DEV_WORDS:
        return _CK_DEV_WORDS[w]
    # One unit per letter: [roman consonant or vowel, its vowel or None (None = a consonant with no vowel), is_cons]
    units, i = [], 0
    while i < len(w):
        ch = w[i]
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if ch in _CK_CONS:
            c = _CK_CONS[ch]
            if nxt == _CK_NUKTA_SIGN:
                c = _CK_NUKTA.get(ch, c)
                i += 1
                nxt = w[i + 1] if i + 1 < len(w) else ""
            if nxt == _CK_VIRAMA:
                units.append([c, None, True, False])
                i += 2
                continue
            if nxt in _CK_SIGNS:
                units.append([c, _CK_SIGNS[nxt], True, False])
                i += 2
                continue
            units.append([c, "a", True, True])        # the vowel every Hindi letter carries unless told otherwise
        elif ch in _CK_VOWELS:
            units.append(["", _CK_VOWELS[ch], False, False])
        elif ch in _CK_NASAL:
            if units:
                units[-1][1] = (units[-1][1] or "") + "n"
                units[-1][3] = False
                units[-1].append("nasal")
            else:
                units.append(["n", None, True, False])
        elif ch in _CK_OTHER:
            units.append([_CK_OTHER[ch], None, False, False])
        elif ch == _CK_NUKTA_SIGN or ch == _CK_VIRAMA:
            pass
        else:
            units.append([ch, None, False, False])
        i += 1
    letters = [u for u in units if u[2] or u[1]]
    # Hindi drops the carried "a" at the end of a word, and between a vowel and another vowel-carrying letter
    # (कहना -> kahna, not kahana). Looked at right to left, as speakers do.
    if len(letters) > 1 and letters[-1][3]:
        letters[-1][1] = None
    for k in range(len(letters) - 2, 0, -1):
        u, before, after = letters[k], letters[k - 1], letters[k + 1]
        if u[3] and before[1] and after[2] and after[1] and "nasal" not in before:
            u[1] = None
    # Long vowels the way people type them: doubled only before a consonant that ends a syllable (baat, kaam,
    # aaj), single elsewhere (hamara, pata, ki, tu).
    short = {"aa": "a", "ee": "i", "oo": "u"}
    for k, u in enumerate(letters):
        if u[1] in short:
            after = letters[k + 1] if k + 1 < len(letters) else None
            if after is None or after[1]:
                u[1] = short[u[1]]
    return "".join(u[0] + (u[1] or "") for u in units)


_CK_UR_WORDS = {"ہے": "hai", "ہیں": "hain", "ہوں": "hoon", "ہو": "ho", "ہی": "hi", "یہ": "ye", "وہ": "wo", "کیا": "kya",
                "کیوں": "kyun", "نہیں": "nahi", "نہ": "na", "میں": "mein", "میرا": "mera", "میری": "meri", "میرے": "mere",
                "تو": "to", "کہ": "ke", "کی": "ki", "کا": "ka", "کے": "ke", "کو": "ko", "سے": "se", "پر": "par",
                "اور": "aur", "بھی": "bhi", "تھا": "tha", "تھی": "thi", "تھے": "the", "گیا": "gaya", "گئی": "gayi", "گئے": "gaye",
                "رہا": "raha", "رہی": "rahi", "رہے": "rahe", "بہت": "bahut", "اچھا": "achha", "یار": "yaar", "بھائی": "bhai",
                "آپ": "aap", "ہم": "hum", "تم": "tum", "مجھے": "mujhe", "تمہیں": "tumhein", "اب": "ab", "کر": "kar",
                "کرو": "karo", "کرنا": "karna", "دیکھو": "dekho", "چلو": "chalo", "ابھی": "abhi", "کبھی": "kabhi",
                "جی": "ji", "ہاں": "haan", "نا": "na", "اللہ": "Allah", "ٹھیک": "theek", "پتا": "pata", "پتہ": "pata",
                "لیکن": "lekin", "پھر": "phir", "سب": "sab", "کچھ": "kuch", "کون": "kaun", "کہاں": "kahan", "یہاں": "yahan",
                "وہاں": "wahan", "ایک": "ek", "دو": "do", "اپنا": "apna", "اپنی": "apni", "لوگ": "log", "بات": "baat",
                "کام": "kaam", "دن": "din", "آج": "aaj", "کل": "kal", "بس": "bas", "شکریہ": "shukriya", "واہ": "wah",
                "گھر": "ghar", "ہوا": "hua", "ہوئی": "hui", "ہوئے": "hue", "اس": "us", "ان": "un", "جو": "jo", "جب": "jab",
                "تب": "tab", "مگر": "magar", "پاکستان": "Pakistan", "زندگی": "zindagi", "پیسے": "paise", "روپے": "rupay"}
_CK_UR_CONS = {"ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s", "ج": "j", "چ": "ch", "ح": "h", "خ": "kh", "د": "d",
               "ڈ": "d", "ذ": "z", "ر": "r", "ڑ": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s", "ض": "z",
               "ط": "t", "ظ": "z", "غ": "gh", "ف": "f", "ق": "q", "ک": "k", "ك": "k", "گ": "g", "ل": "l", "م": "m",
               "ن": "n", "ہ": "h", "ه": "h", "ة": "a", "ھ": "h"}
_CK_UR_MARKS = {"َ": "a", "ِ": "i", "ُ": "u", "ّ": "", "ْ": "", "ً": "an", "ٰ": "a"}
_CK_UR_DIGITS = {**{chr(0x06F0 + i): str(i) for i in range(10)}, **{chr(0x0660 + i): str(i) for i in range(10)}}


def _ck_urdu_word(w: str) -> str:
    if w in _CK_UR_WORDS:
        return _CK_UR_WORDS[w]
    out, prev_cons = [], False
    for i, ch in enumerate(w):
        first, last = i == 0, i == len(w) - 1
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if ch in _CK_UR_MARKS:
            out.append(_CK_UR_MARKS[ch])
            prev_cons = False
            continue
        if ch in _CK_UR_DIGITS:
            out.append(_CK_UR_DIGITS[ch])
            prev_cons = False
            continue
        if ch == "ھ":                     # do chashmi he: the "h" of bh, ph, kh, chh...
            out.append("h")
            continue
        if ch in "اآ":
            out.append("aa" if ch == "آ" else "a")
            prev_cons = False
            continue
        if ch in "وؤ":
            if first or not prev_cons:
                out.append("w" if nxt not in "اآ" else "w")
                prev_cons = True
            else:
                out.append("o")
                prev_cons = False
            continue
        if ch in "یيئى":
            if first:
                out.append("y")
                prev_cons = True
            else:
                out.append("i" if last else "ee" if prev_cons and nxt not in "اآوؤ" else "y")
                prev_cons = out[-1] == "y"
            continue
        if ch == "ے":
            out.append("e")
            prev_cons = False
            continue
        if ch in "ںن" and ch == "ں":
            out.append("n")
            continue
        if ch in "عء":
            continue
        c = _CK_UR_CONS.get(ch)
        if c is None:
            out.append(ch)
            prev_cons = False
            continue
        if ch in "ہه" and last and prev_cons:
            out.append("a")               # a final ہ after a consonant is a vowel: کہ, بچہ -> bacha
            prev_cons = False
            continue
        if prev_cons and i == 1 or (prev_cons and i == 2 and w[1] == "ھ"):
            out.append("a")               # the short "a" Urdu does not write after a word's first letter: گھر -> ghar
        out.append(c)
        prev_cons = True
    s = "".join(out)
    if re.fullmatch(r"[a-z]{2,}", s) and not re.search(r"[aeiou]", s):
        s = s[:-1] + "a" + s[-1]          # no vowel at all: give it the one Urdu leaves out
    return s


# ---------------------------------------------------------------- Russian-style (Cyrillic), Greek and Korean letters
_CK_MORE_SCRIPTS = re.compile(r"[\u0370-\u03FF\u0400-\u04FF\uAC00-\uD7A3]")
_CK_CYR = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "zh", "з": "z", "и": "i", "й": "y",
           "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
           "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
           "я": "ya", "є": "ye", "і": "i", "ї": "yi", "ґ": "g", "ў": "u", "ђ": "dj", "ј": "j", "љ": "lj", "њ": "nj",
           "ћ": "c", "џ": "dz", "ә": "a", "ғ": "g", "қ": "q", "ң": "n", "ө": "o", "ұ": "u", "ү": "u", "һ": "h"}
_CK_GRK = {"α": "a", "β": "v", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "i", "θ": "th", "ι": "i", "κ": "k", "λ": "l",
           "μ": "m", "ν": "n", "ξ": "x", "ο": "o", "π": "p", "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "y", "φ": "f",
           "χ": "ch", "ψ": "ps", "ω": "o"}
_CK_K_FIRST = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
_CK_K_VOWEL = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu",
               "eu", "ui", "i"]
_CK_K_LAST = ["", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l", "l", "p", "l", "m", "p", "p", "t", "t", "ng",
              "t", "t", "k", "t", "p", "t"]
_CK_DEV_RE = re.compile(r"[ऀ-ॿ]")
_CK_OTHER_SCRIPT = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")   # Urdu / Arabic letters


def ck_needs_roman(text: str) -> bool:
    """Is this in a script that captions should show in Roman letters instead (Hindi, Urdu, Arabic, Russian-style,
    Greek or Korean letters)?"""
    text = text or ""
    return bool(_CK_DEV_RE.search(text) or _CK_OTHER_SCRIPT.search(text) or _CK_MORE_SCRIPTS.search(text))


def ck_drawable(text: str) -> bool:
    """Can the caption fonts draw this? They have Latin letters (with accents), digits and punctuation only."""
    return not any(ch.isalpha() and not (ord(ch) < 0x250 or 0x1E00 <= ord(ch) <= 0x1EFF) for ch in text or "")


def _ck_letters(run: str) -> str:
    import unicodedata
    out = []
    for ch in unicodedata.normalize("NFD", run):
        if unicodedata.combining(ch):
            continue                                   # Greek accents
        low = ch.lower()
        r = _CK_CYR.get(low, _CK_GRK.get(low))
        if r is None:
            out.append(ch)
        else:
            out.append(r.capitalize() if ch != low and r else r)
    return "".join(out)


def _ck_korean(run: str) -> str:
    out = []
    for ch in run:
        n = ord(ch) - 0xAC00
        out.append(_CK_K_FIRST[n // 588] + _CK_K_VOWEL[n % 588 // 28] + _CK_K_LAST[n % 28] if 0 <= n < 11172 else ch)
    return "".join(out)


def _ck_to_roman(text: str) -> str:
    """Every Hindi and Urdu word in the text spelled in Roman letters, plus Cyrillic/Greek/Korean.
    Anything else is left exactly as it is."""
    import unicodedata
    text = text or ""
    if _CK_DEV_RE.search(text):
        text = re.sub(r"[ऀ-ॿ]+", lambda m: _ck_dev_word(m.group(0)), text)
    if _CK_OTHER_SCRIPT.search(text):
        text = text.replace("،", ",").replace("؟", "?").replace("۔", ".")
        text = re.sub(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]+", lambda m: _ck_urdu_word(m.group(0)), text)
    if _CK_MORE_SCRIPTS.search(text):
        text = "".join(unicodedata.normalize("NFD", ch) if "\u0370" <= ch <= "\u04FF" or "\u1F00" <= ch <= "\u1FFF" else ch for ch in text)
        text = re.sub(r"[\uAC00-\uD7A3]+", lambda m: _ck_korean(m.group(0)), text)
        text = unicodedata.normalize("NFC", re.sub(r"[\u0370-\u03FF\u0400-\u04FF\u0300-\u036F]+", lambda m: _ck_letters(m.group(0)), text))
    return text


# ---------------------------------------------------------------------------
# Gemini path (optional — user's key from config.json, never hard-coded)
# ---------------------------------------------------------------------------

def load_gemini_key(api_key=None, config_path=None):
    """Read the user's Gemini key.

    Priority: (1) an explicit api_key argument wins; (2) CFB settings —
    clipper.config.settings.gemini_api_key, loaded from the .env
    GEMINI_API_KEY (never hard-coded); (3) legacy fallback: a donor-style
    config.json at config_path, only when config_path is passed
    explicitly. Returns None when absent — the caller must then use the
    local path and say so."""
    if api_key and str(api_key).strip():
        return str(api_key).strip()
    try:
        from .config import settings as _settings
        key = (getattr(_settings, "gemini_api_key", "") or "").strip()
        if key:
            return key
    except Exception:
        pass
    if config_path is None:
        return None
    try:
        d = json.load(open(config_path))
        key = (d.get("gemini_api_key") or "").strip()
        return key or None
    except Exception:
        return None


def gemini_transliterate(text, api_key, src="Hindi/Urdu", model="gemini-2.0-flash",
                         timeout=30):
    """Transliterate with Gemini (natural Hinglish / Roman Urdu spelling).
    Returns the Roman text, or None on any failure (caller falls back to
    the local path). Only called when the user configured a key."""
    import urllib.request
    prompt = ("Transliterate this %s text into Roman (Latin) script, using "
              "natural Hinglish / Roman Urdu spelling as people write it on "
              "social media (e.g. azad, karna, mujhe, pakistan). Return ONLY "
              "the transliterated text, no explanation, no quotes.\n\n%s"
              % (src, text))
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    url = ("https://generativelanguage.googleapis.com/v1beta/models/%s:"
           "generateContent?key=%s" % (model, api_key))
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        return d["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"[transliterate:gemini] failed ({e}) — falling back to local")
        return None


# ---------------------------------------------------------------------------
# main entry: words of a transcript -> display words + script
# ---------------------------------------------------------------------------

def to_roman_word(word, src_script, roman_mode="local", gemini_key=None):
    """One word -> Roman. roman_mode: 'local' (default) or 'gemini'."""
    if src_script == "devanagari":
        if roman_mode == "gemini" and gemini_key:
            return None  # sentence-level Gemini handles whole chunks; see below
        return devanagari_to_roman(word)
    if src_script == "urdu":
        if roman_mode == "gemini" and gemini_key:
            return None
        return urdu_to_roman(word)
    return word


def apply_caption_script(words, detected_lang, caption_mode="auto",
                         roman_mode="local", gemini_key=None):
    """Decide the caption script for a transcript and add a 'display' field
    to each word (the text the caption renderer shows).

    caption_mode:
      auto       — Hindi/Urdu speech -> Roman captions (sohaib's order);
                   English/other -> that language's own script.
      en         — English, Latin as-is.
      hi         — Hindi, Devanagari.
      hi-roman   — Hindi, Roman.
      ur         — Urdu, Nastaliq.
      ur-roman   — Urdu, Roman (also 'roman-urdu').
    Returns (words, script, method) where script is 'latin' | 'devanagari'
    | 'urdu' and method is 'local' | 'gemini' | 'native' | 'none'.
    """
    mode = (caption_mode or "auto").lower().replace("_", "-")
    if mode in ("ur-roman", "roman-urdu"):
        want, src = "roman", "ur"
    elif mode == "hi-roman":
        want, src = "roman", "hi"
    elif mode == "hi":
        want, src = "native", "hi"
    elif mode == "ur":
        want, src = "native", "ur"
    elif mode == "en":
        want, src = "latin", "en"
    else:  # auto
        if detected_lang in ("hi", "ur"):
            want, src = "roman", detected_lang
        elif detected_lang == "en":
            want, src = "latin", "en"
        else:
            want, src = "native", detected_lang

    if want == "latin":
        for w in words:
            w["display"] = w["word"]
        return words, "latin", "none"

    if want == "native":
        script = "devanagari" if src == "hi" else ("urdu" if src == "ur" else "latin")
        for w in words:
            # the caption fonts are Latin-only: a word they cannot draw
            # (Devanagari, Urdu, Cyrillic, Greek, Korean) is romanized
            # instead of rendering as tofu boxes.
            if not ck_drawable(w["word"]):
                w["display"] = _ck_to_roman(w["word"])
            else:
                w["display"] = w["word"]
        return words, script, "native"

    # want == "roman": transliterate word by word (local) or chunk by chunk (gemini)
    script_of_words = "devanagari" if src == "hi" else "urdu"
    if roman_mode == "gemini" and gemini_key:
        # Gemini works best on sentences, not single words: transliterate
        # each segment's text, then re-split onto the word timestamps.
        # Simpler robust approach: transliterate the joined word stream
        # in chunks of <= 40 words, then map back positionally.
        method = "gemini"
        flat = [w["word"] for w in words]
        roman = []
        ok = True
        for i in range(0, len(flat), 40):
            chunk = " ".join(flat[i:i + 40])
            res = gemini_transliterate(chunk, gemini_key,
                                       src="Hindi" if src == "hi" else "Urdu")
            if res is None:
                ok = False
                break
            roman.extend(res.split())
        if ok and len(roman) == len(flat):
            for w, r in zip(words, roman):
                w["display"] = r
            return words, "latin", "gemini"
        print("[transliterate] gemini word-count mismatch — falling back to local")

    method = "local"
    fn = devanagari_to_roman if src == "hi" else urdu_to_roman
    for w in words:
        r = fn(w["word"])
        w["display"] = r if r else w["word"]
    return words, "latin", method


def retime_whisper_words(words, segments):
    """Re-fit Whisper word timings to natural speech pace, per segment.

    Whisper words keep their real timings — but a mis-timed word can
    span seconds (UCHVA lit 2.84s on the 2026-11-21 real-clip test:
    Whisper merged the word with a pause). This applies the same
    _natural_word_dur clamp ([0.18, 0.55]s) used for autosubs words:
    within each transcript segment, words run consecutively at natural
    durations, compressed to fit the segment span. Words keep their
    segment (pauses between segments are preserved).
    """
    if not words or not segments:
        return words
    segs = sorted(segments, key=lambda s: s.get("start", 0.0))
    for w in words:
        ws = w.get("start", 0.0)
        seg = None
        for i, s in enumerate(segs):
            if s.get("start", 0.0) <= ws <= s.get("end", 0.0) + 0.5:
                seg = (i, s)
                break
        if seg is None:
            # nearest segment by start time
            j = min(range(len(segs)),
                    key=lambda i: abs(segs[i].get("start", 0.0) - ws))
            seg = (j, segs[j])
        i, s = seg
        w["_cue"] = i
        w["_cue_start"] = s.get("start", w["start"])
        w["_cue_end"] = s.get("end", w["end"])
    words = _retime_words_natural(words)
    for w in words:
        w.pop("_cue", None)
        w.pop("_cue_start", None)
        w.pop("_cue_end", None)
    return words


def to_roman_text(text, lang="hi"):
    """A whole string (hook/title) -> Roman when it is Devanagari/Urdu.

    Same display logic as captions: a Roman-caption clip must not show
    a Devanagari hook. Local transliteration only (deterministic, no
    key needed) — a title is short, the mechanical path is fine here.
    """
    if not text:
        return text
    if script_of(text) == "devanagari":
        return devanagari_to_roman(text)
    if script_of(text) == "urdu":
        return urdu_to_roman(text)
    return text


# ---------------------------------------------------------------------------
# Gemini Roman rewrite via the connected custom.gemini helper (no raw key)
# ---------------------------------------------------------------------------
# His clipper's proven Hindi path is YouTube autosubs + Gemini
# sentence-level Hindi -> natural Roman. Local files have no autosubs,
# so Whisper words are the input instead: transcribe with the SMALL
# model, then ask Gemini (through the already-connected helper, the
# same call pattern as his clipper's gemini_pick.py) to rewrite each
# segment in natural Roman. Word count per segment MUST match, else
# that segment falls back to the local transliteration — a rewritten
# sentence with a different word count cannot be mapped back onto the
# word timestamps. Gemini can fix spelling/transliteration, it cannot
# recover words Whisper never heard: if the small-model words are
# nonsense, the Roman output is correctly-spelled nonsense, and the
# caller must say so.

_GEMINI_HELPER = ("/home/hatch/workspace/custom-connectors/custom.gemini/"
                  "bin/gemini_ask.py")
_GEMINI_ROMAN_MODEL = "gemini-flash-lite-latest"

_ROMAN_REWRITE_PROMPT = """Rewrite these Hindi sentences in natural Roman (Latin script) Hinglish, exactly as people write it on social media. Rules:
- Keep the SAME words in the SAME order — do NOT add, remove, reorder or merge words; only fix the spelling/transliteration.
- English words in proper English spelling (police, star, car, tattoo).
- Hindi/Urdu words in natural Roman spelling (mujhe, nahin, gaadi, pakistan).
- Fix obvious mishearings ONLY when the sentence makes the right word clear (e.g. 'kaasya' before 'aagay' is 'kaisa').
- Return ONLY the rewritten sentences, one per line, in the same order. No explanation, no quotes, no numbering.

%s"""


def _ask_gemini_roman(prompt, timeout=60):
    """One Gemini call through the connected helper. Returns the reply
    text or None on any failure. Never raises, never touches a raw key.
    Retries once (the helper can fail transiently on a cold call)."""
    import subprocess
    import time
    for attempt in (1, 2):
        try:
            r = subprocess.run(
                ["python3", _GEMINI_HELPER, prompt, _GEMINI_ROMAN_MODEL],
                capture_output=True, text=True, timeout=timeout)
        except Exception as e:
            print(f"[roman:gemini] helper call failed ({e})", flush=True)
            return None
        if r.returncode == 0 and (r.stdout or "").strip():
            return r.stdout.strip()
        if attempt == 1:
            time.sleep(2)
    print(f"[roman:gemini] helper failed (rc={r.returncode}) "
          f"{(r.stderr or '')[:120]}", flush=True)
    return None


def gemini_romanize_words(words, segments, lang="hi"):
    """Rewrite Whisper words' display text with Gemini natural Roman.

    Per segment: send the segment's words (Devanagari source) to
    Gemini, get Roman back, require the same word count, assign
    display words positionally. Segments that fail or mismatch keep
    their local-transliteration display. Returns (words, n_rewritten,
    n_fallback) so the caller can report how much Gemini actually did.
    """
    if not words or not segments:
        return words, 0, 0
    # index words per segment by object identity (words are shared dicts)
    seg_words = {}
    for w in words:
        ws = w.get("start", 0.0)
        best = None
        for i, s in enumerate(segments):
            if s.get("start", 0.0) <= ws <= s.get("end", 0.0) + 0.5:
                best = i
                break
        if best is None:
            best = min(range(len(segments)),
                       key=lambda i: abs(segments[i].get("start", 0.0) - ws))
        seg_words.setdefault(best, []).append(w)
    rewritten, fallback = 0, 0
    for i, ws in sorted(seg_words.items()):
        src = " ".join(w["word"] for w in ws)
        if not src.strip():
            continue
        # skip segments that are already Latin (English game dialogue)
        if script_of(src) == "latin":
            for w in ws:
                w["display"] = w["word"]
            continue
        res = _ask_gemini_roman(_ROMAN_REWRITE_PROMPT % src)
        if res is None:
            fallback += 1
            continue
        lines = [ln.strip() for ln in res.splitlines() if ln.strip()]
        roman = " ".join(lines).split()
        if len(roman) != len(ws):
            print(f"[roman:gemini] segment {i}: word-count mismatch "
                  f"({len(roman)} vs {len(ws)}) — local fallback", flush=True)
            fallback += 1
            continue
        for w, r in zip(ws, roman):
            w["display"] = r
        rewritten += 1
    return words, rewritten, fallback


# ---------------------------------------------------------------------------
# Auto-subs -> transcript + natural word retiming (ported from donor vod.py)
# ---------------------------------------------------------------------------
# YouTube auto-subs repeat each line across rolling cues, so tokens are
# deduped by suffix-prefix match, and word timings use natural speech
# pace (~0.30s/word from display length, fit to each cue's span), NOT an
# even split across the cue: the even split stretched words to 3-5s and
# the red caption highlight lagged the voice (proven 2026-11-20, fixed
# by re-timing to Whisper-measured pace and verified frame by frame).
# Whisper words keep their real timings — retiming applies ONLY to
# autosubs-derived words (their cue spans are not word timings).


def _ytdlp_cmd():
    """Command prefix for yt-dlp.
    Source checkout: [python, -m, yt_dlp, --no-check-certificates].
    PyInstaller exe: the bundled yt-dlp.exe (extracted to _MEIPASS, or
    sitting next to the exe in the portable layout) — sys.executable is
    the CFB exe itself there, so `-m yt_dlp` would not work."""
    import sys
    if getattr(sys, "frozen", False):
        from pathlib import Path
        for base in (getattr(sys, "_MEIPASS", None),
                     str(Path(sys.executable).resolve().parent)):
            if base:
                p = Path(base) / "yt-dlp.exe"
                if p.exists():
                    return [str(p), "--no-check-certificates"]
        return ["yt-dlp", "--no-check-certificates"]
    return [sys.executable, "-m", "yt_dlp", "--no-check-certificates"]


def fetch_autosubs(url, work_dir, lang_hint=None):
    """Fetch YouTube auto-captions for a URL with yt-dlp (proven 2026-11-16
    on the Kakori video: Hindi auto-subs in ~8s, 1,464 segments).
    Returns (cues, lang) where cues = [{start, end, text}] (rolling-window
    duplicates merged), or (None, None) when no subtitles exist.
    Auto-subs are usually far cleaner than local Whisper for Hindi/Urdu,
    so VOD prefers them when available. Local files have no subs."""
    import glob
    import subprocess
    from pathlib import Path
    langs = []
    if (lang_hint or "").lower() in ("hi", "ur", "roman-urdu", "hi-roman", "ur-roman"):
        langs = ["hi.*", "ur.*", "en.*"]
    elif (lang_hint or "").lower() == "en":
        langs = ["en.*", "hi.*"]
    else:
        langs = ["hi.*", "ur.*", "en.*"]
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    out_tmpl = os.path.join(work_dir, "subs.%(ext)s")
    cmd = _ytdlp_cmd() + ["--write-auto-subs", "--write-subs",
           "--sub-lang", ",".join(langs), "--sub-format", "vtt",
           "--skip-download", "-o", out_tmpl, url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"[roman_captions:subs] yt-dlp subs failed: {r.stderr[-300:]}", flush=True)
        return None, None
    # prefer hi, then ur, then en
    for want in ("hi", "ur", "en"):
        for f in sorted(glob.glob(os.path.join(work_dir, "subs.*.vtt"))):
            if f".{want}" in f or f.endswith(f".{want}.vtt"):
                cues = _parse_vtt(f)
                if cues:
                    print(f"[roman_captions:subs] got {len(cues)} cues ({want}) from {f}",
                          flush=True)
                    return cues, want
    print("[roman_captions:subs] no usable subtitles found", flush=True)
    return None, None


def _parse_vtt(path):
    """Parse a VTT file into [{start, end, text}] with rolling-window
    duplicates merged (YouTube auto-subs repeat each line across cues)."""
    import re as _re
    ts_re = _re.compile(r"(\d+):(\d+):(\d+)\.(\d+)\s*-->\s*(\d+):(\d+):(\d+)\.(\d+)")

    def _ts(h, m, s, ms):
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
    cues, cur = [], None
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        m = ts_re.match(line)
        if m:
            if cur:
                cues.append(cur)
            cur = {"start": _ts(*m.groups()[:4]), "end": _ts(*m.groups()[4:]),
                   "text": ""}
        elif cur is not None and line and not line.startswith(("WEBVTT", "NOTE")):
            txt = _re.sub(r"<[^>]+>", "", line).strip()
            if txt:
                cur["text"] = (cur["text"] + " " + txt).strip() if cur["text"] else txt
    if cur:
        cues.append(cur)
    # merge: drop a cue whose text is fully contained in the previous cue's text
    merged = []
    for c in cues:
        if not c["text"]:
            continue
        if merged and c["text"] in merged[-1]["text"]:
            merged[-1]["end"] = max(merged[-1]["end"], c["end"])
            continue
        if merged and merged[-1]["text"] in c["text"]:
            merged[-1] = c
            continue
        merged.append(c)
    return merged


def _natural_word_dur(display_word):
    """Natural seconds a word stays highlighted, from its display length.

    Calibrated 2026-11-20 to Whisper's measured median word duration
    (~0.30s) on the Ducky Reloaded GTA 6 VOD: ~0.055s per character +
    0.08s base, clamped to [0.18, 0.55]s. The old even split across a
    cue's span stretched words to 3-5s, so the red highlight lagged the
    voice; this keeps the highlight moving at speech pace.
    """
    w = (display_word or "").strip(".,!?")
    return max(0.18, min(0.55, 0.055 * len(w) + 0.08))


def _retime_words_natural(words):
    """Re-fit word timings to natural speech pace, per cue (phrase).

    Words carry _cue_start/_cue_end (the cue they came from). Within a
    cue, words run consecutively from the cue start at natural durations
    (from the DISPLAY text length). If the natural total exceeds the cue
    span, compress proportionally to fit (min 0.08s/word) so words never
    spill into the next cue. If it is shorter, words end early — a natural
    pause before the next cue. Returns words sorted by start, non-overlapping.
    """
    # group word indices by cue, preserving order
    by_cue = {}
    order = []
    for i, w in enumerate(words):
        key = w.get("_cue", 0)
        if key not in by_cue:
            by_cue[key] = []
            order.append(key)
        by_cue[key].append(i)
    for key in order:
        idxs = by_cue[key]
        first = words[idxs[0]]
        cue_start = first.get("_cue_start", first["start"])
        cue_end = first.get("_cue_end", first["end"])
        span = max(0.01, cue_end - cue_start)
        durs = [_natural_word_dur(w.get("display") or w["word"])
                for w in (words[i] for i in idxs)]
        total = sum(durs)
        if total > span:
            scale = span / total
            durs = [max(0.08, d * scale) for d in durs]
        cur = cue_start
        for i, d in zip(idxs, durs):
            words[i]["start"] = round(cur, 3)
            words[i]["end"] = round(cur + d, 3)
            cur += d
    words.sort(key=lambda w: w["start"])
    # cursor normalisation: strictly increasing, >= 0.08s each
    cursor = None
    for w in words:
        s = w["start"]
        if cursor is not None:
            s = max(s, cursor)
        e = max(w["end"], s + 0.08)
        w["start"], w["end"] = round(s, 3), round(e, 3)
        cursor = e
    return words


def autosubs_to_transcript(cues, duration, caption_mode="auto",
                           roman_mode="local", detected_lang="hi",
                           gemini_key=None):
    """Auto-sub cues -> transcript dict in the CFB (transcriber) shape.

    YouTube auto-subs repeat each line across rolling cues, so tokens are
    deduped by suffix-prefix match against what was already emitted —
    only NEW words per cue are kept. Word timings use natural speech pace
    (~0.30s/word from display length, fit to each cue's span), NOT an
    even split across the cue: the even split stretched words to 3-5s and
    the red caption highlight lagged the voice (proven 2026-11-20, fixed
    by re-timing to Whisper-measured pace and verified frame by frame).

    CFB shape (differs from the donor, which returned segments without
    words and no "language"/"full_text"): segments carry their "words"
    so clipper.candidate_picker works unchanged, "language" mirrors
    detected_lang, and "full_text" is the deduped word stream.
    """
    words, segs = [], []
    acc = []  # tokens already emitted, for suffix-prefix dedup
    for ci, c in enumerate(cues):
        toks = c["text"].split()
        if not toks:
            continue
        k = 0
        maxk = min(len(acc), len(toks))
        for kk in range(maxk, 0, -1):
            if acc[-kk:] == toks[:kk]:
                k = kk
                break
        new_toks = toks[k:]
        if not new_toks:
            continue
        seg = {"start": round(c["start"], 2), "end": round(c["end"], 2),
               "text": c["text"], "words": []}
        segs.append(seg)
        for tok in new_toks:
            w = {"start": 0.0, "end": 0.0, "word": tok,
                 "_cue": ci, "_cue_start": c["start"],
                 "_cue_end": c["end"]}
            words.append(w)
            seg["words"].append(w)
        acc.extend(new_toks)
    words, cap_script, method = apply_caption_script(
        words, detected_lang, caption_mode, roman_mode, gemini_key)
    words = _retime_words_natural(words)
    for w in words:
        w.pop("_cue", None)
        w.pop("_cue_start", None)
        w.pop("_cue_end", None)
    return {"language": detected_lang, "duration": duration,
            "segments": segs, "full_text": " ".join(w["word"] for w in words),
            "words": words,
            "detected_language": detected_lang, "language_probability": 1.0,
            "caption_script": cap_script, "translit_method": method,
            "transcription_model": "youtube-autosubs", "language_forced": None,
            "script_guard_retried": False}
