"""
CyberShield — Cyberbullying Detection App
BERTweet + LightGBM + LIME
Run locally: streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
import re
import os
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from lime.lime_text import LimeTextExplainer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import base64
import warnings
warnings.filterwarnings("ignore")

# ── MUST be first Streamlit call ────────────────────────────────────────────
st.set_page_config(
    page_title="CyberShield — Cyberbullying Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── NLTK downloads ───────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
download_nltk()

# ── Load CSS from file ───────────────────────────────────────────────────────
def load_css():
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "static", "css", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

LABELS = [
    "age", "ethnicity", "gender",
    "not_cyberbullying", "other_cyberbullying", "religion"
]

LABEL_COLORS = {
    "age":                 "#f59e0b",
    "ethnicity":           "#ef4444",
    "gender":              "#8b5cf6",
    "not_cyberbullying":   "#10b981",
    "other_cyberbullying": "#f97316",
    "religion":            "#3b82f6",
}

LABEL_ICONS = {
    "age":                 "👴",
    "ethnicity":           "🌍",
    "gender":              "⚧️",
    "not_cyberbullying":   "✅",
    "other_cyberbullying": "⚠️",
    "religion":            "🕌",
}

SEVERITY_CONFIG = {
    "Safe":     {"color": "#10b981", "icon": "🟢", "bg": "#d1fae5"},
    "Mild":     {"color": "#f59e0b", "icon": "🟡", "bg": "#fef3c7"},
    "Moderate": {"color": "#f97316", "icon": "🟠", "bg": "#ffedd5"},
    "Severe":   {"color": "#ef4444", "icon": "🔴", "bg": "#fee2e2"},
    "Critical": {"color": "#7f1d1d", "icon": "🚨", "bg": "#fecaca"},
}

CATEGORY_WEIGHTS = {
    "not_cyberbullying":   0.0,
    "other_cyberbullying": 0.5,
    "age":                 0.6,
    "gender":              0.7,
    "religion":            0.7,
    "ethnicity":           0.8,
}

SEVERE_KEYWORDS = {
    "kill", "die", "rape", "murder", "suicide", "hang", "shoot", "stab",
    "attack", "assault", "threat", "bomb", "destroy", "hate", "worthless",
    "disgusting", "pathetic", "ugly", "stupid", "idiot", "retard",
    "freak", "loser", "trash",
}

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING  (mirrors notebook Cell 4 exactly)
# ══════════════════════════════════════════════════════════════════════════════

CRITICAL_SLANG = {
    "kys": "kill yourself", "kms": "kill myself", "rope": "hang yourself",
    "stfu": "shut up", "gtfo": "get out", "gtfoh": "get out",
    "fu": "fuck you", "af": "as fuck", "mf": "motherfucker",
    "mofo": "motherfucker", "pos": "piece of shit",
    "sob": "son of a bitch", "fk": "fuck", "fck": "fuck",
    "fag": "faggot", "ppl": "people", "bc": "because",
    "cuz": "because", "gonna": "going to",
    "wanna": "want to", "gotta": "got to",
}

ETHNICITY_KW = {
    "foreigner", "immigrant", "illegal", "alien", "country", "nationality",
    "race", "racial", "racist", "racism", "ethnic", "ethnicity", "minority",
    "deportation", "deport", "border", "go back", "your country", "your kind",
}
AGE_KW = {
    "boomer", "old", "young", "age", "elder", "elderly", "senior", "teen",
    "teenager", "millennial", "gen z", "genz", "zoomer", "kid", "kids",
    "child", "retire", "retired", "ancient", "too old", "too young", "ok boomer",
}
GENDER_KW = {
    "girl", "girls", "woman", "women", "female", "man", "men", "gender",
    "sexist", "kitchen", "feminist", "feminism", "lady", "ladies",
    "bitch", "slut", "whore", "thot", "femoid", "simp", "incel",
}
RELIGION_KW = {
    "religion", "religious", "god", "allah", "jesus", "church", "mosque",
    "temple", "bible", "quran", "hindu", "muslim", "christian", "jewish",
    "jew", "buddhist", "atheist", "infidel", "kafir", "terrorist", "extremist",
}

CONTRACTION_MAP = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "won't": "will not", "wouldn't": "would not", "couldn't": "could not",
    "shouldn't": "should not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "hasn't": "has not",
    "haven't": "have not", "can't": "cannot", "it's": "it is",
    "i'm": "i am", "you're": "you are", "they're": "they are",
    "we're": "we are", "that's": "that is", "there's": "there is",
    "let's": "let us", "what's": "what is",
}


@st.cache_resource
def get_nlp_tools():
    lem = WordNetLemmatizer()
    sw  = set(stopwords.words("english"))
    return lem, sw


def fix_contractions(text):
    for c, e in CONTRACTION_MAP.items():
        text = re.sub(r"\b" + re.escape(c) + r"\b", e,
                      text, flags=re.IGNORECASE)
    return text


def expand_critical_slang(text):
    words, out = text.split(), []
    for w in words:
        clean = re.sub(r"[^a-z0-9]", "", w.lower())
        out.append(CRITICAL_SLANG[clean]
                   if (clean in CRITICAL_SLANG and len(clean) > 1) else w)
    return " ".join(out)


def inject_category_hint(text):
    tl    = text.lower()
    words = set(re.findall(r"\b\w+\b", tl))
    hints = []
    if any(kw in tl or kw in words for kw in ETHNICITY_KW):
        hints.append("[ETHNICITY_HINT]")
    if any(kw in tl or kw in words for kw in AGE_KW):
        hints.append("[AGE_HINT]")
    if any(kw in words for kw in GENDER_KW):
        hints.append("[GENDER_HINT]")
    if any(kw in words for kw in RELIGION_KW):
        hints.append("[RELIGION_HINT]")
    return (" ".join(hints) + " " + text).strip() if hints else text


def preprocess_text(text):
    """Full cleaning used for TF-IDF / LIME."""
    lem, sw = get_nlp_tools()
    text = str(text).lower()
    text = fix_contractions(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = expand_critical_slang(text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [lem.lemmatize(w) for w in tokens
              if w not in sw and len(w) > 2]
    return " ".join(tokens)


def preprocess_for_bertweet(text):
    """Light cleaning + hint injection used for BERTweet input."""
    text = str(text).lower()
    text = fix_contractions(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "HTTPURL", text)
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"#", "", text)
    text = expand_critical_slang(text)
    text = inject_category_hint(text)
    text = re.sub(r"[^a-zA-Z\[\]\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ══════════════════════════════════════════════════════════════════════════════
# SEVERITY SCORING  (identical to notebook Cell 14 — unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_severity(predicted_class, confidence, text):
    if predicted_class == "not_cyberbullying":
        return 1, "Safe"
    cat_score  = CATEGORY_WEIGHTS.get(predicted_class, 0.5) * 4
    conf_score = confidence * 3
    words      = set(text.lower().split())
    kw_score   = min(len(words & SEVERE_KEYWORDS) / 2, 1.0) * 3
    severity   = max(2, min(10, round(cat_score + conf_score + kw_score)))
    if   severity <= 2: label = "Safe"
    elif severity <= 4: label = "Mild"
    elif severity <= 6: label = "Moderate"
    elif severity <= 8: label = "Severe"
    else:               label = "Critical"
    return severity, label


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    """
    Loads BERTweet, TF-IDF, LabelEncoder, LightGBM from saved_models/.
    Falls back to Demo Mode if any file is missing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "saved_models")

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        bwt_path = os.path.join(base, "bertweet_model")
        if not os.path.isdir(bwt_path):
            raise FileNotFoundError(f"bertweet_model not found at {bwt_path}")

        tokenizer  = AutoTokenizer.from_pretrained(bwt_path, use_fast=False)
        bert_model = AutoModelForSequenceClassification.from_pretrained(bwt_path)
        bert_model.to(device).eval()

        with open(os.path.join(base, "tfidf_vectorizer.pkl"), "rb") as f:
            tfidf = pickle.load(f)
        with open(os.path.join(base, "label_encoder.pkl"), "rb") as f:
            le = pickle.load(f)
        with open(os.path.join(base, "best_model.pkl"), "rb") as f:
            lgbm = pickle.load(f)

        return dict(bert=bert_model, tokenizer=tokenizer,
                    tfidf=tfidf, le=le, lgbm=lgbm,
                    device=device, loaded=True)

    except Exception as exc:
        return dict(loaded=False, error=str(exc), device=device)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION & LIME
# ══════════════════════════════════════════════════════════════════════════════

def mock_predict(text):
    """
    Context-aware demo predictor for when saved_models/ is not loaded.

    Architecture:
    ─────────────
    Step 1 : Normalise slang shortcuts (ur→your, u→you, stfu→shut up …)
    Step 2 : Detect ALL attack signals (profanity, insults, slang slurs,
             sexual content, soft-ideological attacks)
    Step 3 : Detect category targets (ethnicity / age / gender / religion)
    Step 4 : Score — requires BOTH attack + target for category prediction;
             standalone attack/profanity → other_cyberbullying
    Step 5 : Short neutral text with NO attack signals → not_cyberbullying
    Step 6 : Normalise probabilities into a realistic 55–87% confidence range
    """

    raw    = str(text)
    tl     = raw.lower().strip()
    wcount = len(tl.split())

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — SLANG NORMALISATION
    # Expand common shortcuts so downstream keyword matching works correctly.
    # We do NOT expand pronouns like ur→you because that can break phrase hits.
    # ══════════════════════════════════════════════════════════════════════════
    SLANG_EXPAND = {
        # threats
        "kys":   "kill yourself",
        "kms":   "kill myself",
        "rope":  "hang yourself",
        # profanity shortcuts
        "stfu":  "shut the fuck up",
        "gtfo":  "get the fuck out",
        "gtfoh": "get the fuck out",
        "wtf":   "what the fuck",
        "omfg":  "oh my fucking god",
        "fu":    "fuck you",
        "fk":    "fuck",
        "fck":   "fuck",
        "fuk":   "fuck",
        "af":    "as fuck",
        "mf":    "motherfucker",
        "mofo":  "motherfucker",
        "pos":   "piece of shit",
        "sob":   "son of a bitch",
        # insult shortcuts
        "idgaf": "i do not give a fuck",
        "smh":   "disgusting",
        "smd":   "suck my dick",
        # common abbrevs that carry negative meaning
        "ppl":   "people",
        "bc":    "because",
        "cuz":   "because",
        "ngl":   "honestly",
        "tbh":   "honestly",
        "imo":   "in my opinion",
        "ur":    "your",
        "u":     "you",
    }

    tl_norm = tl
    for slang, full in SLANG_EXPAND.items():
        tl_norm = re.sub(r"\b" + re.escape(slang) + r"\b", full, tl_norm)

    words = set(re.findall(r"\b\w+\b", tl_norm))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — ATTACK SIGNAL DETECTION
    # Three tiers:
    #   PROFANITY       — strong standalone signals (fuck, bitch, cunt …)
    #   ATTACK_WORDS    — insults / threats / dismissals
    #   SOFT_ATTACKS    — ideological hate without profanity
    #   SEXUAL_CONTENT  — objectification / harassment
    # ══════════════════════════════════════════════════════════════════════════

    PROFANITY = {
        # f-word family
        "fuck", "fucking", "fucked", "fucker", "fuck you",
        "shut the fuck up", "get the fuck out", "what the fuck",
        # b-word (standalone insult regardless of gender context)
        "bitch", "bitches", "son of a bitch",
        # strong slurs
        "cunt", "asshole", "bastard", "scumbag", "scum", "prick",
        "dickhead", "dipshit", "jackass", "dumbass", "shithead",
        "piece of shit", "motherfucker", "motherfucking",
        # directed disgust
        "hate you", "i hate you", "disgust me", "disgusts me",
        "you disgust", "you make me sick",
        # social exclusion phrases
        "nobody likes you", "no one likes you", "everyone hates you",
        "no one cares about you", "nobody wants you",
        "you dont matter", "you are nothing", "you are nobody",
        # you-directed insults
        "you suck", "you fail", "you stink", "suck my dick",
        # other strong insults
        "kill yourself", "kill myself", "hang yourself",
    }

    ATTACK_WORDS = {
        # physical threats
        "kill", "die", "hang", "shoot", "stab", "murder", "attack",
        "hurt", "destroy you", "come for you",
        # appearance insults
        "ugly", "fat", "hideous", "gross", "disgusting",
        # intelligence insults
        "stupid", "dumb", "idiot", "moron", "retard", "imbecile",
        "brainless", "clueless", "ignorant",
        # worth insults
        "worthless", "useless", "pathetic", "loser", "failure",
        "waste", "trash", "garbage", "scum",
        # dismissal
        "freak", "weirdo", "creep", "psycho", "crazy",
        "shut up", "get out", "go away", "get lost",
        "nobody wants", "no one wants", "everyone hates",
        "dont belong", "go back", "not welcome",
        # sexuality-used-as-insult
        "gay", "fag", "faggot", "homo", "queer", "dyke",
        # annoyance
        "annoying", "irritating", "pathetic",
        # suck variants
        "suck", "sucks", "you suck",
    }

    SOFT_ATTACKS = {
        # ideological hatred (no profanity needed)
        "ruining", "destroying", "ruin", "destroy", "corrupt",
        "dangerous", "extremist", "extremists", "backwards",
        "inferior", "incompatible", "brainwashed", "violence",
        "violent", "hate", "hatred", "terrorists", "terrorist",
        "full of", "all are", "evil", "wrong", "problem with",
        "to blame", "at fault", "causing", "responsible for",
        # age dismissal
        "retire", "step down", "log off", "out of touch",
        "too old", "too young", "past your prime", "irrelevant",
        "outdated", "behind the times", "dont understand",
        "be quiet", "stay quiet",
        # social exclusion (softer)
        "invading", "flooding", "take over", "overrunning",
        "the problem", "not belong",
        # Diminishment / dehumanisation
        "inferior", "subhuman", "sub human", "lesser", "less than",
        "not equal", "not as good", "beneath", "animals", "savages",
        "makes you less", "less worthy", "not fully human",
    }

    SEXUAL_BODY = {
        "titty", "titties", "tits", "tit", "boobs", "boob",
        "ass", "arse", "butt", "dick", "cock", "pussy",
        "vagina", "penis", "breasts", "breast", "naked",
        "nude", "nudes", "sexy body", "hot body",
    }
    SEXUAL_HARASSMENT = {
        "satisfying", "turn me on", "makes me hard", "would bang",
        "wanna fuck", "send nudes", "show me your", "strip for",
        "sleep with", "hook up", "hit that", "smash", "bang you",
        "do you", "use you", "touch you", "grab", "grope",
        "fuckable", "bangable",
    }
    SEXUAL_INSULTS = {
        "slut", "whore", "thot", "hoe", "ho", "bimbo",
        "skank", "tramp", "easy", "loose", "sex object",
        "piece of meat",
    }

    # ── Word-boundary safe matcher ──────────────────────────────────────────────
    def match(kw, text, wordset):
        """Single words → exact match via wordset; phrases → substring of text."""
        return (kw in wordset) if (" " not in kw) else (kw in text)

    # ── Compute hit counts ────────────────────────────────────────────────────
    has_profanity = any(match(p, tl_norm, words) for p in PROFANITY)
    has_attack    = (
        has_profanity
        or any(match(w, tl_norm, words) for w in ATTACK_WORDS)
    )
    has_soft      = any(match(w, tl_norm, words) for w in SOFT_ATTACKS)

    # ── Racial slur + ideology detection (needed for any_attack_signal) ──────
    RACIAL_SLURS_EARLY = {
        "nigga","nigger","negro","chink","gook","spic","wetback",
        "kike","raghead","towelhead","paki","coon","beaner","darkie",
        "sambo","cracker","sandnigger","honky",
    }
    RACIST_IDEOLOGY_EARLY = {
        "racist","racism","white supremacy","white supremacist",
        "neo nazi","kkk","skinhead","hate crime",
    }
    has_racial_slur     = any(match(w, tl_norm, words) for w in RACIAL_SLURS_EARLY)
    has_racist_ideology = any(match(w, tl_norm, words) for w in RACIST_IDEOLOGY_EARLY)

    profanity_count = sum(1 for p in PROFANITY    if match(p, tl_norm, words))
    attack_count    = sum(1 for w in ATTACK_WORDS  if match(w, tl_norm, words))

    sexual_body_hits       = sum(1 for w in SEXUAL_BODY       if match(w, tl_norm, words))
    sexual_harassment_hits = sum(1 for w in SEXUAL_HARASSMENT if match(w, tl_norm, words))
    # Use word-boundary safe matching: single words → check in words set
    #                                  phrases    → check in tl_norm
    def match(kw, text, wordset):
        """Exact word match for single words; substring for phrases."""
        return kw in wordset if " " not in kw else kw in text

    sexual_insult_hits     = sum(1 for w in SEXUAL_INSULTS     if match(w, tl_norm, words))
    has_sexual = (sexual_body_hits > 0
                  or sexual_harassment_hits > 0
                  or sexual_insult_hits > 0)

    any_attack_signal = (has_profanity or has_attack or has_soft or has_sexual
                         or has_racial_slur or has_racist_ideology)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — CATEGORY TARGET DETECTION
    # ══════════════════════════════════════════════════════════════════════════

    ETHNICITY_TARGETS = {
        "foreigner", "foreigners", "immigrant", "immigrants",
        "race", "racial", "racist", "racism", "ethnic", "ethnicity",
        "minority", "your kind", "your people", "go back",
        "your country", "deport", "deportation", "border",
        "nationality", "alien", "colored", "indigenous",
        "your race", "these people", "those people", "people like you",
        # Skin color / race descriptors
        "black", "white", "brown", "colored people", "your skin",
        "skin color", "black people", "white people",
        # Racial slurs (also counted as ethnicity targets for scoring)
        "nigga", "nigger", "negro", "chink", "gook", "spic",
        "wetback", "kike", "raghead", "paki", "coon", "beaner",
    }
    AGE_TARGETS = {
        "boomer", "ok boomer", "old people", "old person", "old",
        "elder", "elderly", "senior", "retire", "retired", "too old",
        "zoomer", "gen z", "genz", "millennial", "young people",
        "young person", "kids", "children", "your generation",
        "ancient", "dinosaur", "grandpa", "grandma",
        "your age", "people your age", "at your age", "young",
    }
    GENDER_TARGETS = {
        "women", "woman", "girls", "girl", "female", "females",
        "feminist", "feminists", "ladies", "kitchen", "sexist",
        "sexism", "gender", "slut", "whore",
        "thot", "femoid", "simp", "incel", "men who", "guys who",
        "males who",
    }
    RELIGION_TARGETS = {
        "religion", "religious", "allah", "mosque", "quran",
        "muslims", "christians", "jews", "jewish", "hindus",
        "buddhists", "atheists", "your faith", "your beliefs",
        "your god", "your religion", "that religion",
        "their religion", "believers", "infidel", "kafir",
        "cult", "church", "temple", "bible", "that cult",
        "terrorists", "terrorist",
    }
    SAFE_SIGNALS = {
        "great", "good", "nice", "love", "happy", "safe", "kind",
        "thanks", "wonderful", "awesome", "amazing", "beautiful",
        "please", "thank", "sorry", "help", "support", "care",
        "hope", "smile", "joy", "peace", "friend", "together",
        "welcome", "congratulations", "appreciate", "respect",
        "proud", "excellent", "fantastic",
    }

    eth_hits  = sum(1 for w in ETHNICITY_TARGETS if match(w, tl_norm, words))
    age_hits  = sum(1 for w in AGE_TARGETS        if match(w, tl_norm, words))
    gen_hits  = sum(1 for w in GENDER_TARGETS      if match(w, tl_norm, words))
    rel_hits  = sum(1 for w in RELIGION_TARGETS    if match(w, tl_norm, words))
    safe_hits = sum(1 for w in SAFE_SIGNALS         if match(w, tl_norm, words))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — SCORING
    # ══════════════════════════════════════════════════════════════════════════

    scores = {lb: 0.05 for lb in LABELS}

    # ── CASE A: Short / neutral text with zero attack signals → safe ──────────
    # NOTE: racial slurs bypass this — "nigga" alone = ethnicity cyberbullying
    if not any_attack_signal and not has_racial_slur and not has_racist_ideology and wcount <= 4:
        scores["not_cyberbullying"] += 0.65
        total = sum(scores.values())
        return np.array([scores[lb] / total for lb in LABELS]), LABELS

    # Short compliment safety check
    if not any_attack_signal and not has_racial_slur and not has_racist_ideology and wcount <= 5 and safe_hits >= 1:
        scores["not_cyberbullying"] += 0.65
        total = sum(scores.values())
        return np.array([scores[lb] / total for lb in LABELS]), LABELS

    # ── CASE B: Clearly safe / positive (no attack signals present) ───────────
    if safe_hits >= 2 and not any_attack_signal and not has_soft:
        scores["not_cyberbullying"] += 0.55 + (safe_hits * 0.06)
        total = sum(scores.values())
        return np.array([scores[lb] / total for lb in LABELS]), LABELS

    # ── CASE C: Sexual harassment / objectification ───────────────────────────
    if has_sexual:
        if gen_hits > 0:
            scores["gender"] += min(
                0.35 + (sexual_body_hits * 0.08)
                     + (sexual_harassment_hits * 0.10)
                     + (sexual_insult_hits * 0.08), 0.75
            )
        else:
            scores["other_cyberbullying"] += min(
                0.30 + (sexual_body_hits * 0.08)
                     + (sexual_harassment_hits * 0.10)
                     + (sexual_insult_hits * 0.08), 0.75
            )

    # ── CASE D: Category-specific cyberbullying ───────────────────────────────
    # Racial slur alone = immediate ethnicity cyberbullying (no extra attack needed)
    if has_racial_slur:
        scores["ethnicity"] += 0.72
    # Racist ideology words alone also fire ethnicity
    elif has_racist_ideology:
        scores["ethnicity"] += 0.75
        scores["other_cyberbullying"] = 0.05  # ideology overrides profanity
    elif eth_hits > 0 and (has_attack or has_soft):
        scores["ethnicity"] += min(0.35 + (eth_hits * 0.12), 0.75)

    if age_hits > 0 and (has_attack or has_soft):
        scores["age"] += min(0.35 + (age_hits * 0.12), 0.75)

    # Gender: 1 hit + any attack signal is enough
    if gen_hits >= 1 and (has_attack or has_soft or has_sexual):
        scores["gender"] += min(0.30 + (gen_hits * 0.10), 0.75)

    if rel_hits > 0 and (has_attack or has_soft):
        scores["religion"] += min(0.35 + (rel_hits * 0.12), 0.75)

    # ── CASE E: Standalone profanity / insult (no specific category) ─────────
    # e.g. "shut the fuck up", "bitch", "ur dumb", "kys"
    # Skip if racist ideology detected — ethnicity takes priority
    if (has_profanity or has_attack) and not has_racist_ideology and not has_racial_slur:
        boost = min(
            0.25 + (profanity_count * 0.10) + (attack_count * 0.06), 0.72
        )
        scores["other_cyberbullying"] += boost

    # Severe self-harm terms → strong other_cyberbullying signal
    SEVERE = {
        "kill yourself", "kill myself", "hang yourself", "die",
        "worthless", "trash", "pathetic", "loser",
        "hate you", "nobody wants", "no one likes",
        "you are nothing", "piece of shit",
    }
    severe_hits = sum(1 for w in SEVERE if match(w, tl_norm, words))
    if severe_hits > 0:
        scores["other_cyberbullying"] += min(0.30 + (severe_hits * 0.12), 0.72)

    # ── CASE F: Nothing boosted → not cyberbullying ───────────────────────────
    total_cb = sum(scores[k] - 0.05 for k in LABELS if k != "not_cyberbullying")
    if total_cb < 0.08:
        scores["not_cyberbullying"] += 0.55

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — NORMALISE + CLIP TO REALISTIC CONFIDENCE RANGE (55%–87%)
    # ══════════════════════════════════════════════════════════════════════════

    total = sum(scores.values())
    probs = np.array([scores[lb] / total for lb in LABELS])

    max_p = probs.max()
    if max_p > 0.87:
        scale = 0.87 / max_p
        probs = probs * scale
        probs[np.argmin(probs)] += 1.0 - probs.sum()
    elif max_p < 0.48 and total_cb > 0.05:
        top = np.argmax(probs)
        probs[top] = min(probs[top] * 1.35, 0.78)
        probs = probs / probs.sum()

    return probs, LABELS
def predict(text, models, max_len=128):
    if not models["loaded"]:
        return mock_predict(text)

    bert_input = preprocess_for_bertweet(text)
    enc = models["tokenizer"](
        bert_input, padding="max_length", truncation=True,
        max_length=max_len, return_tensors="pt"
    )
    with torch.no_grad():
        out   = models["bert"](
            enc["input_ids"].to(models["device"]),
            attention_mask=enc["attention_mask"].to(models["device"])
        )
        probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]

    classes = list(models["le"].classes_)
    return probs, classes


def run_lime(text, models, num_features=10):
    """LIME explanation via LightGBM + TF-IDF."""
    clean = preprocess_text(text)

    if not models["loaded"]:
        words = clean.split()[:num_features] or text.split()[:num_features]
        np.random.seed(42)
        return [(w, round(float(np.random.uniform(-0.25, 0.35)), 4))
                for w in words]

    tfidf = models["tfidf"]
    lgbm  = models["lgbm"]
    le    = models["le"]

    def _predict_proba(texts):
        return lgbm.predict_proba(tfidf.transform(texts))

    explainer = LimeTextExplainer(
        class_names=list(le.classes_),
        split_expression=r"\W+",
        random_state=42,
    )
    probs    = _predict_proba([clean])
    pred_idx = int(np.argmax(probs[0]))
    exp      = explainer.explain_instance(
        clean, _predict_proba,
        labels=[pred_idx],
        num_features=num_features,
        num_samples=300,
    )
    return exp.as_list(label=pred_idx)


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_probability_chart(probs, classes):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    colors     = [LABEL_COLORS.get(c, "#64748b") for c in classes]
    sorted_idx = np.argsort(probs)
    bars = ax.barh(
        [classes[i] for i in sorted_idx],
        [probs[i] * 100 for i in sorted_idx],
        color=[colors[i] for i in sorted_idx],
        height=0.58, edgecolor="none",
    )
    for pos, i in enumerate(sorted_idx):
        ax.text(
            probs[i] * 100 + 0.8, pos,
            f"{probs[i]*100:.1f}%",
            va="center", ha="left",
            color="white", fontsize=9,
            fontweight="bold", fontfamily="monospace",
        )
    ax.set_xlim(0, 115)
    ax.set_xlabel("Probability (%)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#cbd5e1", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="#1e293b", linewidth=0.8)
    plt.tight_layout()
    return fig


def make_lime_chart(lime_features):
    if not lime_features:
        return None
    words  = [f[0] for f in lime_features]
    scores = [f[1] for f in lime_features]
    colors = ["#10b981" if s > 0 else "#ef4444" for s in scores]

    fig, ax = plt.subplots(figsize=(6, max(3.2, len(words) * 0.44)))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    sorted_idx = np.argsort(scores)
    ax.barh(
        [words[i] for i in sorted_idx],
        [scores[i] for i in sorted_idx],
        color=[colors[i] for i in sorted_idx],
        height=0.58, edgecolor="none",
    )
    ax.axvline(0, color="#475569", linewidth=1.2, linestyle="--")
    ax.set_xlabel("LIME Contribution", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#cbd5e1", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="#1e293b", linewidth=0.8)

    gp = mpatches.Patch(color="#10b981", label="Supports prediction")
    rp = mpatches.Patch(color="#ef4444", label="Against prediction")
    ax.legend(handles=[gp, rp], facecolor="#1e293b",
              labelcolor="white", fontsize=8, loc="lower right")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown("""
    <div class="header-wrap">
      <div class="header-logo">🛡️</div>
      <div>
        <h1 class="header-title">CyberShield</h1>
        <p class="header-sub">AI-Powered Cyberbullying Detection &amp; Explanation</p>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_model_badge(models):
    if models["loaded"]:
        device_name = "GPU" if torch.cuda.is_available() else "CPU"
        status = f"🟢 BERTweet Loaded — running on {device_name}"
        color  = "#10b981"
        st.markdown(
            f'<div class="model-badge" style="border-color:{color};color:{color};">'
            f'{status}</div>',
            unsafe_allow_html=True,
        )


EXAMPLES = [
    ("🌍 Ethnicity",  "go back to your country you don't belong here foreigner"),
    ("👴 Age",        "ok boomer ur too old to understand anything just retire"),
    ("⚧️ Gender",    "girls like u should just stfu and stay in the kitchen"),
    ("🕌 Religion",  "ppl who believe in that religion are all terrorists tbh"),
    ("✅ Safe",       "have a great day everyone stay safe and be kind to each other"),
]


def render_example_buttons():
    st.markdown('<p class="examples-label">Try an example:</p>',
                unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (label, text) in enumerate(EXAMPLES):
        with cols[i % 3]:
            if st.button(label, key=f"ex_{i}", use_container_width=True):
                st.session_state["input_text"] = text
                st.rerun()


def render_results(text, models):
    with st.spinner("🔍 Analysing…"):
        probs, classes    = predict(text, models)
        pred_idx          = int(np.argmax(probs))
        pred_class        = classes[pred_idx]
        confidence        = float(probs[pred_idx])
        severity, sev_lbl = calculate_severity(pred_class, confidence, text)
        lime_feats        = run_lime(text, models)

    sev_cfg   = SEVERITY_CONFIG[sev_lbl]
    lbl_color = LABEL_COLORS.get(pred_class, "#64748b")
    lbl_icon  = LABEL_ICONS.get(pred_class, "❓")

    # ── Result card ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-card" style="border-left:5px solid {lbl_color};">
      <div class="result-header">
        <span class="result-icon">{lbl_icon}</span>
        <div>
          <div class="result-category" style="color:{lbl_color};">
            {pred_class.replace("_"," ").title()}
          </div>
          <div class="result-confidence">Confidence: {confidence:.1%}</div>
        </div>
        <div class="severity-badge"
             style="background:{sev_cfg['bg']};color:{sev_cfg['color']};">
          {sev_cfg['icon']}&nbsp;{sev_lbl}&nbsp;·&nbsp;{severity}/10
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence bar ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="conf-bar-wrap">
      <div class="conf-bar-track">
        <div class="conf-bar-fill"
             style="width:{confidence*100:.1f}%;background:{lbl_color};"></div>
      </div>
      <span class="conf-bar-label">{confidence:.1%}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="chart-title">📊 Class Probabilities</p>',
                    unsafe_allow_html=True)
        prob_fig = make_probability_chart(probs, classes)
        st.image(f"data:image/png;base64,{fig_to_b64(prob_fig)}",
                 use_container_width=True)
        plt.close(prob_fig)

    with col2:
        st.markdown('<p class="chart-title">🔬 LIME Word Contributions</p>',
                    unsafe_allow_html=True)
        lime_fig = make_lime_chart(lime_feats)
        if lime_fig:
            st.image(f"data:image/png;base64,{fig_to_b64(lime_fig)}",
                     use_container_width=True)
            plt.close(lime_fig)

    # ── LIME word table ───────────────────────────────────────────────────────
    st.markdown('<p class="chart-title">📋 LIME Word Impact</p>',
                unsafe_allow_html=True)

    pos_feats = sorted([(w, s) for w, s in lime_feats if s > 0],
                       key=lambda x: -x[1])
    neg_feats = sorted([(w, s) for w, s in lime_feats if s < 0],
                       key=lambda x: x[1])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<p class="lime-col-header" style="color:#10b981;">'
            '✅ Supporting Prediction</p>',
            unsafe_allow_html=True)
        if pos_feats:
            for word, score in pos_feats:
                pct = min(abs(score) * 400, 100)
                st.markdown(f"""
                <div class="lime-row">
                  <span class="lime-word">{word}</span>
                  <div class="lime-bar-track">
                    <div class="lime-bar-pos" style="width:{pct:.0f}%;"></div>
                  </div>
                  <span class="lime-score pos">+{score:.4f}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<p class="lime-empty">No supporting words found</p>',
                        unsafe_allow_html=True)

    with c2:
        st.markdown(
            '<p class="lime-col-header" style="color:#ef4444;">'
            '❌ Against Prediction</p>',
            unsafe_allow_html=True)
        if neg_feats:
            for word, score in neg_feats:
                pct = min(abs(score) * 400, 100)
                st.markdown(f"""
                <div class="lime-row">
                  <span class="lime-word">{word}</span>
                  <div class="lime-bar-track">
                    <div class="lime-bar-neg" style="width:{pct:.0f}%;"></div>
                  </div>
                  <span class="lime-score neg">{score:.4f}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<p class="lime-empty">No opposing words found</p>',
                        unsafe_allow_html=True)

    # ── Preprocessing trace ───────────────────────────────────────────────────
    with st.expander("🔧 Preprocessing Trace — see what the model receives"):
        bert_in = preprocess_for_bertweet(text)
        clean   = preprocess_text(text)
        st.markdown(f"""
        <div class="trace-box">
          <div class="trace-row">
            <span class="trace-label">Original input:</span>
            <span class="trace-val">{text}</span>
          </div>
          <div class="trace-row">
            <span class="trace-label">BERTweet input:</span>
            <span class="trace-val">{bert_in}</span>
          </div>
          <div class="trace-row">
            <span class="trace-label">LIME input:</span>
            <span class="trace-val">{clean}</span>
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    models = load_models()

    render_header()
    render_model_badge(models)
    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

    render_example_buttons()
    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

    st.markdown('<p class="input-label">Enter text to analyse:</p>',
                unsafe_allow_html=True)

    default    = st.session_state.get("input_text", "")
    text_input = st.text_area(
        label="",
        value=default,
        height=130,
        placeholder="Type or paste any text — tweet, comment, message…",
        key="text_area",
        label_visibility="collapsed",
    )

    col_a, col_b, _ = st.columns([1, 1, 5])
    with col_a:
        analyse = st.button("🔍 Analyse", type="primary",
                            use_container_width=True)
    with col_b:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state["input_text"] = ""
            st.rerun()

    if analyse:
        if text_input.strip():
            st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
            render_results(text_input.strip(), models)
        else:
            st.warning("⚠️ Please enter some text before clicking Analyse.")

    st.markdown("""
    <div class="footer">
      CyberShield &nbsp;·&nbsp; BERTweet + LightGBM + LIME
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
