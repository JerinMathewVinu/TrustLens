"""
ml_engine.py — TrustLens ML Engine v3
=======================================
Serves both sentiment analysis and fake news detection APIs.

Key upgrades from v1/v2:
  - Dual-model fake news detection (full-text + title-only models)
  - Length-aware routing: short inputs use title model, long inputs blend both
  - Sensationalism scoring adjusts probability before final decision
  - Domain-gap correction: factual science/health/finance text gets fair treatment
  - Trust/Misleading scores derived from model probabilities (no more keyword hack)
  - Confidence calculated from real predict_proba (no more decision_function abuse)

Required files in same directory:
  sentiment_model.pkl
  fake_news_full.pkl       <- trained by train_fake_news.py
  fake_news_title.pkl      <- trained by train_fake_news.py

Start server:
    python ml_engine.py

Endpoints:
    POST /analyze-sentiment    { "text": "..." }
    POST /analyze-fakenews     { "text": "..." }
    GET  /health
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import re
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("TrustLens-ML")

app = Flask(__name__)
CORS(app)

# ── Model paths ────────────────────────────────────────────────────────────────
SENTIMENT_MODEL_PATH  = "sentiment_model.pkl"
FAKE_NEWS_FULL_PATH   = "fake_news_full.pkl"    # trained on title + full text
FAKE_NEWS_TITLE_PATH  = "fake_news_title.pkl"   # trained on title only

# ── Global model holders ───────────────────────────────────────────────────────
sentiment_model   = None
fake_news_full    = None
fake_news_title   = None


# ════════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ════════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Normalise raw text: strip URLs, HTML, source prefixes, lowercase."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    # Strip news source datelines: "WASHINGTON (Reuters) —"
    text = re.sub(r"^[A-Z][A-Z\s,\.\-]+\s*\([^)]+\)\s*[-–]\s*", "", text)
    text = re.sub(r"\(reuters\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\(ap\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


# ════════════════════════════════════════════════════════════════════════════════
# SENSATIONALISM & FACTUAL SCORING
# Used to adjust model probabilities for domain-gap cases
# ════════════════════════════════════════════════════════════════════════════════

# Words/phrases strongly associated with fake / sensationalist content
_SENSATIONAL = [
    "breaking", "shocking", "bombshell", "exposed", "disgusting",
    "unbelievable", "secret", "hidden", "leaked", "insider", "explosive",
    "witch hunt", "deep state", "mainstream media", "you wont believe",
    "they dont want you to know", "banned", "censored", "cover up",
    "illuminati", "agenda", "globalist", "wake up", "they are lying",
    "fake news media", "rigged", "hoax", "fraud", "must share",
    "share before deleted", "wake up sheeple", "truth they hide",
]

# Regex patterns strongly associated with factual/credible reporting
_FACTUAL_PATTERNS = [
    r"\d+\s*percent", r"according to", r"study shows?", r"study found",
    r"report says?", r"officials? said", r"researchers?", r"scientists?",
    r"announced", r"approved", r"signed into law", r"voted",
    r"survey", r"research", r"university", r"hospital",
    r"government", r"minister", r"senate", r"congress", r"court ruled",
    r"\bnasa\b", r"\bfda\b", r"\bcdc\b", r"\bwho\b", r"united nations",
    r"published", r"clinical trial", r"peer.reviewed",
    r"data shows?", r"evidence", r"confirmed", r"discovered", r"launched",
]

# Strong factual signals that override model bias for non-political topics
# (the training data is mostly US political news, so science/health/finance
#  headlines are underrepresented in the REAL class)
_STRONG_FACTUAL = [
    r"scientists? (discover|found|report|study|publish)",
    r"researchers? (at|from|publish|found|discover|report)",
    r"nasa (announce|launch|discover|confirm|complete|find)",
    r"fda (approve|clear|authorize|grant)",
    r"study (shows?|finds?|reveals?|published|confirm)",
    r"clinical (trial|study|research)",
    r"stock market", r"dow jones", r"s.p 500",
    r"gdp (grew|fell|rose|dropped|expand)",
    r"unemployment (rate|fell|rose|drop)",
    r"central bank", r"federal reserve",
    r"university (of|researchers?|scientists?)",
    r"published in (the )?journal",
    r"percent (increase|decrease|rise|fall|drop|gain)",
    r"world health organization",
    r"international monetary fund",
]


def _sensationalism_score(text: str) -> float:
    """
    Returns a float:  positive = more sensational (fake-leaning)
                      negative = more factual (real-leaning)
    """
    t = text.lower()
    sens_hits   = sum(1 for w in _SENSATIONAL if w in t)
    factual_hits = sum(1 for p in _FACTUAL_PATTERNS if re.search(p, t))
    return sens_hits - factual_hits * 0.6


def _is_strong_factual(text: str) -> bool:
    """True if the text contains at least one strong factual indicator."""
    t = text.lower()
    return any(re.search(p, t) for p in _STRONG_FACTUAL)


# ════════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════════

def load_models():
    global sentiment_model, fake_news_full, fake_news_title

    for path, name, var_name in [
        (SENTIMENT_MODEL_PATH,  "Sentiment",         "sentiment_model"),
        (FAKE_NEWS_FULL_PATH,   "Fake News (full)",  "fake_news_full"),
        (FAKE_NEWS_TITLE_PATH,  "Fake News (title)", "fake_news_title"),
    ]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                obj = pickle.load(f)
            globals()[var_name] = obj
            log.info(f"✅ {name} model loaded from {path}")
        else:
            log.warning(f"⚠️  {name} model not found at {path}")


# ════════════════════════════════════════════════════════════════════════════════
# FAKE NEWS PREDICTION LOGIC
# ════════════════════════════════════════════════════════════════════════════════

def _predict_fake_news(raw_text: str) -> dict:
    """
    Core fake news prediction with:
      - Length-aware model routing (title model for short, ensemble for long)
      - Sensationalism adjustment
      - Domain-gap correction for factual non-political topics
    """
    clean = clean_text(raw_text)
    words = clean.split()

    # Route: use title model for short inputs, ensemble for long
    if len(words) < 30:
        probs = fake_news_title.predict_proba([clean])[0]
        classes = fake_news_title.classes_
    else:
        p_full  = fake_news_full.predict_proba([clean])[0]
        p_title = fake_news_title.predict_proba([clean])[0]
        # Weighted ensemble: full-text model gets 70% weight
        probs = 0.70 * p_full + 0.30 * p_title
        classes = fake_news_full.classes_

    prob_map = dict(zip(classes, probs))
    real_p = float(prob_map.get("REAL", 0.5))

    # Adjust for sensationalism / factual language
    sens = _sensationalism_score(raw_text)
    real_p = max(0.01, min(0.99, real_p - sens * 0.06))

    # Domain-gap correction: if text has strong factual language AND
    # no sensational markers, ensure it isn't penalised for being non-political
    sens_raw = sum(1 for w in _SENSATIONAL if w in raw_text.lower())
    if _is_strong_factual(raw_text) and sens_raw == 0:
        real_p = max(real_p, 0.65)

    fake_p = 1.0 - real_p

    # Decision threshold: require 60% confidence to call FAKE
    # This prevents neutral sentences from being misclassified
    prediction   = "FAKE" if fake_p >= 0.60 else "REAL"
    trust_score      = int(round(real_p * 100))
    misleading_score = 100 - trust_score
    confidence       = round(max(real_p, fake_p), 4)

    label = "genuine" if prediction == "REAL" else "likely fake or misleading"
    reason = (
        f"This news appears {label}. "
        f"Model confidence: {round(confidence * 100, 1)}%. "
        f"Trust score reflects the probability of authentic reporting "
        f"based on language patterns from ~44,000 real and fake news articles."
    )

    return {
        "prediction":      prediction,
        "confidence":      confidence,
        "trustScore":      trust_score,
        "misleadingScore": misleading_score,
        "reason":          reason,
    }


# ════════════════════════════════════════════════════════════════════════════════
# LEXICON-BASED SENTIMENT SCORING
# Handles nuanced phrases the ML model may miss:
#   "feel relieved", "not bad", "worked wonders", "could be better", etc.
# Returns an integer:  +N = positive, -N = negative, 0 = neutral/uncertain
# ════════════════════════════════════════════════════════════════════════════════

# --- Positive lexicon: relief, satisfaction, benefit, improvement words ---
_POS_PHRASES = [
    # Direct positive
    "love it", "love this", "love the", "loved it", "loved this",
    "great product", "great quality", "great value", "great purchase",
    "highly recommend", "would recommend", "must buy", "must have",
    "excellent", "outstanding", "superb", "magnificent", "brilliant",
    "fantastic", "wonderful", "amazing", "awesome", "incredible",
    "perfect", "perfectly", "works perfectly", "works great", "works well",
    "exactly as described", "as advertised",
    "best purchase", "best product", "best i've ever",
    "very happy", "very pleased", "very satisfied", "really happy",
    "so happy", "so pleased", "so satisfied",
    "glad i bought", "glad i purchased", "glad i got",
    "very impressed", "blown away",
    "exceeded expectations", "exceeded my expectations",
    "worth every penny", "worth the money",
    "top quality", "top notch", "five stars", "5 stars",
    # Relief / comfort / wellness
    "feel relieved", "feeling relieved", "felt relieved",
    "feel better", "feeling better", "felt better", "makes me feel better",
    "feel good", "feeling good", "felt good", "makes me feel good",
    "feel great", "feeling great", "felt great", "makes me feel great",
    "feel wonderful", "feeling wonderful",
    "feel amazing", "feeling amazing",
    "feel much better", "feeling much more", "feels much better",
    "feel relaxed", "feeling relaxed", "felt relaxed",
    "feel calm", "feeling calm", "feel at ease", "put me at ease",
    "feel comfortable", "feeling comfortable", "very comfortable",
    "instant relief", "immediate relief", "great relief",
    "soothing", "soothes", "soothe", "calming", "calms",
    "energized", "more energized", "energetic", "refreshed",
    "refreshing", "revitalized", "rejuvenated",
    "helped me", "really helped", "helped a lot", "helped so much",
    "made a big difference", "makes a big difference",
    "made a difference", "makes a difference",
    "works wonders", "worked wonders", "works like a charm",
    "noticed improvement", "noticed a difference", "big improvement",
    "fast relief", "quick relief", "effective",
    "would buy again", "will buy again", "will definitely buy",
    "definitely buy again", "ordering again", "reordering",
    # Satisfaction signals
    "no complaints", "zero complaints", "no issues", "no problems",
    "easy to use", "user friendly",
    "good quality", "nice quality", "solid quality",
    "happy with", "pleased with", "satisfied with",
    "delighted", "thrilled",
]

# --- Negative lexicon: harm, dissatisfaction, failure words ---
_NEG_PHRASES = [
    # Direct negative — only specific, unambiguous phrases
    "do not buy", "don't buy", "avoid this", "avoid at all costs",
    "not reliable", "can't trust", "cannot trust", "not good",
    "waste of money", "complete waste", "total waste", "waste of time",
    "worst product", "worst purchase", "worst ever",
    "very disappointed", "so disappointed", "really disappointed",
    "disappointed with", "not happy", "not satisfied", "not pleased",
    "very unhappy", "extremely unhappy",
    "stopped working", "doesn't work", "does not work",
    "not working", "never worked", "fell apart", "broke down",
    "defective", "faulty", "poor quality",
    "false advertising", "scam", "fraud",
    "overpriced", "not worth it", "not worth the price",
    "regret buying", "regret purchasing", "wish i hadn't", "bad purchase",
    "returned it", "sent it back", "asked for refund",
    "one star", "1 star",
    # Health / harm signals
    "made me sick", "feel sick", "feeling sick", "felt sick",
    "made me nauseous", "feeling nauseous", "feel nauseous",
    "bad side effects", "negative side effects",
    "allergic reaction", "burning sensation",
    "gave me a headache", "caused headaches",
    "stomachache", "stomach pain", "stomach ache",
    "feel worse", "feeling worse", "made it worse",
    "very irritating",
    "harmful", "dangerous", "unsafe",
    # Failure signals
    "stopped after", "broke after", "died after",
    "never again", "not again", "do not recommend",
    "would not recommend", "wouldn't recommend",
    "misleading description", "not as described",
    "poor customer service",
]

# --- Neutral / mixed-sentiment phrases that cancel strong override ---
# If ANY of these appear, we resist forcing Positive/Negative
_NEUTRAL_PHRASES = [
    "nothing special", "nothing remarkable", "nothing great",
    "okay", "it's okay", "its okay", "alright", "all right",
    "average", "mediocre", "so-so", "so so",
    "not bad", "not great", "not terrible",
    "neither", "hard to say", "hard to tell",
    "does what it says", "does what it claims",
    "as expected", "what i expected",
    "decent enough", "decent",
    "might try", "could be better", "expected better",
    "on time", "arrived on time", "arrived late",
    "packaging was", "standard product", "nothing out of the ordinary",
    "not sure", "unsure", "undecided", "mixed feelings",
]

# Negative single tokens — only truly unambiguous negatives
_NEG_TOKENS = {
    "disappointed", "frustrating", "frustrated", "angry", "upset",
    "horrible", "terrible", "awful", "poor",
    "worse", "worst", "useless", "defective",
    "painful", "sick", "nauseous",
    "regret", "scam", "fraud", "fake",
    "fail", "fails", "breaking", "broken", "ruined"
}

# Negation words that flip the sentiment of the next word/phrase
_NEGATIONS = {
    "not", "no", "never", "doesn't", "don't", "didn't", "isn't",
    "aren't", "wasn't", "weren't", "won't", "wouldn't", "can't",
    "cannot", "couldn't", "hardly", "barely", "neither", "nor",
}

# Intensifiers that strengthen the sentiment
_INTENSIFIERS = {
    "very", "really", "extremely", "incredibly", "absolutely",
    "totally", "utterly", "completely", "highly", "so", "such",
    "deeply", "truly", "quite", "super", "massively", "severely",
}

# --- Sarcasm & Contrast Helpers ---
_SARCASM_PATTERNS = ["not!", "yeah right", "as if", "sure it is", "what a joke", "totally...", "yeah, right", "not really"]

def _is_sarcastic(text: str) -> bool:
    t = text.lower()
    for p in _SARCASM_PATTERNS:
        if p in t:
            return True
    return False

def _extract_latter_part(text: str) -> str:
    # Match contrast words, split and return the focal part
    t_lower = text.lower()
    
    # Handle "although", "though", "even though" at the beginning
    if t_lower.startswith(("although", "though", "even though")):
        # Find the first comma and take everything after it
        idx = text.find(',')
        if idx != -1:
            return text[idx+1:].strip()
            
    # Mid-sentence contrast words
    contrast_words = [" but ", " however ", " yet "]
    last_idx = -1
    for cw in contrast_words:
        idx = t_lower.rfind(cw)
        if idx > last_idx:
            last_idx = idx + len(cw)
            
    if last_idx != -1:
        return text[last_idx:].strip()
    return text


def _lexicon_sentiment_score(text: str) -> dict:
    """
    Multi-signal lexicon scoring.
    Returns a dict:
      'score'   : int  (positive = POS, negative = NEG, 0 = neutral)
      'neutral' : bool (True if neutral phrases detected → resist override)
      'pos_hits': int
      'neg_hits': int
    """
    t = text.lower()
    score = 0
    pos_hits = 0
    neg_hits = 0

    # 0. Check for neutral/mixed-sentiment phrases first
    has_neutral = any(phrase in t for phrase in _NEUTRAL_PHRASES)

    # 1. Phrase matching (multi-word first to avoid partial matches)
    for phrase in _POS_PHRASES:
        if phrase in t:
            score += 2
            pos_hits += 1

    for phrase in _NEG_PHRASES:
        if phrase in t:
            score -= 2
            neg_hits += 1

    # 2. Token-level negation + intensifier pass
    tokens = re.findall(r"\b\w+\b", t)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        negated  = False
        intensify = 1

        # Look back up to 3 tokens for negation / intensifier
        for j in range(max(0, i - 3), i):
            if tokens[j] in _NEGATIONS:
                negated = True
            if tokens[j] in _INTENSIFIERS:
                intensify = 2

        # Positive single-word signals
        if tok in ("relieved", "relief", "soothing", "refreshed", "relaxed",
                   "satisfied", "pleased", "happy", "glad", "delighted",
                   "effective", "helpful", "beneficial", "improved",
                   "better", "wonderful", "excellent", "awesome",
                   "fantastic", "amazing", "love", "loved",
                   "enjoying", "enjoyed", "enjoy", "trust", "reliable"):
            delta = intensify
            if negated:
                score -= delta
                neg_hits += 1
            else:
                score += delta
                pos_hits += 1

        # Negative single-word signals (only unambiguous ones)
        elif tok in _NEG_TOKENS:
            delta = intensify
            if negated:
                score += delta
                pos_hits += 1
            else:
                score -= delta
                neg_hits += 1

        i += 1

    return {"score": score, "neutral": has_neutral, "pos_hits": pos_hits, "neg_hits": neg_hits}


# ════════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route("/analyze-fakenews", methods=["POST"])
def analyze_fakenews():
    try:
        if fake_news_full is None or fake_news_title is None:
            return jsonify({"error": "Fake news models not loaded. Run train_fake_news.py first."}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        raw_text = data.get("text", "").strip()
        if not raw_text:
            return jsonify({"error": "No text provided"}), 400

        result = _predict_fake_news(raw_text)
        return jsonify(result)

    except Exception as e:
        log.exception("Fake news analysis error")
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-sentiment", methods=["POST"])
def analyze_sentiment():
    try:
        if sentiment_model is None:
            return jsonify({"error": "Sentiment model not loaded. Run train_model.py first."}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        raw_text = data.get("text", "").strip()
        if not raw_text:
            return jsonify({"error": "No text provided"}), 400

        # 1. Sarcasm detection
        sarcasm_detected = _is_sarcastic(raw_text)

        # 2. Contrast words handling
        focal_text = _extract_latter_part(raw_text)

        text = clean_text(focal_text)

        prediction = sentiment_model.predict([text])[0]
        probs      = sentiment_model.predict_proba([text])[0]
        classes    = sentiment_model.classes_

        prob_map   = dict(zip(classes, probs))
        confidence = float(max(probs))

        # --- LEXICON-BASED SENTIMENT CORRECTION ---
        # Analyze the FULL text to catch mixed signals properly
        lex_result = _lexicon_sentiment_score(raw_text)
        lex_score  = lex_result["score"]
        is_neutral = lex_result["neutral"]
        pos_hits   = lex_result.get("pos_hits", 0)
        neg_hits   = lex_result.get("neg_hits", 0)
        notes      = []

        raw_lower = raw_text.lower()

        # Contextual Sarcasm check
        if not sarcasm_detected:
            if any(p in raw_lower for p in ["great job", "wow", "amazing", "perfect"]) and neg_hits > 0:
                sarcasm_detected = True

        # Uncertainty check
        uncertain_detected = any(p in raw_lower for p in ["doubt", "skeptical", "not sure", "hard to say", "uncertain"])

        if sarcasm_detected:
            prediction = "NEGATIVE"
            confidence = 0.95
            prob_map["Negative"] = 0.95
            prob_map["Positive"] = 0.02
            prob_map["Neutral"]  = 0.03
            notes.append("Sarcasm detected; sentiment forced to NEGATIVE.")
        elif uncertain_detected:
            prediction = "UNCERTAIN"
            confidence = min(confidence, 0.45)
            notes.append("Uncertainty/Doubt detected; classifying as UNCERTAIN.")
        elif pos_hits > 0 and neg_hits > 0:
            prediction = "MIXED SENTIMENT"
            # reduce confidence if conflicting signals
            confidence = max(0.40, confidence * 0.8)
            notes.append("Both positive and negative signals found (Mixed Sentiment). Confidence reduced.")
        else:
            # Rule 1: Neutral-phrase guard
            # If neutral/mixed phrases are present AND the model has moderate confidence,
            # force Neutral unless the lexicon has a very strong (de-contextualised) signal.
            if is_neutral and abs(lex_score) < 5:
                # Neutral phrases + no overwhelming sentiment → keep/force Neutral
                if confidence < 0.95 or prediction != "Neutral":
                    prediction = "Neutral"
                    prob_map["Neutral"]  = max(prob_map.get("Neutral", 0.5), 0.65)
                    prob_map["Positive"] = min(prob_map.get("Positive", 0.3), 0.25)
                    prob_map["Negative"] = min(prob_map.get("Negative", 0.3), 0.25)
                    confidence = prob_map["Neutral"]
            else:
                # Rule 2: Positive/Negative override (no neutral context)
                pos_threshold = 1
                neg_threshold = 1

                if lex_score >= pos_threshold:
                    prediction = "Positive"
                    boost = min(0.97, 0.70 + lex_score * 0.03)
                    prob_map["Positive"] = boost
                    prob_map["Negative"] = round((1 - boost) * 0.25, 4)
                    prob_map["Neutral"]  = round((1 - boost) * 0.75, 4)
                    confidence = boost
                elif lex_score <= -neg_threshold:
                    prediction = "Negative"
                    boost = min(0.97, 0.70 + abs(lex_score) * 0.03)
                    prob_map["Negative"] = boost
                    prob_map["Positive"] = round((1 - boost) * 0.25, 4)
                    prob_map["Neutral"]  = round((1 - boost) * 0.75, 4)
                    confidence = boost
            
            if focal_text != raw_text:
                notes.append("Contrast word detected; analyzed the latter part of the sentence.")

        # Low Confidence / Uncertain check
        if confidence < 0.50 and prediction not in ["MIXED SENTIMENT", "NEGATIVE"]:
            prediction = "UNCERTAIN"
            notes.append("Model confidence is low; classifying as UNCERTAIN.")
        # ------------------------------------------

        # Trust score: positive sentiment → high trust, negative → lower
        positive_p = float(
            prob_map.get("Positive", 0) + prob_map.get("Neutral", 0) * 0.5
        )
        trust_score      = int(round(positive_p * 100))
        misleading_score = 100 - trust_score
        fake_prediction  = "GENUINE" if trust_score >= 50 else "FAKE"

        final_note = " ".join(notes) if notes else (
            f"Sentiment classified as {prediction.upper()} with "
            f"{round(confidence * 100, 2)}% model confidence."
        )

        return jsonify({
            # New requested keys
            "sentiment":           prediction.upper(),
            "confidence":          round(confidence, 4),
            "note":                final_note,

            # Existing keys for backward compatibility
            "sentimentConfidence": round(confidence, 4),
            "fakePrediction":      fake_prediction,
            "misleadingScore":     misleading_score,
            "trustScore":          trust_score,
            "reason":              final_note,
        })

    except Exception as e:
        log.exception("Sentiment analysis error")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "UP",
        "sentiment_model":        "loaded" if sentiment_model  else "not loaded",
        "fake_news_full_model":   "loaded" if fake_news_full   else "not loaded",
        "fake_news_title_model":  "loaded" if fake_news_title  else "not loaded",
    })


# ════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    load_models()
    log.info("🚀 TrustLens ML Engine v3 running at http://localhost:5050")
    app.run(host="0.0.0.0", port=5050)
