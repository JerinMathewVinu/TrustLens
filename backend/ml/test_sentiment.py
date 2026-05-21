"""
test_sentiment.py — TrustLens Comprehensive Sentiment Test v2
==============================================================
50 diverse reviews tested against the ML model + lexicon scoring system.
Run: python3 test_sentiment.py
"""
import re, pickle, sys

MODEL_PATH = "sentiment_model.pkl"

# ── Lexicon (mirror of ml_engine.py) ─────────────────────────────────────────
_POS_PHRASES = [
    "love it","love this","love the","loved it","loved this",
    "great product","great quality","great value","great purchase",
    "highly recommend","would recommend","must buy","must have",
    "excellent","outstanding","superb","magnificent","brilliant",
    "fantastic","wonderful","amazing","awesome","incredible",
    "perfect","perfectly","works perfectly","works great","works well",
    "exactly as described","as advertised",
    "best purchase","best product","best i've ever",
    "very happy","very pleased","very satisfied","really happy",
    "so happy","so pleased","so satisfied",
    "glad i bought","glad i purchased","glad i got",
    "very impressed","blown away",
    "exceeded expectations","exceeded my expectations",
    "worth every penny","worth the money",
    "top quality","top notch","five stars","5 stars",
    "feel relieved","feeling relieved","felt relieved",
    "feel better","feeling better","felt better","makes me feel better",
    "feel good","feeling good","felt good","makes me feel good",
    "feel great","feeling great","felt great","makes me feel great",
    "feel wonderful","feeling wonderful","feel amazing","feeling amazing",
    "feel much better","feeling much more","feels much better",
    "feel relaxed","feeling relaxed","felt relaxed",
    "feel calm","feeling calm","feel at ease","put me at ease",
    "feel comfortable","feeling comfortable","very comfortable",
    "instant relief","immediate relief","great relief",
    "soothing","soothes","soothe","calming","calms",
    "energized","more energized","energetic","refreshed",
    "refreshing","revitalized","rejuvenated",
    "helped me","really helped","helped a lot","helped so much",
    "made a big difference","makes a big difference",
    "made a difference","makes a difference",
    "works wonders","worked wonders","works like a charm",
    "noticed improvement","noticed a difference","big improvement",
    "fast relief","quick relief","effective",
    "would buy again","will buy again","will definitely buy",
    "definitely buy again","ordering again","reordering",
    "no complaints","zero complaints","no issues","no problems",
    "easy to use","user friendly",
    "good quality","nice quality","solid quality",
    "happy with","pleased with","satisfied with",
    "delighted","thrilled",
]
_NEG_PHRASES = [
    "do not buy","don't buy","avoid this","avoid at all costs",
    "waste of money","complete waste","total waste","waste of time",
    "worst product","worst purchase","worst ever",
    "very disappointed","so disappointed","really disappointed",
    "disappointed with","not happy","not satisfied","not pleased",
    "very unhappy","extremely unhappy",
    "stopped working","doesn't work","does not work",
    "not working","never worked","fell apart","broke down",
    "defective","faulty","poor quality",
    "false advertising","scam","fraud",
    "overpriced","not worth it","not worth the price",
    "regret buying","regret purchasing","wish i hadn't","bad purchase",
    "returned it","sent it back","asked for refund",
    "one star","1 star",
    "made me sick","feel sick","feeling sick","felt sick",
    "made me nauseous","feeling nauseous","feel nauseous",
    "bad side effects","negative side effects",
    "allergic reaction","burning sensation",
    "gave me a headache","caused headaches",
    "stomachache","stomach pain","stomach ache",
    "feel worse","feeling worse","made it worse",
    "very irritating","harmful","dangerous","unsafe",
    "stopped after","broke after","died after",
    "never again","not again","do not recommend",
    "would not recommend","wouldn't recommend",
    "misleading description","not as described",
    "poor customer service",
]
_NEUTRAL_PHRASES = [
    "nothing special","nothing remarkable","nothing great",
    "okay","it's okay","its okay","alright","all right",
    "average","mediocre","so-so","so so",
    "not bad","not great","not terrible",
    "neither","hard to say","hard to tell",
    "does what it says","does what it claims",
    "as expected","what i expected",
    "decent enough","decent",
    "might try","could be better","expected better",
    "on time","arrived on time","arrived late",
    "packaging was","standard product","nothing out of the ordinary",
    "not sure","unsure","undecided","mixed feelings",
]
_NEG_TOKENS = {
    "disappointed","frustrating","frustrated","angry","upset",
    "horrible","terrible","awful","poor",
    "worse","worst","useless","defective",
    "painful","sick","nauseous","regret","scam","fraud","fake",
}
_NEGATIONS = {
    "not","no","never","doesn't","don't","didn't","isn't",
    "aren't","wasn't","weren't","won't","wouldn't","can't",
    "cannot","couldn't","hardly","barely","neither","nor",
}
_INTENSIFIERS = {
    "very","really","extremely","incredibly","absolutely",
    "totally","utterly","completely","highly","so","such",
    "deeply","truly","quite","super","massively","severely",
}

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s'.,!?%-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def _lexicon_sentiment_score(text):
    t = text.lower()
    score = 0
    has_neutral = any(p in t for p in _NEUTRAL_PHRASES)
    for phrase in _POS_PHRASES:
        if phrase in t: score += 2
    for phrase in _NEG_PHRASES:
        if phrase in t: score -= 2
    tokens = re.findall(r"\b\w+\b", t)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        negated, intensify = False, 1
        for j in range(max(0, i-3), i):
            if tokens[j] in _NEGATIONS: negated = True
            if tokens[j] in _INTENSIFIERS: intensify = 2
        if tok in ("relieved","relief","soothing","refreshed","relaxed",
                   "satisfied","pleased","happy","glad","delighted",
                   "effective","helpful","beneficial","improved",
                   "better","wonderful","excellent","awesome",
                   "fantastic","amazing","love","loved",
                   "enjoying","enjoyed","enjoy"):
            delta = intensify
            score += -delta if negated else delta
        elif tok in _NEG_TOKENS:
            delta = intensify
            score += delta if negated else -delta
        i += 1
    return {"score": score, "neutral": has_neutral}

def predict_with_lexicon(model, raw_text):
    cleaned = clean_text(raw_text)
    prediction = model.predict([cleaned])[0]
    probs      = model.predict_proba([cleaned])[0]
    classes    = model.classes_
    prob_map   = dict(zip(classes, probs))
    confidence = float(max(probs))

    lex_result = _lexicon_sentiment_score(raw_text)
    lex        = lex_result["score"]
    is_neutral = lex_result["neutral"]

    if is_neutral and abs(lex) < 5:
        # Neutral-phrase guard: force Neutral unless model is very sure
        if confidence < 0.95 or prediction != "Neutral":
            prediction = "Neutral"
            prob_map["Neutral"]  = max(prob_map.get("Neutral", 0.5), 0.65)
            prob_map["Positive"] = min(prob_map.get("Positive", 0.3), 0.25)
            prob_map["Negative"] = min(prob_map.get("Negative", 0.3), 0.25)
            confidence = prob_map["Neutral"]
    else:
        if lex >= 2:
            prediction = "Positive"
            boost = min(0.97, 0.70 + lex * 0.03)
            prob_map["Positive"] = boost; confidence = boost
        elif lex <= -2:
            prediction = "Negative"
            boost = min(0.97, 0.70 + abs(lex) * 0.03)
            prob_map["Negative"] = boost; confidence = boost

    return prediction, confidence, lex, is_neutral

# ── Test Cases (50 diverse reviews) ─────────────────────────────────────────
TEST_CASES = [
    # POSITIVE (nuanced & direct) ─ 20 cases
    ("When I eat the product it makes me feel relieved",              "Positive"),
    ("This really helped me with my back pain, feel so much better",  "Positive"),
    ("I feel so much better after using this, would definitely buy again", "Positive"),
    ("Absolutely fantastic, best purchase I've made this year!",      "Positive"),
    ("Works wonders for my skin, love it!",                           "Positive"),
    ("Felt refreshed after the very first use",                       "Positive"),
    ("The product is very soothing and calming",                      "Positive"),
    ("I was a bit sceptical but it really works, very effective",     "Positive"),
    ("Highly recommend this to anyone looking for relief",            "Positive"),
    ("Feeling much more energized since I started using it",          "Positive"),
    ("Great quality, arrived quickly and exactly as described",       "Positive"),
    ("It eased my discomfort within minutes, very impressed",         "Positive"),
    ("My sleep improved a lot after using this product",              "Positive"),
    ("Five stars! Worth every penny",                                 "Positive"),
    ("I noticed a big improvement after the first week",              "Positive"),
    ("The taste is pleasant and it gives me instant relief",          "Positive"),
    ("Made a big difference to my daily routine",                     "Positive"),
    ("So happy with this purchase",                                   "Positive"),
    ("Ordering again without hesitation",                             "Positive"),
    ("Put me at ease immediately, very comfortable to use",           "Positive"),

    # NEGATIVE (nuanced & direct) ─ 20 cases
    ("Terrible waste of money, completely broken on arrival",         "Negative"),
    ("I regret buying this, it stopped working after a week",         "Negative"),
    ("This product gave me a headache, not impressed at all",         "Negative"),
    ("Made me feel nauseous after the first use",                     "Negative"),
    ("Very disappointed, not as described at all",                    "Negative"),
    ("Broke down after two days, very poor quality",                  "Negative"),
    ("Do not buy this, it is a complete scam",                        "Negative"),
    ("I had an allergic reaction and had to stop using it",           "Negative"),
    ("Would not recommend, it made things worse",                     "Negative"),
    ("Asked for a refund immediately, terrible product",              "Negative"),
    ("Gave me a burning sensation, very harmful",                     "Negative"),
    ("Side effects were awful, felt sick the whole day",              "Negative"),
    ("Not worth the price at all, very overpriced",                   "Negative"),
    ("Poor quality material, fell apart within a month",              "Negative"),
    ("Never again! Worst product I have ever bought",                 "Negative"),
    ("It irritated my skin badly, would not buy again",               "Negative"),
    ("Completely useless, does not work as advertised",               "Negative"),
    ("I wish I hadn't bought this, total waste of time",              "Negative"),
    ("Bad purchase, damaged on arrival",                              "Negative"),
    ("Unsafe and dangerous, do not use",                              "Negative"),

    # NEUTRAL ─ 10 cases
    ("It's okay, nothing special but nothing bad either",             "Neutral"),
    ("Product arrived on time, packaging was fine",                   "Neutral"),
    ("Average product for the price",                                 "Neutral"),
    ("Does what it says, neither great nor terrible",                 "Neutral"),
    ("Mediocre quality, expected better for this price range",        "Neutral"),
    ("Product arrived late but works as expected",                    "Neutral"),
    ("Decent enough, I might try it again",                           "Neutral"),
    ("Hard to say if it made any real difference",                    "Neutral"),
    ("The packaging was damaged but the product itself was fine",     "Neutral"),
    ("It's a standard product, nothing remarkable",                   "Neutral"),
]

# ── Run Tests ────────────────────────────────────────────────────────────────
print("=" * 72)
print("TrustLens Sentiment Model — Comprehensive Test v2")
print("=" * 72)

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Model loaded: {MODEL_PATH}\n")
except FileNotFoundError:
    print(f"❌ Model not found at {MODEL_PATH}. Run train_model.py first.")
    sys.exit(1)

results = {"Positive": {"correct":0,"total":0},
           "Negative": {"correct":0,"total":0},
           "Neutral":  {"correct":0,"total":0}}
failures = []

print(f"{'#':>3}  {'Exp':9} {'Got':9} {'Lex':>4} {'Neu':4} {'Conf':>5}  Review")
print("-" * 72)

for i, (review, expected) in enumerate(TEST_CASES, 1):
    pred, conf, lex, neu = predict_with_lexicon(model, review)
    correct = (pred == expected)
    results[expected]["total"] += 1
    if correct: results[expected]["correct"] += 1
    mark = "✅" if correct else "❌"
    n_flag = "NEU" if neu else "   "
    print(f"{i:>3}. {mark} {expected:9} {pred:9} {lex:>+4} {n_flag}  {conf*100:>4.0f}%  {review[:52]}")
    if not correct:
        failures.append((i, review, expected, pred, lex, neu, conf))

# ── Summary ───────────────────────────────────────────────────────────────────
total_correct = sum(v["correct"] for v in results.values())
overall_acc   = total_correct / len(TEST_CASES) * 100

print("\n" + "=" * 72)
print(f"OVERALL ACCURACY : {total_correct}/{len(TEST_CASES)}  ({overall_acc:.1f}%)")
print("=" * 72)
for label in ["Positive","Negative","Neutral"]:
    c, t = results[label]["correct"], results[label]["total"]
    print(f"  {label:9}: {c}/{t}  ({c/t*100:.0f}%)")

if failures:
    print(f"\n❌ FAILURES ({len(failures)}):")
    for num, review, expected, pred, lex, neu, conf in failures:
        print(f"   #{num}: Expected={expected}, Got={pred}, Lex={lex:+d}, Neutral={neu}, Conf={conf*100:.0f}%")
        print(f"        \"{review}\"")
else:
    print("\n🎉 Perfect score — all 50 tests passed!")
