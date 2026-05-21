"""
train_model.py — TrustLens Sentiment Model Trainer v4
=======================================================
Major upgrades over v3:
  - Dual vectoriser: word TF-IDF (1-3 grams) + char TF-IDF (2-5 grams)
    → Captures nuanced phrases like "feel relieved", "not bad", "could be better"
  - Classifier: LinearSVC wrapped in CalibratedClassifierCV
    → Much faster and more accurate than GradientBoosting on text; gives real probs
  - Stratified over-sampling of minority class (Neutral) via resample
  - Trains on BOTH twitter_training.csv AND twitter_validation.csv for more data
  - Saves as sentiment_model.pkl (same filename — drop-in replacement)

Run:
    python train_model.py

Output:
    sentiment_model.pkl
"""

import re
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)        # strip URLs
    text = re.sub(r"@\w+", " ", text)                     # strip @mentions
    text = re.sub(r"#(\w+)", r"\1", text)                 # keep hashtag word
    text = re.sub(r"<[^>]+>", " ", text)                  # strip HTML
    # Preserve contractions and common punctuation important for sentiment
    text = re.sub(r"[^a-zA-Z0-9\s'.,!?%-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


# ── Load & merge datasets ────────────────────────────────────────────────────

print("=" * 60)
print("TrustLens Sentiment Model Trainer v4")
print("=" * 60)

frames = []

# Primary dataset
print("\n[1/3] Loading twitter_training.csv...")
df_train = pd.read_csv("twitter_training.csv", header=None,
                       names=["id", "entity", "sentiment", "text"],
                       on_bad_lines="skip")
frames.append(df_train)
print(f"      → {len(df_train):,} rows loaded")

# Validation dataset (use as extra training data)
import os
if os.path.exists("twitter_validation.csv"):
    print("[2/3] Loading twitter_validation.csv...")
    df_val = pd.read_csv("twitter_validation.csv", header=None,
                         names=["id", "entity", "sentiment", "text"],
                         on_bad_lines="skip")
    frames.append(df_val)
    print(f"      → {len(df_val):,} rows loaded")
else:
    print("[2/3] twitter_validation.csv not found — skipping.")

df = pd.concat(frames, ignore_index=True)

# ── Pre-process ───────────────────────────────────────────────────────────────

print("\n[3/3] Cleaning and filtering...")
valid_sentiments = ["Positive", "Negative", "Neutral"]
df = df[df["sentiment"].isin(valid_sentiments)].copy()
df = df.dropna(subset=["text", "sentiment"])
df["text"] = df["text"].apply(clean_text)
df = df[df["text"].str.strip().str.len() > 3]          # drop near-empty rows
df = df.drop_duplicates(subset=["text"])                # remove duplicate tweets

print(f"\nSamples after cleaning : {len(df):,}")
print(df["sentiment"].value_counts().to_string())

# ── Balance classes ──────────────────────────────────────────────────────────
# Over-sample minority classes to the majority class size

print("\nBalancing classes...")
counts = df["sentiment"].value_counts()
max_count = int(counts.max())

parts = []
for label in valid_sentiments:
    subset = df[df["sentiment"] == label]
    if len(subset) < max_count:
        subset = resample(subset, replace=True, n_samples=max_count, random_state=42)
    parts.append(subset)

df_balanced = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42)
print(df_balanced["sentiment"].value_counts().to_string())

X = df_balanced["text"].values
y = df_balanced["sentiment"].values

# ── Train / test split ───────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Model pipeline ────────────────────────────────────────────────────────────
#
# Two complementary vectorisers:
#   word_tfidf : 1–3 word n-grams (phrases like "not happy", "feel relieved")
#   char_tfidf : 2–5 char n-grams (morphological patterns, spelling variants)
#
# Classifier: LinearSVC (very fast, excellent on sparse text) wrapped in
# CalibratedClassifierCV so we get reliable probability estimates.

word_tfidf = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=60000,
    sublinear_tf=True,
    min_df=2,
    max_df=0.92,
    strip_accents="unicode",
)

char_tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    max_features=40000,
    sublinear_tf=True,
    min_df=3,
)

combined_features = FeatureUnion([
    ("word", word_tfidf),
    ("char", char_tfidf),
])

base_clf = LinearSVC(
    C=0.8,
    max_iter=2000,
    class_weight="balanced",
    dual=True,
)

# Platt scaling for calibrated probabilities
calibrated_clf = CalibratedClassifierCV(base_clf, cv=5, method="sigmoid")

model = Pipeline([
    ("features", combined_features),
    ("clf",      calibrated_clf),
])

# ── Train ────────────────────────────────────────────────────────────────────

print("\nTraining model (this may take a few minutes)...")
model.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
preds = model.predict(X_test)
acc   = accuracy_score(y_test, preds)
print(f"Test Accuracy : {acc * 100:.2f}%")
print("=" * 60)

print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=valid_sentiments))

print("Confusion Matrix (rows=actual, cols=predicted):")
cm = confusion_matrix(y_test, preds, labels=valid_sentiments)
print(f"{'':12}", "  ".join(f"{l:8}" for l in valid_sentiments))
for label, row in zip(valid_sentiments, cm):
    print(f"{label:12}", "  ".join(f"{v:8,}" for v in row))

# ── 5-fold cross-validation on original (unbalanced) data ─────────────────
print("\nRunning 5-fold cross-validation on original dataset...")
X_orig = df["text"].values
y_orig = df["sentiment"].values

cv_pipeline = Pipeline([
    ("features", combined_features),
    ("clf", CalibratedClassifierCV(
        LinearSVC(C=0.8, max_iter=2000, class_weight="balanced", dual=True),
        cv=5, method="sigmoid"
    )),
])

cv_scores = cross_val_score(
    cv_pipeline, X_orig, y_orig,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy", n_jobs=-1
)
print(f"CV Accuracy: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

# ── Quick sanity-check on nuanced phrases ───────────────────────────────────

sanity_pairs = [
    ("When I eat the product it makes me feel relieved", "Positive"),
    ("I feel so much better after using this, would definitely buy again", "Positive"),
    ("This product gave me a headache, not impressed at all", "Negative"),
    ("It's okay, nothing special", "Neutral"),
    ("Terrible waste of money, completely broken on arrival", "Negative"),
    ("Absolutely fantastic, best purchase I've made this year!", "Positive"),
    ("Product arrived late but works as expected", "Neutral"),
    ("I regret buying this, it stopped working after a week", "Negative"),
    ("Feeling much more energized since I started using it", "Positive"),
    ("Not bad but not great either", "Neutral"),
]

print("\n" + "=" * 60)
print("Sanity-check on nuanced phrases:")
print("=" * 60)
clean_tests = [clean_text(t) for t, _ in sanity_pairs]
sanity_preds = model.predict(clean_tests)
sanity_probs = model.predict_proba(clean_tests)
classes      = model.classes_

correct = 0
for (phrase, expected), pred, probs in zip(sanity_pairs, sanity_preds, sanity_probs):
    prob_map = dict(zip(classes, probs))
    mark = "✅" if pred == expected else "❌"
    correct += (pred == expected)
    print(f"{mark} [{expected:8} → {pred:8}]  "
          f"pos={prob_map.get('Positive', 0):.2f}  "
          f"neu={prob_map.get('Neutral',  0):.2f}  "
          f"neg={prob_map.get('Negative', 0):.2f}  "
          f'"{phrase[:60]}"')

print(f"\nSanity score: {correct}/{len(sanity_pairs)}")

# ── Save ─────────────────────────────────────────────────────────────────────

with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\n✅  sentiment_model.pkl saved successfully")
print("    Reload ml_engine.py to pick up the new model.")
