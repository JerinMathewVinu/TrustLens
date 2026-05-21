"""
train_fake_news.py — TrustLens Fake News Model Trainer
========================================================
Trains TWO complementary models on True.csv + Fake.csv (~44k articles):

  Model A (fake_news_full.pkl)  — trained on title + full article text
                                   best for long inputs (30+ words)
  Model B (fake_news_title.pkl) — trained on title text only
                                   best for short inputs / single sentences

Both .pkl files are loaded by ml_engine.py at runtime.

Usage:
    python train_fake_news.py

Requirements:
    pip install scikit-learn pandas numpy

Place True.csv and Fake.csv in the same folder before running.
Training takes ~3-5 minutes on a standard CPU.
"""

import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def clean_text(text: str) -> str:
    """Strip source markers, URLs, HTML then lowercase."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove dateline prefixes: "WASHINGTON (Reuters) -", "NEW YORK (AP) -"
    text = re.sub(r"^[A-Z][A-Z\s,\.\-]+\s*\([^)]+\)\s*[-–]\s*", "", text)
    text = re.sub(r"\(reuters\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\(ap\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


print("=" * 55)
print("TrustLens — Fake News Model Training")
print("=" * 55)

# ── Load & merge ──────────────────────────────────────────────
print("\nLoading datasets...")
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

print(f"  True articles : {len(true_df):,}")
print(f"  Fake articles : {len(fake_df):,}")

true_df["label"] = "REAL"
fake_df["label"] = "FAKE"

df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df["content_full"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)
df["content_title"] = df["title"].fillna("").apply(clean_text)
df = df[(df["content_full"].str.strip() != "") & (df["content_title"].str.strip() != "")]

print(f"\nClean dataset: {len(df):,}  |  REAL: {(df['label']=='REAL').sum():,}  FAKE: {(df['label']=='FAKE').sum():,}")

# ── Split ─────────────────────────────────────────────────────
(X_train_full, X_test_full,
 X_train_title, X_test_title,
 y_train, y_test) = train_test_split(
    df["content_full"], df["content_title"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)

print(f"Train: {len(X_train_full):,}  |  Test: {len(X_test_full):,}")


def build_model(X_train, X_test, y_train, y_test, max_features, name):
    print(f"\n[{name}] Training...")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english", max_features=max_features,
            ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95,
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=42), cv=3
        )),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print(f"[{name}] Accuracy: {accuracy_score(y_test, preds)*100:.2f}%")
    print(classification_report(y_test, preds))
    print("Confusion Matrix [REAL / FAKE]:")
    print(confusion_matrix(y_test, preds, labels=["REAL", "FAKE"]))
    return pipe


# ── Train & save ──────────────────────────────────────────────
model_full = build_model(X_train_full, X_test_full, y_train, y_test, 50000, "Model-A full-text")
with open("fake_news_full.pkl", "wb") as f:
    pickle.dump(model_full, f)
print("✅ fake_news_full.pkl saved")

model_title = build_model(X_train_title, X_test_title, y_train, y_test, 30000, "Model-B title-only")
with open("fake_news_title.pkl", "wb") as f:
    pickle.dump(model_title, f)
print("✅ fake_news_title.pkl saved")

print("\n" + "=" * 55)
print("Done! Copy both .pkl files into your ml/ folder.")
print("=" * 55)
