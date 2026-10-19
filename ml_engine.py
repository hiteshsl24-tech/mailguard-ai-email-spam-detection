"""Small, explainable baseline email spam classifier for the MailGuard demo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

TEXT_ALIASES = ("text", "message", "email", "body", "content", "mail", "email_text", "message_text")
LABEL_ALIASES = ("label", "category", "class", "target", "type", "spam", "status", "classification")

SPAM_LABELS = {
    "spam", "junk", "phishing", "phishing email", "malicious", "scam", "fraud",
    "unwanted", "1", "true", "yes", "positive", "suspicious"
}
HAM_LABELS = {
    "ham", "not spam", "not_spam", "non spam", "non-spam", "legitimate", "legit",
    "safe", "normal", "0", "false", "no", "negative", "genuine", "valid"
}

SUSPICIOUS_PATTERNS = [
    ("Urgent or threatening language", re.compile(r"\b(urgent|immediately|act now|account suspended|final warning|within 24 hours|verify now)\b", re.I)),
    ("Credential or account request", re.compile(r"\b(password|login|log in|verify your account|confirm your identity|one[- ]time password|otp|credentials)\b", re.I)),
    ("Payment or prize lure", re.compile(r"\b(prize|winner|won|lottery|claim your|gift card|cash reward|wire transfer|bank details|refund)\b", re.I)),
    ("Link present", re.compile(r"(?:https?://|www\.)\S+", re.I)),
    ("Email address or domain impersonation wording", re.compile(r"\b(impersonat(?:e|ion)|spoof(?:ed|ing)?|lookalike domain)\b", re.I)),
    ("Attachment mentioned", re.compile(r"\b(attachment|attached file|enable macros|invoice\.zip|\.exe\b)\b", re.I)),
]


def infer_column(columns, aliases: tuple[str, ...]) -> str | None:
    """Infer a column by normalized header; return None if no match is clear."""
    normalized = {str(col).strip().lower().replace(" ", "_"): col for col in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    # A few common dataset conventions.
    for normalized_name, original in normalized.items():
        if any(alias in normalized_name for alias in aliases):
            return original
    return None


def normalize_label(value: Any) -> str | None:
    """Map common labels to spam/ham. Numeric 1 means spam; 0 means ham."""
    label = str(value).strip().lower()
    label = re.sub(r"[_\-]+", " ", label)
    label = re.sub(r"\s+", " ", label)
    if label in SPAM_LABELS:
        return "spam"
    if label in HAM_LABELS:
        return "ham"
    # Support common text such as "spam email" while avoiding arbitrary labels.
    if "spam" in label or "phish" in label or "scam" in label or "fraud" in label or "malicious" in label:
        return "spam"
    if "ham" in label or "legitimate" in label or "not spam" in label or "safe email" in label:
        return "ham"
    return None


def prepare_training_frame(
    frame: pd.DataFrame,
    text_column: str | None = None,
    label_column: str | None = None,
) -> tuple[pd.Series, pd.Series, str, str]:
    if frame.empty:
        raise ValueError("The uploaded CSV has no data rows.")

    text_column = text_column or infer_column(frame.columns, TEXT_ALIASES)
    label_column = label_column or infer_column(frame.columns, LABEL_ALIASES)
    if not text_column or text_column not in frame.columns:
        raise ValueError(
            "Could not identify the email-text column. Use a column named 'text', 'message', 'email', or 'body', "
            "or enter its exact name in the Text column field."
        )
    if not label_column or label_column not in frame.columns:
        raise ValueError(
            "Could not identify the label column. Use a column named 'label', 'category', 'class', or 'target', "
            "or enter its exact name in the Label column field."
        )
    if text_column == label_column:
        raise ValueError("Text column and label column must be different.")

    work = frame[[text_column, label_column]].copy()
    work.columns = ["text", "raw_label"]
    work["text"] = work["text"].fillna("").astype(str).str.strip()
    work["label"] = work["raw_label"].map(normalize_label)

    unknown = sorted(work.loc[work["label"].isna(), "raw_label"].astype(str).unique().tolist())
    if unknown:
        shown = ", ".join(unknown[:6])
        extra = " …" if len(unknown) > 6 else ""
        raise ValueError(
            f"Unsupported label value(s): {shown}{extra}. Labels must identify spam or legitimate/ham emails. "
            "For numeric labels, this app treats 1 as spam and 0 as ham."
        )

    work = work[work["text"].str.len() > 0]
    if len(work) < 20:
        raise ValueError(f"At least 20 non-empty labeled emails are required; found {len(work)}.")
    counts = work["label"].value_counts()
    if len(counts) < 2 or any(counts.get(label, 0) < 2 for label in ("ham", "spam")):
        raise ValueError("The dataset must contain at least two spam emails and two legitimate (ham) emails.")

    return work["text"], work["label"], str(text_column), str(label_column)


def make_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                max_features=30000,
                sublinear_tf=True,
            )),
            ("classifier", MultinomialNB(alpha=0.35)),
        ]
    )


def train_classifier(
    frame: pd.DataFrame,
    text_column: str | None = None,
    label_column: str | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    texts, labels, resolved_text_col, resolved_label_col = prepare_training_frame(frame, text_column, label_column)
    class_counts = labels.value_counts()
    can_stratify = all(class_counts.get(name, 0) >= 2 for name in ("ham", "spam"))
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels if can_stratify else None,
    )

    evaluation_model = make_pipeline()
    evaluation_model.fit(x_train, y_train)
    predictions = evaluation_model.predict(x_test)
    cm = confusion_matrix(y_test, predictions, labels=["ham", "spam"])

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)) * 100, 2),
        "precision": round(float(precision_score(y_test, predictions, pos_label="spam", zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_test, predictions, pos_label="spam", zero_division=0)) * 100, 2),
        "f1": round(float(f1_score(y_test, predictions, pos_label="spam", zero_division=0)) * 100, 2),
        "test_size": int(len(y_test)),
        "dataset_size": int(len(texts)),
        "spam_count": int((labels == "spam").sum()),
        "ham_count": int((labels == "ham").sum()),
        "confusion_matrix": {
            "true_ham_pred_ham": int(cm[0][0]),
            "true_ham_pred_spam": int(cm[0][1]),
            "true_spam_pred_ham": int(cm[1][0]),
            "true_spam_pred_spam": int(cm[1][1]),
        },
        "text_column": resolved_text_col,
        "label_column": resolved_label_col,
    }

    # Evaluate on held-out data above, then train the deployed model on all available rows.
    final_model = make_pipeline()
    final_model.fit(texts, labels)
    return final_model, metrics


def inspect_indicators(text: str) -> list[str]:
    return [description for description, pattern in SUSPICIOUS_PATTERNS if pattern.search(text or "")]


def predict_email(model: Pipeline, text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Enter an email subject or body before checking it.")
    label = str(model.predict([cleaned])[0])
    probabilities = model.predict_proba([cleaned])[0]
    classes = list(model.named_steps["classifier"].classes_)
    spam_probability = float(probabilities[classes.index("spam")]) if "spam" in classes else 0.0
    confidence = max(float(value) for value in probabilities)
    return {
        "label": label,
        "spam_probability": round(spam_probability * 100, 2),
        "confidence": round(confidence * 100, 2),
        "indicators": inspect_indicators(cleaned),
    }
