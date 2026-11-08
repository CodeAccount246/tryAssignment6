"""Shared helpers used by the Streamlit app (kept free of Streamlit so they can be unit-tested)."""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, classification_report,
                             roc_curve)
from features import FEATURE_COLUMNS, TARGET

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib",
}
CLASS_NAMES = ["Movie (0)", "TV Show (1)"]


def load_model(name):
    return joblib.load(MODEL_FILES[name])


def split_xy(df):
    """Validate an uploaded CSV and return (X, y). Raises ValueError with a readable message."""
    missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("Uploaded CSV is missing column(s): " + ", ".join(missing))
    y = df[TARGET].astype(int)
    if y.nunique() < 2:
        raise ValueError("The 'target' column must contain both classes (0 and 1) to compute AUC.")
    return df[FEATURE_COLUMNS], y


def evaluate(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y, pred), "AUC": roc_auc_score(y, proba),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0), "MCC": matthews_corrcoef(y, pred),
    }
    return metrics, pred, proba


def confusion(y, pred):
    return confusion_matrix(y, pred, labels=[0, 1])


def report_text(y, pred):
    return classification_report(y, pred, target_names=CLASS_NAMES, digits=4, zero_division=0)


def roc_points(y, proba):
    fpr, tpr, _ = roc_curve(y, proba)
    return fpr, tpr
