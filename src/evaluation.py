from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def evaluate_classifier(model, X_test, y_test):
    """
    Returns dict with accuracy, confusion matrix, and classification report text.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "y_pred": y_pred,
    }


def try_roc_auc(model, X_test, y_test):
    """
    Compute ROC-AUC if the model supports predict_proba.
    Returns None if not supported.
    """
    if not hasattr(model, "predict_proba"):
        return None
    y_proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_proba)
