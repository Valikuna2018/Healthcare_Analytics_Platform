from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_confusion_matrix(cm, title: str = "Confusion Matrix"):
    """
    Plot a confusion matrix as a heatmap.
    """
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def save_current_figure(path: str | Path, dpi: int = 200):
    """
    Save the current matplotlib figure to a path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_feature_importance(feature_importances, feature_names, top_n: int = 15, title: str = "Feature Importance"):
    """
    Plot top_n feature importances (e.g., from Decision Tree).
    """
    s = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    s.sort_values().plot(kind="barh")
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    return s
