from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def train_logistic_regression(X_train, y_train, max_iter: int = 1000, random_state: int = 42):
    """
    Train Logistic Regression classifier.
    """
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, random_state: int = 42):
    """
    Train Decision Tree classifier.
    """
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model
