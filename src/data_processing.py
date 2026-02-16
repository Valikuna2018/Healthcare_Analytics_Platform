from __future__ import annotations

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_features_target(df: pd.DataFrame, target_col: str = "target") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataframe into features (X) and target (y).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Stratified train/test split to keep class balance.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def scale_standard(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Fit StandardScaler on train and transform train/test.
    Returns scaled arrays and the fitted scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
