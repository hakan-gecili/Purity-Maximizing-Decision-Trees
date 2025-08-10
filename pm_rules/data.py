"""Utilities for loading datasets used in examples."""

from typing import Tuple
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from .config import RANDOM_STATE, TEST_SIZE, VAL_SIZE


def load_breast_cancer_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """Load the breast cancer dataset and return train/val/test splits."""
    dt = load_breast_cancer()
    X = pd.DataFrame(dt.data, columns=[c.replace(" ", "_") for c in dt.feature_names])
    y = pd.Series(dt.target, name="y")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train_full
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
