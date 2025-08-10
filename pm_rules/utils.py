"""Common utilities for rule construction and evaluation."""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd


@dataclass
class PathConstraint:
    """Interval constraints for a single feature."""
    low: float = -np.inf
    high: float = np.inf


def update_constraint(cons: PathConstraint, cond: Tuple[str, float]) -> PathConstraint:
    """Intersect existing constraint with a new condition."""
    op, thr = cond
    if op == "le":
        cons.high = min(cons.high, thr)
    elif op == "gt":
        cons.low = max(cons.low, thr)
    return cons


def rule_to_mask(rule: Dict[str, PathConstraint], X: pd.DataFrame) -> np.ndarray:
    """Return mask of rows satisfying the rule."""
    mask = np.ones(len(X), dtype=bool)
    for f_name, cons in rule.items():
        mask &= (X[f_name] > cons.low) & (X[f_name] <= cons.high)
    return mask


def rule_to_string(rule: Dict[str, PathConstraint]) -> str:
    parts = []
    for f_name, cons in rule.items():
        if cons.low != -np.inf:
            parts.append(f"{f_name} > {cons.low:.6g}")
        if cons.high != np.inf:
            parts.append(f"{f_name} <= {cons.high:.6g}")
    return " & ".join(parts) if parts else "(True)"


def rule_recall_global(mask: np.ndarray, y_true: pd.Series, pos_label: int = 1) -> float:
    pos_total = (y_true == pos_label).sum()
    if pos_total == 0:
        return 0.0
    tp = ((mask) & (y_true == pos_label)).sum()
    return tp / pos_total


def f1_from_pr(precision: float, recall: float) -> float:
    return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
