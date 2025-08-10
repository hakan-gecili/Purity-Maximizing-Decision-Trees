"""Forward Sequential Rule Selection (FSRS)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from .utils import PathConstraint, rule_to_mask, update_constraint


@dataclass
class AtomicCond:
    feature: str
    op: str
    thr: float


def conditions_to_rule_dict(conds: List[AtomicCond]) -> Dict[str, PathConstraint]:
    rd: Dict[str, PathConstraint] = {}
    for c in conds:
        if c.feature not in rd:
            rd[c.feature] = PathConstraint()
        rd[c.feature] = update_constraint(rd[c.feature], (c.op, float(c.thr)))
    return rd


def expand_rule_dict_canonically(rule_dict: Dict[str, PathConstraint]) -> List[AtomicCond]:
    conds: List[AtomicCond] = []
    for f in sorted(rule_dict.keys()):
        pc = rule_dict[f]
        if np.isfinite(pc.low):
            conds.append(AtomicCond(f, "gt", float(pc.low)))
        if np.isfinite(pc.high):
            conds.append(AtomicCond(f, "le", float(pc.high)))
    return conds


def rule_row_to_ordered_conditions(rule_row: pd.Series) -> List[AtomicCond]:
    if "cond_list" in rule_row and isinstance(rule_row["cond_list"], list):
        return [AtomicCond(c["feature"], c["op"], float(c["thr"])) for c in rule_row["cond_list"]]
    return expand_rule_dict_canonically(rule_row.get("rule_dict", {}))


def apply_prefix_mask(conds: List[AtomicCond], X: pd.DataFrame, prefix_len: int) -> np.ndarray:
    if prefix_len == 0:
        return np.ones(len(X), dtype=bool)
    rd = conditions_to_rule_dict(conds[:prefix_len])
    return rule_to_mask(rd, X)


def estimate_class_prob_for_mask(mask: np.ndarray, y: pd.Series, class_label: int) -> float:
    if mask.sum() == 0:
        return 0.0
    return float((y[mask] == class_label).mean())


def majority_label_on_mask(mask: np.ndarray, y: pd.Series, pos_label: int = 1) -> int:
    if mask.sum() == 0:
        return pos_label
    mean_pos = float((y[mask] == pos_label).mean())
    return pos_label if mean_pos >= 0.5 else 1 - pos_label


def mcnemar_between_prefixes(
    conds: List[AtomicCond],
    X: pd.DataFrame,
    y: pd.Series,
    pred_label: int,
    i: int,
) -> Tuple[float, int, int]:
    mask_i = apply_prefix_mask(conds, X, i)
    mask_ip1 = apply_prefix_mask(conds, X, i + 1)
    yhat_i = np.where(mask_i, pred_label, 1 - pred_label)
    yhat_ip1 = np.where(mask_ip1, pred_label, 1 - pred_label)
    b = int(np.sum((yhat_i == y.values) & (yhat_ip1 != y.values)))
    c = int(np.sum((yhat_i != y.values) & (yhat_ip1 == y.values)))
    table = [[0, b], [c, 0]]
    res = mcnemar(table, exact=(b + c) < 25, correction=(b + c) >= 25)
    return float(res.pvalue), b, c


def forward_sequential_rule_selection(
    conds: List[AtomicCond],
    X: pd.DataFrame,
    y: pd.Series,
    pos_label: int = 1,
    alpha: float = 0.05,
    delta: float = 0.10,
    check_flip: bool = True,
    min_covered: int = 5,
    pred_label_strategy: str = "majority_full",
) -> Dict[str, Any]:
    k = len(conds)
    if k == 0:
        mask_full = np.ones(len(X), dtype=bool)
        prob_full = estimate_class_prob_for_mask(mask_full, y, pos_label)
        return {
            "short_conds": [],
            "selected_prefix_len": 0,
            "pvals": [],
            "prob_full": prob_full,
            "prob_short": prob_full,
            "pred_label": pos_label,
            "coverage_full": int(mask_full.sum()),
            "coverage_short": int(mask_full.sum()),
        }

    mask_full = apply_prefix_mask(conds, X, k)
    coverage_full = int(mask_full.sum())
    pred_label = majority_label_on_mask(mask_full, y, pos_label) if pred_label_strategy == "majority_full" else pos_label
    if coverage_full < min_covered:
        prob_full = estimate_class_prob_for_mask(mask_full, y, pred_label)
        return {
            "short_conds": conds,
            "selected_prefix_len": k,
            "pvals": [],
            "prob_full": prob_full,
            "prob_short": prob_full,
            "pred_label": pred_label,
            "coverage_full": coverage_full,
            "coverage_short": coverage_full,
        }

    prob_full = estimate_class_prob_for_mask(mask_full, y, pred_label)
    optimal_idx = 0
    pval_log: List[Tuple[float, float, int, int]] = []

    for i in range(k):
        alpha_i = alpha / (k - i)
        pval, b, c = mcnemar_between_prefixes(conds, X, y, pred_label, i)
        pval_log.append((pval, alpha_i, b, c))

        mask_short = apply_prefix_mask(conds, X, i + 1)
        coverage_short = int(mask_short.sum())
        if coverage_short < min_covered:
            break
        prob_short = estimate_class_prob_for_mask(mask_short, y, pred_label)
        stat_sig = pval < alpha_i
        prob_close = abs(prob_short - prob_full) < delta
        flip = (prob_short >= 0.5) != (prob_full >= 0.5)
        if stat_sig and prob_close and (not check_flip or not flip):
            optimal_idx = i + 1
        else:
            break

    short_conds = conds[:optimal_idx]
    mask_short_final = apply_prefix_mask(conds, X, optimal_idx)
    prob_short_final = estimate_class_prob_for_mask(mask_short_final, y, pred_label)

    return {
        "short_conds": short_conds,
        "selected_prefix_len": optimal_idx,
        "pvals": pval_log,
        "prob_full": prob_full,
        "prob_short": prob_short_final,
        "pred_label": pred_label,
        "coverage_full": coverage_full,
        "coverage_short": int(mask_short_final.sum()),
    }


def fsrs_shorten_rule_row(
    rule_row: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    pos_label: int = 1,
    alpha: float = 0.05,
    delta: float = 0.10,
    check_flip: bool = True,
    min_covered: int = 5,
    pred_label_strategy: str = "majority_full",
) -> Tuple[Dict[str, PathConstraint], dict]:
    conds = rule_row_to_ordered_conditions(rule_row)
    diag = forward_sequential_rule_selection(
        conds=conds,
        X=X_val,
        y=y_val,
        pos_label=pos_label,
        alpha=alpha,
        delta=delta,
        check_flip=check_flip,
        min_covered=min_covered,
        pred_label_strategy=pred_label_strategy,
    )
    new_rule_dict = conditions_to_rule_dict(diag["short_conds"])
    return new_rule_dict, diag
