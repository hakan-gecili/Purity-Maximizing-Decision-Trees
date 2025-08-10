"""Rule extraction utilities for LightGBM models."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

from .config import (
    TOP_K_FEATURES_PER_INSTANCE,
    MIN_SUPPORT,
    MIN_PRECISION,
    MIN_RECALL,
    MIN_F1,
    POS_LABEL,
)
from .utils import (
    PathConstraint,
    update_constraint,
    rule_to_mask,
    rule_to_string,
    rule_recall_global,
    f1_from_pr,
)


BoosterDump = Dict[str, Any]


def load_trees_dump(model: lgb.LGBMClassifier) -> BoosterDump:
    """Return LightGBM booster dump."""
    booster = model.booster_
    return booster.dump_model()


def traverse_tree_for_row(
    tree: Dict[str, Any], row: pd.Series, feat_idx_to_name: Dict[int, str]
) -> Tuple[List[Tuple[str, Tuple[str, float]]], float]:
    """Return path followed by a row and leaf value."""
    path = []
    node = tree
    while "split_feature" in node:
        f_idx = node["split_feature"]
        f_name = feat_idx_to_name[f_idx]
        thr = node["threshold"]
        if row[f_name] <= float(thr):
            path.append((f_name, ("le", float(thr))))
            node = node["left_child"]
        else:
            path.append((f_name, ("gt", float(thr))))
            node = node["right_child"]
    return path, float(node.get("leaf_value", 0.0))


def build_rule_from_paths(
    paths: List[Tuple[str, Tuple[str, float]]], keep_features: set
) -> Dict[str, PathConstraint]:
    """Merge path conditions into feature intervals."""
    rule: Dict[str, PathConstraint] = {}
    for f_name, (op, thr) in paths:
        if f_name not in keep_features:
            continue
        if f_name not in rule:
            rule[f_name] = PathConstraint()
        rule[f_name] = update_constraint(rule[f_name], (op, thr))
    return rule


def extract_local_rules(
    clf: lgb.LGBMClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> pd.DataFrame:
    """Generate rules by tracing SHAP-important paths for positives."""
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_val)
    if isinstance(shap_vals, list):  # binary format
        shap_vals = shap_vals[1]
    shap_abs = np.abs(shap_vals)

    y_val_pred = clf.predict(X_val)
    dump = load_trees_dump(clf)
    tree_infos = dump["tree_info"]
    feat_idx_to_name = {i: n for i, n in enumerate(X_val.columns)}

    Xv = X_val.reset_index(drop=True)
    yv = y_val.reset_index(drop=True)
    yv_pred = pd.Series(y_val_pred).reset_index(drop=True)

    candidates: List[Dict[str, Any]] = []

    for i in range(len(Xv)):
        if not (yv[i] == POS_LABEL and yv_pred[i] == POS_LABEL):
            continue
        row = Xv.iloc[i]
        shap_order = np.argsort(-shap_abs[i])[:TOP_K_FEATURES_PER_INSTANCE]
        keep_feats = set(Xv.columns[j] for j in shap_order)

        all_conditions: List[Tuple[str, Tuple[str, float]]] = []
        for t in tree_infos:
            tree = t["tree_structure"]
            path, _ = traverse_tree_for_row(tree, row, feat_idx_to_name)
            all_conditions.extend(path)

        rule_dict = build_rule_from_paths(all_conditions, keep_feats)
        if not rule_dict:
            continue

        mask = rule_to_mask(rule_dict, Xv)
        support = int(mask.sum())
        if support < MIN_SUPPORT:
            continue
        ppv = float((yv[mask] == POS_LABEL).mean())
        rec = rule_recall_global(mask, yv, POS_LABEL)
        f1 = f1_from_pr(ppv, rec)
        if (ppv < MIN_PRECISION) or (rec < MIN_RECALL) or (f1 < MIN_F1):
            continue

        candidates.append(
            {
                "source": "TreeSHAPLocal",
                "instance_idx": i,
                "rule_dict": rule_dict,
                "rule_str": rule_to_string(rule_dict),
                "features": list(rule_dict.keys()),
                "support": support,
                "precision": ppv,
                "recall": rec,
                "f1": f1,
            }
        )

    df = pd.DataFrame(candidates)
    if not df.empty:
        df = df.sort_values(["f1", "precision", "support"], ascending=[False, False, False])
        df = df.drop_duplicates(subset=["rule_str"]).reset_index(drop=True)
    return df


def dfs_collect_paths(
    node: Dict[str, Any],
    feat_idx_to_name: Dict[int, str],
    acc_path: List[Tuple[str, Tuple[str, float]]],
    out_paths: List[Tuple[List[Tuple[str, Tuple[str, float]]], float]],
) -> None:
    """Depth-first traversal collecting root-to-leaf paths."""
    if "split_feature" not in node:
        leaf_value = float(node.get("leaf_value", 0.0))
        out_paths.append((acc_path.copy(), leaf_value))
        return
    f_idx = node["split_feature"]
    f_name = feat_idx_to_name[f_idx]
    thr = float(node["threshold"])

    acc_path.append((f_name, ("le", thr)))
    dfs_collect_paths(node["left_child"], feat_idx_to_name, acc_path, out_paths)
    acc_path.pop()

    acc_path.append((f_name, ("gt", thr)))
    dfs_collect_paths(node["right_child"], feat_idx_to_name, acc_path, out_paths)
    acc_path.pop()


def build_interval_rule_from_path(path: List[Tuple[str, Tuple[str, float]]]) -> Dict[str, PathConstraint]:
    rule: Dict[str, PathConstraint] = {}
    for f_name, (op, thr) in path:
        if f_name not in rule:
            rule[f_name] = PathConstraint()
        rule[f_name] = update_constraint(rule[f_name], (op, thr))
    return rule


def extract_global_rules(
    clf: lgb.LGBMClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> pd.DataFrame:
    """Mine rules by enumerating all positive leaf paths."""
    dump = load_trees_dump(clf)
    tree_infos = dump["tree_info"]
    feat_idx_to_name = {i: n for i, n in enumerate(X_val.columns)}

    Xv = X_val.reset_index(drop=True)
    yv = y_val.reset_index(drop=True)

    candidates: List[Dict[str, Any]] = []
    for t in tree_infos:
        tree = t["tree_structure"]
        paths: List[Tuple[List[Tuple[str, Tuple[str, float]]], float]] = []
        dfs_collect_paths(tree, feat_idx_to_name, [], paths)
        for path_conditions, leaf_val in paths:
            if leaf_val <= 0.0:
                continue
            rule_dict = build_interval_rule_from_path(path_conditions)
            mask = rule_to_mask(rule_dict, Xv)
            support = int(mask.sum())
            if support < MIN_SUPPORT:
                continue
            ppv = float((yv[mask] == POS_LABEL).mean())
            rec = rule_recall_global(mask, yv, POS_LABEL)
            f1 = f1_from_pr(ppv, rec)
            if (ppv < MIN_PRECISION) or (rec < MIN_RECALL) or (f1 < MIN_F1):
                continue
            candidates.append(
                {
                    "source": "GlobalPath",
                    "rule_dict": rule_dict,
                    "rule_str": rule_to_string(rule_dict),
                    "features": list(rule_dict.keys()),
                    "support": support,
                    "precision": ppv,
                    "recall": rec,
                    "f1": f1,
                    "leaf_value": leaf_val,
                }
            )
    df = pd.DataFrame(candidates)
    if not df.empty:
        df = df.sort_values(["f1", "precision", "support"], ascending=[False, False, False])
        df = df.drop_duplicates(subset=["rule_str"]).reset_index(drop=True)
    return df


def merge_rule_sets(local_df: pd.DataFrame, global_df: pd.DataFrame) -> pd.DataFrame:
    """Combine local and global candidate rules."""
    frames = [df for df in [local_df, global_df] if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["f1", "precision", "support"], ascending=[False, False, False])
    merged = merged.drop_duplicates(subset=["rule_str"]).reset_index(drop=True)
    return merged
