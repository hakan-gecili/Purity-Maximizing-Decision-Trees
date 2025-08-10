"""Rule-based ensemble classifiers."""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegressionCV

from .utils import rule_to_mask


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _build_rule_matrix(rules_df: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    masks = []
    for _, row in rules_df.iterrows():
        rd = row.get("rule_dict") or {}
        mask = rule_to_mask(rd, X) if rd else np.ones(len(X), dtype=bool)
        masks.append(mask.astype(np.float32))
    return np.vstack(masks)


def _pick_threshold_for_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float, float]:
    uniq = np.unique(np.round(y_prob, 6))
    grid = np.linspace(0.05, 0.95, 19)
    cand = np.unique(np.concatenate([uniq, grid]))
    best = (0.5, 0.0, 0.0, 0.0)
    for t in cand:
        y_hat = (y_prob >= t).astype(int)
        P = precision_score(y_true, y_hat, zero_division=0)
        R = recall_score(y_true, y_hat, zero_division=0)
        F = f1_score(y_true, y_hat, zero_division=0)
        if F > best[3]:
            best = (float(t), float(P), float(R), float(F))
    return best


class VotingRulesetClassifier:
    """Combine rules via weighted voting with a tuned threshold."""

    def __init__(self, rules_df: pd.DataFrame, weighting: str = "precision_logit"):
        self.rules_df = rules_df.copy()
        self.weighting = weighting
        self.betas_: Optional[np.ndarray] = None
        self.threshold_: Optional[float] = None

    def _compute_betas(self) -> np.ndarray:
        if self.weighting == "precision_logit":
            p = np.clip(self.rules_df["precision"].to_numpy(float), 1e-3, 1 - 1e-3)
            return np.log(p / (1 - p))
        if self.weighting == "leaf_value":
            return self.rules_df["leaf_value"].to_numpy(float)
        if self.weighting == "uniform":
            return np.ones(len(self.rules_df))
        raise ValueError(f"Unknown weighting: {self.weighting}")

    def fit(self, X_val: pd.DataFrame, y_val: pd.Series) -> "VotingRulesetClassifier":
        M_val = _build_rule_matrix(self.rules_df, X_val)
        betas = self._compute_betas()
        z_val = (betas.reshape(-1, 1) * M_val).sum(axis=0)
        p_val = _sigmoid(z_val)
        thr, _, _, _ = _pick_threshold_for_f1(y_val.to_numpy(), p_val)
        self.betas_ = betas
        self.threshold_ = thr
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.betas_ is None:
            raise RuntimeError("Call fit() first")
        M = _build_rule_matrix(self.rules_df, X)
        z = (self.betas_.reshape(-1, 1) * M).sum(axis=0)
        return _sigmoid(z)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Call fit() first")
        p = self.predict_proba(X)
        return (p >= self.threshold_).astype(int)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        p = self.predict_proba(X)
        y_hat = (p >= self.threshold_).astype(int)
        out = {
            "precision": precision_score(y, y_hat, zero_division=0),
            "recall": recall_score(y, y_hat, zero_division=0),
            "f1": f1_score(y, y_hat, zero_division=0),
        }
        try:
            out["auc_roc"] = roc_auc_score(y, p)
            out["ap"] = average_precision_score(y, p)
        except Exception:
            pass
        return out


class LearnedRulesetClassifier:
    """Sparse logistic regression over rule indicators."""

    def __init__(self, rules_df: pd.DataFrame, Cs=(0.01, 0.1, 1, 10), l1_ratios=(0.5, 1.0)):
        self.rules_df = rules_df.copy().reset_index(drop=True)
        self.Cs = Cs
        self.l1_ratios = l1_ratios
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.threshold_: Optional[float] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> "LearnedRulesetClassifier":
        M_train = _build_rule_matrix(self.rules_df, X_train).T
        lr = LogisticRegressionCV(
            Cs=self.Cs,
            cv=5,
            penalty="elasticnet",
            solver="saga",
            l1_ratios=self.l1_ratios,
            scoring="f1",
            max_iter=5000,
            n_jobs=-1,
            refit=True,
        )
        lr.fit(M_train, y_train.to_numpy())
        self.coef_ = lr.coef_.ravel()
        self.intercept_ = float(lr.intercept_.ravel()[0])
        p_val = self.predict_proba(X_val)
        thr, _, _, _ = _pick_threshold_for_f1(y_val.to_numpy(), p_val)
        self.threshold_ = thr
        return self

    def _logit(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Call fit() first")
        M = _build_rule_matrix(self.rules_df, X).T
        return M @ self.coef_ + self.intercept_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return _sigmoid(self._logit(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Call fit() first")
        p = self.predict_proba(X)
        return (p >= self.threshold_).astype(int)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        p = self.predict_proba(X)
        y_hat = (p >= self.threshold_).astype(int)
        out = {
            "precision": precision_score(y, y_hat, zero_division=0),
            "recall": recall_score(y, y_hat, zero_division=0),
            "f1": f1_score(y, y_hat, zero_division=0),
        }
        try:
            out["auc_roc"] = roc_auc_score(y, p)
            out["ap"] = average_precision_score(y, p)
        except Exception:
            pass
        return out
