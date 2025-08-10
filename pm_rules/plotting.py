"""Visualization utilities for rule performance."""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _short_rule(s: str, maxlen: int = 60) -> str:
    return s if len(s) <= maxlen else s[: maxlen - 3] + "..."


def _normalize_sizes(values, min_size=30, max_size=220):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        return np.full_like(v, (min_size + max_size) / 2.0)
    return min_size + (v - vmin) * (max_size - min_size) / (vmax - vmin)


def generate_rules_performance_report(
    merged: pd.DataFrame,
    out_pdf: str = "generated_rules_performances.pdf",
    min_precision: Optional[float] = None,
    min_recall: Optional[float] = None,
) -> None:
    required_cols = {"source", "rule_str", "support", "precision", "recall", "f1"}
    if not required_cols.issubset(merged.columns):
        missing = required_cols - set(merged.columns)
        raise ValueError(f"missing columns: {sorted(missing)}")
    merged = merged.copy()
    for col in ["support", "precision", "recall", "f1"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    with PdfPages(out_pdf) as pdf:
        fig1 = plt.figure(figsize=(8, 6))
        for src in merged["source"].astype(str).unique():
            sub = merged[merged["source"].astype(str) == src]
            plt.scatter(
                sub["recall"],
                sub["precision"],
                s=_normalize_sizes(sub["support"]),
                alpha=0.8,
                label=str(src),
            )
        if min_recall is not None:
            plt.axvline(min_recall, linestyle="--", alpha=0.6)
        if min_precision is not None:
            plt.axhline(min_precision, linestyle="--", alpha=0.6)
        plt.xlabel("Recall (global coverage)")
        plt.ylabel("Precision (PPV)")
        plt.title("Precision vs Recall by Rule")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(title="Source")
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        fig2 = plt.figure(figsize=(8, 6))
        sc = plt.scatter(
            merged["support"],
            merged["f1"],
            c=merged["precision"],
            s=_normalize_sizes(merged["support"]),
            alpha=0.85,
        )
        plt.xlabel("Support (covered samples)")
        plt.ylabel("F1 Score")
        plt.title("Support vs F1 (color ~ Precision)")
        cbar = plt.colorbar(sc)
        cbar.set_label("Precision")
        plt.grid(True, linestyle="--", alpha=0.4)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        fig3 = plt.figure(figsize=(8, 6))
        top = merged.sort_values(["f1", "precision", "support"], ascending=[False, False, False]).head(10)
        y_pos = np.arange(len(top))
        labels = [_short_rule(s) for s in top["rule_str"].tolist()]
        plt.barh(y_pos, top["f1"])
        plt.yticks(y_pos, labels)
        plt.gca().invert_yaxis()
        plt.xlabel("F1 Score")
        plt.title("Top 10 Rules by F1")
        for i, (f1, ppv, rec) in enumerate(zip(top["f1"], top["precision"], top["recall"])):
            plt.text(f1 + 0.01, i, f"P={ppv:.2f}, R={rec:.2f}", va="center")
        xmax = float(top["f1"].max()) if len(top) else 1.0
        plt.xlim(0, max(1.0, xmax + 0.1))
        plt.grid(True, axis="x", linestyle="--", alpha=0.3)
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

    print(f"Saved plots to: {out_pdf}")
