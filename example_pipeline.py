"""Example pipeline demonstrating rule extraction and combination."""

import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score

from pm_rules.config import RANDOM_STATE, PLOT_RULE_PERFORMANCE, POS_LABEL, MIN_SUPPORT, MIN_PRECISION, MIN_RECALL, MIN_F1
from pm_rules.data import load_breast_cancer_split
from pm_rules.rule_generation import extract_local_rules, extract_global_rules, merge_rule_sets
from pm_rules.fsrs import fsrs_shorten_rule_row
from pm_rules.classifiers import VotingRulesetClassifier
from pm_rules.plotting import generate_rules_performance_report
from pm_rules.utils import rule_to_string, rule_to_mask, rule_recall_global, f1_from_pr

# Load data
X_train, X_val, X_test, y_train, y_val, y_test = load_breast_cancer_split()

# Train LightGBM
clf = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=-1,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
)
clf.fit(X_train, y_train)

# Quick sanity metrics
y_pred = clf.predict(X_test)
print({
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
})

# Stage A: rule generation
local_df = extract_local_rules(clf, X_val, y_val)
global_df = extract_global_rules(clf, X_val, y_val)
merged = merge_rule_sets(local_df, global_df)

# Stage B: FSRS shortening
fsrs_df = merged.copy().reset_index(drop=True)
new_rule_dicts = []
for _, row in fsrs_df.iterrows():
    new_rd, diag = fsrs_shorten_rule_row(
        rule_row=row,
        X_val=X_val,
        y_val=y_val,
        pos_label=POS_LABEL,
        min_covered=MIN_SUPPORT,
    )
    new_rule_dicts.append(new_rd)
    row["fsrs_diag"] = diag
fsrs_df["rule_dict"] = new_rule_dicts
fsrs_df["rule_str"] = fsrs_df["rule_dict"].apply(rule_to_string)

# Re-score rules after FSRS
supports, ppvs, recs, f1s = [], [], [], []
for rd in fsrs_df["rule_dict"]:
    mask = rule_to_mask(rd, X_val)
    support = int(mask.sum())
    ppv = float((y_val[mask] == POS_LABEL).mean()) if support > 0 else 0.0
    rec = rule_recall_global(mask, y_val, POS_LABEL)
    f1 = f1_from_pr(ppv, rec)
    supports.append(support); ppvs.append(ppv); recs.append(rec); f1s.append(f1)
fsrs_df["support"] = supports
fsrs_df["precision"] = ppvs
fsrs_df["recall"] = recs
fsrs_df["f1"] = f1s
fsrs_df = fsrs_df[(fsrs_df["support"] >= MIN_SUPPORT) & (fsrs_df["precision"] >= MIN_PRECISION) & (fsrs_df["recall"] >= MIN_RECALL) & (fsrs_df["f1"] >= MIN_F1)].reset_index(drop=True)

# Stage C: simple voting combiner
fsrs_df = fsrs_df.sort_values(["f1", "precision", "support"], ascending=[False, False, False]).reset_index(drop=True)
voter = VotingRulesetClassifier(fsrs_df)
voter.fit(X_val, y_val)
print("Validation metrics:", voter.evaluate(X_val, y_val))
print("Test metrics:", voter.evaluate(X_test, y_test))

if PLOT_RULE_PERFORMANCE:
    generate_rules_performance_report(merged, "generated_rules_performances.pdf", min_precision=0.9, min_recall=0.15)
