import pandas as pd

df = pd.read_csv("full_scan_results.csv")
v_cols = [f"V{i}" for i in range(1, 29)]
base_cols = ["Time", "Amount", "Risk_Tier", "Class"] + v_cols

TIERS = ["Low", "Medium", "High", "Critical"]
rows = []
seen_idx = set()

def add(subset, scenario, label, n=2, sort_col=None, ascending=True):
    global rows, seen_idx
    s = subset[~subset.index.isin(seen_idx)]
    if sort_col:
        s = s.sort_values(sort_col, ascending=ascending)
    picked = s.head(n)
    for idx, r in picked.iterrows():
        seen_idx.add(idx)
        d = {c: r[c] for c in base_cols}
        d["scenario"] = scenario
        d["scenario_label"] = label
        d["m2_pred"] = r["m2_pred"]
        d["m2_defer_prob"] = r["m2_defer_prob"]
        d["m2_deferred"] = r["m2_deferred"]
        d["m3_pred"] = r["m3_pred"]
        d["m3_defer_prob"] = r["m3_defer_prob"]
        d["m3_deferred"] = r["m3_deferred"]
        rows.append(d)
    return len(picked)

# 1. Both auto-decide, correct -- baseline agreement, per tier x class
for tier in TIERS:
    fraud_ok = df[(df["Risk_Tier"] == tier) & (df["Class"] == 1) & (~df["m2_deferred"]) & (df["m2_pred"] == "Fraud")
                  & (~df["m3_deferred"]) & (df["m3_pred"] == "Fraud")]
    add(fraud_ok, "baseline_agree", "Both agree, no dispute (real fraud, both auto-flag correctly)", n=3, sort_col="Amount", ascending=False)

    normal_ok = df[(df["Risk_Tier"] == tier) & (df["Class"] == 0) & (~df["m2_deferred"]) & (df["m2_pred"] == "Normal")
                   & (~df["m3_deferred"]) & (df["m3_pred"] == "Normal")]
    add(normal_ok, "baseline_agree", "Both agree, no dispute (legitimate transaction, both auto-approve correctly)", n=3, sort_col="Amount", ascending=False)

# 2. Standard L2D defers, Risk-Sensitive doesn't (explicitly requested)
for tier in TIERS:
    only_m2 = df[(df["Risk_Tier"] == tier) & (df["m2_deferred"]) & (~df["m3_deferred"])]
    add(only_m2, "only_m2_defers", "Standard L2D defers, Risk-Sensitive L2D auto-decides", n=5, sort_col="Amount", ascending=False)

# 3. Risk-Sensitive L2D defers, Standard doesn't (thesis's headline mechanism) -- prioritise real frauds
only_m3_fraud = df[(df["Class"] == 1) & (~df["m2_deferred"]) & (df["m3_deferred"])]
add(only_m3_fraud, "only_m3_defers", "Risk-Sensitive L2D defers, Standard L2D auto-decides (real fraud)", n=10, sort_col="Amount", ascending=False)
for tier in TIERS:
    only_m3_normal = df[(df["Risk_Tier"] == tier) & (df["Class"] == 0) & (~df["m2_deferred"]) & (df["m3_deferred"])]
    add(only_m3_normal, "only_m3_defers", "Risk-Sensitive L2D defers, Standard L2D auto-decides (legitimate transaction)", n=2, sort_col="Amount", ascending=False)

# 4. Both defer -- split fraud vs normal, across tiers
both_defer_fraud = df[(df["Class"] == 1) & (df["m2_deferred"]) & (df["m3_deferred"])]
add(both_defer_fraud, "both_defer", "Both models defer to a human (real fraud)", n=6, sort_col="Amount", ascending=False)
for tier in TIERS:
    both_defer_normal = df[(df["Risk_Tier"] == tier) & (df["Class"] == 0) & (df["m2_deferred"]) & (df["m3_deferred"])]
    add(both_defer_normal, "both_defer", "Both models defer to a human (legitimate transaction)", n=2, sort_col="Amount", ascending=False)

# 5. Classification disagreement -- models see the transaction differently
disagree_fraud = df[(df["Class"] == 1) & (df["m2_pred"] != df["m3_pred"])]
add(disagree_fraud, "classification_disagreement", "Models disagree on Fraud vs Normal itself (real fraud)", n=10, sort_col="Amount", ascending=False)
for tier in TIERS:
    disagree_normal = df[(df["Risk_Tier"] == tier) & (df["Class"] == 0) & (df["m2_pred"] != df["m3_pred"])]
    add(disagree_normal, "classification_disagreement", "Models disagree on Fraud vs Normal itself (legitimate transaction)", n=3, sort_col="Amount", ascending=False)

# 6. Both models miss a real fraud (most concerning failure mode) -- include all real examples
both_miss = df[(df["Class"] == 1) & (~df["m2_deferred"]) & (df["m2_pred"] == "Normal") & (~df["m3_deferred"]) & (df["m3_pred"] == "Normal")]
add(both_miss, "both_miss_fraud", "Both models silently miss a real fraud (auto-approved, wrong)", n=20, sort_col="Amount", ascending=False)

# 7. False alarm -- both auto-flag a legitimate transaction as fraud
for tier in TIERS:
    false_alarm = df[(df["Risk_Tier"] == tier) & (df["Class"] == 0) & (~df["m2_deferred"]) & (df["m2_pred"] == "Fraud")
                      & (~df["m3_deferred"]) & (df["m3_pred"] == "Fraud")]
    add(false_alarm, "false_alarm", "Both models auto-flag a legitimate transaction as fraud (false alarm)", n=3, sort_col="Amount", ascending=False)

# 8. Boundary / edge-case amounts -- nearest real transactions to each tier threshold, both sides
for threshold in [22, 77, 500]:
    below = df[df["Amount"] <= threshold]
    add(below, "edge_boundary", f"Edge case: amount just below the €{threshold} tier boundary", n=1, sort_col="Amount", ascending=False)
    above = df[df["Amount"] > threshold]
    add(above, "edge_boundary", f"Edge case: amount just above the €{threshold} tier boundary", n=1, sort_col="Amount", ascending=True)

# smallest and largest real amounts in the dataset
add(df, "edge_boundary", "Edge case: smallest real transaction amount in the dataset", n=1, sort_col="Amount", ascending=True)
add(df, "edge_boundary", "Edge case: largest real transaction amount in the dataset", n=1, sort_col="Amount", ascending=False)

out = pd.DataFrame(rows)
out["is_known_divergent"] = out["m2_deferred"] != out["m3_deferred"]
out = out.rename(columns={"m2_deferred": "deferred_m2", "m3_deferred": "deferred_m3"})
out = out.drop_duplicates(subset=["Time", "Amount"])
out.to_csv("sample_transactions.csv", index=False)

print("Total rows:", len(out))
print()
print(out["scenario"].value_counts())
print()
print(out["scenario_label"].value_counts())
