"""
Deferral is rare (<1% even in Critical tier per the thesis's own numbers), so
a small random sample of examples will almost never show a live Model 2 vs
Model 3 divergence. This scans the full dataset once to find real rows where
the two models actually disagree, so the prototype can guarantee at least
one genuine "Standard L2D lets this through, Risk-Sensitive L2D escalates it"
moment instead of leaving it to chance.
"""
import json

import pandas as pd
import torch

from l2d_model import L2DNetwork, build_input_vector

HERE = r"E:\NCI\Sem3\Thesis\implementation\prototype"
SRC = r"E:\NCI\Sem3\Thesis\implementation\data\creditcard_with_tiers.csv"

standard = L2DNetwork()
standard.load_state_dict(torch.load(f"{HERE}/standard_l2d.pth", map_location="cpu"))
standard.eval()

risk_sensitive = L2DNetwork()
risk_sensitive.load_state_dict(torch.load(f"{HERE}/risk_sensitive_l2d.pth", map_location="cpu"))
risk_sensitive.eval()

with open(f"{HERE}/scaler_stats.json") as f:
    scaler_stats = json.load(f)

df = pd.read_csv(SRC)

v_cols = [f"V{i}" for i in range(1, 29)]
time_scaled = (df["Time"] - scaler_stats["time_mean"]) / scaler_stats["time_std"]
amount_scaled = (df["Amount"] - scaler_stats["amount_mean"]) / scaler_stats["amount_std"]

X = torch.tensor(
    pd.concat([time_scaled, df[v_cols], amount_scaled], axis=1).values,
    dtype=torch.float32,
)

with torch.no_grad():
    _, defer2 = standard(X)
    _, defer3 = risk_sensitive(X)

df["defer_prob_m2"] = defer2.squeeze().numpy()
df["defer_prob_m3"] = defer3.squeeze().numpy()
df["deferred_m2"] = df["defer_prob_m2"] > 0.5
df["deferred_m3"] = df["defer_prob_m3"] > 0.5

print("Total M2 deferrals:", df["deferred_m2"].sum())
print("Total M3 deferrals:", df["deferred_m3"].sum())
print()
print("M3 deferrals by tier:")
print(df[df["deferred_m3"]]["Risk_Tier"].value_counts())
print()

divergent = df[(df["deferred_m3"]) & (~df["deferred_m2"])]
print(f"True divergence cases (M3 defers, M2 does not): {len(divergent)}")
if len(divergent):
    cols = ["Risk_Tier", "Amount", "Class", "defer_prob_m2", "defer_prob_m3"]
    print(divergent[cols].sort_values("Risk_Tier"))
    divergent.to_csv(f"{HERE}/divergent_cases.csv", index=False)
    print(f"\nSaved {len(divergent)} divergent cases to divergent_cases.csv")

# Also save the top M3-deferred Critical-tier cases regardless of M2, as a
# fallback in case true divergence is thin on the ground.
critical_m3_deferred = df[(df["deferred_m3"]) & (df["Risk_Tier"] == "Critical")]
critical_m3_deferred.to_csv(f"{HERE}/critical_m3_deferred.csv", index=False)
print(f"\nSaved {len(critical_m3_deferred)} Critical-tier M3-deferred cases (fallback set)")
