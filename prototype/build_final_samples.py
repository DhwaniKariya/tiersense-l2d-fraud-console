"""
Combines the original tier/fraud-variety samples with real, discovered
Model 2 vs Model 3 divergence cases, so the prototype's example picker always
has at least a few genuine "the two policies disagree on this exact
transaction" moments, rather than leaving that to chance on a small random
sample (deferral overall is under 1%, so a handful of random picks will
almost always land on cases where nothing interesting happens).
"""
import os
import json

import pandas as pd
import torch

from l2d_model import L2DNetwork

HERE = os.path.dirname(os.path.abspath(__file__))

standard = L2DNetwork()
standard.load_state_dict(torch.load(f"{HERE}/standard_l2d.pth", map_location="cpu"))
standard.eval()

risk_sensitive = L2DNetwork()
risk_sensitive.load_state_dict(torch.load(f"{HERE}/risk_sensitive_l2d.pth", map_location="cpu"))
risk_sensitive.eval()

with open(f"{HERE}/scaler_stats.json") as f:
    scaler_stats = json.load(f)

original = pd.read_csv(f"{HERE}/sample_transactions.csv")
divergent = pd.read_csv(f"{HERE}/divergent_cases.csv")

# Pick a spread of divergence cases: 3 Critical (highest amounts, most
# dramatic), 2 High/Medium, 1 Low, so every tier has at least one "watch them
# disagree" example available.
picks = []
for tier, n in [("Critical", 3), ("High", 2), ("Medium", 2), ("Low", 1)]:
    tier_rows = divergent[divergent["Risk_Tier"] == tier].sort_values("Amount", ascending=False)
    picks.append(tier_rows.head(n))
divergent_picked = pd.concat(picks, ignore_index=True)
divergent_picked = divergent_picked.drop(columns=["defer_prob_m2", "defer_prob_m3"])
divergent_picked["is_known_divergent"] = True

original["is_known_divergent"] = False

combined = pd.concat([original, divergent_picked], ignore_index=True)
combined = combined.drop_duplicates(subset=["Time", "Amount", "Class"])
combined = combined.sort_values(["Risk_Tier", "Amount"]).reset_index(drop=True)

combined.to_csv(f"{HERE}/sample_transactions.csv", index=False)
print(f"Final sample set: {len(combined)} transactions")
print(combined[["Risk_Tier", "Amount", "Class", "is_known_divergent"]].to_string())
