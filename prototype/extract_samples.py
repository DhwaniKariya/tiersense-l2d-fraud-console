"""
Builds the two small data files the prototype needs, so it never has to load
the full 284,807-row ULB dataset at runtime:
  - sample_transactions.csv: a curated set of real transactions spanning all
    four risk tiers and both classes (fraud / normal), for the "pick an
    example" mode.
  - scaler_stats.json: the Amount/Time mean and std from the FULL dataset,
    matching exactly what 04_Risk_Sensitive_L2D.ipynb fit its StandardScaler
    on (fit before the train/test split), so any new input gets scaled the
    same way the models were trained to expect.
"""
import os
import json
import pandas as pd

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "creditcard_with_tiers.csv")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(SRC)

# Scaler stats, fit on the full dataset (matches the notebook's fit_transform
# call order: Amount first, then Time, each with its own mean/std).
scaler_stats = {
    "amount_mean": float(df["Amount"].mean()),
    "amount_std": float(df["Amount"].std()),
    "time_mean": float(df["Time"].mean()),
    "time_std": float(df["Time"].std()),
}
with open(f"{OUT_DIR}/scaler_stats.json", "w") as f:
    json.dump(scaler_stats, f, indent=2)

# Also save the mean V1-V28 profile of NORMAL transactions, used as the
# background feature vector for the "type your own amount" mode, so the
# demo isolates the effect of amount without also faking a fraud pattern.
v_cols = [f"V{i}" for i in range(1, 29)]
normal_profile = df[df["Class"] == 0][v_cols].mean().to_dict()
with open(f"{OUT_DIR}/normal_v_profile.json", "w") as f:
    json.dump({k: float(v) for k, v in normal_profile.items()}, f, indent=2)

# Curated example transactions: up to 2 normal + 2 fraud per tier.
rows = []
for tier in ["Low", "Medium", "High", "Critical"]:
    tier_df = df[df["Risk_Tier"] == tier]
    normals = tier_df[tier_df["Class"] == 0].sample(n=2, random_state=42)
    frauds = tier_df[tier_df["Class"] == 1]
    n_fraud = min(2, len(frauds))
    frauds = frauds.sample(n=n_fraud, random_state=42) if n_fraud else frauds
    rows.append(normals)
    if n_fraud:
        rows.append(frauds)

sample = pd.concat(rows, ignore_index=True)
sample = sample.sort_values(["Risk_Tier", "Amount"]).reset_index(drop=True)
sample.to_csv(f"{OUT_DIR}/sample_transactions.csv", index=False)

print(f"Saved {len(sample)} sample transactions.")
print(sample[["Risk_Tier", "Amount", "Class"]].to_string())
print()
print("Scaler stats:", scaler_stats)
