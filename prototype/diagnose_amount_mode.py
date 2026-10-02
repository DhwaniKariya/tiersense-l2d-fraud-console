"""
Diagnose why the "type your own amount" mode shows identical, unchanging
output for both models across the whole amount range.
"""
import os
import json
import pandas as pd
import torch

from l2d_model import L2DNetwork, build_input_vector, run_inference

HERE = os.path.dirname(os.path.abspath(__file__))

standard = L2DNetwork()
standard.load_state_dict(torch.load(f"{HERE}/standard_l2d.pth", map_location="cpu"))
standard.eval()

risk_sensitive = L2DNetwork()
risk_sensitive.load_state_dict(torch.load(f"{HERE}/risk_sensitive_l2d.pth", map_location="cpu"))
risk_sensitive.eval()

with open(f"{HERE}/scaler_stats.json") as f:
    scaler_stats = json.load(f)

with open(f"{HERE}/normal_v_profile.json") as f:
    normal_v_profile = json.load(f)

print("=== Test 1: mean-of-all-normals V-profile, sweeping amount ===")
for amount in [1, 20, 50, 100, 300, 600, 1000, 2000, 5000, 10000]:
    x = build_input_vector(50000.0, normal_v_profile, float(amount), scaler_stats)
    r2 = run_inference(standard, x)
    r3 = run_inference(risk_sensitive, x)
    print(f"amount={amount:>7} | M2 fraud={r2['fraud_probability']*100:6.2f}% defer={r2['defer_probability']*100:6.2f}% | "
          f"M3 fraud={r3['fraud_probability']*100:6.2f}% defer={r3['defer_probability']*100:6.2f}%")

print("\n=== Test 2: check variance of the mean V-profile values themselves ===")
import statistics
vals = list(normal_v_profile.values())
print("mean-of-normals V-profile stats: min", min(vals), "max", max(vals), "mean", statistics.mean(vals))

print("\n=== Test 3: real transaction's V-profile (a Low-tier normal), sweeping amount ===")
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "creditcard_with_tiers.csv"))
real_row = df[(df["Risk_Tier"] == "Low") & (df["Class"] == 0)].iloc[100]
real_v = {f"V{i}": float(real_row[f"V{i}"]) for i in range(1, 29)}
for amount in [1, 20, 50, 100, 300, 600, 1000, 2000, 5000, 10000]:
    x = build_input_vector(float(real_row["Time"]), real_v, float(amount), scaler_stats)
    r2 = run_inference(standard, x)
    r3 = run_inference(risk_sensitive, x)
    print(f"amount={amount:>7} | M2 fraud={r2['fraud_probability']*100:6.2f}% defer={r2['defer_probability']*100:6.2f}% | "
          f"M3 fraud={r3['fraud_probability']*100:6.2f}% defer={r3['defer_probability']*100:6.2f}%")

print("\n=== Test 4: real Critical-tier fraud row's V-profile, sweeping amount ===")
real_row2 = df[(df["Risk_Tier"] == "Critical") & (df["Class"] == 0)].iloc[5]
real_v2 = {f"V{i}": float(real_row2[f"V{i}"]) for i in range(1, 29)}
for amount in [1, 20, 50, 100, 300, 600, 1000, 2000, 5000, 10000]:
    x = build_input_vector(float(real_row2["Time"]), real_v2, float(amount), scaler_stats)
    r2 = run_inference(standard, x)
    r3 = run_inference(risk_sensitive, x)
    print(f"amount={amount:>7} | M2 fraud={r2['fraud_probability']*100:6.2f}% defer={r2['defer_probability']*100:6.2f}% | "
          f"M3 fraud={r3['fraud_probability']*100:6.2f}% defer={r3['defer_probability']*100:6.2f}%")
