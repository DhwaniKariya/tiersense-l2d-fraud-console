"""
Test whether ANY real transaction's V-profile is actually amount-sensitive,
i.e. whether varying Amount alone (holding V1-V28 fixed) moves the deferral
decision, by sweeping amount around rows we already know are in the
"interesting" zone (the divergent cases found earlier).
"""
import json
import pandas as pd
import torch

from l2d_model import L2DNetwork, build_input_vector, run_inference

HERE = r"E:\NCI\Sem3\Thesis\implementation\prototype"

standard = L2DNetwork()
standard.load_state_dict(torch.load(f"{HERE}/standard_l2d.pth", map_location="cpu"))
standard.eval()

risk_sensitive = L2DNetwork()
risk_sensitive.load_state_dict(torch.load(f"{HERE}/risk_sensitive_l2d.pth", map_location="cpu"))
risk_sensitive.eval()

with open(f"{HERE}/scaler_stats.json") as f:
    scaler_stats = json.load(f)

divergent = pd.read_csv(f"{HERE}/divergent_cases.csv")
divergent = divergent.sort_values("Amount", ascending=False)

test_amounts = [1, 20, 50, 100, 300, 500, 700, 1000, 1500, 2000, 3000, 5000]

for idx in [0, 5, 10, 50, 150, 300]:
    if idx >= len(divergent):
        continue
    row = divergent.iloc[idx]
    v = {f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)}
    print(f"\n=== Row: original amount={row['Amount']:.2f}, tier={row['Risk_Tier']}, class={row['Class']} ===")
    for amount in test_amounts:
        x = build_input_vector(float(row["Time"]), v, float(amount), scaler_stats)
        r2 = run_inference(standard, x)
        r3 = run_inference(risk_sensitive, x)
        marker = " <-- ORIGINAL" if abs(amount - row["Amount"]) < 1 else ""
        print(f"  amount={amount:>6} | M2 defer={r2['defer_probability']*100:6.2f}% | M3 defer={r3['defer_probability']*100:6.2f}%{marker}")
