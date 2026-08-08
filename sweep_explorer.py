import json
import torch
from l2d_model import amount_to_tier, build_input_vector, load_models, run_inference

standard, risk_sensitive, scaler_stats, _ = load_models()
with open("borderline_profiles.json") as f:
    profiles = json.load(f)

for label, p in profiles.items():
    print(f"=== {label} (orig €{p['original_amount']:.2f}, {p['original_tier']}) ===")
    tier_counts = {}
    for amount in range(1, 2001):
        x = build_input_vector(p["time"], p["v_values"], float(amount), scaler_stats)
        r2 = run_inference(standard, x)
        r3 = run_inference(risk_sensitive, x)
        tier = amount_to_tier(amount)
        key = tier
        tier_counts.setdefault(key, {"n": 0, "m2_defer": 0, "m3_defer": 0})
        tier_counts[key]["n"] += 1
        tier_counts[key]["m2_defer"] += int(r2["deferred"])
        tier_counts[key]["m3_defer"] += int(r3["deferred"])
    for tier in ["Low", "Medium", "High", "Critical"]:
        if tier in tier_counts:
            c = tier_counts[tier]
            m2r = c["m2_defer"] / c["n"] * 100
            m3r = c["m3_defer"] / c["n"] * 100
            print(f"  {tier:8s} n={c['n']:4d}  M2 defer rate={m2r:6.2f}%  M3 defer rate={m3r:6.2f}%")
    print()
