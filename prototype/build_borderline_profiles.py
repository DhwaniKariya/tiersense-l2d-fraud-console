"""
Replaces the flat "population mean" V-profile (which sits in a dead, always-
normal region of the network, so varying amount alone never did anything)
with a small set of REAL transactions' V1-V28 patterns that are actually
amount-sensitive, found via diagnose_amount_sensitivity.py. The "type your
own amount" mode lets the user pick one of these real behaviour patterns and
slide the amount on top of it, which is also a more honest framing: "how
would the system treat a transaction that behaves like X, at different
sizes" rather than a meaningless flat hypothetical.
"""
import os
import json

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

divergent = pd.read_csv(f"{HERE}/divergent_cases.csv")

# Hand-picked from the sensitivity scan: real rows whose V-pattern is
# genuinely amount-sensitive across a wide range, each telling a slightly
# different story.
picks = {
    "Small everyday-purchase pattern": divergent[divergent["Amount"].between(15, 16)].iloc[0],
    "Occasional larger-purchase pattern": divergent[divergent["Amount"].between(236, 237)].iloc[0],
    "High-value purchase pattern": divergent[divergent["Amount"].between(583, 584)].iloc[0],
}

profiles = {}
for label, row in picks.items():
    profiles[label] = {
        "time": float(row["Time"]),
        "v_values": {f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)},
        "original_amount": float(row["Amount"]),
        "original_tier": row["Risk_Tier"],
    }

with open(f"{HERE}/borderline_profiles.json", "w") as f:
    json.dump(profiles, f, indent=2)

print("Saved profiles:", list(profiles.keys()))
for label, p in profiles.items():
    print(f"  {label}: original amount=€{p['original_amount']:.2f} ({p['original_tier']} tier)")
