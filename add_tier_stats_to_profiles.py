import json
from l2d_model import amount_to_tier, build_input_vector, load_models, run_inference

standard, risk_sensitive, scaler_stats, _ = load_models()
with open("borderline_profiles.json") as f:
    profiles = json.load(f)

for label, p in profiles.items():
    tier_counts = {}
    for amount in range(1, 2001):
        x = build_input_vector(p["time"], p["v_values"], float(amount), scaler_stats)
        r2 = run_inference(standard, x)
        r3 = run_inference(risk_sensitive, x)
        tier = amount_to_tier(amount)
        tier_counts.setdefault(tier, {"n": 0, "m2_defer": 0, "m3_defer": 0})
        tier_counts[tier]["n"] += 1
        tier_counts[tier]["m2_defer"] += int(r2["deferred"])
        tier_counts[tier]["m3_defer"] += int(r3["deferred"])

    tier_rates = {}
    for tier, c in tier_counts.items():
        tier_rates[tier] = {
            "m2_pct": round(c["m2_defer"] / c["n"] * 100, 1),
            "m3_pct": round(c["m3_defer"] / c["n"] * 100, 1),
        }
    p["tier_defer_rates"] = tier_rates

    # does this profile's own ordering match "Critical defers most under M3"?
    tier_order = ["Low", "Medium", "High", "Critical"]
    present = [t for t in tier_order if t in tier_rates]
    if present:
        crit_rate = tier_rates.get("Critical", {}).get("m3_pct")
        max_rate = max(tier_rates[t]["m3_pct"] for t in present)
        max_tier = [t for t in present if tier_rates[t]["m3_pct"] == max_rate][0]
        p["matches_critical_defers_most"] = (crit_rate is not None and crit_rate == max_rate)
        p["highest_m3_defer_tier"] = max_tier

with open("borderline_profiles.json", "w") as f:
    json.dump(profiles, f, indent=2)

for label, p in profiles.items():
    print(label, "-> matches_critical_defers_most:", p["matches_critical_defers_most"], "| highest tier:", p["highest_m3_defer_tier"])
    print("  ", p["tier_defer_rates"])
