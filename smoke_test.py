import pandas as pd
from l2d_model import load_models, build_input_vector, run_inference, amount_to_tier

standard, risk_sensitive, scaler_stats, normal_v_profile = load_models()
samples = pd.read_csv("sample_transactions.csv")

print(f"{'Tier':<10} {'Amount':>10} {'Actual':<8} {'M2 fraud%':>10} {'M2 decision':<24} {'M3 fraud%':>10} {'M3 decision':<24} {'DIVERGE?'}")
for _, row in samples.iterrows():
    v_values = {f"V{i}": row[f"V{i}"] for i in range(1, 29)}
    x = build_input_vector(row["Time"], v_values, row["Amount"], scaler_stats)
    r2 = run_inference(standard, x)
    r3 = run_inference(risk_sensitive, x)
    actual = "Fraud" if row["Class"] == 1 else "Normal"
    diverge = "<<<" if r2["deferred"] != r3["deferred"] else ""
    print(f"{row['Risk_Tier']:<10} {row['Amount']:>10.2f} {actual:<8} "
          f"{r2['fraud_probability']*100:>9.1f}% {r2['decision']:<24} "
          f"{r3['fraud_probability']*100:>9.1f}% {r3['decision']:<24} {diverge}")

# Also test the custom-amount path (pure amount sweep on a "typical normal" profile)
print("\n--- Custom amount sweep on a typical-normal V-profile ---")
for amount in [5, 20, 50, 100, 300, 600, 2000]:
    x = build_input_vector(50000, normal_v_profile, amount, scaler_stats)
    r2 = run_inference(standard, x)
    r3 = run_inference(risk_sensitive, x)
    tier = amount_to_tier(amount)
    print(f"amount={amount:>7} tier={tier:<8} M2={r2['decision']:<24} M3={r3['decision']:<24}")
