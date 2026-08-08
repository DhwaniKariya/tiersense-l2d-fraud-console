# TierSense — Risk-Sensitive Deferral Console

**Live demo:** https://tiersense-l2d-fraud-console.streamlit.app

An interactive prototype built on top of the trained models from the MSc thesis *"A
Risk-Sensitive Learning-to-Defer Framework for Responsible AI in Financial Fraud Detection"*
(National College of Ireland, 2026). It compares a standard Learning-to-Defer (L2D) policy
against the thesis's risk-sensitive one on the same transaction, live, and lets you see why the
two disagree.

This is a research prototype for demonstration purposes only. It is not a production fraud
system, and no organisation's live data has been run through it.

## What it does

**Why Risk-Sensitive L2D** — the headline evidence across all three datasets tested in the
report (ULB, IEEE-CIS, PaySim), pulled directly from the Evaluation chapter. Shows why a
risk-sensitive deferral cost is needed at all, before looking at any single transaction.

**ULB Transaction Console** — explore individual real transactions from the ULB Credit Card
Fraud dataset:
- A curated flagship case (a real €549.06 confirmed-fraud transaction) where the two policies
  make opposite calls.
- **Browse real transactions**: 116 real transactions from the held-out test set, organised into
  8 named scenario categories (both models agree, only one defers in either direction, both
  defer, models disagree on the label, both miss a fraud, false alarms, boundary edge cases).
  Not a random sample — deliberately picked to cover every scenario the two policies can produce.
- **Same pattern, different amount**: take one real transaction's underlying behaviour and slide
  its amount from €1 to €2,000, everything else held fixed, and watch each model's decision
  change as the amount alone changes.

## Run it locally

```
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501 by default.

## Files

- `app.py` — the Streamlit UI.
- `l2d_model.py` — model architecture (matches `implementation/notebooks/04_Risk_Sensitive_L2D.ipynb`
  in the main repo exactly) and inference helpers.
- `standard_l2d.pth`, `risk_sensitive_l2d.pth` — the trained model weights (Model 2 and Model 3
  from the thesis, Section 4.3), copied from `implementation/models/`.
- `scaler_stats.json` — Amount/Time mean and std, fit on the full ULB training set, so new inputs
  are scaled the same way the models were trained.
- `normal_v_profile.json` — mean V1-V28 profile of legitimate transactions, used as the
  background feature vector when sweeping amount, so that view isolates the effect of amount
  rather than also faking a fraud pattern.
- `sample_transactions.csv` — 116 curated real transactions spanning all four risk tiers and all
  8 scenario categories, built by `extract_samples.py`, `build_final_samples.py`, and
  `build_scenario_catalog.py`.
- `borderline_profiles.json`, `divergent_cases.csv`, `critical_m3_deferred.csv`,
  `flagship_case.json` — supporting curated case data used by the console.
- `ulb_weight_sensitivity.json` — 5 real, independently retrained ULB models from the report's
  own weight-sensitivity sweep (Section 6.1.4), used by the "What if the weight gradient were
  gentler or steeper?" section. Not a live-retraining control; the risk weight is baked into the
  training loss, so changing it means retraining from scratch.
- The `build_*.py` / `diagnose_*.py` / `find_*.py` scripts were one-off tools used to generate the
  curated sample files above from the full 284,807-row ULB dataset. They are included for
  transparency but are not needed to run the app, and they require the raw dataset (not bundled
  here, see below).

## A note on the data

The full raw datasets (ULB Credit Card Fraud, IEEE-CIS Fraud Detection, PaySim) are not bundled
in this repository. They are large, public Kaggle datasets that the college's own ethics
guidance and Kaggle's terms do not permit redistributing. This app only ships the small, curated,
already-derived files listed above (well under 1MB total), which is everything the live demo
needs. Full instructions for reproducing the training pipeline from the raw data are in the
thesis's Configuration Manual.
