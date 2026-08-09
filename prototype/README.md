# TierSense (Streamlit version)

This is the original build of the console, before I rewrote it as a static site. That newer
version lives in [`../docs/`](../docs) and is the one I'd actually point people to:
https://dhwanikariya.github.io/tiersense-l2d-fraud-console/. It has the same features and
doesn't rely on Streamlit Cloud staying up. This folder is kept for reference.

Interactive prototype built on top of the trained models from my MSc thesis, *"A Risk-Sensitive
Learning-to-Defer Framework for Responsible AI in Financial Fraud Detection"* (NCI, 2026). It
compares a standard Learning-to-Defer policy against the risk-sensitive one on the same
transaction, live, and lets you see where they disagree.

This is a research prototype, not a production system. No real organisation's data has gone
through it.

Full training pipeline (notebooks, scripts, results) is in the repo root, [`../`](..).

## What it does

**Why Risk-Sensitive L2D** walks through the headline evidence across all three datasets in the
report (ULB, IEEE-CIS, PaySim), taken straight from the Evaluation chapter. This is the case for
why a risk-sensitive deferral cost matters at all, before you look at any single transaction.

**ULB Transaction Console** lets you dig into real transactions from the ULB dataset:

- A flagship case: a real €549.06 confirmed fraud where the two policies make opposite calls.
- Browse real transactions: 116 from the held-out test set, sorted into 8 categories (both
  models agree, only one defers, both defer, they disagree on the label, both miss a fraud,
  false alarms, edge cases). I picked these deliberately to cover every scenario, not randomly.
- Same pattern, different amount: take one transaction's underlying behaviour and slide the
  amount from €1 to €2,000 with everything else held fixed, and watch each model's decision move.

## Running it

```
pip install -r requirements.txt
streamlit run app.py
```

Opens at localhost:8501.

## Files

- `app.py` - the Streamlit UI.
- `l2d_model.py` - model architecture (matches `../notebooks/04_Risk_Sensitive_L2D.ipynb`) plus
  the inference helpers.
- `standard_l2d.pth`, `risk_sensitive_l2d.pth` - trained weights for Model 2 and Model 3
  (Section 4.3 of the report), copied over from `../models/`.
- `scaler_stats.json` - Amount/Time mean and std from the ULB training set, so new inputs get
  scaled the same way the models were trained on.
- `normal_v_profile.json` - average V1-V28 profile of legit transactions, used as the background
  vector when sweeping amount so that view isolates the effect of amount alone.
- `sample_transactions.csv` - the 116 curated transactions, built by `extract_samples.py`,
  `build_final_samples.py` and `build_scenario_catalog.py`.
- `borderline_profiles.json`, `divergent_cases.csv`, `critical_m3_deferred.csv`,
  `flagship_case.json` - supporting curated data for the console.
- `ulb_weight_sensitivity.json` - 5 separately retrained ULB models from the weight sensitivity
  sweep in the report (Section 6.1.4). Not a live control, the risk weight is baked into the
  training loss so changing it means retraining from scratch.
- The `build_*.py`, `diagnose_*.py` and `find_*.py` scripts are one-off tools I used to build the
  curated sample files from the full 284,807-row dataset. Not needed to run the app, and they
  need the raw dataset which isn't included here.

## About the data

The raw datasets (ULB, IEEE-CIS, PaySim) aren't bundled here. They're public Kaggle datasets and
redistributing them isn't allowed under Kaggle's terms, plus the college's ethics guidance. The
app only ships the small curated files above (well under 1MB total), which is all the demo
actually needs. Full setup instructions for the raw data are in the Configuration Manual.
