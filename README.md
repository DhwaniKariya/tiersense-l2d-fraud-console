# A Risk-Sensitive Learning-to-Defer Framework for Responsible AI in Financial Fraud Detection

Everything behind the MSc thesis of the same name (Dhwani Sanjay Kariya, MSc AI for
Business, National College of Ireland): the full training pipeline, trained model weights, every
results file referenced in the report's Evaluation and Discussion chapters, and the interactive
live demo built on top of the trained models.

**Live demo:** https://tiersense-l2d-fraud-console.streamlit.app (no install needed)

This repo exists so every finding in the report traces back to a real, runnable artefact in one
place, not just a number typed into a table.

## Structure

- [`prototype/`](prototype) — the Streamlit console (`app.py`) that lets you compare the standard
  vs. risk-sensitive L2D policy on real ULB transactions, live. See
  [`prototype/README.md`](prototype/README.md) for details.
- `notebooks/` — the 7 Jupyter notebooks that build and evaluate the three models (baseline,
  Standard L2D, Risk-Sensitive L2D) across ULB, IEEE-CIS and PaySim.
- `scripts/` — 5 standalone scripts written later to stress-test the headline result: a
  weight-sensitivity sweep, rule-based baselines, and a multi-seed robustness check on IEEE-CIS.
- `models/` — trained weights for Model 2 (Standard L2D) and Model 3 (Risk-Sensitive L2D) on ULB,
  the primary dataset. IEEE-CIS and PaySim models are retrained in-notebook and not separately
  checkpointed.
- `results/` — every JSON result, CSV, and chart produced by the notebooks and scripts above.
  Nothing in this folder is hand-edited; it is all direct pipeline output.
- `docs/` — configuration manual and project presentation (coming soon).
- `requirements-pipeline.txt` — pinned package versions for the notebooks/scripts pipeline
  (PyTorch, scikit-learn, imbalanced-learn, Jupyter). The console has its own lighter
  [`prototype/requirements.txt`](prototype/requirements.txt).

## Running the training pipeline

See the thesis's own Configuration Manual for full step-by-step setup, dataset download links,
and run order. In short:

```
pip install -r requirements-pipeline.txt
jupyter notebook   # run notebooks/ in numeric order, then scripts/ in numeric order
```

## Running the console locally

```
cd prototype
pip install -r requirements.txt
streamlit run app.py
```

## A note on the data

The three raw datasets (ULB Credit Card Fraud, IEEE-CIS Fraud Detection, PaySim) are not included
in this repository. They are large, public Kaggle datasets that Kaggle's own terms do not permit
redistributing. Download links and instructions for reproducing the full pipeline from raw data
are in the Configuration Manual.
