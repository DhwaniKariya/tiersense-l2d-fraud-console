# A Risk-Sensitive Learning-to-Defer Framework for Responsible AI in Financial Fraud Detection

Everything behind the MSc thesis of the same name (Dhwani Sanjay Kariya, MSc AI for
Business, National College of Ireland): the full training pipeline, trained model weights, every
results file referenced in the report's Evaluation and Discussion chapters, and the interactive
live demo built on top of the trained models.

**Live demo:** https://dhwanikariya.github.io/tiersense-l2d-fraud-console/ (static site, no install,
nothing sent to any server — the two trained models run as plain JavaScript in your browser)

This repo exists so every finding in the report traces back to a real, runnable artefact in one
place, not just a number typed into a table.

## Structure

- [`docs/`](docs) — the live console: a static HTML/CSS/JS site (deployed via GitHub Pages) that
  ports the two trained ULB models (Standard L2D, Risk-Sensitive L2D) to run entirely client-side,
  so it has no server to keep running and nothing to deploy/break. Compares both policies on real
  transactions live, plus the cross-dataset evidence page. See [`docs/README.md`](docs/README.md)
  if present, or just open `docs/index.html`.
- [`prototype/`](prototype) — the original Streamlit console (`app.py`) with the same features,
  kept for reference. See [`prototype/README.md`](prototype/README.md) for details.
- `notebooks/` — the 7 Jupyter notebooks that build and evaluate the three models (baseline,
  Standard L2D, Risk-Sensitive L2D) across ULB, IEEE-CIS and PaySim.
- `scripts/` — 5 standalone scripts written later to stress-test the headline result: a
  weight-sensitivity sweep, rule-based baselines, and a multi-seed robustness check on IEEE-CIS.
- `models/` — trained weights for Model 2 (Standard L2D) and Model 3 (Risk-Sensitive L2D) on ULB,
  the primary dataset. IEEE-CIS and PaySim models are retrained in-notebook and not separately
  checkpointed.
- `results/` — every JSON result, CSV, and chart produced by the notebooks and scripts above.
  Nothing in this folder is hand-edited; it is all direct pipeline output.
- `requirements-pipeline.txt` — pinned package versions for the notebooks/scripts pipeline
  (PyTorch, scikit-learn, imbalanced-learn, Jupyter). The Streamlit console has its own lighter
  [`prototype/requirements.txt`](prototype/requirements.txt).

## Running the live console locally

No install needed — it's a static site:

```
cd docs
python -m http.server 8000
# open http://localhost:8000
```

Or just open `docs/index.html` directly in a browser.

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
