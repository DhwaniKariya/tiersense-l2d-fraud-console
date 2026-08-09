# A Risk-Sensitive Learning-to-Defer Framework for Responsible AI in Financial Fraud Detection

This is the code behind my MSc thesis (Dhwani Sanjay Kariya, MSc AI for Business,
National College of Ireland). Notebooks, trained models, results, and a small demo console so
the numbers in the report aren't just numbers, you can actually poke at the thing.

**Live demo:** https://dhwanikariya.github.io/tiersense-l2d-fraud-console/

It's a static page, nothing to install. Both trained models run in your browser in plain JS, so
nothing gets sent anywhere.

## What's in here

- `docs/` - the live console. Compares the standard L2D policy against the risk-sensitive one on
  real transactions, plus a page walking through the evidence across all three datasets. See
  `docs/README.md`, or just open `docs/index.html`.
- `prototype/` - the original version of the console, built with Streamlit. Same features, kept
  around for reference. Moved off it because Streamlit Cloud kept needing manual redeploys and
  the free tier sleeps the app after a while, which got annoying for something that's supposed to
  just be a link people click.
- `notebooks/` - the 7 notebooks that build and evaluate the three models (baseline, standard
  L2D, risk-sensitive L2D) across ULB, IEEE-CIS and PaySim.
- `scripts/` - a handful of scripts I wrote later to stress-test the result: weight sensitivity
  sweep, rule-based baselines, multi-seed check on IEEE-CIS.
- `models/` - the two trained ULB checkpoints (standard and risk-sensitive). IEEE-CIS and PaySim
  get retrained inside their notebooks and aren't saved separately.
- `results/` - every JSON/CSV/PNG the notebooks and scripts spit out. None of this is hand edited.
- `requirements-pipeline.txt` - what you need for the notebooks/scripts. The console has its own
  lighter `prototype/requirements.txt`.

## Running the console

It's static, so:

```
cd docs
python -m http.server 8000
```

then open localhost:8000. Or just double-click `docs/index.html`.

## Running the training pipeline

Full setup instructions and dataset links are in the Configuration Manual submitted with the
thesis. Short version:

```
pip install -r requirements-pipeline.txt
jupyter notebook
```

then run the notebooks in order, then the scripts.

## Running the old Streamlit console

```
cd prototype
pip install -r requirements.txt
streamlit run app.py
```

## About the data

I'm not including the raw datasets (ULB Credit Card Fraud, IEEE-CIS, PaySim) here, they're
Kaggle datasets and redistributing them isn't allowed under Kaggle's terms. Download links and
setup instructions are in the Configuration Manual.
