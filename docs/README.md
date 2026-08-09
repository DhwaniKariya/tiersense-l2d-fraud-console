# TierSense

Live at https://dhwanikariya.github.io/tiersense-l2d-fraud-console/

Same console as the Streamlit version in [`../prototype/`](../prototype), rewritten as a plain
HTML/CSS/JS site. No backend, no build step. Both trained models (standard L2D and
risk-sensitive L2D) run right in the browser, ported from the saved PyTorch weights.

I moved off Streamlit because Streamlit Cloud kept needing manual redeploys whenever the repo
structure changed, and the free tier sleeps the app if nobody's visited it in a while. Neither of
those seemed worth it for a demo that's supposed to just work when someone clicks the link.

## Files

- `index.html` - the page.
- `app.js` - wires up the tabs, charts and interactive bits.
- `model.js` - the model forward pass, plain JS matching `l2d_model.py`. Checked it against the
  real PyTorch models on all 116 sample transactions before trusting it, same decisions
  everywhere, probabilities within about 1e-6 (just float32 vs float64 rounding).
- `charts.js` - small SVG bar/scatter chart helpers, no chart library.
- `constants.js` - the static numbers from the report (Tables 3-16) used on the evidence page.
- `assets/bundle.js` - the trained model weights plus the curated transaction data, bundled into
  one file so the page works without a server.

## Running it locally

```
cd docs
python -m http.server 8000
```

or just open `index.html` directly, it doesn't need a server at all.
