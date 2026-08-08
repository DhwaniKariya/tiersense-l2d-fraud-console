"""
TierSense — Risk-Sensitive Deferral Console, a fraud-review prototype for
"A Risk-Sensitive Learning-to-Defer Framework for Responsible AI in Financial
Fraud Detection" (MSc thesis, National College of Ireland, 2026).

Loads the two trained ULB models (Standard L2D and Risk-Sensitive L2D) and
lets a user step through real transactions, or take a real behaviour
pattern and vary its amount, to see the two deferral policies make (and
sometimes disagree on) a live decision. A separate evidence page presents
the cross-dataset case for the method using figures taken directly from
the report's Evaluation chapter (Tables 3-16).

Run with:  py -3.13 -m streamlit run app.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from l2d_model import amount_to_tier, build_input_vector, load_models, run_inference

st.set_page_config(page_title="TierSense — Risk-Sensitive Deferral Console", page_icon="🛡️", layout="wide")

TIER_COLOR = {
    "Low": "#2ecc71",
    "Medium": "#f1c40f",
    "High": "#e67e22",
    "Critical": "#e74c3c",
}
BASELINE_COLOR = "#64748b"
STANDARD_COLOR = "#60a5fa"
RISK_COLOR = "#f59e0b"

FULL_DATASET_SIZE = 284807
CATEGORY_REAL_FREQUENCY = {
    "only_m2_defers": 107,
    "only_m3_defers": 550,
    "both_defer": 610,
    "classification_disagreement": 342,
    "both_miss_fraud": 14,
    "false_alarm": 101,
}

# Every figure below is taken directly from the report's Evaluation chapter
# (Tables 3, 4, 5, 8, 9, 10, 12, 13, 16) -- nothing here is recomputed live.
DATASET_METRICS = {
    "ULB Credit Card Fraud": {
        "baseline": {"f1": 0.1094, "roc_auc": 0.9698},
        "standard": {"f1": 0.7736, "roc_auc": 0.9787},
        "risk_sensitive": {"f1": 0.7378, "roc_auc": 0.9800},
        "tiers": {
            "Low": {"standard": 0.28, "risk_sensitive": 0.42},
            "Medium": {"standard": 0.18, "risk_sensitive": 0.22},
            "High": {"standard": 0.25, "risk_sensitive": 0.49},
            "Critical": {"standard": 0.21, "risk_sensitive": 0.96},
        },
        "critical_headline_ratio": "4.6x",
        "seeds": [42, 43, 44, 45, 46],
        "seed_ratios": [4.6, 24.7, 6.2, 23.5, 19.1],  # Table 5
    },
    "IEEE-CIS": {
        "baseline": {"f1": 0.343, "roc_auc": 0.905},
        "standard": {"f1": 0.400, "roc_auc": 0.913},
        "risk_sensitive": {"f1": 0.559, "roc_auc": 0.944},
        "tiers": {
            "Low": {"standard": 12.74, "risk_sensitive": 18.60},
            "Medium": {"standard": 3.60, "risk_sensitive": 13.62},
            "High": {"standard": 8.68, "risk_sensitive": 41.94},
            "Critical": {"standard": 10.67, "risk_sensitive": 70.56},
        },
        "critical_headline_ratio": "6.6x",
        "seeds": [42, 43, 44, 45],
        "seed_ratios": [6.61, 5.73, 6.78, 5.49],  # Table 10, M2->M3 Critical-tier ratio
    },
    "PaySim": {
        "baseline": {"f1": 0.648, "roc_auc": 0.998},
        "standard": {"f1": 0.599, "roc_auc": 0.999},
        "risk_sensitive": {"f1": 0.617, "roc_auc": 0.999},
        "tiers": {
            "Low": {"standard": 0.00, "risk_sensitive": 0.02},
            "Medium": {"standard": 0.44, "risk_sensitive": 0.51},
            "High": {"standard": 0.69, "risk_sensitive": 0.74},
            "Critical": {"standard": 0.68, "risk_sensitive": 3.84},
        },
        "critical_headline_ratio": "5.65x",
        "seeds": None,
        "seed_ratios": None,  # not run on PaySim -- Table 16: "Not run (compute cost)"
    },
}

CUSTOM_CSS = """
<style>
.ts-header {
    padding: 1.1rem 1.4rem;
    border-radius: 10px;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    margin-bottom: 1rem;
}
.ts-header h1 {
    color: #ffffff !important;
    font-size: 1.6rem;
    margin: 0 0 0.2rem 0;
}
.ts-header p {
    color: #cbd5e1 !important;
    margin: 0;
    font-size: 0.95rem;
}
.pill {
    display: inline-block;
    padding: 2px 11px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    vertical-align: middle;
}
.pill-research { background: #475569; color: #f1f5f9; }
.pill-live { background: #15803d; color: #f0fdf4; }
.pill-planned { background: #92400e; color: #fff7ed; }
.chip {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 700;
}
.chip-fraud { background: #7f1d1d; color: #fef2f2; }
.chip-normal { background: #14532d; color: #f0fdf4; }
.tier-chip {
    padding: 3px 12px;
    border-radius: 6px;
    color: white;
    font-size: 0.85rem;
    font-weight: 700;
}
.decision-line-auto { color: #15803d; font-weight: 700; font-size: 1.05rem; }
.decision-line-defer { color: #b91c1c; font-weight: 700; font-size: 1.05rem; }
.roadmap-card {
    border: 1px solid rgba(128,128,128,0.35);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    height: 100%;
}
.flagship-box {
    border: 2px solid #b45309;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    background: rgba(180,83,9,0.08);
    margin-bottom: 0.8rem;
}
.pill-flagship { background: #b45309; color: #fff7ed; }
.intro-box {
    border-left: 4px solid #3b82f6;
    padding: 0.7rem 1rem;
    background: rgba(59,130,246,0.08);
    border-radius: 4px;
    margin-bottom: 1rem;
}
.problem-box {
    border: 2px solid #b91c1c;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    background: rgba(185,28,28,0.08);
    margin-bottom: 1rem;
}
.kpi-card {
    border: 1px solid rgba(148,163,184,0.35);
    border-radius: 10px;
    padding: 1rem 0.6rem;
    text-align: center;
    height: 100%;
}
.kpi-number { font-size: 1.8rem; font-weight: 800; color: #f59e0b; }
.kpi-label { font-size: 0.78rem; color: #cbd5e1; margin-top: 0.3rem; line-height: 1.25; }
.cta-box {
    border-left: 4px solid #3b82f6;
    border-radius: 4px;
    padding: 1rem 1.3rem;
    background: rgba(59,130,246,0.06);
    text-align: left;
    margin-top: 1rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def style_dark_fig(fig, axes):
    fig.patch.set_facecolor("#0e1117")
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#cbd5e1", labelsize=8.5)
        ax.xaxis.label.set_color("#e2e8f0")
        ax.yaxis.label.set_color("#e2e8f0")
        ax.title.set_color("#f1f5f9")
        for spine in ax.spines.values():
            spine.set_color("#475569")
        ax.grid(axis="y", color="#334155", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)


@st.cache_resource
def get_models():
    return load_models()


@st.cache_data
def get_borderline_profiles():
    with open("borderline_profiles.json") as f:
        return json.load(f)


@st.cache_data
def get_flagship_case():
    with open("flagship_case.json") as f:
        return json.load(f)


standard_model, risk_sensitive_model, scaler_stats, _normal_v_profile = get_models()
samples = pd.read_csv("sample_transactions.csv")
borderline_profiles = get_borderline_profiles()
flagship_case = get_flagship_case()

st.markdown(
    """
    <div class="ts-header">
        <h1>🛡️ TierSense <span style="font-weight:400; font-size:0.65em;">— Risk-Sensitive Deferral Console</span></h1>
        <p>Compares a standard Learning-to-Defer policy against a risk-sensitive one on the same transaction, live.
        <span class="pill pill-research" style="margin-left:6px;">RESEARCH PROTOTYPE · MSc THESIS</span></p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### About this console")
    st.markdown(
        "This app has two views: **Why Risk-Sensitive L2D** compares all three trained models across "
        "all three datasets from the report; **ULB Transaction Console** lets you explore individual "
        "real transactions from the ULB dataset only. This is a thesis research prototype, not a "
        "production system — no organisation's live data has been run through it."
    )
    st.markdown("---")
    st.markdown("### Demo set disclosure")
    st.markdown(
        f"The **Browse real transactions** tab holds {len(samples)} real transactions, deliberately "
        f"picked to cover 8 named scenario categories (both agree, only one defers in either direction, "
        f"both defer, models disagree on the label, both miss a fraud, false alarms, boundary edge "
        f"cases) — **not a random sample**. Each category is rare or common in very different ways in "
        f"the full {FULL_DATASET_SIZE:,}-row dataset, e.g. only-Standard-defers cases: "
        f"{CATEGORY_REAL_FREQUENCY['only_m2_defers']} ({CATEGORY_REAL_FREQUENCY['only_m2_defers']/FULL_DATASET_SIZE*100:.3f}%); "
        f"only-Risk-Sensitive-defers cases: {CATEGORY_REAL_FREQUENCY['only_m3_defers']} "
        f"({CATEGORY_REAL_FREQUENCY['only_m3_defers']/FULL_DATASET_SIZE*100:.3f}%); "
        f"both-models-miss-a-fraud cases: only {CATEGORY_REAL_FREQUENCY['both_miss_fraud']} in the whole "
        f"dataset. The category picker on that tab shows how many real examples exist for each."
    )
    st.markdown("---")
    st.markdown("### Risk tiers (ULB)")
    for tier, upper in [("Low", "≤ €22"), ("Medium", "≤ €77"), ("High", "≤ €500"), ("Critical", "> €500")]:
        st.markdown(
            f"<span class='tier-chip' style='background:{TIER_COLOR[tier]}'>{tier}</span> &nbsp;{upper}",
            unsafe_allow_html=True,
        )


def render_decision_card(col, title, result):
    with col:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            pred_class = "chip-fraud" if result["predicted_label"] == "Fraud" else "chip-normal"
            st.markdown(
                f"<span class='chip {pred_class}'>PREDICTED: {result['predicted_label'].upper()}</span>",
                unsafe_allow_html=True,
            )
            st.write("")
            st.caption(f"Fraud probability — {result['fraud_probability']*100:.1f}%")
            st.progress(min(result["fraud_probability"], 1.0))
            st.caption(f"Deferral probability — {result['defer_probability']*100:.1f}%")
            st.progress(min(result["defer_probability"], 1.0))
            st.write("")
            if result["deferred"]:
                st.markdown(f"<div class='decision-line-defer'>🚨 {result['decision']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='decision-line-auto'>✅ {result['decision']}</div>", unsafe_allow_html=True)


def render_transaction_header(amount, tier):
    tier_color = TIER_COLOR[tier]
    st.markdown(
        f"### Transaction: €{amount:,.2f} &nbsp;"
        f"<span class='tier-chip' style='background:{tier_color}'>{tier} tier</span>",
        unsafe_allow_html=True,
    )


def render_divergence(r2, r3):
    if r2["deferred"] != r3["deferred"]:
        st.warning(
            "**These two policies disagree on this exact transaction.** "
            + (
                "Standard L2D would let it through automatically; Risk-Sensitive L2D sends it to a human reviewer."
                if r3["deferred"]
                else "Risk-Sensitive L2D would let it through automatically; Standard L2D sends it to a human reviewer."
            )
        )
    else:
        st.info("Both policies make the same call on this transaction.")


TIER_STATS = {
    "Low": {
        "m2": "0.28%", "m3": "0.42%", "ratio": "1.5x",
        "note": (
            "Low tier carries no cost reduction in this design (risk weight 1.0, unchanged from Standard "
            "L2D), so it's expected to barely move — this small residual shift reflects incidental "
            "shared-trunk training variation, not the mechanism this thesis is about."
        ),
    },
    "Medium": {
        "m2": "0.18%", "m3": "0.22%", "ratio": "1.2x",
        "note": (
            "Medium tier gets a modest cost reduction (risk weight 0.8), and shows a correspondingly "
            "modest increase — smaller than High or Critical, as the graded design intends."
        ),
    },
    "High": {
        "m2": "0.25%", "m3": "0.49%", "ratio": "2.0x",
        "note": (
            "High tier gets a larger cost reduction (risk weight 0.5), producing a bigger increase than "
            "Medium but smaller than Critical, sitting where the graded weighting is designed to put it."
        ),
    },
    "Critical": {
        "m2": "0.21%", "m3": "0.96%", "ratio": "4.6x",
        "note": (
            "Critical tier gets the largest cost reduction (risk weight 0.2), producing the largest "
            "relative increase — though that ratio itself ranged 4.6x-24.7x across five training seeds, "
            "and needs roughly a 4-5x deferral-cost gradient between tiers to show up at all "
            "(Sections 6.1.3-6.1.4)."
        ),
    },
}


def render_profile_sensitivity_note(profile, tier):
    rates = profile.get("tier_defer_rates", {})
    cur = rates.get(tier)
    if cur is None:
        return
    st.caption(
        f"For this specific behaviour pattern, sweeping the full €1-2,000 range: Risk-Sensitive L2D defers "
        f"**{cur['m3_pct']:.0f}%** of amounts landing in {tier} tier (Standard L2D: {cur['m2_pct']:.0f}%)."
    )
    if not profile.get("matches_critical_defers_most", True):
        highest = profile.get("highest_m3_defer_tier")
        st.warning(
            f"For this specific pattern, **{highest} tier is deferred most by Risk-Sensitive L2D, not "
            f"Critical** — the opposite ordering from the thesis's population-level result (Table 4). "
            f"'Critical defers most' is an aggregate statistic across thousands of transactions, not a "
            f"guarantee for every individual behaviour pattern — the closest discussion of why the effect "
            f"isn't uniform is the weight-sensitivity finding in Section 6.1.4. Try the other two patterns "
            f"below to compare."
        )


def render_tier_context_note(tier, r3):
    if r3["deferred"]:
        return
    s = TIER_STATS[tier]
    st.caption(
        f"Why is this still AUTO-DECIDED? On the full ULB test set, {tier} tier is deferred {s['m3']} of "
        f"the time under Risk-Sensitive L2D (vs {s['m2']} under Standard L2D, a {s['ratio']} change) — so "
        f"seeing AUTO-DECIDED here is the typical case, not an exception. {s['note']} See report Section "
        f"6.1.2 for the full tier breakdown."
    )


page_evidence, page_console = st.tabs(["📊 Why Risk-Sensitive L2D", "🔍 ULB Transaction Console"])

# ============================================================================
# PAGE 1 — cross-dataset evidence for the method, using report figures only
# ============================================================================
with page_evidence:
    st.markdown(
        """
        <div class="intro-box">
        <b>Before looking at any single transaction:</b> here is the evidence, across all three fraud
        datasets tested in the report, for why a risk-sensitive deferral cost is needed at all. Every
        number on this page comes straight from the report's Evaluation chapter (Tables 3-16) — nothing
        here is recomputed live. Scroll down, then switch to the ULB console to see the same mechanism
        operate on individual transactions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Why Risk-Sensitive L2D?")

    # Computed, not asserted: which tier gets the largest absolute percentage-point
    # increase in each dataset. Using pp-increase rather than a ratio avoids the
    # zero-denominator problem on PaySim, where Low tier moves from 0.00% to 0.02% --
    # a ratio there is literally undefined, not "smaller than Critical's".
    def largest_pp_increase_tier(d):
        deltas = {t: d["tiers"][t]["risk_sensitive"] - d["tiers"][t]["standard"] for t in d["tiers"]}
        top_tier = max(deltas, key=deltas.get)
        return top_tier, deltas[top_tier]

    critical_wins = 0
    for _dname, _d in DATASET_METRICS.items():
        _top_tier, _ = largest_pp_increase_tier(_d)
        if _top_tier == "Critical":
            critical_wins += 1

    kcols = st.columns(5)
    kpis = [
        ("4.6x", "ULB: Critical-tier deferral increase, Standard → Risk-Sensitive (headline run)"),
        ("6.6x", "IEEE-CIS: Critical-tier deferral increase, Standard → Risk-Sensitive (headline run)"),
        ("5.65x", "PaySim: Critical-tier deferral increase, Standard → Risk-Sensitive"),
        (
            f"{critical_wins} / {len(DATASET_METRICS)}",
            "datasets where Critical tier gets the largest percentage-point increase in deferral of any tier",
        ),
        ("1 term", "changed in the loss function — same architecture, retrained independently on each dataset"),
    ]
    for col, (number, label) in zip(kcols, kpis):
        with col:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-number'>{number}</div>"
                f"<div class='kpi-label'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown(
        """
        <div class="problem-box">
        <b>The problem, in one real number:</b> on IEEE-CIS, Standard L2D defers <b>more</b> on Low-tier
        transactions (12.74%) than on Critical-tier transactions (10.67%) — backwards from what
        risk-proportional human oversight should look like, and the exact structural flaw that motivated
        this thesis (Section 6.2.2). Risk-Sensitive L2D, trained on the same data with the same
        architecture, fixes it: Critical-tier deferral rises to <b>70.56%</b>, now correctly the highest
        of all four tiers, while Low rises only to 18.60%.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Deferral by risk tier — all three datasets")
    st.caption(
        "Standard L2D (blue) vs Risk-Sensitive L2D (amber), percentage of transactions deferred to a "
        "human, per risk tier. Each dataset has its own scale — ULB's effect is sub-1%, IEEE-CIS's is "
        "10-70%, PaySim's is 0-4% — because base fraud rates and transaction volumes differ hugely "
        "across the three. What matters is the shape: Critical tier should be the tallest amber bar."
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (dataset_name, d) in zip(axes, DATASET_METRICS.items()):
        tiers = list(d["tiers"].keys())
        std_vals = [d["tiers"][t]["standard"] for t in tiers]
        rs_vals = [d["tiers"][t]["risk_sensitive"] for t in tiers]
        x = np.arange(len(tiers))
        width = 0.35
        bars_std = ax.bar(x - width / 2, std_vals, width, label="Standard L2D", color=STANDARD_COLOR, edgecolor="none")
        bars_rs = ax.bar(x + width / 2, rs_vals, width, label="Risk-Sensitive L2D", color=RISK_COLOR, edgecolor="none")
        # Exact values printed on every bar -- each panel has its own y-axis scale
        # (ULB sub-1%, IEEE-CIS 10-70%, PaySim 0-4%), so the real numbers must be
        # readable directly rather than inferred from bar height alone.
        ax.bar_label(bars_std, fmt="%.2f", fontsize=6.5, color="#cbd5e1", padding=2)
        ax.bar_label(bars_rs, fmt="%.2f", fontsize=6.5, color="#fde68a", padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(tiers, fontsize=8.5)
        ax.set_ylabel("Deferral rate (%)", fontsize=9)
        ax.margins(y=0.15)
        ax.set_title(f"{dataset_name}\nCritical-tier ratio: {d['critical_headline_ratio']}", fontsize=10)
        if ax is axes[0]:
            ax.legend(fontsize=7.5, loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0")
    style_dark_fig(fig, axes)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Exact values are printed on every bar precisely because each panel's y-axis is independently "
        "scaled — read the numbers, not just the bar heights, when comparing across datasets."
    )

    st.markdown("### 2. Is this a fluke of one lucky training run?")
    st.caption(
        "No formal significance test was run (McNemar's test and confidence intervals were scoped out "
        "given the project timeline — see report Section 3.4). What was checked instead: retraining "
        "each model from scratch across multiple random seeds and confirming the direction holds every "
        "time. PaySim's multi-seed check was not run due to compute cost (Table 16)."
    )

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
    for ax, dataset_name in zip(axes2, ["ULB Credit Card Fraud", "IEEE-CIS"]):
        d = DATASET_METRICS[dataset_name]
        seeds = d["seeds"]
        ratios = d["seed_ratios"]
        ax.scatter(seeds, ratios, color=RISK_COLOR, s=70, zorder=3)
        mean_ratio = np.mean(ratios)
        ax.axhline(mean_ratio, color="#94a3b8", linestyle="--", linewidth=1.2, label=f"mean {mean_ratio:.1f}x")
        ax.set_xlabel("Training seed")
        ax.set_ylabel("Critical-tier deferral ratio (M2 → M3)")
        ax.set_title(f"{dataset_name}\n{len(seeds)} independent training runs", fontsize=10)
        ax.legend(fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0")
    style_dark_fig(fig2, axes2)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown(
        "On **ULB**, the direction held in all 5 seeds (ratio range 4.6x-24.7x), though 2 of those 5 "
        "seeds saw the underlying classifier itself collapse (F1 = 0.000) under severe class imbalance — "
        "a limitation of the small shared-trunk architecture, not of the risk-sensitive mechanism, which "
        "kept separating tiers correctly even then (Section 6.1.3). On **IEEE-CIS**, the result was "
        "steadier: ratio range 5.49x-6.78x across 4 seeds (Section 6.2.4)."
    )

    st.markdown("### 3. Does this cost prediction accuracy?")
    st.caption(
        "Honestly, mixed. ROC-AUC (how well the classifier ranks fraud vs normal, independent of any "
        "decision threshold) holds or improves on all three datasets. F1 (the classifier's precision/"
        "recall balance on autonomous decisions) dips slightly on ULB, since deferring more of the "
        "hardest cases removes some it would otherwise have guessed right on autonomously — but rises "
        "on IEEE-CIS and PaySim."
    )

    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4.2))
    dataset_names = list(DATASET_METRICS.keys())
    x = np.arange(len(dataset_names))
    width = 0.25
    for metric, ax in zip(["f1", "roc_auc"], axes3):
        baseline_vals = [DATASET_METRICS[d]["baseline"][metric] for d in dataset_names]
        standard_vals = [DATASET_METRICS[d]["standard"][metric] for d in dataset_names]
        rs_vals = [DATASET_METRICS[d]["risk_sensitive"][metric] for d in dataset_names]
        ax.bar(x - width, baseline_vals, width, label="Baseline", color=BASELINE_COLOR)
        ax.bar(x, standard_vals, width, label="Standard L2D", color=STANDARD_COLOR)
        ax.bar(x + width, rs_vals, width, label="Risk-Sensitive L2D", color=RISK_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels(["ULB", "IEEE-CIS", "PaySim"], fontsize=9)
        ax.set_title("F1 Score (autonomous decisions)" if metric == "f1" else "ROC-AUC", fontsize=10)
        if metric == "f1":
            ax.legend(fontsize=7.5, facecolor="#1e293b", labelcolor="#e2e8f0")
    style_dark_fig(fig3, axes3)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.markdown("### Before you go further: what this page doesn't show")
    st.markdown(
        """
        <div class="intro-box">
        This page is the headline evidence, not the full account. Three things the charts above don't
        make visible: each dataset needed its <b>own hand-tuned cost configuration</b> (ULB uses risk
        weights 1.0/0.8/0.5/0.2 with base cost 0.13; IEEE-CIS uses gentler weights 1.0/0.9/0.7/0.5 with
        base cost 0.63 — not one universal setting that works everywhere, Section 4.6). <b>PaySim's
        calibration was fragile</b> — a systematic sweep across base costs found the model's behaviour
        collapsed into deferring almost everything or almost nothing for most values tried before a usable
        setting was found (Section 6.3, Discussion). And <b>2 of 5 ULB training seeds saw the underlying
        classifier collapse entirely</b> (F1 = 0.000), even though the deferral mechanism itself still
        separated tiers correctly in those runs (Section 6.1.3). None of this contradicts the headline
        result — the direction held in every seed and every dataset tested — but the full report treats
        these results with more caution than this page does, and that caution is deliberate, not an
        oversight. See Sections 4.6, 6.1.3-6.1.4, and the PaySim discussion in 6.3 for the complete picture.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="cta-box">
        Curious how this actually plays out, transaction by transaction? Switch to the
        <b>🔍 ULB Transaction Console</b> tab above to see this mechanism operate live on real, individual
        ULB transactions — including a real fraud case where it made the difference.
        <br><br>
        <span style="font-size:0.82rem; color:#94a3b8;">Note: that console explores ULB only. The
        IEEE-CIS and PaySim results above are the report's figures, not a live, clickable build for
        those two datasets.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# PAGE 2 — the existing interactive ULB console
# ============================================================================
with page_console:
    st.markdown(
        """
        <div class="intro-box">
        <b>What you're looking at, in plain terms:</b> two AI systems were trained on the same real fraud
        data. Both look at a transaction and choose one of two things: handle it automatically, or send it
        to a human reviewer. The <i>only</i> difference between them is how much the transaction's size
        matters to that choice. Standard L2D treats a €5 purchase and a €50,000 purchase the same way.
        Risk-Sensitive L2D (this thesis) does not. The example below is a real transaction where that
        difference changed what happened.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "## ⭐ The case this thesis is about",
    )
    st.markdown(
        "<span class='pill pill-flagship'>REAL TRANSACTION · CONFIRMED FRAUD</span>",
        unsafe_allow_html=True,
    )
    st.write("")

    fc = flagship_case
    fc_x = build_input_vector(fc["time"], fc["v_values"], float(fc["amount"]), scaler_stats)
    fc_r2 = run_inference(standard_model, fc_x)
    fc_r3 = run_inference(risk_sensitive_model, fc_x)

    st.markdown(
        f"### €{fc['amount']:,.2f} transaction &nbsp;"
        f"<span class='tier-chip' style='background:{TIER_COLOR[fc['tier']]}'>{fc['tier']} tier</span> &nbsp;"
        f"<span class='chip chip-fraud'>ACTUAL: FRAUD</span>",
        unsafe_allow_html=True,
    )

    fc_col1, fc_col2 = st.columns(2)
    render_decision_card(fc_col1, "Standard L2D (uniform cost)", fc_r2)
    render_decision_card(fc_col2, "Risk-Sensitive L2D (this thesis)", fc_r3)

    st.markdown(
        f"""
    <div class="flagship-box">
    Both models were almost equally sure this was fraud: Standard L2D was <b>{fc_r2['fraud_probability']*100:.1f}% confident</b>,
    Risk-Sensitive L2D was <b>{fc_r3['fraud_probability']*100:.1f}% confident</b> — essentially the same read on the transaction.
    Standard L2D processed it <b>completely automatically anyway. No human ever saw it. No audit trail.</b>
    Risk-Sensitive L2D sent it to a human reviewer, <b>purely because of the €{fc['amount']:,.0f} size of the transaction</b>,
    not because it was any less sure. That is the entire mechanism this thesis proposes: not smarter fraud
    detection (Standard L2D actually scores slightly higher on raw accuracy, Table 3), but <b>proportional human
    oversight</b> — the bigger the transaction, the more a human gets a say, regardless of how confident the
    model already is. This is the real-world case for EU AI Act Article 14 human oversight made concrete.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Explore more transactions yourself")
    st.caption(
        "The example above is one fixed, curated case. The two tabs below let you check more real "
        "transactions and see this same mechanism play out (or not) elsewhere in the data."
    )

    tab_review, tab_explorer = st.tabs(["🔍 Browse real transactions", "🎚 Same pattern, different amount"])

    CATEGORY_ORDER = [
        "baseline_agree", "only_m2_defers", "only_m3_defers", "both_defer",
        "classification_disagreement", "both_miss_fraud", "false_alarm", "edge_boundary",
    ]
    CATEGORY_INFO = {
        "baseline_agree": {
            "name": "✅ Both models agree (baseline)",
            "desc": "Both models make the same call, no dispute — what most real transactions look like. "
                    "Included so you see the ordinary case, not only the interesting ones.",
        },
        "only_m2_defers": {
            "name": "🟦 Standard L2D defers, Risk-Sensitive doesn't",
            "desc": "Standard L2D is a real Learning-to-Defer model, not a plain classifier — it has its own "
                    "deferral head and does send some cases to a human. Every real example of this in the "
                    "dataset turns out to be an actual normal transaction the classifier found hard to read, "
                    "never a real fraud — Standard L2D's deferrals are driven by classification difficulty, "
                    "not by transaction risk.",
        },
        "only_m3_defers": {
            "name": "🟥 Risk-Sensitive L2D defers, Standard doesn't",
            "desc": "The mechanism this thesis proposes: Risk-Sensitive L2D routes a case to a human "
                    "specifically because of its size, in situations Standard L2D would just process automatically.",
        },
        "both_defer": {
            "name": "🤝 Both models defer to a human",
            "desc": "Both policies agree a human should look at this one.",
        },
        "classification_disagreement": {
            "name": "🔀 Models disagree on Fraud vs Normal itself",
            "desc": "These two networks were trained separately and don't always read a transaction the same "
                    "way, independent of deferral. Ordinary variance between two trained models, not part of "
                    "the risk-sensitive mechanism.",
        },
        "both_miss_fraud": {
            "name": "⚠️ Both models miss a real fraud",
            "desc": "The most important failure mode to show honestly: real frauds both models auto-approved "
                    "and got wrong. Risk-sensitive weighting does not fix this — it changes who reviews what, "
                    "not how good the underlying classifier is.",
        },
        "false_alarm": {
            "name": "🚨 Both flag a normal transaction as fraud",
            "desc": "Real, legitimate transactions both models auto-blocked as fraud — the other kind of "
                    "error, a false alarm rather than a missed fraud.",
        },
        "edge_boundary": {
            "name": "📏 Edge cases: tier boundaries & extremes",
            "desc": "Real transactions sitting right at a tier threshold (€22 / €77 / €500) on either side, "
                    "plus the smallest and largest real amounts in the dataset.",
        },
    }

    with tab_review:
        st.caption(
            f"{len(samples)} real transactions from the held-out test set, organised into named categories "
            "covering every scenario the two policies can produce — not just the interesting ones."
        )

        available_categories = [c for c in CATEGORY_ORDER if c in samples["scenario"].unique()]
        category = st.selectbox(
            "Scenario category",
            options=available_categories,
            format_func=lambda c: f"{CATEGORY_INFO[c]['name']}  ({(samples['scenario'] == c).sum()})",
        )
        st.caption(CATEGORY_INFO[category]["desc"])

        def label_row(row):
            actual = "Fraud" if row["Class"] == 1 else "Normal"
            return f"{row['Risk_Tier']:<8} | €{row['Amount']:>10,.2f} | actual: {actual}"

        samples_r = samples[samples["scenario"] == category].reset_index(drop=True)
        labels = [label_row(r) for _, r in samples_r.iterrows()]
        choice = st.selectbox("Choose a transaction", options=range(len(samples_r)), format_func=lambda i: labels[i])
        row = samples_r.iloc[choice]

        v_values = {f"V{i}": row[f"V{i}"] for i in range(1, 29)}
        amount = float(row["Amount"])
        time_val = float(row["Time"])
        actual_label = "Fraud" if row["Class"] == 1 else "Normal"
        x = build_input_vector(time_val, v_values, amount, scaler_stats)
        tier = amount_to_tier(amount)

        render_transaction_header(amount, tier)

        r2 = run_inference(standard_model, x)
        r3 = run_inference(risk_sensitive_model, x)

        col1, col2 = st.columns(2)
        render_decision_card(col1, "Standard L2D (uniform cost)", r2)
        render_decision_card(col2, "Risk-Sensitive L2D (this thesis)", r3)

        render_divergence(r2, r3)
        render_tier_context_note(tier, r3)

        with st.expander("Reveal actual outcome (ground truth)"):
            st.write(f"This transaction was actually **{actual_label}** in the dataset.")

    with tab_explorer:
        st.caption(
            "Take one real transaction's underlying behaviour and slide its amount up or down — everything "
            "else about the transaction stays fixed. Watch how each model's decision changes as the amount alone changes."
        )
        profile_label = st.selectbox("Behaviour pattern (taken from a real transaction)", options=list(borderline_profiles.keys()))
        profile = borderline_profiles[profile_label]
        st.caption(
            f"This pattern originally appeared as a €{profile['original_amount']:,.2f} transaction "
            f"({profile['original_tier']} tier). Everything about it is held fixed except the amount below."
        )
        amount = st.slider("Transaction amount (€)", min_value=1, max_value=2000, value=int(profile["original_amount"]), step=1)
        x = build_input_vector(profile["time"], profile["v_values"], float(amount), scaler_stats)
        tier = amount_to_tier(amount)

        render_transaction_header(amount, tier)

        r2 = run_inference(standard_model, x)
        r3 = run_inference(risk_sensitive_model, x)

        col1, col2 = st.columns(2)
        render_decision_card(col1, "Standard L2D (uniform cost)", r2)
        render_decision_card(col2, "Risk-Sensitive L2D (this thesis)", r3)

        render_divergence(r2, r3)
        render_profile_sensitivity_note(profile, tier)

        st.caption(
            "Note: the deferral probability does not move smoothly or monotonically as amount changes -- "
            "an empirical property of this small neural network found while building this console, not a "
            "claim made in the report itself. Try sliding slowly through the full range to see it."
        )

    with st.expander("Technical detail: how the cost difference actually works"):
        st.write(
            "Both models share the same architecture and were trained on the same ULB Credit Card Fraud data. "
            "The only difference is the deferral cost: Standard L2D uses one fixed cost for every transaction; "
            "Risk-Sensitive L2D scales that cost down for higher risk tiers, so deferring a Critical-tier case is "
            "cheaper (and therefore more likely) than deferring a Low-tier one. This console lets you see that "
            "difference operate on individual transactions instead of only as an aggregate percentage in a table."
        )

    st.markdown("---")
    st.markdown("### Product roadmap")
    st.caption("What exists today versus what is proposed future work. See the report's Future Work section for the full reasoning.")

    rcol1, rcol2, rcol3, rcol4 = st.columns(4)
    with rcol1:
        st.markdown(
            "<div class='roadmap-card'><span class='pill pill-live'>LIVE</span><br><br>"
            "<b>Policy comparison engine</b><br><br>"
            "The console you're using: two trained deferral policies scored side by side on the same "
            "transaction, in real time.</div>",
            unsafe_allow_html=True,
        )
    with rcol2:
        st.markdown(
            "<div class='roadmap-card'><span class='pill pill-planned'>PLANNED — not implemented</span><br><br>"
            "<b>Diagnose</b><br><br>"
            "Upload an organisation's own model decisions and transaction data to check whether its current "
            "human-review policy is risk-proportional.</div>",
            unsafe_allow_html=True,
        )
    with rcol3:
        st.markdown(
            "<div class='roadmap-card'><span class='pill pill-planned'>PLANNED — not implemented</span><br><br>"
            "<b>Calibrate</b><br><br>"
            "Compute a risk matrix and weighting scheme from the organisation's own transaction distribution, "
            "sized to its reviewer capacity.</div>",
            unsafe_allow_html=True,
        )
    with rcol4:
        st.markdown(
            "<div class='roadmap-card'><span class='pill pill-planned'>PLANNED — not implemented</span><br><br>"
            "<b>Monitor</b><br><br>"
            "An ongoing dashboard tracking whether risk-proportionality holds as transaction patterns drift.</div>",
            unsafe_allow_html=True,
        )
