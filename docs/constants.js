/* Static content transcribed from app.py (the Streamlit prototype this page replaces).
 * Every figure in DATASET_METRICS is taken directly from the report's Evaluation
 * chapter (Tables 3, 4, 5, 8, 9, 10, 12, 13, 16) -- nothing here is recomputed live. */

const TIER_COLOR = {
  Low: "#2ecc71",
  Medium: "#f1c40f",
  High: "#e67e22",
  Critical: "#e74c3c",
};
const BASELINE_COLOR = "#64748b";
const STANDARD_COLOR = "#60a5fa";
const RISK_COLOR = "#f59e0b";

const FULL_DATASET_SIZE = 284807;
const CATEGORY_REAL_FREQUENCY = {
  only_m2_defers: 107,
  only_m3_defers: 550,
  both_defer: 610,
  classification_disagreement: 342,
  both_miss_fraud: 14,
  false_alarm: 101,
};

const DATASET_METRICS = {
  "ULB Credit Card Fraud": {
    baseline: { f1: 0.1094, roc_auc: 0.9698 },
    standard: { f1: 0.7736, roc_auc: 0.9787 },
    risk_sensitive: { f1: 0.7378, roc_auc: 0.98 },
    tiers: {
      Low: { standard: 0.28, risk_sensitive: 0.42 },
      Medium: { standard: 0.18, risk_sensitive: 0.22 },
      High: { standard: 0.25, risk_sensitive: 0.49 },
      Critical: { standard: 0.21, risk_sensitive: 0.96 },
    },
    critical_headline_ratio: "4.6x",
    seeds: [42, 43, 44, 45, 46],
    seed_ratios: [4.6, 24.7, 6.2, 23.5, 19.1],
  },
  "IEEE-CIS": {
    baseline: { f1: 0.343, roc_auc: 0.905 },
    standard: { f1: 0.4, roc_auc: 0.913 },
    risk_sensitive: { f1: 0.559, roc_auc: 0.944 },
    tiers: {
      Low: { standard: 12.74, risk_sensitive: 18.6 },
      Medium: { standard: 3.6, risk_sensitive: 13.62 },
      High: { standard: 8.68, risk_sensitive: 41.94 },
      Critical: { standard: 10.67, risk_sensitive: 70.56 },
    },
    critical_headline_ratio: "6.6x",
    seeds: [42, 43, 44, 45],
    seed_ratios: [6.61, 5.73, 6.78, 5.49],
  },
  PaySim: {
    baseline: { f1: 0.648, roc_auc: 0.998 },
    standard: { f1: 0.599, roc_auc: 0.999 },
    risk_sensitive: { f1: 0.617, roc_auc: 0.999 },
    tiers: {
      Low: { standard: 0.0, risk_sensitive: 0.02 },
      Medium: { standard: 0.44, risk_sensitive: 0.51 },
      High: { standard: 0.69, risk_sensitive: 0.74 },
      Critical: { standard: 0.68, risk_sensitive: 3.84 },
    },
    critical_headline_ratio: "5.65x",
    seeds: null,
    seed_ratios: null,
  },
};

const TIER_STATS = {
  Low: {
    m2: "0.28%",
    m3: "0.42%",
    ratio: "1.5x",
    note:
      "Low tier carries no cost reduction in this design (risk weight 1.0, unchanged from Standard " +
      "L2D), so it's expected to barely move — this small residual shift reflects incidental " +
      "shared-trunk training variation, not the mechanism this thesis is about.",
  },
  Medium: {
    m2: "0.18%",
    m3: "0.22%",
    ratio: "1.2x",
    note:
      "Medium tier gets a modest cost reduction (risk weight 0.8), and shows a correspondingly " +
      "modest increase — smaller than High or Critical, as the graded design intends.",
  },
  High: {
    m2: "0.25%",
    m3: "0.49%",
    ratio: "2.0x",
    note:
      "High tier gets a larger cost reduction (risk weight 0.5), producing a bigger increase than " +
      "Medium but smaller than Critical, sitting where the graded weighting is designed to put it.",
  },
  Critical: {
    m2: "0.21%",
    m3: "0.96%",
    ratio: "4.6x",
    note:
      "Critical tier gets the largest cost reduction (risk weight 0.2), producing the largest " +
      "relative increase — though that ratio itself ranged 4.6x-24.7x across five training seeds, " +
      "and needs roughly a 4-5x deferral-cost gradient between tiers to show up at all " +
      "(Sections 6.1.3-6.1.4).",
  },
};

const CATEGORY_ORDER = [
  "baseline_agree",
  "only_m2_defers",
  "only_m3_defers",
  "both_defer",
  "classification_disagreement",
  "both_miss_fraud",
  "false_alarm",
  "edge_boundary",
];

const CATEGORY_INFO = {
  baseline_agree: {
    name: "✅ Both models agree (baseline)",
    desc:
      "Both models make the same call, no dispute — what most real transactions look like. " +
      "Included so you see the ordinary case, not only the interesting ones.",
  },
  only_m2_defers: {
    name: "🟦 Standard L2D defers, Risk-Sensitive doesn't",
    desc:
      "Standard L2D is a real Learning-to-Defer model, not a plain classifier — it has its own " +
      "deferral head and does send some cases to a human. Every real example of this in the " +
      "dataset turns out to be an actual normal transaction the classifier found hard to read, " +
      "never a real fraud — Standard L2D's deferrals are driven by classification difficulty, " +
      "not by transaction risk.",
  },
  only_m3_defers: {
    name: "🟥 Risk-Sensitive L2D defers, Standard doesn't",
    desc:
      "The mechanism this thesis proposes: Risk-Sensitive L2D routes a case to a human " +
      "specifically because of its size, in situations Standard L2D would just process automatically.",
  },
  both_defer: {
    name: "🤝 Both models defer to a human",
    desc: "Both policies agree a human should look at this one.",
  },
  classification_disagreement: {
    name: "🔀 Models disagree on Fraud vs Normal itself",
    desc:
      "These two networks were trained separately and don't always read a transaction the same " +
      "way, independent of deferral. Ordinary variance between two trained models, not part of " +
      "the risk-sensitive mechanism.",
  },
  both_miss_fraud: {
    name: "⚠️ Both models miss a real fraud",
    desc:
      "The most important failure mode to show honestly: real frauds both models auto-approved " +
      "and got wrong. Risk-sensitive weighting does not fix this — it changes who reviews what, " +
      "not how good the underlying classifier is.",
  },
  false_alarm: {
    name: "🚨 Both flag a normal transaction as fraud",
    desc:
      "Real, legitimate transactions both models auto-blocked as fraud — the other kind of " +
      "error, a false alarm rather than a missed fraud.",
  },
  edge_boundary: {
    name: "📏 Edge cases: tier boundaries & extremes",
    desc:
      "Real transactions sitting right at a tier threshold (€22 / €77 / €500) on either side, " +
      "plus the smallest and largest real amounts in the dataset.",
  },
};
