const B = window.TIERSENSE_BUNDLE;
const scalerStats = B.scalerStats;
const standardWeights = B.standardWeights;
const riskWeights = B.riskWeights;

const FULL_DATASET_SIZE_FMT = FULL_DATASET_SIZE.toLocaleString("en-US");

/* ================= tab switching ================= */
function wireTabs(barSelector, panelPrefixAttr) {
  document.querySelectorAll(barSelector + " .tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const bar = btn.closest(".tabbar");
      bar.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const key = btn.dataset[panelPrefixAttr];
      const isSub = panelPrefixAttr === "subtab";
      const panels = document.querySelectorAll(isSub ? "[id^='subtab-']" : "[id^='tab-']:not([id^='subtab-'])");
      panels.forEach((p) => {
        const match = isSub ? p.id === `subtab-${key}` : p.id === `tab-${key}`;
        if (isSub) p.style.display = match ? "block" : "none";
        p.classList.toggle("active", match);
      });
    });
  });
}
wireTabs(".tabbar:not([style])", "tab"); // main tabbar (no inline style)
document.querySelectorAll(".tabbar").forEach((bar) => {
  const btns = bar.querySelectorAll(".tab-btn");
  if (btns.length && btns[0].dataset.subtab) {
    wireSubtabs(bar);
  }
});
function wireSubtabs(bar) {
  bar.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      bar.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const key = btn.dataset.subtab;
      document.getElementById("subtab-review").style.display = key === "review" ? "block" : "none";
      document.getElementById("subtab-explorer").style.display = key === "explorer" ? "block" : "none";
    });
  });
}

/* ================= sidebar ================= */
const samples = B.sampleTransactions;
document.getElementById("sidebar-demo-disclosure").innerHTML =
  `The <b>Browse real transactions</b> tab holds ${samples.length} real transactions, deliberately ` +
  `picked to cover 8 named scenario categories (both agree, only one defers in either direction, ` +
  `both defer, models disagree on the label, both miss a fraud, false alarms, boundary edge ` +
  `cases) — <b>not a random sample</b>. Each category is rare or common in very different ways in ` +
  `the full ${FULL_DATASET_SIZE_FMT}-row dataset, e.g. only-Standard-defers cases: ` +
  `${CATEGORY_REAL_FREQUENCY.only_m2_defers} (${(CATEGORY_REAL_FREQUENCY.only_m2_defers / FULL_DATASET_SIZE * 100).toFixed(3)}%); ` +
  `only-Risk-Sensitive-defers cases: ${CATEGORY_REAL_FREQUENCY.only_m3_defers} ` +
  `(${(CATEGORY_REAL_FREQUENCY.only_m3_defers / FULL_DATASET_SIZE * 100).toFixed(3)}%); ` +
  `both-models-miss-a-fraud cases: only ${CATEGORY_REAL_FREQUENCY.both_miss_fraud} in the whole ` +
  `dataset. The category picker on that tab shows how many real examples exist for each.`;

/* ================= decision card + divergence rendering ================= */
function renderDecisionCard(container, title, result, tier, policyId) {
  const predClass = result.predictedLabel === "Fraud" ? "chip-fraud" : "chip-normal";
  const decisionHtml = result.deferred
    ? `<div class="decision-line-defer">🚨 ${result.decision}</div>`
    : `<div class="decision-line-auto">✅ ${result.decision}</div>`;
  const reasonHtml =
    tier != null
      ? `<div class="reason-code">reason: risk_score=${(result.fraudProbability * 100).toFixed(1)}% ` +
        `· tier=${tier} · deferral_prob=${(result.deferProbability * 100).toFixed(1)}% ` +
        `· policy=${policyId}</div>`
      : "";
  container.innerHTML = `
    <div class="card">
      <b class="card-title">${title}</b><br/><br/>
      <span class="chip ${predClass}">PREDICTED: ${result.predictedLabel.toUpperCase()}</span>
      <div class="prob-label">Fraud probability — ${(result.fraudProbability * 100).toFixed(1)}%</div>
      <div class="prob-bar"><div class="prob-bar-fill" style="width:${Math.min(result.fraudProbability, 1) * 100}%"></div></div>
      <div class="prob-label">Deferral probability — ${(result.deferProbability * 100).toFixed(1)}%</div>
      <div class="prob-bar"><div class="prob-bar-fill" style="width:${Math.min(result.deferProbability, 1) * 100}%"></div></div>
      ${decisionHtml}
      ${reasonHtml}
    </div>`;
}

function renderDivergence(container, r2, r3) {
  if (r2.deferred !== r3.deferred) {
    const msg = r3.deferred
      ? "Standard L2D would let it through automatically; Risk-Sensitive L2D sends it to a human reviewer."
      : "Risk-Sensitive L2D would let it through automatically; Standard L2D sends it to a human reviewer.";
    container.innerHTML = `<div class="warn-box"><b>These two policies disagree on this exact transaction.</b> ${msg}</div>`;
  } else {
    container.innerHTML = `<div class="info-box">Both policies make the same call on this transaction.</div>`;
  }
}

function tierHeaderHtml(amount, tier) {
  return `Transaction: €${amount.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} &nbsp;` +
    `<span class="tier-chip" style="background:${TIER_COLOR[tier]}">${tier} tier</span>`;
}

function tierContextNote(tier, r3) {
  if (r3.deferred) return "";
  const s = TIER_STATS[tier];
  return `Why is this still AUTO-DECIDED? On the full ULB test set, ${tier} tier is deferred ${s.m3} of ` +
    `the time under Risk-Sensitive L2D (vs ${s.m2} under Standard L2D, a ${s.ratio} change) — so ` +
    `seeing AUTO-DECIDED here is the typical case, not an exception. ${s.note} See report Section ` +
    `6.1.2 for the full tier breakdown.`;
}

/* ================= EVIDENCE TAB ================= */
function largestPpIncreaseTier(d) {
  let topTier = null;
  let topDelta = -Infinity;
  for (const t in d.tiers) {
    const delta = d.tiers[t].risk_sensitive - d.tiers[t].standard;
    if (delta > topDelta) {
      topDelta = delta;
      topTier = t;
    }
  }
  return topTier;
}
let criticalWins = 0;
for (const dname in DATASET_METRICS) {
  if (largestPpIncreaseTier(DATASET_METRICS[dname]) === "Critical") criticalWins++;
}

const kpis = [
  ["4.6x", "ULB: Critical-tier deferral increase, Standard → Risk-Sensitive (headline run)"],
  ["6.6x", "IEEE-CIS: Critical-tier deferral increase, Standard → Risk-Sensitive (headline run)"],
  ["5.65x", "PaySim: Critical-tier deferral increase, Standard → Risk-Sensitive"],
  [`${criticalWins} / ${Object.keys(DATASET_METRICS).length}`, "datasets where Critical tier gets the largest percentage-point increase in deferral of any tier"],
  ["1 term", "changed in the loss function — same architecture, retrained independently on each dataset"],
];
const kpiRow = document.getElementById("kpi-row");
kpis.forEach(([number, label]) => {
  const div = document.createElement("div");
  div.className = "kpi-card";
  div.innerHTML = `<div class="kpi-number">${number}</div><div class="kpi-label">${label}</div>`;
  kpiRow.appendChild(div);
});

/* tier charts (3 panels) */
const tierChartsContainer = document.getElementById("tier-charts");
Object.entries(DATASET_METRICS).forEach(([dname, d]) => {
  const wrap = document.createElement("div");
  wrap.className = "chart-wrap";
  tierChartsContainer.appendChild(wrap);
  const tiers = Object.keys(d.tiers);
  renderGroupedBarChart(wrap, {
    title: `${dname}\nCritical-tier ratio: ${d.critical_headline_ratio}`,
    categories: tiers,
    series: [
      { label: "Standard L2D", values: tiers.map((t) => d.tiers[t].standard), color: STANDARD_COLOR },
      { label: "Risk-Sensitive L2D", values: tiers.map((t) => d.tiers[t].risk_sensitive), color: RISK_COLOR },
    ],
    yLabel: "Deferral rate (%)",
    valueFmt: (v) => v.toFixed(2),
    legend:
      dname === "ULB Credit Card Fraud"
        ? [
            { label: "Standard L2D", color: STANDARD_COLOR },
            { label: "Risk-Sensitive L2D", color: RISK_COLOR },
          ]
        : null,
  });
});

/* seed scatter charts (2 panels) */
const seedChartsContainer = document.getElementById("seed-charts");
["ULB Credit Card Fraud", "IEEE-CIS"].forEach((dname) => {
  const d = DATASET_METRICS[dname];
  const wrap = document.createElement("div");
  wrap.className = "chart-wrap";
  seedChartsContainer.appendChild(wrap);
  renderScatterChart(wrap, {
    title: `${dname}\n${d.seeds.length} independent training runs`,
    xValues: d.seeds,
    yValues: d.seed_ratios,
    color: RISK_COLOR,
    xLabel: "Training seed",
  });
});

/* F1 / ROC-AUC comparison (2 panels, 3 series) */
const f1AucContainer = document.getElementById("f1-auc-charts");
const datasetNames = Object.keys(DATASET_METRICS);
["f1", "roc_auc"].forEach((metric) => {
  const wrap = document.createElement("div");
  wrap.className = "chart-wrap";
  f1AucContainer.appendChild(wrap);
  renderGroupedBarChart(wrap, {
    title: metric === "f1" ? "F1 Score (autonomous decisions)" : "ROC-AUC",
    categories: ["ULB", "IEEE-CIS", "PaySim"],
    series: [
      { label: "Baseline", values: datasetNames.map((d) => DATASET_METRICS[d].baseline[metric]), color: BASELINE_COLOR },
      { label: "Standard L2D", values: datasetNames.map((d) => DATASET_METRICS[d].standard[metric]), color: STANDARD_COLOR },
      { label: "Risk-Sensitive L2D", values: datasetNames.map((d) => DATASET_METRICS[d].risk_sensitive[metric]), color: RISK_COLOR },
    ],
    valueFmt: (v) => v.toFixed(3),
    legend:
      metric === "f1"
        ? [
            { label: "Baseline", color: BASELINE_COLOR },
            { label: "Standard L2D", color: STANDARD_COLOR },
            { label: "Risk-Sensitive L2D", color: RISK_COLOR },
          ]
        : null,
  });
});

/* weight sensitivity selector */
const weightConfigs = B.weightSensitivity;
const weightSelect = document.getElementById("weight-config-select");
Object.keys(weightConfigs).forEach((name) => {
  const opt = document.createElement("option");
  opt.value = name;
  opt.textContent = name;
  weightSelect.appendChild(opt);
});
function renderWeightConfig() {
  const configName = weightSelect.value;
  const cfg = weightConfigs[configName];
  document.getElementById("weight-config-weights").innerHTML =
    `<b>Risk weights</b> — Low ${cfg.weights.Low.toFixed(2)} · Medium ${cfg.weights.Medium.toFixed(2)} · ` +
    `High ${cfg.weights.High.toFixed(2)} · Critical ${cfg.weights.Critical.toFixed(2)}`;
  document.getElementById("weight-config-f1").textContent = cfg.f1.toFixed(4);
  document.getElementById("weight-config-ratio").textContent = `${cfg.critical_over_low_ratio.toFixed(2)}x`;

  const original = weightConfigs["original (thesis)"];
  const tiers = Object.keys(cfg.tier_rates);
  const chartDiv = document.getElementById("weight-config-chart");
  renderGroupedBarChart(chartDiv, {
    title: "Deferral rate by tier — this configuration vs the thesis's own",
    categories: tiers,
    series: [
      { label: "original (thesis)", values: tiers.map((t) => original.tier_rates[t]), color: BASELINE_COLOR },
      { label: configName, values: tiers.map((t) => cfg.tier_rates[t]), color: RISK_COLOR },
    ],
    yLabel: "Deferral rate (%)",
    valueFmt: (v) => v.toFixed(2),
    legend: [
      { label: "original (thesis)", color: BASELINE_COLOR },
      { label: configName, color: RISK_COLOR },
    ],
  });
}
weightSelect.addEventListener("change", renderWeightConfig);
renderWeightConfig();

/* ================= CONSOLE TAB: flagship case ================= */
const fc = B.flagshipCase;
const fcX = buildInputVector(fc.time, fc.v_values, fc.amount, scalerStats);
const fcR2 = runInference(standardWeights, fcX);
const fcR3 = runInference(riskWeights, fcX);

document.getElementById("flagship-header").innerHTML =
  `€${fc.amount.toLocaleString("en-US", { minimumFractionDigits: 2 })} transaction &nbsp;` +
  `<span class="tier-chip" style="background:${TIER_COLOR[fc.tier]}">${fc.tier} tier</span> &nbsp;` +
  `<span class="chip chip-fraud">ACTUAL: FRAUD</span>`;

const fcCards = document.getElementById("flagship-cards");
const fcCol1 = document.createElement("div");
const fcCol2 = document.createElement("div");
fcCards.appendChild(fcCol1);
fcCards.appendChild(fcCol2);
renderDecisionCard(fcCol1, "Standard L2D (uniform cost)", fcR2, fc.tier, "standard-l2d");
renderDecisionCard(fcCol2, "Risk-Sensitive L2D (this thesis)", fcR3, fc.tier, "risk-sensitive-l2d");

document.getElementById("flagship-narrative").innerHTML =
  `Both models were almost equally sure this was fraud: Standard L2D was <b>${(fcR2.fraudProbability * 100).toFixed(1)}% confident</b>, ` +
  `Risk-Sensitive L2D was <b>${(fcR3.fraudProbability * 100).toFixed(1)}% confident</b> — essentially the same read on the transaction. ` +
  `Standard L2D processed it <b>completely automatically anyway. No human ever saw it. No audit trail.</b> ` +
  `Risk-Sensitive L2D sent it to a human reviewer, <b>purely because of the €${fc.amount.toFixed(0)} size of the transaction</b>, ` +
  `not because it was any less sure. That is the entire mechanism this thesis proposes: not smarter fraud ` +
  `detection (Standard L2D actually scores slightly higher on raw accuracy, Table 3), but <b>proportional human ` +
  `oversight</b> — the bigger the transaction, the more a human gets a say, regardless of how confident the ` +
  `model already is. This is the real-world case for EU AI Act Article 14 human oversight made concrete.`;

/* ================= Browse real transactions ================= */
document.getElementById("review-caption").textContent =
  `${samples.length} real transactions from the held-out test set, organised into named categories ` +
  `covering every scenario the two policies can produce — not just the interesting ones.`;

const availableCategories = CATEGORY_ORDER.filter((c) => samples.some((s) => s.scenario === c));
const categorySelect = document.getElementById("category-select");
availableCategories.forEach((c) => {
  const count = samples.filter((s) => s.scenario === c).length;
  const opt = document.createElement("option");
  opt.value = c;
  opt.textContent = `${CATEGORY_INFO[c].name}  (${count})`;
  categorySelect.appendChild(opt);
});

const transactionSelect = document.getElementById("transaction-select");

function labelRow(row) {
  const actual = row.Class === 1 ? "Fraud" : "Normal";
  return `${row.Risk_Tier.padEnd(8, " ")} | €${row.Amount.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }).padStart(10, " ")} | actual: ${actual}`;
}

function populateTransactionSelect() {
  const category = categorySelect.value;
  document.getElementById("category-desc").textContent = CATEGORY_INFO[category].desc;
  const rows = samples.filter((s) => s.scenario === category);
  transactionSelect.innerHTML = "";
  rows.forEach((row, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = labelRow(row);
    transactionSelect.appendChild(opt);
  });
  renderReviewTransaction();
}

function renderReviewTransaction() {
  const category = categorySelect.value;
  const rows = samples.filter((s) => s.scenario === category);
  const row = rows[Number(transactionSelect.value)];
  if (!row) return;
  const amount = row.Amount;
  const tier = amountToTier(amount);
  const x = buildInputVector(row.Time, row.V, amount, scalerStats);
  const r2 = runInference(standardWeights, x);
  const r3 = runInference(riskWeights, x);

  document.getElementById("review-tx-header").innerHTML = tierHeaderHtml(amount, tier);
  const cardsDiv = document.getElementById("review-cards");
  cardsDiv.innerHTML = "";
  const c1 = document.createElement("div");
  const c2 = document.createElement("div");
  cardsDiv.appendChild(c1);
  cardsDiv.appendChild(c2);
  renderDecisionCard(c1, "Standard L2D (uniform cost)", r2, tier, "standard-l2d");
  renderDecisionCard(c2, "Risk-Sensitive L2D (this thesis)", r3, tier, "risk-sensitive-l2d");
  renderDivergence(document.getElementById("review-divergence"), r2, r3);
  document.getElementById("review-tier-note").textContent = tierContextNote(tier, r3);

  const actualLabel = row.Class === 1 ? "Fraud" : "Normal";
  document.getElementById("review-ground-truth").textContent = `This transaction was actually ${actualLabel} in the dataset.`;
}

categorySelect.addEventListener("change", populateTransactionSelect);
transactionSelect.addEventListener("change", renderReviewTransaction);
populateTransactionSelect();

/* ================= Same pattern, different amount ================= */
const borderlineProfiles = B.borderlineProfiles;
const profileSelect = document.getElementById("profile-select");
Object.keys(borderlineProfiles).forEach((name) => {
  const opt = document.createElement("option");
  opt.value = name;
  opt.textContent = name;
  profileSelect.appendChild(opt);
});

const amountSlider = document.getElementById("amount-slider");
const amountSliderValue = document.getElementById("amount-slider-value");

function onProfileChange() {
  const profile = borderlineProfiles[profileSelect.value];
  document.getElementById("profile-desc").textContent =
    `This pattern originally appeared as a €${profile.original_amount.toLocaleString("en-US", { minimumFractionDigits: 2 })} transaction ` +
    `(${profile.original_tier} tier). Everything about it is held fixed except the amount below.`;
  amountSlider.value = Math.round(profile.original_amount);
  amountSliderValue.textContent = amountSlider.value;
  renderExplorerTransaction();
}

function renderExplorerTransaction() {
  const profile = borderlineProfiles[profileSelect.value];
  const amount = Number(amountSlider.value);
  amountSliderValue.textContent = amount;
  const tier = amountToTier(amount);
  const x = buildInputVector(profile.time, profile.v_values, amount, scalerStats);
  const r2 = runInference(standardWeights, x);
  const r3 = runInference(riskWeights, x);

  document.getElementById("explorer-tx-header").innerHTML = tierHeaderHtml(amount, tier);
  const cardsDiv = document.getElementById("explorer-cards");
  cardsDiv.innerHTML = "";
  const c1 = document.createElement("div");
  const c2 = document.createElement("div");
  cardsDiv.appendChild(c1);
  cardsDiv.appendChild(c2);
  renderDecisionCard(c1, "Standard L2D (uniform cost)", r2, tier, "standard-l2d");
  renderDecisionCard(c2, "Risk-Sensitive L2D (this thesis)", r3, tier, "risk-sensitive-l2d");
  renderDivergence(document.getElementById("explorer-divergence"), r2, r3);

  const rates = profile.tier_defer_rates || {};
  const cur = rates[tier];
  const warnDiv = document.getElementById("explorer-warning");
  if (cur) {
    document.getElementById("explorer-sensitivity-note").textContent =
      `For this specific behaviour pattern, sweeping the full €1-2,000 range: Risk-Sensitive L2D defers ` +
      `${cur.m3_pct.toFixed(0)}% of amounts landing in ${tier} tier (Standard L2D: ${cur.m2_pct.toFixed(0)}%).`;
  } else {
    document.getElementById("explorer-sensitivity-note").textContent = "";
  }
  if (profile.matches_critical_defers_most === false) {
    const highest = profile.highest_m3_defer_tier;
    warnDiv.innerHTML = `<div class="warn-box">For this specific pattern, <b>${highest} tier is deferred most by Risk-Sensitive L2D, not ` +
      `Critical</b> — the opposite ordering from the thesis's population-level result (Table 4). ` +
      `'Critical defers most' is an aggregate statistic across thousands of transactions, not a ` +
      `guarantee for every individual behaviour pattern — the closest discussion of why the effect ` +
      `isn't uniform is the weight-sensitivity finding in Section 6.1.4. Try the other two patterns ` +
      `below to compare.</div>`;
  } else {
    warnDiv.innerHTML = "";
  }
}

profileSelect.addEventListener("change", onProfileChange);
amountSlider.addEventListener("input", renderExplorerTransaction);
onProfileChange();

/* re-render charts on resize (SVGs are sized from container width) */
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    tierChartsContainer.innerHTML = "";
    Object.entries(DATASET_METRICS).forEach(([dname, d]) => {
      const wrap = document.createElement("div");
      wrap.className = "chart-wrap";
      tierChartsContainer.appendChild(wrap);
      const tiers = Object.keys(d.tiers);
      renderGroupedBarChart(wrap, {
        title: `${dname}\nCritical-tier ratio: ${d.critical_headline_ratio}`,
        categories: tiers,
        series: [
          { label: "Standard L2D", values: tiers.map((t) => d.tiers[t].standard), color: STANDARD_COLOR },
          { label: "Risk-Sensitive L2D", values: tiers.map((t) => d.tiers[t].risk_sensitive), color: RISK_COLOR },
        ],
        yLabel: "Deferral rate (%)",
        valueFmt: (v) => v.toFixed(2),
      });
    });
    renderWeightConfig();
  }, 200);
});
