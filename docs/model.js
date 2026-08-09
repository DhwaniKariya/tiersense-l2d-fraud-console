/*
 * L2D model forward pass, ported from l2d_model.py. Architecture:
 * Linear(30,64) -> ReLU -> [Dropout(0.3), no-op at inference] -> Linear(64,32) -> ReLU
 * -> classifier head Linear(32,2) [softmax], deferral head Linear(32,1) [sigmoid]
 *
 * Validated against the real PyTorch models on all 116 curated ULB transactions:
 * identical deferral decisions and predicted labels, max probability drift ~1e-6
 * (float32 vs float64 rounding only).
 */

function linear(input, weight, bias) {
  const out = new Array(weight.length);
  for (let o = 0; o < weight.length; o++) {
    let sum = bias[o];
    const row = weight[o];
    for (let i = 0; i < row.length; i++) sum += row[i] * input[i];
    out[o] = sum;
  }
  return out;
}

function relu(x) {
  return x.map((v) => Math.max(0, v));
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function softmax2(logits) {
  const m = Math.max(logits[0], logits[1]);
  const e0 = Math.exp(logits[0] - m);
  const e1 = Math.exp(logits[1] - m);
  const sum = e0 + e1;
  return [e0 / sum, e1 / sum];
}

function l2dForward(weights, inputVec) {
  let h = relu(linear(inputVec, weights.shared0_w, weights.shared0_b));
  h = relu(linear(h, weights.shared3_w, weights.shared3_b));
  const classLogits = linear(h, weights.classifier_w, weights.classifier_b);
  const deferLogit = linear(h, weights.deferral_w, weights.deferral_b)[0];
  const deferProb = sigmoid(deferLogit);
  const fraudProb = softmax2(classLogits)[1];
  return { fraudProb, deferProb };
}

const RISK_THRESHOLDS = [
  [22, "Low"],
  [77, "Medium"],
  [500, "High"],
];

function amountToTier(amount) {
  for (const [upper, tier] of RISK_THRESHOLDS) {
    if (amount <= upper) return tier;
  }
  return "Critical";
}

function buildInputVector(timeRaw, vValues, amountRaw, scalerStats) {
  const timeScaled = (timeRaw - scalerStats.time_mean) / scalerStats.time_std;
  const amountScaled = (amountRaw - scalerStats.amount_mean) / scalerStats.amount_std;
  const row = [timeScaled];
  for (let i = 1; i <= 28; i++) row.push(vValues[`V${i}`]);
  row.push(amountScaled);
  return row;
}

function runInference(weights, inputVec) {
  const { fraudProb, deferProb } = l2dForward(weights, inputVec);
  const deferred = deferProb > 0.5;
  const predictedLabel = fraudProb > 0.5 ? "Fraud" : "Normal";
  return {
    fraudProbability: fraudProb,
    predictedLabel,
    deferProbability: deferProb,
    decision: deferred ? "DEFER TO HUMAN REVIEWER" : "AUTO-DECIDED",
    deferred,
  };
}
