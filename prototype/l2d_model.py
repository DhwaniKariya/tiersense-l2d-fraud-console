"""
Shared model definition and inference helpers for the fraud-review console
prototype. Architecture matches notebooks/04_Risk_Sensitive_L2D.ipynb
exactly (input_dim=30: Time, V1-V28, Amount), so the saved .pth checkpoints
load without modification.
"""
import json
import os

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))

RISK_THRESHOLDS = {
    # (upper bound, tier name), in ascending order. Matches Table 1 in the
    # thesis report: Low <=22, Medium <=77, High <=500, Critical above.
    22: "Low",
    77: "Medium",
    500: "High",
}


def amount_to_tier(amount: float) -> str:
    for upper, tier in RISK_THRESHOLDS.items():
        if amount <= upper:
            return tier
    return "Critical"


class L2DNetwork(nn.Module):
    def __init__(self, input_dim=30, num_classes=2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.classifier_head = nn.Linear(32, num_classes)
        self.deferral_head = nn.Linear(32, 1)

    def forward(self, x):
        shared_out = self.shared(x)
        class_logits = self.classifier_head(shared_out)
        defer_prob = torch.sigmoid(self.deferral_head(shared_out))
        return class_logits, defer_prob


def load_models():
    standard = L2DNetwork()
    standard.load_state_dict(torch.load(os.path.join(HERE, "standard_l2d.pth"), map_location="cpu"))
    standard.eval()

    risk_sensitive = L2DNetwork()
    risk_sensitive.load_state_dict(torch.load(os.path.join(HERE, "risk_sensitive_l2d.pth"), map_location="cpu"))
    risk_sensitive.eval()

    with open(os.path.join(HERE, "scaler_stats.json")) as f:
        scaler_stats = json.load(f)

    with open(os.path.join(HERE, "normal_v_profile.json")) as f:
        normal_v_profile = json.load(f)

    return standard, risk_sensitive, scaler_stats, normal_v_profile


def build_input_vector(time_raw, v_values, amount_raw, scaler_stats):
    """
    v_values: dict V1..V28 -> raw (already roughly standardised, PCA output,
    used as-is per the thesis's own preprocessing). Returns a 1x30 tensor in
    the exact column order the model was trained on: Time, V1..V28, Amount.
    """
    time_scaled = (time_raw - scaler_stats["time_mean"]) / scaler_stats["time_std"]
    amount_scaled = (amount_raw - scaler_stats["amount_mean"]) / scaler_stats["amount_std"]

    row = [time_scaled] + [v_values[f"V{i}"] for i in range(1, 29)] + [amount_scaled]
    return torch.tensor([row], dtype=torch.float32)


def run_inference(model, x_tensor):
    with torch.no_grad():
        class_logits, defer_prob = model(x_tensor)
        fraud_prob = torch.softmax(class_logits, dim=1)[0, 1].item()
        defer_p = defer_prob.item()
    decision = "DEFER TO HUMAN REVIEWER" if defer_p > 0.5 else "AUTO-DECIDED"
    predicted_label = "Fraud" if fraud_prob > 0.5 else "Normal"
    return {
        "fraud_probability": fraud_prob,
        "predicted_label": predicted_label,
        "defer_probability": defer_p,
        "decision": decision,
        "deferred": defer_p > 0.5,
    }
