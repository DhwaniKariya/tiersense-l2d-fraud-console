# Risk-weight sensitivity sweep on ULB.
#
# Retrains Model 3 with the seed and base cost held fixed at their headline
# values (seed 42, base_cost 0.13) and only the risk-tier weight ratios
# changed, across five configurations. This checks whether the
# Critical-tier-defers-more result depends on the specific weights used
# elsewhere in this project (1.0/0.8/0.5/0.2) or holds more generally.
#
# Run with: py -3.13 07_ulb_weight_sensitivity.py
# (the notebook packages live under the py launcher's 3.13 install, not the
# system python3)

import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

T0 = time.time()
OUT_JSON = os.path.join(ROOT, "results", "ulb_weight_sensitivity.json")

df = pd.read_csv(os.path.join(ROOT, "data", "creditcard_with_tiers.csv"))
X = df.drop(columns=['Class', 'Risk_Tier'])
y = df['Class']
risk_tiers = df['Risk_Tier']
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
X['Time'] = scaler.fit_transform(X[['Time']])
X_train, X_test, y_train, y_test, rt_train, rt_test = train_test_split(
    X, y, risk_tiers, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

rt_train_values = rt_train.values
original_size = len(rt_train_values)
smote_size = len(X_train_bal) - original_size
rt_train_bal = list(rt_train_values) + ['Low'] * smote_size

X_train_tensor = torch.FloatTensor(X_train_bal.values)
y_train_tensor = torch.LongTensor(y_train_bal.values)
X_test_tensor = torch.FloatTensor(X_test.values)
y_np = y_test.values
rt_np = rt_test.values
input_dim = X_train_tensor.shape[1]
BASE_COST = 0.13

print(f"Data ready in {time.time()-T0:.0f}s. input_dim={input_dim}", flush=True)


class L2DNetwork(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU())
        self.classifier_head = nn.Linear(32, num_classes)
        self.deferral_head = nn.Linear(32, 1)

    def forward(self, x):
        s = self.shared(x)
        return self.classifier_head(s), torch.sigmoid(self.deferral_head(s))


class FraudDatasetWithTiers(torch.utils.data.Dataset):
    def __init__(self, X, y, tiers):
        self.X, self.y, self.tiers = X, y, tiers

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.tiers[idx]


def rs_loss(class_logits, defer_prob, targets, tier_batch, weights_dict):
    ce = nn.CrossEntropyLoss(reduction='none')
    class_loss = ce(class_logits, targets)
    defer_p = defer_prob.squeeze()
    weights = torch.tensor([weights_dict[t] for t in tier_batch], dtype=torch.float32)
    instance_cost = weights * BASE_COST
    return ((1 - defer_p) * class_loss + defer_p * instance_cost).mean()


def train_and_eval(weights_dict, seed=42):
    torch.manual_seed(seed)
    model = L2DNetwork(input_dim=input_dim)
    opt = optim.Adam(model.parameters(), lr=0.001)
    ds = FraudDatasetWithTiers(X_train_tensor, y_train_tensor, rt_train_bal)
    loader = DataLoader(ds, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))
    for epoch in range(15):
        model.train()
        for X_b, y_b, t_b in loader:
            opt.zero_grad()
            cl, dp = model(X_b)
            loss = rs_loss(cl, dp, y_b, list(t_b), weights_dict)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        cl, dp = model(X_test_tensor)
        cp = torch.argmax(cl, dim=1).numpy()
        dp = dp.squeeze().numpy()
        deferred = dp > 0.5
        autonomous = ~deferred
    f1 = f1_score(y_np[autonomous], cp[autonomous], zero_division=0) if autonomous.sum() > 0 else 0.0
    tier_rates = {}
    for tier in ['Low', 'Medium', 'High', 'Critical']:
        mask = rt_np == tier
        tier_rates[tier] = round(float(deferred[mask].mean() * 100), 3)
    return f1, tier_rates


CONFIGS = {
    'original (thesis)':      {'Low': 1.0, 'Medium': 0.8, 'High': 0.5, 'Critical': 0.2},
    'gentle (IEEE-CIS-style)': {'Low': 1.0, 'Medium': 0.9, 'High': 0.7, 'Critical': 0.5},
    'steep':                   {'Low': 1.0, 'Medium': 0.6, 'High': 0.3, 'Critical': 0.05},
    'moderate':                {'Low': 1.0, 'Medium': 0.75, 'High': 0.5, 'Critical': 0.25},
    'minimal':                 {'Low': 1.0, 'Medium': 0.95, 'High': 0.85, 'Critical': 0.7},
}

results = {}
for name, weights in CONFIGS.items():
    ts = time.time()
    print(f"\n=== Config: {name} {weights} ===", flush=True)
    f1, tier_rates = train_and_eval(weights)
    ratio = (tier_rates['Critical'] / tier_rates['Low']) if tier_rates['Low'] > 0 else None
    print(f"  F1={f1:.4f}  tiers={tier_rates}  Critical/Low ratio={ratio}  ({time.time()-ts:.0f}s)", flush=True)
    results[name] = {'weights': weights, 'f1': round(f1, 4), 'tier_rates': tier_rates,
                      'critical_over_low_ratio': round(ratio, 2) if ratio else None}
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)

print("\n\n=== SWEEP COMPLETE ===", flush=True)
print(json.dumps(results, indent=2), flush=True)
print(f"Total elapsed: {time.time()-T0:.0f}s", flush=True)
