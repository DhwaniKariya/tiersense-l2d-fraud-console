# -*- coding: utf-8 -*-
"""
Trivial rule-based deferral baseline for ULB, using Model 3's OWN trained
classifier head (loaded from the saved risk_sensitive_l2d.pth), so the ONLY
thing that changes versus Model 3 is the deferral policy: a hand-written rule
(defer iff Critical tier) instead of the learned deferral head. This isolates
whether the learned mechanism adds anything beyond hard-coding the same risk
tiers it was given, holding the classifier fixed -- the same single-variable
discipline used for the Model 2 vs Model 3 comparison elsewhere in this thesis.
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class L2DNetwork(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU())
        self.classifier_head = nn.Linear(32, num_classes)
        self.deferral_head = nn.Linear(32, 1)

    def forward(self, x):
        s = self.shared(x)
        return self.classifier_head(s), torch.sigmoid(self.deferral_head(s))


df = pd.read_csv(r"E:\NCI\Sem3\Thesis\implementation\data\creditcard_with_tiers.csv")
X = df.drop(columns=['Class', 'Risk_Tier'])
y = df['Class']
risk_tiers = df['Risk_Tier']
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
X['Time'] = scaler.fit_transform(X[['Time']])
X_train, X_test, y_train, y_test, rt_train, rt_test = train_test_split(
    X, y, risk_tiers, test_size=0.2, random_state=42, stratify=y)

X_test_tensor = torch.FloatTensor(X_test.values)
y_np = y_test.values
rt_np = rt_test.values

model = L2DNetwork(input_dim=X_test_tensor.shape[1])
model.load_state_dict(torch.load(r"E:\NCI\Sem3\Thesis\implementation\models\risk_sensitive_l2d.pth", map_location='cpu'))
model.eval()

with torch.no_grad():
    cl, dp_learned = model(X_test_tensor)
    y_pred = torch.argmax(cl, dim=1).numpy()
    y_proba = torch.softmax(cl, dim=1)[:, 1].numpy()
    dp_learned = dp_learned.squeeze().numpy()

# Rule-based deferral policy: defer iff Critical tier (same classifier as Model 3)
deferred = (rt_np == 'Critical')
autonomous = ~deferred

f1 = f1_score(y_np[autonomous], y_pred[autonomous], zero_division=0)
acc = accuracy_score(y_np[autonomous], y_pred[autonomous])
auc = roc_auc_score(y_np[autonomous], y_proba[autonomous])

fraud_total = int((y_np == 1).sum())
fraud_deferred = int(deferred[y_np == 1].sum())
fraud_caught = int(((y_pred[autonomous] == 1) & (y_np[autonomous] == 1)).sum())
fraud_missed = int(((y_pred[autonomous] == 0) & (y_np[autonomous] == 1)).sum())

tier_rates = {}
for tier in ['Low', 'Medium', 'High', 'Critical']:
    mask = rt_np == tier
    tier_rates[tier] = round(float(deferred[mask].mean() * 100), 2)

result = {
    'model': 'Rule-based deferral baseline (Model 3 classifier + hand-written rule: defer iff Critical tier)',
    'dataset': 'ULB',
    'deferral_rate': round(float(deferred.mean() * 100), 2),
    'f1_score': round(float(f1), 4),
    'accuracy': round(float(acc), 4),
    'roc_auc': round(float(auc), 4),
    'frauds_total': fraud_total,
    'frauds_deferred': fraud_deferred,
    'frauds_caught': fraud_caught,
    'frauds_missed': fraud_missed,
    'deferral_by_tier': tier_rates,
    'note': 'Uses the SAME trained classifier as Model 3 (risk_sensitive_l2d.pth); only the '
            'deferral decision differs (hand-written rule vs learned deferral head), to isolate '
            'the effect of the learned mechanism specifically.'
}

print(json.dumps(result, indent=2))
with open(r"E:\NCI\Sem3\Thesis\implementation\results\ulb_trivial_rule_baseline.json", 'w') as f:
    json.dump(result, f, indent=2)
print("Saved.")
