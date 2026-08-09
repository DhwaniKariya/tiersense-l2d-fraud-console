# -*- coding: utf-8 -*-
"""
Trivial rule-based deferral baseline for PaySim: trains a fresh classifier
identical to Model 3's own architecture/training (same as the notebook's Model 3
routine, base_cost=0.4, seed=42), then evaluates it TWICE: once with its own
learned deferral head (should reproduce the existing Model 3 result almost
exactly, as a sanity check), and once with a hand-written rule (defer iff
Critical tier) substituted for the learned deferral head, holding the
classifier fixed. Isolates whether the learned mechanism adds anything beyond
hard-coding the tiers.
"""
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
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

T0 = time.time()
print("Loading PaySim...", flush=True)
df_pay = pd.read_csv(r"E:\NCI\Sem3\Thesis\implementation\data\PS_20174392719_1491204439457_log.csv")
df_pay_filtered = df_pay[df_pay['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()

p25 = df_pay_filtered['amount'].quantile(0.25)
p50 = df_pay_filtered['amount'].quantile(0.50)
p75 = df_pay_filtered['amount'].quantile(0.75)


def assign_tier(amount):
    if amount <= p25:
        return 'Low'
    elif amount <= p50:
        return 'Medium'
    elif amount <= p75:
        return 'High'
    else:
        return 'Critical'


df_pay_filtered['Risk_Tier'] = df_pay_filtered['amount'].apply(assign_tier)
features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step']
df_pay_filtered['type_num'] = (df_pay_filtered['type'] == 'TRANSFER').astype(int)
features.append('type_num')

X_pay = df_pay_filtered[features].copy()
y_pay = df_pay_filtered['isFraud']
rt_pay = df_pay_filtered['Risk_Tier']

scaler_pay = StandardScaler()
X_pay_scaled = pd.DataFrame(scaler_pay.fit_transform(X_pay), columns=X_pay.columns)
X_tr, X_te, y_tr, y_te, rt_tr, rt_te = train_test_split(
    X_pay_scaled, y_pay, rt_pay, test_size=0.2, random_state=42, stratify=y_pay)

print("Applying SMOTE...", flush=True)
smote = SMOTE(random_state=42)
X_tr_bal, y_tr_bal = smote.fit_resample(X_tr, y_tr)

rt_tr_values = rt_tr.values
original_size = len(rt_tr_values)
smote_size = len(X_tr_bal) - original_size
rt_tr_bal = list(rt_tr_values) + ['Low'] * smote_size

X_tr_tensor = torch.FloatTensor(X_tr_bal.values)
y_tr_tensor = torch.LongTensor(y_tr_bal.values)
X_te_tensor = torch.FloatTensor(X_te.values)
y_np = y_te.values
rt_np = rt_te.values
input_dim = X_tr_tensor.shape[1]
print(f"Data ready in {time.time()-T0:.0f}s. train={X_tr_tensor.shape}", flush=True)

PAY_BASE_COST = 0.4
RISK_WEIGHTS = {'Low': 1.0, 'Medium': 0.8, 'High': 0.5, 'Critical': 0.2}


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


def rs_loss(class_logits, defer_prob, targets, tier_batch):
    ce = nn.CrossEntropyLoss(reduction='none')
    class_loss = ce(class_logits, targets)
    defer_p = defer_prob.squeeze()
    weights = torch.tensor([RISK_WEIGHTS[t] for t in tier_batch], dtype=torch.float32)
    instance_cost = weights * PAY_BASE_COST
    return ((1 - defer_p) * class_loss + defer_p * instance_cost).mean()


torch.manual_seed(42)
model = L2DNetwork(input_dim=input_dim)
opt = optim.Adam(model.parameters(), lr=0.001)
ds = FraudDatasetWithTiers(X_tr_tensor, y_tr_tensor, rt_tr_bal)
loader = DataLoader(ds, batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(42))

print("Training Model 3 (risk-sensitive) fresh copy for the ablation...", flush=True)
ts = time.time()
for epoch in range(15):
    model.train()
    for X_b, y_b, t_b in loader:
        opt.zero_grad()
        cl, dp = model(X_b)
        loss = rs_loss(cl, dp, y_b, list(t_b))
        loss.backward()
        opt.step()
    if (epoch + 1) % 5 == 0:
        print(f"  epoch {epoch+1}/15  ({time.time()-ts:.0f}s)", flush=True)

model.eval()
with torch.no_grad():
    cl, dp_learned = model(X_te_tensor)
    y_pred = torch.argmax(cl, dim=1).numpy()
    y_proba = torch.softmax(cl, dim=1)[:, 1].numpy()
    dp_learned = dp_learned.squeeze().numpy()

# Sanity check: learned-deferral evaluation (should match existing Model 3 result closely)
deferred_learned = dp_learned > 0.5
autonomous_learned = ~deferred_learned
f1_learned = f1_score(y_np[autonomous_learned], y_pred[autonomous_learned], zero_division=0)
tier_rates_learned = {t: round(float(deferred_learned[rt_np == t].mean() * 100), 3) for t in ['Low', 'Medium', 'High', 'Critical']}
print(f"\nSanity check (learned deferral, fresh copy): F1={f1_learned:.4f} overall={deferred_learned.mean()*100:.2f}% tiers={tier_rates_learned}", flush=True)

# Rule-based deferral: defer iff Critical tier, SAME classifier
deferred = (rt_np == 'Critical')
autonomous = ~deferred
f1 = f1_score(y_np[autonomous], y_pred[autonomous], zero_division=0)
acc = accuracy_score(y_np[autonomous], y_pred[autonomous])
auc = roc_auc_score(y_np[autonomous], y_proba[autonomous])

fraud_total = int((y_np == 1).sum())
fraud_deferred = int(deferred[y_np == 1].sum())
fraud_caught = int(((y_pred[autonomous] == 1) & (y_np[autonomous] == 1)).sum())
fraud_missed = int(((y_pred[autonomous] == 0) & (y_np[autonomous] == 1)).sum())

tier_rates = {t: round(float(deferred[rt_np == t].mean() * 100), 3) for t in ['Low', 'Medium', 'High', 'Critical']}

result = {
    'model': 'Rule-based deferral baseline (fresh Model-3-architecture classifier + hand-written rule: defer iff Critical tier)',
    'dataset': 'PaySim',
    'sanity_check_learned_deferral': {'f1': round(float(f1_learned), 4), 'overall_deferral_rate': round(float(deferred_learned.mean()*100), 2), 'tier_rates': tier_rates_learned},
    'deferral_rate': round(float(deferred.mean() * 100), 2),
    'f1_score': round(float(f1), 4),
    'accuracy': round(float(acc), 4),
    'roc_auc': round(float(auc), 4),
    'frauds_total': fraud_total,
    'frauds_deferred': fraud_deferred,
    'frauds_caught': fraud_caught,
    'frauds_missed': fraud_missed,
    'deferral_by_tier': tier_rates,
}

print(json.dumps(result, indent=2), flush=True)
with open(r"E:\NCI\Sem3\Thesis\implementation\results\paysim_trivial_rule_baseline.json", 'w') as f:
    json.dump(result, f, indent=2)
print(f"Saved. Total elapsed {time.time()-T0:.0f}s", flush=True)
