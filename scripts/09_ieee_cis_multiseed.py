# -*- coding: utf-8 -*-
"""
IEEE-CIS multi-seed robustness check (extends the existing seed-42 result with
seeds 43, 44, 45), mirroring the ULB 5-seed robustness check in Section 6.1.3.
Data split and SMOTE are kept fixed at random_state=42 throughout (same
convention as the ULB check); only model weight init and batch order vary
by seed.
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
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

T0 = time.time()
OUT_JSON = r"E:\NCI\Sem3\Thesis\implementation\results\ieee_cis_multiseed_robustness.json"

print("Loading IEEE-CIS...", flush=True)
train_transaction = pd.read_csv(r"E:\NCI\Sem3\Thesis\implementation\data\ieee-cis\train_transaction.csv")
train_identity = pd.read_csv(r"E:\NCI\Sem3\Thesis\implementation\data\ieee-cis\train_identity.csv")
df_ieee = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

missing_pct = df_ieee.isnull().mean()
cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
df_ieee = df_ieee.drop(columns=cols_to_drop)
df_ieee_num = df_ieee.select_dtypes(include=[np.number])
df_ieee_num = df_ieee_num.fillna(df_ieee_num.median())
y_ieee = df_ieee_num['isFraud']
X_ieee = df_ieee_num.drop(columns=['isFraud', 'TransactionID'])

p25 = df_ieee['TransactionAmt'].quantile(0.25)
p50 = df_ieee['TransactionAmt'].quantile(0.50)
p75 = df_ieee['TransactionAmt'].quantile(0.75)

def assign_tier(amount):
    if amount <= p25:
        return 'Low'
    elif amount <= p50:
        return 'Medium'
    elif amount <= p75:
        return 'High'
    else:
        return 'Critical'

df_ieee['Risk_Tier'] = df_ieee['TransactionAmt'].apply(assign_tier)
X_ieee['Risk_Tier'] = df_ieee['Risk_Tier'].values
risk_tiers_ieee = X_ieee['Risk_Tier']
X_ieee = X_ieee.drop(columns=['Risk_Tier'])

scaler_ieee = StandardScaler()
X_ieee_scaled = pd.DataFrame(scaler_ieee.fit_transform(X_ieee), columns=X_ieee.columns)

X_tr, X_te, y_tr, y_te, rt_tr, rt_te = train_test_split(
    X_ieee_scaled, y_ieee, risk_tiers_ieee, test_size=0.2, random_state=42, stratify=y_ieee)

print("Applying SMOTE (data split fixed at seed 42 for all model seeds)...", flush=True)
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

print(f"Data ready. train={X_tr_tensor.shape}, test={X_te_tensor.shape}. "
      f"Elapsed {time.time()-T0:.0f}s", flush=True)

IEEE_BASE_COST = 0.63
IEEE_RISK_WEIGHTS = {'Low': 1.0, 'Medium': 0.9, 'High': 0.7, 'Critical': 0.5}


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


def l2d_loss(class_logits, defer_prob, targets, cost):
    ce = nn.CrossEntropyLoss(reduction='none')
    class_loss = ce(class_logits, targets)
    defer_p = defer_prob.squeeze()
    return ((1 - defer_p) * class_loss + defer_p * cost).mean()


def rs_loss(class_logits, defer_prob, targets, tier_batch):
    ce = nn.CrossEntropyLoss(reduction='none')
    class_loss = ce(class_logits, targets)
    defer_p = defer_prob.squeeze()
    weights = torch.tensor([IEEE_RISK_WEIGHTS[t] for t in tier_batch], dtype=torch.float32)
    instance_cost = weights * IEEE_BASE_COST
    return ((1 - defer_p) * class_loss + defer_p * instance_cost).mean()


def train_standard(seed, input_dim):
    torch.manual_seed(seed)
    model = L2DNetwork(input_dim=input_dim)
    opt = optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(TensorDataset(X_tr_tensor, y_tr_tensor), batch_size=256, shuffle=True,
                         generator=torch.Generator().manual_seed(seed))
    for epoch in range(15):
        model.train()
        for X_b, y_b in loader:
            opt.zero_grad()
            cl, dp = model(X_b)
            loss = l2d_loss(cl, dp, y_b, cost=IEEE_BASE_COST)
            loss.backward()
            opt.step()
    return model


def train_risk_sensitive(seed, input_dim):
    torch.manual_seed(seed)
    model = L2DNetwork(input_dim=input_dim)
    opt = optim.Adam(model.parameters(), lr=0.001)
    ds = FraudDatasetWithTiers(X_tr_tensor, y_tr_tensor, rt_tr_bal)
    loader = DataLoader(ds, batch_size=256, shuffle=True,
                         generator=torch.Generator().manual_seed(seed))
    for epoch in range(15):
        model.train()
        for X_b, y_b, t_b in loader:
            opt.zero_grad()
            cl, dp = model(X_b)
            loss = rs_loss(cl, dp, y_b, list(t_b))
            loss.backward()
            opt.step()
    return model


def evaluate(model):
    model.eval()
    with torch.no_grad():
        cl, dp = model(X_te_tensor)
        cp = torch.argmax(cl, dim=1).numpy()
        dp = dp.squeeze().numpy()
        deferred = dp > 0.5
        autonomous = ~deferred
    f1 = f1_score(y_np[autonomous], cp[autonomous], zero_division=0) if autonomous.sum() > 0 else 0.0
    tier_rates = {}
    for tier in ['Low', 'Medium', 'High', 'Critical']:
        mask = rt_np == tier
        tier_rates[tier] = round(float(deferred[mask].mean() * 100), 2)
    return f1, tier_rates


input_dim = X_tr_tensor.shape[1]
SEEDS = [43, 44, 45]
runs = []

# seed 42 is the existing, already-reported result (kept here for a complete table)
runs.append({
    'seed': 42,
    'm2_f1': 0.4001, 'm2_tier': {'Low': 12.74, 'Medium': 3.60, 'High': 8.68, 'Critical': 10.67},
    'm3_f1': 0.5593, 'm3_tier': {'Low': 18.60, 'Medium': 13.62, 'High': 41.94, 'Critical': 70.56},
})

for seed in SEEDS:
    ts = time.time()
    print(f"\n=== SEED {seed} ===", flush=True)
    m2 = train_standard(seed, input_dim)
    m2_f1, m2_tier = evaluate(m2)
    print(f"  Model 2 (seed {seed}): F1={m2_f1:.4f} tiers={m2_tier}  ({time.time()-ts:.0f}s so far)", flush=True)

    m3 = train_risk_sensitive(seed, input_dim)
    m3_f1, m3_tier = evaluate(m3)
    print(f"  Model 3 (seed {seed}): F1={m3_f1:.4f} tiers={m3_tier}  ({time.time()-ts:.0f}s total)", flush=True)

    ratio = (m3_tier['Critical'] / m2_tier['Critical']) if m2_tier['Critical'] > 0 else None
    low_ratio = (m3_tier['Critical'] / m3_tier['Low']) if m3_tier['Low'] > 0 else None
    runs.append({
        'seed': seed,
        'm2_f1': round(m2_f1, 4), 'm2_tier': m2_tier,
        'm3_f1': round(m3_f1, 4), 'm3_tier': m3_tier,
        'critical_m2_to_m3_ratio': round(ratio, 2) if ratio else None,
        'm3_critical_over_low_ratio': round(low_ratio, 2) if low_ratio else None,
    })
    with open(OUT_JSON, 'w') as f:
        json.dump({'data_seed': 42, 'model_seeds': [42] + SEEDS, 'runs': runs}, f, indent=2)
    print(f"  Saved progress to {OUT_JSON}", flush=True)

ratios = [r['critical_m2_to_m3_ratio'] for r in runs if r.get('critical_m2_to_m3_ratio')]
# also compute seed 42's own ratio for the summary
runs[0]['critical_m2_to_m3_ratio'] = round(runs[0]['m3_tier']['Critical'] / runs[0]['m2_tier']['Critical'], 2)
ratios = [r['critical_m2_to_m3_ratio'] for r in runs]
summary = {
    'data_seed': 42,
    'model_seeds': [42] + SEEDS,
    'runs': runs,
    'critical_m2_to_m3_ratio_mean': round(float(np.mean(ratios)), 3),
    'critical_m2_to_m3_ratio_std': float(np.std(ratios)),
    'critical_m2_to_m3_ratio_min': round(float(min(ratios)), 2),
    'critical_m2_to_m3_ratio_max': round(float(max(ratios)), 2),
}
with open(OUT_JSON, 'w') as f:
    json.dump(summary, f, indent=2)

print("\n\n=== FINAL SUMMARY ===", flush=True)
print(json.dumps(summary, indent=2), flush=True)
print(f"\nTotal elapsed: {time.time()-T0:.0f}s", flush=True)
print("DONE", flush=True)
