from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests
from canonical_analysis import load_data, build_models, RANDOM_STATE, OUT


def compute_midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def fast_delong(preds, n_pos):
    m = n_pos
    n = preds.shape[1] - m
    k = preds.shape[0]
    pos, neg = preds[:, :m], preds[:, m:]
    tx, ty, tz = np.empty((k, m)), np.empty((k, n)), np.empty((k, m+n))
    for r in range(k):
        tx[r] = compute_midrank(pos[r])
        ty[r] = compute_midrank(neg[r])
        tz[r] = compute_midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1)/(m*n) - (m+1)/(2*n)
    v01 = (tz[:, :m] - tx)/n
    v10 = 1 - (tz[:, m:] - ty)/m
    return aucs, np.cov(v01)/m + np.cov(v10)/n


def delong_pvalue(y, p1, p2):
    y = np.asarray(y)
    order = np.argsort(-y)
    preds = np.vstack([p1, p2])[:, order]
    aucs, cov = fast_delong(preds, int(y.sum()))
    contrast = np.array([[1, -1]])
    var = float(contrast @ cov @ contrast.T)
    if var <= 0:
        return aucs[0], aucs[1], 1.0
    z = abs(aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], float(2 * norm.sf(z))


X, y, _ = load_data()
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
models = build_models()
base = ['Logistic Regression', 'SVM (RBF)', 'Random Forest', 'XGBoost']
probs, preds, stack = {}, {}, []
for name in base:
    model = clone(models[name]).fit(Xtr, ytr)
    probs[name] = model.predict_proba(Xte)[:, 1]
    preds[name] = (probs[name] >= 0.50).astype(int)
    stack.append(probs[name])
probs['Soft Voting Ensemble'] = np.mean(np.vstack(stack), axis=0)
preds['Soft Voting Ensemble'] = (probs['Soft Voting Ensemble'] >= 0.50).astype(int)

rows, mcnemar_p, delong_p = [], [], []
y_arr = np.asarray(yte)
for a, b in itertools.combinations(probs.keys(), 2):
    a_correct = preds[a] == y_arr
    b_correct = preds[b] == y_arr
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    pm = binomtest(min(a_only, b_only), n=a_only+b_only, p=0.5,
                   alternative='two-sided').pvalue if a_only+b_only else 1.0
    auc_a, auc_b, pdv = delong_pvalue(yte, probs[a], probs[b])
    rows.append({
        'Model A': a, 'Model B': b,
        'A correct/B wrong': a_only, 'A wrong/B correct': b_only,
        'McNemar exact p': pm, 'A AUC': auc_a, 'B AUC': auc_b,
        'DeLong p': pdv,
    })
    mcnemar_p.append(pm)
    delong_p.append(pdv)

m_adj = multipletests(mcnemar_p, method='holm')[1]
d_adj = multipletests(delong_p, method='holm')[1]
for row, pm, pdv in zip(rows, m_adj, d_adj):
    row['McNemar Holm p'] = float(pm)
    row['DeLong Holm p'] = float(pdv)

out = pd.DataFrame(rows)
out.to_csv(OUT / 'all_pairwise_tests.csv', index=False)
print(out.to_string(index=False))
