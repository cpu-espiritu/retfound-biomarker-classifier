#!/usr/bin/env python3
"""Ensemble folds, Youden thresholds, size-stratified recall, patient bootstrap."""
import argparse, glob
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score

T = ['IRF', 'SRF', 'PED']

ap_ = argparse.ArgumentParser()
ap_.add_argument('--preds', required=True)
ap_.add_argument('--manifest', required=True)
ap_.add_argument('--arm', required=True)
a = ap_.parse_args()

fs = sorted(glob.glob(f"{a.preds}/preds_{a.arm}_f*_s*.npz"))
assert fs, f"no preds for {a.arm}"
D = [np.load(f, allow_pickle=True) for f in fs]
Pt = np.mean([d['test_p'] for d in D], axis=0)
Yt = D[0]['test_y']
Gt = D[0]['test_g']
Pv = np.concatenate([d['val_p'] for d in D])
Yv = np.concatenate([d['val_y'] for d in D])

man = pd.read_csv(a.manifest)
te = man[man.split == 'test'].reset_index(drop=True)
assert len(te) == len(Yt), (len(te), len(Yt))


def youden(y, p):
    npos, nneg = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    ts = np.linspace(0.02, 0.98, 97)
    j = [((p >= t) & (y == 1)).sum() / npos + ((p < t) & (y == 0)).sum() / nneg - 1
         for t in ts]
    return ts[int(np.argmax(j))]


def boot(y, p, thr, g, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    gi = {u: np.nonzero(g == u)[0] for u in ug}
    A, R = [], []
    for _ in range(n):
        i = np.concatenate([gi[u] for u in rng.choice(ug, len(ug))])
        if y[i].min() == y[i].max():
            continue
        A.append(average_precision_score(y[i], p[i]))
        R.append(recall_score(y[i], p[i] >= thr, zero_division=0))
    q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return q(A), q(R)


print(f"=== {a.arm} | {len(fs)} folds ensembled | {len(np.unique(Gt))} test patients ===")
for i, c in enumerate(T):
    y, p = Yt[:, i], Pt[:, i]
    thr = youden(Yv[:, i], Pv[:, i])
    pred = p >= thr
    spec = ((~pred) & (y == 0)).sum() / max((y == 0).sum(), 1)
    prec = (pred & (y == 1)).sum() / max(pred.sum(), 1)
    (al, ah), (rl, rh) = boot(y, p, thr, Gt)
    print(f"\n{c}  prev={y.mean():.3f}  thr={thr:.2f}")
    print(f"  AUPRC {average_precision_score(y, p):.3f} [{al:.3f},{ah:.3f}]"
          f"   AUROC {roc_auc_score(y, p):.3f}")
    print(f"  recall {recall_score(y, pred, zero_division=0):.3f} [{rl:.3f},{rh:.3f}]"
          f"   spec {spec:.3f}   prec {prec:.3f}")
    area = te[f'area_{c}'].values[y == 1]
    hit = pred[y == 1]
    if len(area) > 20:
        q1, q3 = np.percentile(area, [25, 75])
        for nm, sel in [('small', area < q1),
                        ('med', (area >= q1) & (area < q3)),
                        ('large', area >= q3)]:
            print(f"    recall {nm:<6} (n={sel.sum():>3}) {hit[sel].mean():.3f}")
