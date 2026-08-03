#!/usr/bin/env python3
"""Step 4: linear probe heads on cached AMD-SD features."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score

TARGETS = ['IRF', 'SRF', 'PED']
CGRID = [0.001, 0.01, 0.1, 1.0]


def fit_predict(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=C, max_iter=3000)
    lr.fit(sc.transform(Xtr), ytr)
    return lr.predict_proba(sc.transform(Xte))[:, 1]


def best_f1_threshold(y, p):
    """Youden's J: max(sens + spec - 1). Prevalence-independent."""
    best, bt = -1, 0.5
    npos, nneg = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    for t in np.linspace(0.05, 0.95, 91):
        pred = p >= t
        sens = (pred & (y == 1)).sum() / npos
        spec = ((~pred) & (y == 0)).sum() / nneg
        j = sens + spec - 1
        if j > best: best, bt = j, t
    return bt


def boot(y, p, thr, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    gi = {g: np.nonzero(groups == g)[0] for g in ug}
    ap, rc = [], []
    for _ in range(n):
        idx = np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])
        yy, pp = y[idx], p[idx]
        if yy.min() == yy.max(): continue
        ap.append(average_precision_score(yy, pp))
        rc.append(recall_score(yy, pp >= thr, zero_division=0))
    q = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
    return q(ap), q(rc)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--features', required=True)
    ap_.add_argument('--manifest', required=True)
    a = ap_.parse_args()

    X = np.load(a.features)
    df = pd.read_csv(a.manifest)
    assert len(X) == len(df)
    pool = df.split == 'pool'
    test = df.split == 'test'
    print(f"pool {pool.sum()} scans / {df[pool].group.nunique()} patients | "
          f"test {test.sum()} / {df[test].group.nunique()}")

    for c in TARGETS:
        y = df[f'label_{c}'].values
        v = df[f'valid_{c}'].values.astype(bool)

        # --- CV over C, out-of-fold predictions
        best_C, best_ap, oof_best = None, -1, None
        for C in CGRID:
            oof = np.full(len(df), np.nan)
            for k in range(5):
                tr = pool & (df.fold != k) & v
                va = pool & (df.fold == k) & v
                oof[va.values] = fit_predict(X[tr.values], y[tr.values],
                                             X[va.values], C)
            m = pool & v
            s = average_precision_score(y[m.values], oof[m.values])
            if s > best_ap: best_C, best_ap, oof_best = C, s, oof

        m = (pool & v).values
        thr = best_f1_threshold(y[m], oof_best[m])

        # --- refit on pool, evaluate on test
        tr, te = (pool & v).values, (test & v).values
        p = fit_predict(X[tr], y[tr], X[te], best_C)
        yt, gt = y[te], df.group.values[te]
        prev = yt.mean()

        auprc = average_precision_score(yt, p)
        auroc = roc_auc_score(yt, p)
        r50 = recall_score(yt, p >= 0.5, zero_division=0)
        rth = recall_score(yt, p >= thr, zero_division=0)
        (al, ah), (rl, rh) = boot(yt, p, thr, gt)

        print(f"\n=== {c} ===  C={best_C}  thr={thr:.2f}  test prevalence={prev:.3f}")
        print(f"  AUPRC  {auprc:.3f}  [{al:.3f}, {ah:.3f}]   (chance {prev:.3f})")
        print(f"  AUROC  {auroc:.3f}")
        print(f"  recall @0.50  {r50:.3f}")
        print(f"  recall @{thr:.2f}  {rth:.3f}  [{rl:.3f}, {rh:.3f}]")
        pred = p >= thr
        spec = ((~pred) & (yt == 0)).sum() / max((yt == 0).sum(), 1)
        prec = (pred & (yt == 1)).sum() / max(pred.sum(), 1)
        print(f"  specificity {spec:.3f}   precision {prec:.3f}   pos-rate {pred.mean():.3f}")

        area = df[f'area_{c}'].values[te]
        pos = area[yt == 1]
        if len(pos) > 20:
            q1, q3 = np.percentile(pos, [25, 75])
            for name, sel in [('small', pos < q1),
                              ('med', (pos >= q1) & (pos < q3)),
                              ('large', pos >= q3)]:
                sub = (p[yt == 1] >= thr)[sel]
                print(f"    recall {name:<6} (n={sel.sum():>3}) {sub.mean():.3f}")


if __name__ == '__main__':
    main()
