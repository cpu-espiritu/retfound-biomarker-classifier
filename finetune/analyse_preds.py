#!/usr/bin/env python3
"""Ensemble folds, Youden thresholds, size-stratified recall, patient bootstrap.

One seed = one model: its folds are ensembled and scored as a unit. Given several
seeds, each is scored independently and the spread across them is reported. That
spread is the run-to-run noise floor — a difference between two arms only counts
if it clears it. Seeds are deliberately NOT pooled into one large ensemble, which
would improve the point estimate while hiding the variance being measured.
"""
import argparse, glob, re
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score

T = ['IRF', 'SRF', 'PED']


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


def score(files, te, n_boot):
    """Ensemble one seed's folds, then score every class on the test set."""
    D = [np.load(f, allow_pickle=True) for f in files]
    for d in D[1:]:
        assert np.array_equal(d['test_y'], D[0]['test_y']), "folds disagree on test labels"
        assert np.array_equal(d['test_g'], D[0]['test_g']), "folds disagree on test groups"
    assert list(D[0]['classes']) == T, f"class order {list(D[0]['classes'])} != {T}"

    Pt = np.mean([d['test_p'] for d in D], axis=0)
    Yt, Gt = D[0]['test_y'], D[0]['test_g']
    Pv = np.concatenate([d['val_p'] for d in D])
    Yv = np.concatenate([d['val_y'] for d in D])
    assert len(te) == len(Yt), (len(te), len(Yt))

    res = {}
    for i, c in enumerate(T):
        y, p = Yt[:, i], Pt[:, i]
        thr = youden(Yv[:, i], Pv[:, i])
        pred = p >= thr
        (al, ah), (rl, rh) = boot(y, p, thr, Gt, n=n_boot)
        r = {'prev': y.mean(), 'thr': thr,
             'auprc': average_precision_score(y, p), 'auprc_ci': (al, ah),
             'auroc': roc_auc_score(y, p),
             'recall': recall_score(y, pred, zero_division=0), 'recall_ci': (rl, rh),
             'spec': ((~pred) & (y == 0)).sum() / max((y == 0).sum(), 1),
             'prec': (pred & (y == 1)).sum() / max(pred.sum(), 1),
             'size': []}
        area, hit = te[f'area_{c}'].values[y == 1], pred[y == 1]
        if len(area) > 20:
            q1, q3 = np.percentile(area, [25, 75])
            for nm, sel in [('small', area < q1),
                            ('med', (area >= q1) & (area < q3)),
                            ('large', area >= q3)]:
                r['size'].append((nm, int(sel.sum()), hit[sel].mean()))
        res[c] = r
    return res


def report(res):
    for c in T:
        r = res[c]
        print(f"\n{c}  prev={r['prev']:.3f}  thr={r['thr']:.2f}")
        print(f"  AUPRC {r['auprc']:.3f} [{r['auprc_ci'][0]:.3f},{r['auprc_ci'][1]:.3f}]"
              f"   AUROC {r['auroc']:.3f}")
        print(f"  recall {r['recall']:.3f} [{r['recall_ci'][0]:.3f},{r['recall_ci'][1]:.3f}]"
              f"   spec {r['spec']:.3f}   prec {r['prec']:.3f}")
        for nm, n, v in r['size']:
            print(f"    recall {nm:<6} (n={n:>3}) {v:.3f}")


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--preds', required=True)
    ap_.add_argument('--manifest', required=True)
    ap_.add_argument('--arm', required=True, help="e.g. last4_224")
    ap_.add_argument('--seeds', default=None,
                     help="comma list, e.g. '0,1,2'. Default: every seed found.")
    ap_.add_argument('--boot', type=int, default=2000, help="bootstrap resamples")
    a = ap_.parse_args()

    fs = sorted(glob.glob(f"{a.preds}/preds_{a.arm}_f*_s*.npz"))
    assert fs, f"no preds for {a.arm}"
    by_seed = {}
    for f in fs:
        by_seed.setdefault(int(re.search(r'_s(\d+)\.npz$', f).group(1)), []).append(f)

    if a.seeds:
        want = [int(s) for s in a.seeds.split(',')]
        missing = [s for s in want if s not in by_seed]
        assert not missing, f"no preds for seed(s) {missing}; found {sorted(by_seed)}"
        by_seed = {s: by_seed[s] for s in want}
    seeds = sorted(by_seed)

    n_folds = {len(v) for v in by_seed.values()}
    if len(n_folds) > 1:
        print(f"[warn] uneven fold counts per seed: "
              f"{ {s: len(v) for s, v in by_seed.items()} } — seeds are not comparable")

    man = pd.read_csv(a.manifest)
    te = man[man.split == 'test'].reset_index(drop=True)
    n_pat = len(np.unique(np.load(by_seed[seeds[0]][0], allow_pickle=True)['test_g']))
    print(f"=== {a.arm} | seeds {seeds} | {len(by_seed[seeds[0]])} folds/seed "
          f"| {n_pat} test patients ===")

    all_res = {}
    for s in seeds:
        all_res[s] = score(by_seed[s], te, a.boot)
        if len(seeds) > 1:
            print(f"\n{'-' * 62}\n-- seed {s} ({len(by_seed[s])} folds ensembled)")
        report(all_res[s])

    if len(seeds) > 1:
        print(f"\n{'=' * 62}\nAcross-seed spread ({len(seeds)} seeds), "
              f"each seed scored as its own fold ensemble")
        print(f"{'':6}{'AUPRC mean':>12}{'SD':>8}{'min':>8}{'max':>8}"
              f"{'recall mean':>13}{'SD':>8}")
        for c in T:
            av = np.array([all_res[s][c]['auprc'] for s in seeds])
            rv = np.array([all_res[s][c]['recall'] for s in seeds])
            print(f"{c:<6}{av.mean():>12.3f}{av.std(ddof=1):>8.3f}"
                  f"{av.min():>8.3f}{av.max():>8.3f}"
                  f"{rv.mean():>13.3f}{rv.std(ddof=1):>8.3f}")
        print("\nSD here is seed noise only. The patient bootstrap CIs above are far "
              "wider and\nremain the honest uncertainty estimate — do not quote this "
              "SD as one.")


if __name__ == '__main__':
    main()
