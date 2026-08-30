#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_pooling import AttnPool, layernorm, pool_fixed, fit_mean_baseline

ROOT = Path(__file__).resolve().parents[2]
T = ['IRF', 'SRF', 'PED']


def youden(y, p, ts=np.linspace(0.02, 0.98, 97)):
    a_, b_ = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    return ts[int(np.argmax([((p >= t) & (y == 1)).sum() / a_ +
                             ((p < t) & (y == 0)).sum() / b_ - 1 for t in ts]))]


def fold_models(X, df, pool, test, y, how, cfg, seed):
    """One model per fold. Returns test predictions per fold and the val
    predictions of the held-out fold, mirroring what train_amdsd.py saves."""
    test_p, val_p, val_y = [], [], []
    for k in range(5):
        tr = pool & (df.fold != k).values
        va = pool & (df.fold == k).values
        if how == 'attn':
            m = AttnPool(X.shape[2], L=int(cfg['L']), lr=float(cfg['lr']),
                         wd=float(cfg['wd']), epochs=int(cfg['epochs']), seed=seed)
            m.fit(np.asarray(X[tr], np.float32), y[tr].astype(float))
            test_p.append(m.decision(np.asarray(X[test], np.float32)))
            val_p.append(m.decision(np.asarray(X[va], np.float32)))
        else:
            Ztr = layernorm(pool_fixed(np.asarray(X[tr], np.float32), 'mean'))
            Zte = layernorm(pool_fixed(np.asarray(X[test], np.float32), 'mean'))
            Zva = layernorm(pool_fixed(np.asarray(X[va], np.float32), 'mean'))
            pt, _ = fit_mean_baseline(Ztr, y[tr], Zte, None, y, tr)
            pv, _ = fit_mean_baseline(Ztr, y[tr], Zva, None, y, tr)
            test_p.append(pt); val_p.append(pv)
        val_y.append(y[va])
    return np.array(test_p), np.concatenate(val_p), np.concatenate(val_y)


def main():
    ap = argparse.ArgumentParser(
        description='Frozen-encoder arms under the RESULTS.md section 1 protocol: '
                    'head-only training on cached features, one model per fold, test '
                    'predictions ensembled, three seeds, predictions saved.')
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--selected', required=True, help='attn_tuned.csv')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--boot', type=int, default=5000)
    ap.add_argument('--out', default=str(ROOT / 'results/frozen_arms.csv'))
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(',')]
    sel = pd.read_csv(a.selected)
    df = pd.read_csv(a.manifest)
    X = np.load(a.tokens, mmap_mode='r')
    pool = (df.split == 'pool').values
    test = (df.split == 'test').values
    G = df.loc[test, 'group'].values
    modal = {c: {k: sel[sel.cls == c][k].mode().iloc[0]
                 for k in ('L', 'lr', 'wd', 'epochs')} for c in T}

    print(f'{int(test.sum())} test scans, {len(np.unique(G))} patients, seeds {seeds}')
    for c in T:
        print(f"  {c}: L={modal[c]['L']} lr={modal[c]['lr']:g} "
              f"wd={modal[c]['wd']:g} epochs={modal[c]['epochs']}")

    out = Path(a.out)
    P = {}
    rows = []
    for how in ('mean', 'attn'):
        for c in T:
            i = T.index(c)
            y = df[f'label_{c}'].values
            for s in (seeds if how == 'attn' else [seeds[0]]):
                tp, vp, vy = fold_models(X, df, pool, test, y, how, modal[c], s)
                ens = tp.mean(0)                       # 5-fold ensemble, as in section 1
                thr = youden(vy, vp)                   # threshold from validation folds
                P[(how, s, c)] = ens
                yt = y[test]
                pred = ens >= thr
                rows.append(dict(
                    pool=how, cls=c, seed=s, auprc=AP(yt, ens),
                    auroc=roc_auc_score(yt, ens),
                    recall=recall_score(yt, pred, zero_division=0),
                    spec=((~pred) & (yt == 0)).sum() / max((yt == 0).sum(), 1),
                    thr=thr, **modal[c]))
                print(f'  {how:<5} {c:<4} s{s}  AUPRC {rows[-1]["auprc"]:.3f}  '
                      f'thr {thr:.2f}', flush=True)
                np.savez(out.with_name(f'frozen_{how}_{c}_s{s}.npz'),
                         test_p=ens, test_p_folds=tp, test_y=y[test], test_g=G,
                         val_p=vp, val_y=vy, thr=thr)
                pd.DataFrame(rows).to_csv(out, index=False)

    R = pd.DataFrame(rows)
    print('\n=== AUPRC, mean over seeds (test set, 5-fold ensembled) ===')
    g = R.groupby(['pool', 'cls']).auprc.agg(['mean', 'std', 'count'])
    print(g.round(4).to_string())

    # paired bootstrap: attention - mean, both frozen, on identical resamples
    ug = np.unique(G)
    gi = {u: np.flatnonzero(G == u) for u in ug}
    rng = np.random.default_rng(0)
    reps = [np.concatenate([gi[u] for u in rng.choice(ug, len(ug))])
            for _ in range(a.boot)]
    print(f'\n=== attention − mean, paired patient bootstrap ({a.boot}) ===')
    print(f"{'cls':<5}{'attn':>8}{'mean':>8}{'Δ':>9}{'95% CI':>20}{'p':>8}")
    tests = []
    for c in T:
        y = df[f'label_{c}'].values[test]
        A_ = np.mean([P[('attn', s, c)] for s in seeds], axis=0)
        B_ = P[('mean', seeds[0], c)]
        d = []
        for ix in reps:
            if y[ix].min() == y[ix].max():
                continue
            d.append(AP(y[ix], A_[ix]) - AP(y[ix], B_[ix]))
        d = np.array(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        n = len(d)
        p = min(2 * min((1 + (d <= 0).sum()) / (n + 1),
                        (1 + (d >= 0).sum()) / (n + 1)), 1.0)
        obs = AP(y, A_) - AP(y, B_)
        print(f'{c:<5}{AP(y, A_):>8.3f}{AP(y, B_):>8.3f}{obs:>+9.3f}'
              f'{f"[{lo:+.3f},{hi:+.3f}]":>20}{p:>8.3f}')
        tests.append(dict(cls=c, attn=AP(y, A_), mean=AP(y, B_), delta=obs,
                          lo=lo, hi=hi, p=p))
    pd.DataFrame(tests).to_csv(out.with_name('frozen_arms_tests.csv'), index=False)
    print(f'\n-> {out}, frozen_arms_tests.csv, and frozen_*_{{cls}}_s{{seed}}.npz')
    print('The .npz files match the layout of data/amdsd_preds/preds_*.npz, so any '
          'arm can be paired against any other.')


if __name__ == '__main__':
    main()
