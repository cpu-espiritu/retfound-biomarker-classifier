#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'prep'))
from frozen_arms import fold_models, youden
from make_size_subsets import groups as subset_groups

ROOT = Path(__file__).resolve().parents[2]
T = ['IRF', 'SRF', 'PED']
FULL = 118          # pool patients; the size the head hyperparameters were tuned at


def main():
    ap = argparse.ArgumentParser(
        description='Frozen-encoder arms across training-set sizes. Same section 1 '
                    'protocol as frozen_arms.py, with the pool split restricted to a '
                    'patient subsample; the test set is never touched.')
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--selected', required=True, help='attn_tuned.csv')
    ap.add_argument('--subsets', default=str(ROOT / 'data/amdsd_splits/size_subsets.csv'))
    ap.add_argument('--sizes', default='15,30,60,118')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--epoch-budget', choices=['steps', 'fixed'], default='steps',
                    help="'steps' scales epochs by FULL/n so every size takes a "
                         "comparable number of gradient steps; 'fixed' keeps the "
                         "tuned epoch count, so small sizes take proportionally fewer")
    ap.add_argument('--pools', default='attn,mean')
    ap.add_argument('--preds-dir', default=str(ROOT / 'data/amdsd_preds'))
    ap.add_argument('--out', default=str(ROOT / 'results/size_curve_frozen.csv'))
    a = ap.parse_args()

    sizes = [int(x) for x in a.sizes.split(',')]
    seeds = [int(x) for x in a.seeds.split(',')]
    pools = a.pools.split(',')

    sel = pd.read_csv(a.selected)
    df = pd.read_csv(a.manifest)
    S = pd.read_csv(a.subsets)
    X = np.load(a.tokens, mmap_mode='r')
    test = (df.split == 'test').values
    G = df.loc[test, 'group'].values
    modal = {c: {k: sel[sel.cls == c][k].mode().iloc[0]
                 for k in ('L', 'lr', 'wd', 'epochs')} for c in T}

    pdir = Path(a.preds_dir); pdir.mkdir(parents=True, exist_ok=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    print(f'{int(test.sum())} test scans, {len(np.unique(G))} patients | '
          f'sizes {sizes} seeds {seeds} | epoch budget {a.epoch_budget}')

    rows = []
    for n in sizes:
        for s in seeds:
            keep = df.group.isin(subset_groups(S, n, s)).values
            pool = (df.split == 'pool').values & keep       # <- the whole subsampling
            for how in pools:
                for c in T:
                    y = df[f'label_{c}'].values
                    cfg = dict(modal[c])
                    if a.epoch_budget == 'steps':
                        cfg['epochs'] = max(1, int(round(cfg['epochs'] * FULL / n)))
                    tp, vp, vy = fold_models(X, df, pool, test, y, how, cfg, s)
                    ens = tp.mean(0)
                    thr = youden(vy, vp)
                    yt, pred = y[test], ens >= thr
                    rows.append(dict(
                        arm=f'frozen_{how}', pool=how, n_patients=n, seed=s, cls=c,
                        n_train_scans=int(pool.sum()), auprc=AP(yt, ens),
                        auroc=roc_auc_score(yt, ens),
                        recall=recall_score(yt, pred, zero_division=0),
                        spec=((~pred) & (yt == 0)).sum() / max((yt == 0).sum(), 1),
                        thr=thr, **cfg))
                    print(f'  n={n:<4} s{s} {how:<5} {c:<4} '
                          f'AUPRC {rows[-1]["auprc"]:.3f}  thr {thr:.2f}', flush=True)
                    np.savez(pdir / f'sizecurve_frozen_{how}_{c}_n{n}_s{s}.npz',
                             test_p=ens, test_p_folds=tp, test_y=yt, test_g=G,
                             val_p=vp, val_y=vy, thr=thr, n_patients=n, seed=s)
                    pd.DataFrame(rows).to_csv(out, index=False)

    R = pd.DataFrame(rows)
    print('\n=== AUPRC, mean over seeds ===')
    print(R.pivot_table(index=['cls', 'arm'], columns='n_patients',
                        values='auprc').round(3).to_string())
    print(f'\n-> {out} and {pdir}/sizecurve_frozen_*.npz')


if __name__ == '__main__':
    main()
