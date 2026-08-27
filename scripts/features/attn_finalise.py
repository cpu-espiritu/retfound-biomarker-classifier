#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_pooling import AttnPool, layernorm, pool_fixed, fit_mean_baseline

ROOT = Path(__file__).resolve().parents[2]
T = ['IRF', 'SRF', 'PED']


def main():
    ap = argparse.ArgumentParser(
        description='Refit the attention head with the hyperparameters already '
                    'selected in attn_tuned.csv, then confirm on the held-out test '
                    'set. No grid search: selection is done.')
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--selected', required=True, help='attn_tuned.csv')
    ap.add_argument('--boot', type=int, default=5000)
    ap.add_argument('--out', default=str(ROOT / 'notebooks/attn_final.csv'))
    a = ap.parse_args()

    sel = pd.read_csv(a.selected)
    df = pd.read_csv(a.manifest)
    X = np.load(a.tokens, mmap_mode='r')
    d = X.shape[2]
    pool = (df.split == 'pool').values
    test = (df.split == 'test').values

    print('selected hyperparameters, and how stable they are:')
    modal = {}
    for c in T:
        s = sel[sel.cls == c]
        modal[c] = {k: s[k].mode().iloc[0] for k in ('L', 'lr', 'wd', 'epochs')}
        uniq = len(s[['L', 'lr', 'wd']].drop_duplicates())
        print(f"  {c}: L={modal[c]['L']} lr={modal[c]['lr']} wd={modal[c]['wd']} "
              f"epochs={modal[c]['epochs']}   {uniq}/{len(s)} distinct (L,lr,wd); "
              f"epochs span {s.epochs.min()}-{s.epochs.max()}")

    rows = []
    for c in T:
        i = T.index(c)
        y = df[f'label_{c}'].values
        oof = np.full(len(df), np.nan)
        for r in sel[sel.cls == c].itertuples():
            tr = pool & (df.fold != r.outer).values
            va = pool & (df.fold == r.outer).values
            m = AttnPool(d, L=int(r.L), lr=float(r.lr), wd=float(r.wd),
                         epochs=int(r.epochs), seed=0)
            m.fit(np.asarray(X[tr], np.float32), y[tr].astype(float))
            oof[va] = m.decision(np.asarray(X[va], np.float32))
            print(f'  {c} fold{r.outer} refit', flush=True)
        np.save(Path(a.out).with_name(f'attn_final_oof_{c}.npy'), oof)

        cfg = modal[c]
        m = AttnPool(d, L=int(cfg['L']), lr=float(cfg['lr']), wd=float(cfg['wd']),
                     epochs=int(cfg['epochs']), seed=0)
        m.fit(np.asarray(X[pool], np.float32), y[pool].astype(float))
        p_attn = m.decision(np.asarray(X[test], np.float32))

        Ztr = layernorm(pool_fixed(np.asarray(X[pool], np.float32), 'mean'))
        Zte = layernorm(pool_fixed(np.asarray(X[test], np.float32), 'mean'))
        p_mean, chosen_C = fit_mean_baseline(Ztr, y[pool], Zte, None, y, pool)

        yt = y[test]
        g = df.loc[test, 'group'].values
        rng = np.random.default_rng(0)
        ug = np.unique(g)
        gi = {u: np.flatnonzero(g == u) for u in ug}
        dd = []
        for _ in range(a.boot):
            ix = np.concatenate([gi[u] for u in rng.choice(ug, len(ug))])
            if yt[ix].min() == yt[ix].max():
                continue
            dd.append(AP(yt[ix], p_attn[ix]) - AP(yt[ix], p_mean[ix]))
        dd = np.array(dd)
        lo, hi = np.percentile(dd, [2.5, 97.5])
        B = len(dd)
        pv = min(2 * min((1 + (dd <= 0).sum()) / (B + 1),
                         (1 + (dd >= 0).sum()) / (B + 1)), 1.0)
        rows.append(dict(cls=c, oof_attn=AP(y[pool], oof[pool]),
                         test_attn=AP(yt, p_attn), test_mean=AP(yt, p_mean),
                         delta=AP(yt, p_attn) - AP(yt, p_mean), lo=lo, hi=hi, p=pv,
                         baseline_C=chosen_C, **cfg))
        pd.DataFrame(rows).to_csv(a.out, index=False)
        r = rows[-1]
        print(f"  -> {c}: OOF {r['oof_attn']:.3f} | test attn {r['test_attn']:.3f} "
              f"vs mean {r['test_mean']:.3f}  Δ {r['delta']:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]  p {pv:.3f}", flush=True)

    print(f'\n-> {a.out}')
    print('OOF is selection-optimistic; the test column is the clean number.')


if __name__ == '__main__':
    main()
