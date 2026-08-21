#!/usr/bin/env python3
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_pooling import AttnPool, layernorm, pool_fixed

ROOT = Path(__file__).resolve().parent.parent
T = ['IRF', 'SRF', 'PED']

GRIDS = {
    'small': dict(L=[32, 64], lr=[1e-3, 3e-3], wd=[1e-4, 1e-3]),
    'full':  dict(L=[16, 32, 64, 128], lr=[3e-4, 1e-3, 3e-3], wd=[1e-5, 1e-4, 1e-3]),
}


class AttnTracked(AttnPool):

    def fit_tracked(self, Z, y, Zv, yv, max_epochs, every=5, batch=64):
        self.epochs = every
        best = (-np.inf, 0)
        curve = []
        for done in range(every, max_epochs + 1, every):
            self.fit(Z, y, batch=batch)              # `every` more epochs each call
            s = self.decision(Zv)
            ap = AP(yv, s) if yv.min() != yv.max() else np.nan
            curve.append((done, float(ap)))
            if np.isfinite(ap) and ap > best[0]:
                best = (ap, done)
        return best[1], best[0], curve


def main():
    ap_ = argparse.ArgumentParser(
        description='Nested CV selection of the attention head: epochs, L, lr, '
                    'weight decay. Selection happens inside the training folds; '
                    'the held-out test set is touched once at the end.')
    ap_.add_argument('--tokens', required=True)
    ap_.add_argument('--manifest', required=True)
    ap_.add_argument('--grid', choices=list(GRIDS), default='small')
    ap_.add_argument('--max-epochs', type=int, default=80)
    ap_.add_argument('--every', type=int, default=5)
    ap_.add_argument('--classes', nargs='*', default=T)
    ap_.add_argument('--boot', type=int, default=5000)
    ap_.add_argument('--out', default=str(ROOT / 'notebooks/attn_tuned.csv'))
    a = ap_.parse_args()

    grid = [dict(L=L, lr=lr, wd=wd) for L, lr, wd in
            itertools.product(GRIDS[a.grid]['L'], GRIDS[a.grid]['lr'],
                              GRIDS[a.grid]['wd'])]
    df = pd.read_csv(a.manifest)
    X = np.load(a.tokens, mmap_mode='r')
    d = X.shape[2]
    pool = (df.split == 'pool').values
    test = (df.split == 'test').values
    folds = sorted(df.loc[pool, 'fold'].unique())

    print(f'{len(grid)} configs x {len(folds)} outer folds x {len(a.classes)} classes')
    print(f'epochs selected from the inner curve, max {a.max_epochs}, every {a.every}\n')

    out = Path(a.out)
    rec = pd.read_csv(out).to_dict('records') if out.exists() else []
    done = {(r['cls'], r['outer']) for r in rec}

    oof = {c: np.full(len(df), np.nan) for c in a.classes}
    for c in a.classes:
        i = T.index(c)
        y = df[f'label_{c}'].values
        for k in folds:
            if (c, k) in done:
                continue
            outer_tr = pool & (df.fold != k).values
            outer_va = pool & (df.fold == k).values
            inner_va_fold = [f for f in folds if f != k][0]
            inner_tr = outer_tr & (df.fold != inner_va_fold).values
            inner_va = outer_tr & (df.fold == inner_va_fold).values

            Zi = np.asarray(X[inner_tr], np.float32)
            Zv = np.asarray(X[inner_va], np.float32)
            best = None
            for cfg in grid:
                m = AttnTracked(d, L=cfg['L'], lr=cfg['lr'], wd=cfg['wd'], seed=0)
                ep, score, _ = m.fit_tracked(Zi, y[inner_tr].astype(float),
                                             Zv, y[inner_va], a.max_epochs, a.every)
                if best is None or score > best[0]:
                    best = (score, cfg, ep)
                print(f'    {c} fold{k} L={cfg["L"]:<4}lr={cfg["lr"]:<7}'
                      f'wd={cfg["wd"]:<7}ep={ep:<4}inner AP {score:.3f}', flush=True)
            score, cfg, ep = best
            # refit on the full outer-training set with the selected config
            m = AttnPool(d, L=cfg['L'], lr=cfg['lr'], wd=cfg['wd'], epochs=ep, seed=0)
            m.fit(np.asarray(X[outer_tr], np.float32), y[outer_tr].astype(float))
            oof[c][outer_va] = m.decision(np.asarray(X[outer_va], np.float32))
            rec.append(dict(cls=c, outer=int(k), L=cfg['L'], lr=cfg['lr'],
                            wd=cfg['wd'], epochs=ep, inner_ap=score))
            print(f'  -> {c} fold{k} selected L={cfg["L"]} lr={cfg["lr"]} '
                  f'wd={cfg["wd"]} epochs={ep}', flush=True)
            pd.DataFrame(rec).to_csv(out, index=False)
            np.save(out.with_name(f'attn_tuned_oof_{c}.npy'), oof[c])

    R = pd.DataFrame(rec)
    print('\n=== selected hyperparameters per outer fold ===')
    print(R.to_string(index=False))
    print('\nstability (mode across folds):')
    modal = {}
    for c in a.classes:
        s = R[R.cls == c]
        modal[c] = {k: s[k].mode().iloc[0] for k in ('L', 'lr', 'wd', 'epochs')}
        print(f'  {c}: {modal[c]}   unique configs chosen: '
              f'{len(s[["L","lr","wd"]].drop_duplicates())}/{len(s)}')

    # ---- confirm once on the held-out test set --------------------------------
    print('\n=== held-out test set, touched once ===')
    rows = []
    for c in a.classes:
        i = T.index(c)
        y = df[f'label_{c}'].values
        cfg = modal[c]
        m = AttnPool(d, L=int(cfg['L']), lr=float(cfg['lr']), wd=float(cfg['wd']),
                     epochs=int(cfg['epochs']), seed=0)
        m.fit(np.asarray(X[pool], np.float32), y[pool].astype(float))
        p_attn = m.decision(np.asarray(X[test], np.float32))

        Ztr = layernorm(pool_fixed(np.asarray(X[pool], np.float32), 'mean'))
        Zte = layernorm(pool_fixed(np.asarray(X[test], np.float32), 'mean'))
        sc = StandardScaler().fit(Ztr)
        lr_ = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(Ztr), y[pool])
        p_mean = lr_.predict_proba(sc.transform(Zte))[:, 1]

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
        p = min(2 * min((1 + (dd <= 0).sum()) / (B + 1),
                        (1 + (dd >= 0).sum()) / (B + 1)), 1.0)
        rows.append(dict(cls=c, oof_attn=AP(y[pool], oof[c][pool]),
                         test_attn=AP(yt, p_attn), test_mean=AP(yt, p_mean),
                         delta=AP(yt, p_attn) - AP(yt, p_mean),
                         lo=lo, hi=hi, p=p, **cfg))
        r = rows[-1]
        print(f"  {c:<5} OOF {r['oof_attn']:.3f}   test attn {r['test_attn']:.3f}  "
              f"mean {r['test_mean']:.3f}   Δ {r['delta']:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]  p {p:.3f}")

    pd.DataFrame(rows).to_csv(out.with_name('attn_tuned_test.csv'), index=False)
    print(f'\n-> {out} and attn_tuned_test.csv')
    print('OOF here is selection-optimistic (config chosen inside the pool). '
          'The test column is the clean number.')


if __name__ == '__main__':
    main()
