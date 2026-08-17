#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_pooling import AttnPool, layernorm, pool_fixed

ROOT = Path(__file__).resolve().parent.parent
T = ['IRF', 'SRF', 'PED']


def param_count(d, L):
    attn = d * L + L + d + 1          # V, w, c, b
    return attn


def mlp_hidden_for(d, L):
    """Hidden width giving an MLP on mean-pooled features a similar budget."""
    # d*h + h + h + 1 ~= d*L + L + d + 1  ->  h ~= (d*L + d) / (d + 2)
    return max(2, int(round((d * L + d) / (d + 2))))


def main():
    ap = argparse.ArgumentParser(
        description='Capacity control for attention pooling: is the gain from '
                    'where it looks, or just from having more parameters? '
                    'Compares attention pooling against a same-budget MLP on '
                    'mean-pooled features, plus a patient bootstrap on the gap.')
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--L', type=int, default=64)
    ap.add_argument('--boot', type=int, default=5000)
    ap.add_argument('--out', default=str(ROOT / 'notebooks/pooling_control.csv'))
    a = ap.parse_args()

    df = pd.read_csv(a.manifest)
    X = np.load(a.tokens, mmap_mode='r')
    d = X.shape[2]
    h = mlp_hidden_for(d, a.L)
    print(f'tokens {X.shape}   attention params ~{param_count(d, a.L):,}   '
          f'matched MLP hidden={h} (~{d * h + 2 * h + 1:,})')

    pool = (df.split == 'pool').values
    G = df['group'].values

    methods = ['mean', 'mean+mlp', 'attn']
    OOF = {}
    rows = []
    for how in methods:
        for i, c in enumerate(T):
            y = df[f'label_{c}'].values
            oof = np.full(len(df), np.nan)
            for k in range(5):
                tr = pool & (df.fold != k).values
                va = pool & (df.fold == k).values
                if how == 'attn':
                    m = AttnPool(d, L=a.L, epochs=a.epochs).fit(
                        np.asarray(X[tr], np.float32), y[tr].astype(float))
                    oof[va] = m.decision(np.asarray(X[va], np.float32))
                else:
                    Ztr = layernorm(pool_fixed(np.asarray(X[tr], np.float32), 'mean'))
                    Zva = layernorm(pool_fixed(np.asarray(X[va], np.float32), 'mean'))
                    sc = StandardScaler().fit(Ztr)
                    if how == 'mean':
                        from sklearn.linear_model import LogisticRegression
                        mdl = LogisticRegression(C=0.01, max_iter=3000)
                    else:
                        mdl = MLPClassifier(hidden_layer_sizes=(h,), alpha=1e-2,
                                            max_iter=800, random_state=0,
                                            early_stopping=True, n_iter_no_change=20)
                    mdl.fit(sc.transform(Ztr), y[tr])
                    oof[va] = mdl.predict_proba(sc.transform(Zva))[:, 1]
            OOF[(how, c)] = oof
            rows.append(dict(pool=how, cls=c, oof_auprc=AP(y[pool], oof[pool])))
            print(f'  {how:<9}{c:<5} OOF AUPRC {rows[-1]["oof_auprc"]:.3f}', flush=True)
            np.save(Path(a.out).with_name(f'oofc_{how}_{c}.npy'), oof)
            pd.DataFrame(rows).to_csv(a.out, index=False)

    # ---- paired patient bootstrap on the gaps -------------------------------
    rng = np.random.default_rng(0)
    idx_pool = np.flatnonzero(pool)
    ug = np.unique(G[idx_pool])
    gi = {u: idx_pool[G[idx_pool] == u] for u in ug}
    reps = [np.concatenate([gi[u] for u in rng.choice(ug, len(ug))])
            for _ in range(a.boot)]

    print(f'\npaired patient bootstrap, {a.boot} resamples\n')
    print(f"{'comparison':<22}{'class':<5}{'Δ AUPRC':>9}{'95% CI':>20}{'p':>8}")
    out = []
    for hi_, lo_ in [('attn', 'mean'), ('attn', 'mean+mlp'), ('mean+mlp', 'mean')]:
        for c in T:
            y = df[f'label_{c}'].values
            A_, B_ = OOF[(hi_, c)], OOF[(lo_, c)]
            dd = []
            for ix in reps:
                yy = y[ix]
                if yy.min() == yy.max():
                    continue
                dd.append(AP(yy, A_[ix]) - AP(yy, B_[ix]))
            dd = np.array(dd)
            l, hh = np.percentile(dd, [2.5, 97.5])
            B = len(dd)
            p = min(2 * min((1 + (dd <= 0).sum()) / (B + 1),
                            (1 + (dd >= 0).sum()) / (B + 1)), 1.0)
            obs = AP(y[pool], A_[pool]) - AP(y[pool], B_[pool])
            print(f"{hi_ + ' − ' + lo_:<22}{c:<5}{obs:>+9.3f}"
                  f"{f'[{l:+.3f}, {hh:+.3f}]':>20}"
                  f"{('<0.001' if p < 0.001 else f'{p:.3f}'):>8}")
            out.append(dict(hi=hi_, lo=lo_, cls=c, delta=obs, ci_lo=l, ci_hi=hh, p=p))

    pd.DataFrame(out).to_csv(Path(a.out).with_name('pooling_control_tests.csv'),
                             index=False)
    print(f'\n-> {a.out} and pooling_control_tests.csv')


if __name__ == '__main__':
    main()
