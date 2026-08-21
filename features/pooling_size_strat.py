#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_pooling import (AttnPool, layernorm, pool_fixed,
                           fit_mean_baseline, CGRID)
from pooling_control import mlp_hidden_for

ROOT = Path(__file__).resolve().parent.parent
T = ['IRF', 'SRF', 'PED']
PATCH = (380 / 14) * (570 / 14)
EDGES = np.logspace(np.log10(0.03), np.log10(30), 8)
MIN_SCANS, MIN_PATIENTS = 15, 5


def youden(y, p):
    ts = np.linspace(0.02, 0.98, 97)
    a_ = max((y == 1).sum(), 1); b_ = max((y == 0).sum(), 1)
    return ts[int(np.argmax([((p >= t) & (y == 1)).sum() / a_ +
                             ((p < t) & (y == 0)).sum() / b_ - 1 for t in ts]))]


def holm(p):
    p = np.asarray(p, float); m = len(p)
    adj, run = np.empty(m), 0.0
    for k, j in enumerate(p.argsort()):
        run = max(run, (m - k) * p[j]); adj[j] = min(run, 1.0)
    return adj


def fit_oof(X, df, pool, y, how, seed, epochs, L, h):
    oof = np.full(len(df), np.nan)
    for k in range(5):
        tr = pool & (df.fold != k).values
        va = pool & (df.fold == k).values
        if how == 'attn':
            m = AttnPool(X.shape[2], L=L, epochs=epochs, seed=seed).fit(
                np.asarray(X[tr], np.float32), y[tr].astype(float))
            oof[va] = m.decision(np.asarray(X[va], np.float32))
        else:
            Ztr = layernorm(pool_fixed(np.asarray(X[tr], np.float32), 'mean'))
            Zva = layernorm(pool_fixed(np.asarray(X[va], np.float32), 'mean'))
            if how == 'mean':
                oof[va], _ = fit_mean_baseline(Ztr, y[tr], Zva, None, y, tr)
            else:
                sc = StandardScaler().fit(Ztr)
                mdl = MLPClassifier(hidden_layer_sizes=(h,), alpha=1e-2,
                                    max_iter=800, random_state=seed,
                                    early_stopping=True, n_iter_no_change=20)
                mdl.fit(sc.transform(Ztr), y[tr])
                oof[va] = mdl.predict_proba(sc.transform(Zva))[:, 1]
    return oof


def main():
    ap = argparse.ArgumentParser(
        description='Does the attention-pooling gain concentrate on small lesions? '
                    'Delta recall by patch-unit bin, three seeds per arm.')
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--L', type=int, default=64)
    ap.add_argument('--boot', type=int, default=5000)
    ap.add_argument('--out', default=str(ROOT / 'notebooks/pooling_size_strat.csv'))
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(',')]
    df = pd.read_csv(a.manifest)
    X = np.load(a.tokens, mmap_mode='r')
    h = mlp_hidden_for(X.shape[2], a.L)
    pool = (df.split == 'pool').values
    G = df['group'].values

    # every hyperparameter recorded, so a rerun can be matched exactly
    print(f'tokens {X.shape}   seeds {seeds}   epochs {a.epochs}   L {a.L}   '
          f'lr 3e-3   wd 1e-4   mlp_hidden {h}   boot {a.boot}')

    OOF, rows = {}, []
    for how in ('mean', 'mean+mlp', 'attn'):
        for s in seeds:
            if how == 'mean' and s != seeds[0]:
                OOF[(how, s)] = OOF[(how, seeds[0])]      # deterministic, no seed effect
                continue
            for i, c in enumerate(T):
                y = df[f'label_{c}'].values
                oof = fit_oof(X, df, pool, y, how, s, a.epochs, a.L, h)
                OOF[(how, s, c)] = oof
                rows.append(dict(pool=how, seed=s, cls=c,
                                 oof_auprc=AP(y[pool], oof[pool])))
                print(f'  {how:<9}s{s} {c:<5} AUPRC {rows[-1]["oof_auprc"]:.3f}',
                      flush=True)
                pd.DataFrame(rows).to_csv(a.out, index=False)
            OOF[(how, s)] = True

    R = pd.DataFrame(rows)
    print('\n=== AUPRC, mean +/- SD over seeds ===')
    g = R.groupby(['pool', 'cls']).oof_auprc.agg(['mean', 'std', 'count'])
    print(g.round(4).to_string())

    # ---------- size-stratified delta recall --------------------------------
    rng = np.random.default_rng(0)
    idx_pool = np.flatnonzero(pool)
    ug = np.unique(G[idx_pool])
    gi = {u: idx_pool[G[idx_pool] == u] for u in ug}
    reps = [np.concatenate([gi[u] for u in rng.choice(ug, len(ug))])
            for _ in range(a.boot)]

    srows = []
    for i, c in enumerate(T):
        y = df[f'label_{c}'].values
        area = df[f'area_{c}'].values / PATCH
        # per-seed hit vectors, each at that arm's own Youden on its own OOF
        HA = np.array([(OOF[('attn', s, c)] >= youden(y[pool], OOF[('attn', s, c)][pool]))
                       .astype(float) for s in seeds])
        hm = OOF[('mean', seeds[0], c)]
        HB = np.array([(hm >= youden(y[pool], hm[pool])).astype(float)])
        for b in range(len(EDGES) - 1):
            m = pool & (y == 1) & (area >= EDGES[b]) & (area < EDGES[b + 1])
            if m.sum() < MIN_SCANS or len(np.unique(G[m])) < MIN_PATIENTS:
                continue
            dA, dB = HA[:, m].mean(1), HB[:, m].mean(1)
            d = np.array([HA[:, ix[m[ix]]].mean() - HB[:, ix[m[ix]]].mean()
                          if m[ix].any() else np.nan for ix in reps])
            d = d[np.isfinite(d)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            B = len(d)
            p = min(2 * min((1 + (d <= 0).sum()) / (B + 1),
                            (1 + (d >= 0).sum()) / (B + 1)), 1.0)
            srows.append(dict(cls=c, bin=f'{EDGES[b]:.2f}-{EDGES[b+1]:.2f}',
                              n=int(m.sum()), n_pat=len(np.unique(G[m])),
                              rec_attn=dA.mean(), sd_attn=dA.std(ddof=1) if len(dA) > 1 else np.nan,
                              rec_mean=dB.mean(), delta=dA.mean() - dB.mean(),
                              lo=lo, hi=hi, p=p))

    S = pd.DataFrame(srows)
    S['p_holm'] = holm(S.p.values)
    S['sig'] = np.where(S.p_holm < 0.05, '**', np.where(S.p < 0.05, '*', ''))
    print(f'\n=== attn - mean recall by size bin ({len(S)} tests, Holm across all) ===')
    print(f"{'cls':<5}{'bin':<13}{'n':>5}{'pat':>5}{'attn':>7}{'SD':>7}{'mean':>7}"
          f"{'Δ':>8}{'95% CI':>20}{'p':>8}{'holm':>8}{'':>4}")
    for r in S.itertuples():
        print(f"{r.cls:<5}{r.bin:<13}{r.n:>5}{r.n_pat:>5}{r.rec_attn:>7.3f}"
              f"{(r.sd_attn if np.isfinite(r.sd_attn) else 0):>7.3f}{r.rec_mean:>7.3f}"
              f"{r.delta:>+8.3f}{f'[{r.lo:+.3f},{r.hi:+.3f}]':>20}"
              f"{('<0.001' if r.p < 0.001 else f'{r.p:.3f}'):>8}"
              f"{('<0.001' if r.p_holm < 0.001 else f'{r.p_holm:.3f}'):>8}{r.sig:>4}")

    sub = S[S.bin.str.startswith(('0.03', '0.08', '0.22'))]
    sup = S[~S.bin.str.startswith(('0.03', '0.08', '0.22'))]
    print(f'\nmean Δ, sub-patch bins  {sub.delta.mean():+.3f}  (n={int(sub.n.sum())})')
    print(f'mean Δ, above-patch bins {sup.delta.mean():+.3f}  (n={int(sup.n.sum())})')
    print('  concentrated in small lesions -> mechanism supported')
    print('  uniform across bins -> just a better head')
    S.to_csv(Path(a.out).with_name('pooling_size_strat_tests.csv'), index=False)
    print(f'\n-> {a.out} and pooling_size_strat_tests.csv')


if __name__ == '__main__':
    main()
