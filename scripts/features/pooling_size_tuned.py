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
PATCH = (380 / 14) * (570 / 14)
EDGES = np.logspace(np.log10(0.03), np.log10(30), 8)
MIN_SCANS, MIN_PATIENTS = 15, 5


def youden(y, p, ts=np.linspace(0.02, 0.98, 97)):
    a_, b_ = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    return ts[int(np.argmax([((p >= t) & (y == 1)).sum() / a_ +
                             ((p < t) & (y == 0)).sum() / b_ - 1 for t in ts]))]


def holm(p):
    p = np.asarray(p, float); m = len(p)
    adj, run = np.empty(m), 0.0
    for k, j in enumerate(p.argsort()):
        run = max(run, (m - k) * p[j]); adj[j] = min(run, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser(
        description='Repeat the size-stratified pooling comparison at the '
                    'hyperparameters chosen by nested CV, not the ones picked by '
                    'hand. The earlier run used 40 epochs; IRF selected 5.')
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--selected', required=True, help='attn_tuned.csv')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--boot', type=int, default=5000)
    ap.add_argument('--out', default=str(ROOT / 'results/pooling_size_tuned.csv'))
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(',')]
    sel = pd.read_csv(a.selected)
    df = pd.read_csv(a.manifest)
    X = np.load(a.tokens, mmap_mode='r')
    d = X.shape[2]
    pool = (df.split == 'pool').values
    G = df['group'].values

    modal = {c: {k: sel[sel.cls == c][k].mode().iloc[0]
                 for k in ('L', 'lr', 'wd', 'epochs')} for c in T}
    print('hyperparameters selected by nested CV (previous run used epochs=40):')
    for c in T:
        print(f"  {c}: L={modal[c]['L']} lr={modal[c]['lr']:g} "
              f"wd={modal[c]['wd']:g} epochs={modal[c]['epochs']}")
    print(f'seeds {seeds}\n')

    OOF = {}
    for c in T:
        y = df[f'label_{c}'].values
        cfg = modal[c]
        for s in seeds:
            oof = np.full(len(df), np.nan)
            for k in range(5):
                tr = pool & (df.fold != k).values
                va = pool & (df.fold == k).values
                m = AttnPool(d, L=int(cfg['L']), lr=float(cfg['lr']),
                             wd=float(cfg['wd']), epochs=int(cfg['epochs']), seed=s)
                m.fit(np.asarray(X[tr], np.float32), y[tr].astype(float))
                oof[va] = m.decision(np.asarray(X[va], np.float32))
            OOF[('attn', s, c)] = oof
            print(f'  attn s{s} {c}  OOF AUPRC {AP(y[pool], oof[pool]):.3f}', flush=True)
        oof = np.full(len(df), np.nan)
        for k in range(5):
            tr = pool & (df.fold != k).values
            va = pool & (df.fold == k).values
            Ztr = layernorm(pool_fixed(np.asarray(X[tr], np.float32), 'mean'))
            Zva = layernorm(pool_fixed(np.asarray(X[va], np.float32), 'mean'))
            oof[va], _ = fit_mean_baseline(Ztr, y[tr], Zva, None, y, tr)
        OOF[('mean', c)] = oof
        print(f'  mean      {c}  OOF AUPRC {AP(y[pool], oof[pool]):.3f}', flush=True)

    rng = np.random.default_rng(0)
    idx = np.flatnonzero(pool)
    ug = np.unique(G[idx])
    gi = {u: idx[G[idx] == u] for u in ug}
    reps = [np.concatenate([gi[u] for u in rng.choice(ug, len(ug))])
            for _ in range(a.boot)]

    rows = []
    for c in T:
        y = df[f'label_{c}'].values
        area = df[f'area_{c}'].values / PATCH
        HA = np.array([(OOF[('attn', s, c)] >=
                        youden(y[pool], OOF[('attn', s, c)][pool])).astype(float)
                       for s in seeds])
        hm = OOF[('mean', c)]
        HB = np.array([(hm >= youden(y[pool], hm[pool])).astype(float)])
        for b in range(len(EDGES) - 1):
            m = pool & (y == 1) & (area >= EDGES[b]) & (area < EDGES[b + 1])
            if m.sum() < MIN_SCANS or len(np.unique(G[m])) < MIN_PATIENTS:
                continue
            dA, dB = HA[:, m].mean(1), HB[:, m].mean(1)
            dd = np.array([HA[:, ix[m[ix]]].mean() - HB[:, ix[m[ix]]].mean()
                           if m[ix].any() else np.nan for ix in reps])
            dd = dd[np.isfinite(dd)]
            lo, hi = np.percentile(dd, [2.5, 97.5])
            B = len(dd)
            p = min(2 * min((1 + (dd <= 0).sum()) / (B + 1),
                            (1 + (dd >= 0).sum()) / (B + 1)), 1.0)
            rows.append(dict(cls=c, bin=f'{EDGES[b]:.2f}-{EDGES[b+1]:.2f}',
                             lo_edge=EDGES[b], n=int(m.sum()),
                             n_pat=len(np.unique(G[m])), rec_attn=dA.mean(),
                             sd_attn=dA.std(ddof=1) if len(dA) > 1 else np.nan,
                             rec_mean=dB.mean(), delta=dA.mean() - dB.mean(),
                             lo=lo, hi=hi, p=p))

    S_ = pd.DataFrame(rows)
    S_['p_holm'] = holm(S_.p.values)
    S_['sig'] = np.where(S_.p_holm < 0.05, '**', np.where(S_.p < 0.05, '*', ''))
    S_.to_csv(a.out, index=False)

    print(f'\nattn - mean recall by size bin ({len(S_)} tests, Holm across all)\n')
    print(f"{'cls':<5}{'bin':<13}{'n':>5}{'attn':>7}{'SD':>7}{'mean':>7}{'Δ':>8}"
          f"{'95% CI':>20}{'holm':>8}{'':>4}")
    for r in S_.itertuples():
        print(f"{r.cls:<5}{r.bin:<13}{r.n:>5}{r.rec_attn:>7.3f}"
              f"{(r.sd_attn if np.isfinite(r.sd_attn) else 0):>7.3f}{r.rec_mean:>7.3f}"
              f"{r.delta:>+8.3f}{f'[{r.lo:+.3f},{r.hi:+.3f}]':>20}"
              f"{('<0.001' if r.p_holm < 0.001 else f'{r.p_holm:.3f}'):>8}{r.sig:>4}")

    sub = S_[S_.lo_edge < 1.0]
    sup = S_[S_.lo_edge >= 1.0]
    for lab, g in [('sub-patch', sub), ('above-patch', sup)]:
        if not len(g):
            continue
        head = 1 - g.rec_mean
        print(f'\n{lab:<12} n={int(g.n.sum()):<5} weighted Δ '
              f'{np.average(g.delta, weights=g.n):+.3f}   '
              f'mean share of gap closed {np.average(g.delta / head, weights=g.n):.0%}')
    print('\nconcentrated in small lesions -> size mechanism supported')
    print('uniform or larger above-patch  -> just a better head')
    print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
