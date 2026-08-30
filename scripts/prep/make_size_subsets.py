#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SIZES = [15, 30, 60, 118]


def _interleave(pos, neg, rng):
    """One patient ordering whose every prefix holds the IRF base rate."""
    items = np.concatenate([pos, neg])
    key = np.concatenate([(np.arange(len(pos)) + 0.5) / max(len(pos), 1),
                          (np.arange(len(neg)) + 0.5) / max(len(neg), 1)])
    return items[np.lexsort((rng.random(len(items)), key))]


def _allocate(sizes, cap):
    """Patients per fold at each size: exact totals, folds kept proportional,
    and monotone in n so every subset nests inside the next one up."""
    take, out, tot = {f: 0 for f in cap}, {}, sum(cap.values())
    for n in sorted(sizes):
        n = min(n, tot)
        while sum(take.values()) < n:
            f = min((f for f in cap if take[f] < cap[f]),
                    key=lambda f: (take[f] / cap[f], f))
            take[f] += 1
        out[n] = dict(take)
    return out


def build(df, sizes, seeds):
    """Nested patient subsets of the pool split.

    Sampling is by patient, so a sampled patient contributes all of their scans.
    Within a seed the subsets nest (15 subset of 30 subset of 60 ...), which is what
    makes a learning curve a curve rather than four unrelated draws. Each subset
    keeps the five preset folds populated in proportion and holds the patient-level
    IRF rate, the scarcest of the three labels. At the full pool size every fold is
    taken whole, so that point reproduces the existing runs exactly.
    """
    pool = df[df.split == 'pool']
    pat = (pool.groupby('group')
              .agg(fold=('fold', 'first'), irf=('label_IRF', 'max'))
              .reset_index())
    assert pool.groupby('group').fold.nunique().max() == 1, 'a patient spans folds'

    rows = []
    for s in seeds:
        order = {}
        for f, g in pat.groupby('fold'):
            rng = np.random.default_rng([s, f])
            order[f] = _interleave(rng.permutation(g[g.irf == 1].group.values),
                                   rng.permutation(g[g.irf == 0].group.values), rng)
        alloc = _allocate(sizes, {f: len(v) for f, v in order.items()})
        for n, take in alloc.items():
            for f, k in take.items():
                for gid in order[f][:k]:
                    rows.append(dict(n_patients=n, seed=s, fold=f, group=gid))
    return pd.DataFrame(rows)


def groups(subsets, n, seed):
    s = subsets[(subsets.n_patients == n) & (subsets.seed == seed)]
    if s.empty:
        raise SystemExit(f'no subset for n={n} seed={seed} in the subset file')
    return s.group.values


def main():
    ap = argparse.ArgumentParser(
        description='Nested patient-level subsamples of the pool split, shared by '
                    'every arm of the training-set-size experiment.')
    ap.add_argument('--manifest', default=str(ROOT / 'data/amdsd_splits/manifest.csv'))
    ap.add_argument('--sizes', default=','.join(map(str, SIZES)))
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--out', default=str(ROOT / 'data/amdsd_splits/size_subsets.csv'))
    a = ap.parse_args()

    sizes = [int(x) for x in a.sizes.split(',')]
    seeds = [int(x) for x in a.seeds.split(',')]
    df = pd.read_csv(a.manifest)
    S = build(df, sizes, seeds)

    # nesting and fold coverage are load-bearing; check them here, not at train time
    for s in seeds:
        prev = None
        for n in sorted(sizes):
            g = set(groups(S, n, s))
            assert len(g) == min(n, df[df.split == 'pool'].group.nunique())
            assert prev is None or prev <= g, f'seed {s}: n={n} does not nest'
            assert S[(S.n_patients == n) & (S.seed == s)].fold.nunique() == 5, \
                f'seed {s} n={n}: a fold is empty'
            prev = g

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    S.to_csv(a.out, index=False)

    sc = df[df.split == 'pool'].groupby('group').size()
    pos = df[df.split == 'pool'].groupby('group')[['label_IRF', 'label_SRF', 'label_PED']].max()
    print(f'-> {a.out}  ({len(S)} rows)')
    print(f"{'n':>5}{'seed':>6}{'scans':>8}{'IRF+':>7}{'SRF+':>7}{'PED+':>7}   folds")
    for n in sorted(sizes):
        for s in seeds:
            g = groups(S, n, s)
            f = S[(S.n_patients == n) & (S.seed == s)].fold.value_counts().sort_index()
            print(f'{n:>5}{s:>6}{sc[g].sum():>8}'
                  + ''.join(f'{int(pos.loc[g, c].sum()):>7}'
                            for c in ('label_IRF', 'label_SRF', 'label_PED'))
                  + '   ' + '/'.join(str(int(v)) for v in f.values))


if __name__ == '__main__':
    main()
