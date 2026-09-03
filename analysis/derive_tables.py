import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'scripts' / 'features'))
import style as S                                              # noqa: E402
from probe_pooling import fit_mean_baseline                     # noqa: E402

T = list(S.CLASSES)
SEEDS = (0, 1, 2)
RESULTS = ROOT / 'results'
PREDS = S.DATA / 'amdsd_preds'
FEATURES = S.DATA / 'amdsd_features'

ENCODER_ARMS = {('RETFound', 'lp'): 'lp_224',
                ('RETFound', 'last4'): 'last4_224',
                ('RETFound', 'full'): 'full_224',
                ('MAE-IN1k', 'lp'): 'mae_in1k_lp_224',
                ('MAE-IN1k', 'last4'): 'mae_in1k_last4_224',
                ('Sup-IN21k', 'lp'): 'sup_in21k_lp_224',
                ('Sup-IN21k', 'last4'): 'sup_in21k_last4_224'}

# Holm is applied within the encoder family only. Depth and resolution carry an
# equivalence claim and a null respectively; a multiplicity correction serves
# neither, so both are reported as differences with intervals.
FAMILIES = [
    ('Depth', [('last-4 - LP', 'last4_224', 'lp_224'),
               ('full FT - last-4', 'full_224', 'last4_224')]),
    ('Resolution', [('384 - 224, full FT', 'full_384', 'full_224'),
                    ('384 - 224, frozen probe', 'probe_384', 'probe_224'),
                    ('448 - 224, last-4', 'last4_448', 'last4_224')]),
    ('Encoder', [('RETFound - MAE-IN1k, LP', 'lp_224', 'mae_in1k_lp_224'),
                 ('RETFound - MAE-IN1k, last-4', 'last4_224',
                  'mae_in1k_last4_224'),
                 ('RETFound - Sup-IN21k, LP', 'lp_224', 'sup_in21k_lp_224'),
                 ('RETFound - Sup-IN21k, last-4', 'last4_224',
                  'sup_in21k_last4_224')]),
]
HOLM_FAMILY = 'Encoder'


def seed_arm(arm):
    """Seed-averaged, fold-ensembled test predictions -> (P, Y, G, per_seed)."""
    Ps, Y, G = [], None, None
    for s in SEEDS:
        fs = sorted(glob.glob(str(PREDS / f'preds_{arm}_f*_s{s}.npz')))
        if not fs:
            continue
        D = [np.load(f, allow_pickle=True) for f in fs]
        for d in D[1:]:
            assert np.array_equal(d['test_y'], D[0]['test_y']), f'{arm}: folds disagree'
        Ps.append(np.mean([d['test_p'] for d in D], axis=0))
        Y, G = D[0]['test_y'], D[0]['test_g']
    if not Ps:
        raise FileNotFoundError(f'no predictions for arm {arm!r}')
    return np.mean(Ps, axis=0), Y, G, Ps


def frozen_probe(man, size):
    """One probe recipe on the cached features, so a size contrast is resolution only."""
    pool, test = (man.split == 'pool').values, (man.split == 'test').values
    F = np.load(FEATURES / f'features_RETFound_mae_natureOCT_{size}.npy')
    F = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-6)
    return np.stack([fit_mean_baseline(F[pool], man[f'label_{c}'].values[pool],
                                       F[test], None, man[f'label_{c}'].values,
                                       pool)[0] for c in T], 1)


def pvalues(man, reps, Y, P, NS):
    rows = []
    for fam, contrasts in FAMILIES:
        for lab, hi_, lo_ in contrasts:
            for i, c in enumerate(T):
                a_, b_ = P[hi_][:, i], P[lo_][:, i]
                d = S.boot_stat(
                    lambda ix: (AP(Y[ix, i], a_[ix]) - AP(Y[ix, i], b_[ix])
                                if Y[ix, i].min() != Y[ix, i].max() else np.nan), reps)
                lo, hi = S.ci(d)
                rows.append(dict(family=fam, comparison=lab, cls=c,
                                 delta=float(AP(Y[:, i], a_) - AP(Y[:, i], b_)),
                                 lo=lo, hi=hi, p=float(S.two_sided_p(d)),
                                 seeds=f'{NS[hi_]}v{NS[lo_]}'))
    F = pd.DataFrame(rows)
    F['p_holm'] = np.nan
    m = F.family == HOLM_FAMILY
    F.loc[m, 'p_holm'] = S.holm(F.loc[m, 'p'].values)
    F['sig'] = np.where(m & (F.p_holm < 0.05), '**',
                        np.where(~m & (F.p < 0.05), '(uncorr)', ''))
    return F


def encoder_grid(reps, G0):
    rows = []
    for (enc, depth), arm in ENCODER_ARMS.items():
        P, Y, G, Ps = seed_arm(arm)
        assert np.array_equal(G, G0), f'{arm}: test groups differ'
        for i, c in enumerate(T):
            d = S.boot_stat(lambda ix: (AP(Y[ix, i], P[ix, i])
                                        if Y[ix, i].min() != Y[ix, i].max()
                                        else np.nan), reps)
            lo, hi = S.ci(d)
            per_seed = [AP(Y[:, i], p[:, i]) for p in Ps]
            rows.append(dict(encoder=enc, depth=depth, cls=c, n_seeds=len(Ps),
                             mean=float(AP(Y[:, i], P[:, i])),
                             sd=float(np.std(per_seed, ddof=1)) if len(Ps) > 1
                             else np.nan, lo=lo, hi=hi))
    return pd.DataFrame(rows).sort_values(['encoder', 'depth', 'cls'])


def main():
    ap = argparse.ArgumentParser(
        description='Regenerate results/pvalues.csv and results/encoder_grid_auprc.csv '
                    'from the saved per-fold predictions.')
    ap.add_argument('--only', nargs='*', choices=['pvalues', 'encoder'])
    a = ap.parse_args()
    want = set(a.only or ['pvalues', 'encoder'])

    man = S.manifest()
    _, Y, G, _ = seed_arm('last4_224')
    reps = S.replicates(np.asarray(G))
    print(f'{len(np.unique(G))} test patients, bootstrap {S.N_BOOT:,}')

    if 'pvalues' in want:
        P, NS = {}, {}
        for arm in sorted({x for _, cs in FAMILIES for c in cs for x in c[1:]}
                          - {'probe_224', 'probe_384'}):
            P[arm], _, _, Ps = seed_arm(arm)
            NS[arm] = len(Ps)
        for size in (224, 384):
            P[f'probe_{size}'], NS[f'probe_{size}'] = frozen_probe(man, size), 1
        F = pvalues(man, reps, Y, P, NS)
        F.to_csv(RESULTS / 'pvalues.csv', index=False)
        print(f"wrote pvalues.csv  ({(F.family == HOLM_FAMILY).sum()} tests in the "
              f"{HOLM_FAMILY} Holm family, {(F.family != HOLM_FAMILY).sum()} uncorrected)")

    if 'encoder' in want:
        E = encoder_grid(reps, G)
        E.to_csv(RESULTS / 'encoder_grid_auprc.csv', index=False)
        print(f'wrote encoder_grid_auprc.csv  ({E.n_seeds.min()}-{E.n_seeds.max()} seeds)')


if __name__ == '__main__':
    main()
