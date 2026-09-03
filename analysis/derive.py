#!/usr/bin/env python3
"""Recompute every number the paper figures draw, and write them to results/*.csv.

Run this once, on a machine that has the restricted inputs (predictions, cached
features, masks). The figure scripts in analysis/figures/ then read only these
CSVs, so the figures rebuild anywhere without the raw data.

AROI-derived outputs are written as results/aroi_*.csv, which .gitignore blocks:
the areas behind them come from masks the AROI licence forbids redistributing.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score as AP, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'scripts/features'))
import style as S
from probe_pooling import fit_mean_baseline, layernorm
from pooling_control import mlp_hidden_for

T = S.CLASSES
ROOT = S.ROOT
RESULTS = ROOT / 'results'
PRED = S.DATA / 'amdsd_preds'
FEATURES = S.DATA / 'amdsd_features'
IMAGES = ROOT.parent / 'data/amdsd/images'
MASKS = ROOT.parent / 'data/amdsd/masks'

PATCH_AMDSD = S.patch_px('amdsd', 224)
PATCH_AROI = S.patch_px('aroi', 224)
EDGES = S.PATCH_EDGES
SUB_BINS = [b for b in range(len(EDGES) - 1) if EDGES[b + 1] <= 1.0]
N_BOOT = S.N_BOOT
MASK_IDX = {'SRF': 1, 'IRF': 2, 'PED': 3}      # as recovered in prep_amdsd.py

# examples for figure 1a: scans whose components straddle one patch
EXAMPLE = {'IRF': '100_18.png', 'SRF': '110_5.png', 'PED': '51_9.png'}
CROP_H = 170

WROTE = []


def write(df, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f'{name}.csv'
    df.to_csv(p, index=False)
    WROTE.append((p, len(df)))
    print(f'  -> results/{name}.csv  ({len(df)} rows)')


def youden(y, p, ts=np.linspace(0.02, 0.98, 97)):
    a_, b_ = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    return float(ts[int(np.argmax([((p >= t) & (y == 1)).sum() / a_ +
                                   ((p < t) & (y == 0)).sum() / b_ - 1 for t in ts]))])


def cluster_reps(groups, n=N_BOOT, seed=S.SEED):
    return S.replicates(np.asarray(groups), n=n, seed=seed)


def ap_ci(y, p, reps):
    b = S.boot_stat(lambda ix: (AP(y[ix], p[ix]) if y[ix].min() != y[ix].max()
                                else np.nan), reps)
    return (float(AP(y, p)), *S.ci(b))


def ap_delta(y, a, b_, reps):
    d = S.boot_stat(lambda ix: (AP(y[ix], a[ix]) - AP(y[ix], b_[ix])
                                if y[ix].min() != y[ix].max() else np.nan), reps)
    return (float(AP(y, a) - AP(y, b_)), *S.ci(d), float(S.two_sided_p(d)))


# --------------------------------------------------------------- shared inputs
def load_amdsd():
    """Manifest, test rows, fold-ensembled last-4 predictions, Youden thresholds."""
    man = S.manifest()
    te = man[man.split == 'test'].reset_index(drop=True)
    p, y, g = S.arm_preds('last4_224')
    thr = dict(zip(T, S.thresholds('last4_224')))
    return man, te, p, y, g, thr


def oof_and_test_scores(man):
    """Per-scan score for every AMD-SD scan: out-of-fold on pool, ensembled on test."""
    D = [np.load(f, allow_pickle=True)
         for f in sorted(glob.glob(str(PRED / 'preds_last4_224_f*_s0.npz')))]
    ens = np.mean([d['test_p'] for d in D], axis=0)
    out = {}
    for i, c in enumerate(T):
        m = dict(zip(man[man.split == 'test'].file, ens[:, i]))
        for k, d in enumerate(D):
            m.update(zip(man[(man.split == 'pool') & (man.fold == k)].file,
                         d['val_p'][:, i]))
        out[c] = m
    return out


def load_aroi():
    p = RESULTS / 'aroi_zeroshot.csv'
    if not p.exists():
        return None
    return pd.read_csv(p)


# --------------------------------------------------------------- figure 1
def fig1(man):
    """Example scans for panel a. Panel b reads the per-lesion CSVs already in
    results/, which prep produces; only the example metadata is new here."""
    if not MASKS.exists():
        print('  masks absent — skipping fig1 examples')
        return
    rows = []
    for c, f in EXAMPLE.items():
        m = np.array(Image.open(MASKS / f).convert('L')) == MASK_IDX[c]
        lab, n = ndimage.label(m, structure=np.ones((3, 3), int))
        px = np.asarray(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)))
        r0 = int(np.clip(round(np.where(m)[0].mean()) - CROP_H // 2,
                         0, m.shape[0] - CROP_H))
        rows.append(dict(cls=c, file=f, crop_top=r0, crop_h=CROP_H, n_components=n,
                         min_px=int(px.min()), max_px=int(px.max()),
                         min_patches=px.min() / PATCH_AMDSD,
                         max_patches=px.max() / PATCH_AMDSD))
    write(pd.DataFrame(rows), 'fig1_examples')


# --------------------------------------------------------------- figure 2
def fig2(y, g):
    arms = ['lp_224', 'last4_224', 'last4_448', 'full_224', 'full_384']
    trainable = {'lp_224': 0.003, 'last4_224': 50.0, 'last4_448': 50.0,
                 'full_224': 303.0, 'full_384': 303.0}
    depth = {'lp_224': 'linear probe', 'last4_224': 'last-4', 'last4_448': 'last-4',
             'full_224': 'full FT', 'full_384': 'full FT'}
    size = {'lp_224': 224, 'last4_224': 224, 'last4_448': 448,
            'full_224': 224, 'full_384': 384}
    reps = cluster_reps(g)
    rows = []
    for a in arms:
        p, ya, _ = S.arm_preds(a)
        assert np.array_equal(ya, y), f'{a}: label vector differs'
        for i, c in enumerate(T):
            est, lo, hi = ap_ci(y[:, i], p[:, i], reps)
            rows.append(dict(arm=a, depth=depth[a], input_size=size[a], cls=c,
                             auprc=est, lo=lo, hi=hi, trainable_M=trainable[a],
                             prevalence=float(y[:, i].mean()),
                             n_seeds=len(glob.glob(str(PRED / f'preds_{a}_f0_s*.npz')))))
    write(pd.DataFrame(rows), 'fig2_arms_auprc')


# --------------------------------------------------------------- figure 3
def recall_bins(u, hit, g, n_boot=2000, seed=0):
    """Recall per patch-unit bin, with a patient bootstrap. Bins failing the
    reporting minimum are still returned, flagged, so the count strip can show
    why a point is missing."""
    rng = np.random.default_rng(seed)
    out = []
    for b in range(len(EDGES) - 1):
        m = (u >= EDGES[b]) & (u < EDGES[b + 1])
        n, npat = int(m.sum()), len(np.unique(g[m])) if m.sum() else 0
        ok = n >= S.MIN_SCANS and npat >= S.MIN_PATIENTS
        row = dict(bin=b, lo_edge=EDGES[b], hi_edge=EDGES[b + 1],
                   x=float(np.sqrt(EDGES[b] * EDGES[b + 1])), n=n, n_patients=npat,
                   reportable=ok, recall=np.nan, lo=np.nan, hi=np.nan)
        if ok:
            h, gg = hit[m], g[m]
            ug = np.unique(gg)
            gi = {v: np.flatnonzero(gg == v) for v in ug}
            bs = [h[np.concatenate([gi[v] for v in rng.choice(ug, len(ug))])].mean()
                  for _ in range(n_boot)]
            row.update(recall=float(h.mean()), lo=float(np.percentile(bs, 2.5)),
                       hi=float(np.percentile(bs, 97.5)))
        out.append(row)
    return pd.DataFrame(out)


def fig3(man, te, thr, scores, Z):
    rows = []
    for c in T:
        pos = (man[f'label_{c}'] == 1).values
        u = (man.loc[pos, f'area_{c}'] / PATCH_AMDSD).values
        hit = (man.file.map(scores[c]).values[pos] >= thr[c]).astype(int)
        rows.append(recall_bins(u, hit, man.loc[pos, 'group'].values)
                    .assign(dataset='AMD-SD', cls=c, n_positive=int(pos.sum())))
    write(pd.concat(rows, ignore_index=True), 'fig3_recall_bins')

    if Z is not None:
        thr_aroi = {c: youden(Z[f'label_{c}'].values, Z[f'p_{c}'].values) for c in T}
        rows = []
        for c in T:
            pos = (Z[f'label_{c}'] == 1).values
            u = (Z.loc[pos, f'area_{c}'] / PATCH_AROI).values
            hit = (Z[f'p_{c}'].values[pos] >= thr_aroi[c]).astype(int)
            rows.append(recall_bins(u, hit, Z.loc[pos, 'patient'].values)
                        .assign(dataset='AROI, recalibrated', cls=c,
                                n_positive=int(pos.sum())))
        write(pd.concat(rows, ignore_index=True), 'aroi_fig3_recall_bins')

    # panel b: the continuous score behind the binary recall, test scans only
    D = [np.load(f, allow_pickle=True)
         for f in sorted(glob.glob(str(PRED / 'preds_last4_224_f*_s0.npz')))]
    Pt, Yt = np.mean([d['test_p'] for d in D], axis=0), D[0]['test_y']
    rows, meta = [], []
    for i, c in enumerate(T):
        u = te[f'area_{c}'].values / PATCH_AMDSD
        rows.append(pd.DataFrame(dict(cls=c, label=Yt[:, i].astype(int),
                                      area_patches=u, score=Pt[:, i])))
        rho, _ = spearmanr(u[Yt[:, i] == 1], Pt[Yt[:, i] == 1, i])
        meta.append(dict(cls=c, threshold=thr[c], spearman_rho=float(rho),
                         n_positive=int((Yt[:, i] == 1).sum())))
    write(pd.concat(rows, ignore_index=True), 'fig3_scores')
    write(pd.DataFrame(meta), 'fig3_thresholds')


# --------------------------------------------------------------- figure 4
def fig4(te, thr, Z, n_grid=499):
    D = [np.load(f, allow_pickle=True)
         for f in sorted(glob.glob(str(PRED / 'preds_last4_224_f*_s0.npz')))]
    Pt, Yt = np.mean([d['test_p'] for d in D], axis=0), D[0]['test_y']
    ts = np.linspace(0.002, 0.998, n_grid)
    rows = []
    for i, c in enumerate(T):
        y, p = Yt[:, i], Pt[:, i]
        sub = (te[f'area_{c}'].values / PATCH_AMDSD < 1.0) & (y == 1)
        neg = y == 0
        rows.append(pd.DataFrame(dict(
            cls=c, threshold=ts,
            specificity=[(p[neg] < t).mean() for t in ts],
            recall_subpatch=[(p[sub] >= t).mean() for t in ts],
            recall_all=[(p[y == 1] >= t).mean() for t in ts],
            n_subpatch=int(sub.sum()), n_positive=int((y == 1).sum()),
            youden_threshold=thr[c])))
    write(pd.concat(rows, ignore_index=True), 'fig4_threshold_sweep')

    if Z is None:
        return
    thr_aroi = {c: youden(Z[f'label_{c}'].values, Z[f'p_{c}'].values) for c in T}
    reps = cluster_reps(Z.patient.values, n=2000)
    rows = []
    for c in T:
        y, p = Z[f'label_{c}'].values, Z[f'p_{c}'].values
        for src, t in (('AMD-SD threshold', thr[c]), ('refit on AROI', thr_aroi[c])):
            b = np.array([(p[ix][y[ix] == 1] >= t).mean() if (y[ix] == 1).any()
                          else np.nan for ix in reps], float)
            b = b[np.isfinite(b)]
            lo, hi = S.ci(b)
            rows.append(dict(cls=c, threshold_source=src, threshold=t,
                             recall=float((p[y == 1] >= t).mean()), lo=lo, hi=hi,
                             specificity=float((p[y == 0] < t).mean()),
                             n_positive=int((y == 1).sum()),
                             n_negative=int((y == 0).sum()),
                             prevalence=float(y.mean())))
    write(pd.DataFrame(rows), 'aroi_fig4_operating_points')


# --------------------------------------------------------------- figure 5
def fig5(man):
    pool = (man.split == 'pool').values
    test = (man.split == 'test').values
    folds = man.fold.values
    F = layernorm(S.features('RETFound'))
    h = mlp_hidden_for(F.shape[1], 64)

    P, Yt, G = {}, None, None
    for i, c in enumerate(T):
        d0 = np.load(PRED / f'frozen_attn_{c}_s0.npz')
        if Yt is None:
            Yt, G = np.zeros((len(d0['test_y']), len(T)), int), d0['test_g']
        Yt[:, i] = d0['test_y']
        P[('frozen', 'attn', c)] = np.mean(
            [np.load(PRED / f'frozen_attn_{c}_s{s}.npz')['test_p'] for s in (0, 1, 2)],
            axis=0)
        P[('frozen', 'mean', c)] = np.load(PRED / f'frozen_mean_{c}_s0.npz')['test_p']

        y = man[f'label_{c}'].values
        mlp = []
        for k in range(5):
            tr = pool & (folds != k)
            sc = StandardScaler().fit(F[tr])
            mdl = MLPClassifier(hidden_layer_sizes=(h,), alpha=1e-2, max_iter=800,
                                random_state=0, early_stopping=True,
                                n_iter_no_change=20)
            mdl.fit(sc.transform(F[tr]), y[tr])
            mlp.append(mdl.predict_proba(sc.transform(F[test]))[:, 1])
        P[('frozen', 'mlp', c)] = np.mean(mlp, axis=0)

    for tag, pat in (('mean', 'preds_last4_224_f*_s0.npz'),
                     ('attn', 'preds_last4_224_attn_f*_s0.npz')):
        D = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(str(PRED / pat)))]
        Pm = np.mean([d['test_p'] for d in D], axis=0)
        for i, c in enumerate(T):
            P[('last-4', tag, c)] = Pm[:, i]

    reps = cluster_reps(G)
    params = {('frozen', 'mean'): 0.003, ('frozen', 'attn'): 0.069,
              ('last-4', 'mean'): 50.0, ('last-4', 'attn'): 50.07}
    rows = []
    for dp in ('frozen', 'last-4'):
        for pl in ('mean', 'attn'):
            for i, c in enumerate(T):
                est, lo, hi = ap_ci(Yt[:, i], P[(dp, pl, c)], reps)
                rows.append(dict(depth=dp, pooling=pl, cls=c, auprc=est, lo=lo, hi=hi,
                                 trainable_M=params[(dp, pl)]))
    write(pd.DataFrame(rows), 'fig5_pooling_arms')

    rows = []
    for a_, b_, lab in (('attn', 'mean', 'attention - mean'),
                        ('attn', 'mlp', 'attention - matched MLP'),
                        ('mlp', 'mean', 'matched MLP - mean')):
        for i, c in enumerate(T):
            d, lo, hi, pv = ap_delta(Yt[:, i], P[('frozen', a_, c)],
                                     P[('frozen', b_, c)], reps)
            rows.append(dict(contrast=lab, cls=c, delta=d, lo=lo, hi=hi, p=pv,
                             mlp_hidden=h,
                             mlp_params=F.shape[1] * h + 2 * h + 1))
    write(pd.DataFrame(rows), 'fig5_capacity_control')


# --------------------------------------------------------------- figure 6
def fig6(man, y, g):
    pool = (man.split == 'pool').values
    test = (man.split == 'test').values
    reps = cluster_reps(g)
    P = {a: S.arm_preds(a)[0] for a in
         ('lp_224', 'last4_224', 'full_224', 'full_384', 'last4_448',
          'mae_in1k_lp_224', 'sup_in21k_lp_224',
          'mae_in1k_last4_224', 'sup_in21k_last4_224')}

    # one recipe at both resolutions, so the contrast is resolution and not head
    for size in (224, 384):
        F = layernorm(np.load(FEATURES / f'features_RETFound_mae_natureOCT_{size}.npy'))
        P[f'probe_{size}'] = np.stack(
            [fit_mean_baseline(F[pool], man[f'label_{c}'].values[pool], F[test], None,
                               man[f'label_{c}'].values, pool)[0] for c in T], 1)

    PANELS = [
        ('Depth', [('last-4 - linear probe', 'last4_224', 'lp_224'),
                   ('full FT - last-4', 'full_224', 'last4_224')]),
        ('Resolution', [('384 - 224, linear probe', 'probe_384', 'probe_224'),
                        ('384 - 224, full FT', 'full_384', 'full_224'),
                        ('448 - 224, last-4', 'last4_448', 'last4_224')]),
        ('Encoder', [('RETFound - Sup-IN21k, LP', 'lp_224', 'sup_in21k_lp_224'),
                     ('RETFound - Sup-IN21k, last-4', 'last4_224',
                      'sup_in21k_last4_224'),
                     ('RETFound - MAE-IN1k, LP', 'lp_224', 'mae_in1k_lp_224'),
                     ('RETFound - MAE-IN1k, last-4', 'last4_224',
                      'mae_in1k_last4_224')]),
    ]
    rows = []
    for panel, groups in PANELS:
        for lab, hi_, lo_ in groups:
            for i, c in enumerate(T):
                d, lo, hi, pv = ap_delta(y[:, i], P[hi_][:, i], P[lo_][:, i], reps)
                rows.append(dict(panel=panel, contrast=lab, cls=c, metric='AUPRC',
                                 delta=d, lo=lo, hi=hi, p=pv, n=np.nan))
    F_ = pd.DataFrame(rows)
    # Holm inside the encoder panel only: the one place a marked row is worth having
    F_['p_holm'] = np.nan
    m = F_.panel == 'Encoder'
    F_.loc[m, 'p_holm'] = S.holm(F_.loc[m, 'p'].values)

    cr = pd.read_csv(RESULTS / 'crossers.csv').query("name == 'crossers'")
    F_ = pd.concat([F_, pd.DataFrame([
        dict(panel='Patch crossing', contrast='448 - 224 on crossing scans', cls=r.cls,
             metric='recall', delta=r.delta, lo=r.lo, hi=r.hi, p=np.nan, n=int(r.n),
             p_holm=np.nan) for r in cr.itertuples()])], ignore_index=True)
    write(F_, 'fig6_forest')


# --------------------------------------------------------------- figure 7
def _binned(D_, c, patch, probs, t, gcol):
    pos = (D_[f'label_{c}'] == 1).values
    u = (D_.loc[pos, f'area_{c}'] / patch).values
    hit = (probs >= t).astype(int)[pos]
    g = D_.loc[pos, gcol].values
    out = {}
    for b in range(len(EDGES) - 1):
        m = (u >= EDGES[b]) & (u < EDGES[b + 1])
        if m.sum() < S.MIN_SCANS or len(np.unique(g[m])) < S.MIN_PATIENTS:
            continue
        out[b] = dict(n=int(m.sum()), recall=float(hit[m].mean()))
    return out


def _standardised(bins, ref):
    """Sub-patch recall reweighted to a reference bin composition."""
    keep = [b for b in SUB_BINS if b in bins and b in ref]
    if not keep:
        return np.nan
    w = np.array([ref[b] for b in keep], float)
    w /= w.sum()
    return float((np.array([bins[b]['recall'] for b in keep]) * w).sum())


def fig7(man, thr, scores, Z):
    if Z is None:
        print('  aroi_zeroshot.csv absent — skipping fig7')
        return
    thr_aroi = {c: youden(Z[f'label_{c}'].values, Z[f'p_{c}'].values) for c in T}
    reps = cluster_reps(Z.patient.values)

    D = [np.load(f, allow_pickle=True)
         for f in sorted(glob.glob(str(PRED / 'preds_last4_224_f*_s0.npz')))]
    Pt, Yt = np.mean([d['test_p'] for d in D], axis=0), D[0]['test_y']

    rows = []
    for i, c in enumerate(T):
        y, p = Z[f'label_{c}'].values, Z[f'p_{c}'].values
        b = [roc_auc_score(y[ix], p[ix]) for ix in reps if len(np.unique(y[ix])) > 1]
        lo, hi = S.ci(b)
        rows.append(dict(cls=c, auroc_aroi=float(roc_auc_score(y, p)), lo=lo, hi=hi,
                         auroc_amdsd=float(roc_auc_score(Yt[:, i], Pt[:, i])),
                         n_positive=int(y.sum()), n_negative=int((y == 0).sum()),
                         prevalence=float(y.mean()),
                         n_patients=int(Z.patient.nunique())))
    write(pd.DataFrame(rows), 'aroi_fig7_auroc')

    rows = []
    for c in T:
        A = _binned(man, c, PATCH_AMDSD, man.file.map(scores[c]).values, thr[c], 'group')
        ref = {b: v['n'] for b, v in A.items()}
        rows.append(dict(cls=c, arm='AMD-SD, own thresholds', threshold=thr[c],
                         recall_std=_standardised(A, ref)))
        for lab, t in (('AROI, AMD-SD thresholds', thr[c]),
                       ('AROI, refit thresholds', thr_aroi[c])):
            rows.append(dict(cls=c, arm=lab, threshold=t,
                             recall_std=_standardised(
                                 _binned(Z, c, PATCH_AROI, Z[f'p_{c}'].values, t,
                                         'patient'), ref)))
    write(pd.DataFrame(rows), 'aroi_fig7_recall')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only', nargs='*', help='figure numbers, e.g. --only 2 5')
    a = ap.parse_args()
    want = set(a.only or [str(i) for i in range(1, 8)])

    man, te, _, y, g, thr = load_amdsd()
    scores = oof_and_test_scores(man)
    Z = load_aroi()
    print(f'{len(man)} scans, {len(te)} test, {len(np.unique(g))} test patients, '
          f'bootstrap {N_BOOT:,}')
    if '1' in want:
        print('figure 1'); fig1(man)
    if '2' in want:
        print('figure 2'); fig2(y, g)
    if '3' in want:
        print('figure 3'); fig3(man, te, thr, scores, Z)
    if '4' in want:
        print('figure 4'); fig4(te, thr, Z)
    if '5' in want:
        print('figure 5'); fig5(man)
    if '6' in want:
        print('figure 6'); fig6(man, y, g)
    if '7' in want:
        print('figure 7'); fig7(man, thr, scores, Z)
    print(f'\n{len(WROTE)} files written')


if __name__ == '__main__':
    main()
