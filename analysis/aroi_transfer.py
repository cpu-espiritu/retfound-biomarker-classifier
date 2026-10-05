import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S

T = S.CLASSES
PATCH = {'AMD-SD': S.patch_px('amdsd', 224), 'AROI': S.patch_px('aroi', 224)}
EDGES = S.PATCH_EDGES
MIN_BIN = 20            # scans; below this a bin is merged into a neighbour
EPS = 1e-6


def merge_bins(u, g, edges=EDGES, min_scans=MIN_BIN, min_patients=S.MIN_PATIENTS):
    """Adjacent patch-unit bins merged until each clears the reporting minimum.

    The thinnest bin absorbs whichever neighbour is itself thinner, so merging
    spreads instead of sweeping everything into one end. Returns the surviving
    edges; with 24 patients this usually costs the top IRF bins.
    """
    cuts = list(range(len(edges)))
    def stats(i):
        m = (u >= edges[cuts[i]]) & (u < edges[cuts[i + 1]])
        return int(m.sum()), (len(np.unique(g[m])) if m.sum() else 0)
    while len(cuts) > 2:
        bad = [i for i in range(len(cuts) - 1)
               if stats(i)[0] < min_scans or stats(i)[1] < min_patients]
        if not bad:
            break
        i = min(bad, key=lambda j: stats(j)[0])
        if i == 0:
            drop = 1
        elif i == len(cuts) - 2:
            drop = i
        else:
            drop = i if stats(i - 1)[0] <= stats(i + 1)[0] else i + 1
        cuts.pop(drop)
    return [edges[i] for i in cuts]


def recall_two_thresholds(u, p, g, thr_a, thr_b, edges, n_boot=S.N_BOOT, seed=S.SEED):
    """Recall in each bin under two thresholds, plus their paired difference.

    Both thresholds score the same scans, so the bootstrap resamples patients once
    per replicate and evaluates both — the difference is paired and patient
    difficulty cancels, as everywhere else in this analysis.
    """
    rng = np.random.default_rng(seed)
    out = []
    for b in range(len(edges) - 1):
        m = (u >= edges[b]) & (u < edges[b + 1])
        n = int(m.sum())
        row = dict(lo_edge=edges[b], hi_edge=edges[b + 1],
                   x=float(np.sqrt(edges[b] * edges[b + 1])), n=n,
                   n_patients=int(len(np.unique(g[m]))) if n else 0)
        if n:
            ha, hb, gg = (p[m] >= thr_a), (p[m] >= thr_b), g[m]
            ug = np.unique(gg)
            gi = {v: np.flatnonzero(gg == v) for v in ug}
            A, B, D = [], [], []
            for _ in range(n_boot):
                ix = np.concatenate([gi[v] for v in rng.choice(ug, len(ug))])
                A.append(ha[ix].mean()); B.append(hb[ix].mean())
                D.append(hb[ix].mean() - ha[ix].mean())
            row.update(
                recall_amdsd_thr=float(ha.mean()),
                amdsd_lo=float(np.percentile(A, 2.5)), amdsd_hi=float(np.percentile(A, 97.5)),
                recall_refit_thr=float(hb.mean()),
                refit_lo=float(np.percentile(B, 2.5)), refit_hi=float(np.percentile(B, 97.5)),
                delta=float(hb.mean() - ha.mean()),
                delta_lo=float(np.percentile(D, 2.5)), delta_hi=float(np.percentile(D, 97.5)),
                p=S.two_sided_p(np.array(D)))
        out.append(row)
    return pd.DataFrame(out)


def test1(Z, thr_amd, thr_aroi):
    rows, edges_by_cls = [], {}
    for c in T:
        pos = (Z[f'label_{c}'] == 1).values
        u = (Z.loc[pos, f'area_{c}'] / PATCH['AROI']).values
        p = Z[f'p_{c}'].values[pos]
        g = Z.loc[pos, 'patient'].values
        e = merge_bins(u, g)
        edges_by_cls[c] = e
        R = recall_two_thresholds(u, p, g, thr_amd[c], thr_aroi[c], e)
        merged = len(EDGES) - len(e)
        rows.append(R.assign(cls=c, n_positive=int(pos.sum()),
                             thr_amdsd=thr_amd[c], thr_refit=thr_aroi[c],
                             bins_merged=merged))
        print(f'  {c}: {int(pos.sum())} positive scans, {len(e)-1} bins '
              f'({merged} merge{"s" if merged != 1 else ""})')
    return pd.concat(rows, ignore_index=True), edges_by_cls


def shift_model_recall(man, scores, thr_amd, shift, edges_by_cls,
                       n_boot=S.N_BOOT, seed=S.SEED):
    """Counterfactual: AMD-SD out-of-fold scores moved down by the fitted shift.

    If AROI were AMD-SD with every logit displaced by one constant, then shifting
    AMD-SD's own scores by that constant and applying AMD-SD's own threshold would
    reproduce AROI's recall curve. Only the dataset coefficient is applied, never
    the interaction — the pure parallel-lines counterfactual is the thing on trial,
    and the bins where it misses are where the slope change lives.
    """
    rng = np.random.default_rng(seed)
    out = []
    for c in T:
        pool = (man.split == 'pool') & (man[f'label_{c}'] == 1)
        d = man[pool]
        p0 = d.file.map(scores[c]).values.astype(float)
        u = (d[f'area_{c}'] / PATCH['AMD-SD']).values
        g = d.group.values
        z = np.log(np.clip(p0, EPS, 1 - EPS) / (1 - np.clip(p0, EPS, 1 - EPS)))
        p1 = 1 / (1 + np.exp(-(z + shift[c])))          # shift is negative
        hit = (p1 >= thr_amd[c]).astype(float)
        e = edges_by_cls[c]
        for b in range(len(e) - 1):
            m = (u >= e[b]) & (u < e[b + 1])
            n = int(m.sum())
            row = dict(cls=c, lo_edge=e[b], hi_edge=e[b + 1],
                       x=float(np.sqrt(e[b] * e[b + 1])), n_amdsd=n,
                       n_patients_amdsd=int(len(np.unique(g[m]))) if n else 0,
                       shift=shift[c], predicted=np.nan, pred_lo=np.nan, pred_hi=np.nan)
            if n:
                h, gg = hit[m], g[m]
                ug = np.unique(gg)
                gi = {v: np.flatnonzero(gg == v) for v in ug}
                bs = [h[np.concatenate([gi[v] for v in rng.choice(ug, len(ug))])].mean()
                      for _ in range(n_boot)]
                row.update(predicted=float(h.mean()),
                           pred_lo=float(np.percentile(bs, 2.5)),
                           pred_hi=float(np.percentile(bs, 97.5)))
            out.append(row)
    return pd.DataFrame(out)


def oof_shift(man, scores, Z):
    """The shift refitted with AMD-SD's out-of-fold scores on the left-hand side.

    Out-of-fold scores are optimistic — each fold stopped on the very split it is
    scored against — so they sit above the ensembled test scores (IRF by 1.4 logits).
    Fitting the shift on that same footing lets the bias cancel instead of leaking
    into the counterfactual, at the cost of a baseline that is not clean out-of-sample.
    Reported beside the test-fitted version, never instead of it.
    """
    rows = []
    for c in T:
        d = man[(man.split == 'pool') & (man[f'label_{c}'] == 1)]
        rows.append(pd.DataFrame(dict(
            dataset='AMD-SD', cls=c, patient=d.group.values,
            area_patches=d[f'area_{c}'].values / PATCH['AMD-SD'],
            score=d.file.map(scores[c]).values.astype(float))))
    sc = pd.concat(rows + [aroi_positive_scores(Z)], ignore_index=True)
    return {c: float(fit_interaction(sc[sc.cls == c], n_boot=2000)
                     .set_index('term').loc['dataset_AROI', 'estimate']) for c in T}


def test3(man, thr_amd, M, R, edges_by_cls, Z):
    scores = oof_and_test_scores(man)
    shift = {c: float(M[(M.cls == c) & (M.term == 'dataset_AROI')].estimate.iloc[0])
             for c in T}
    shift2 = oof_shift(man, scores, Z)
    P2 = shift_model_recall(man, scores, thr_amd, shift2, edges_by_cls)[
        ['cls', 'lo_edge', 'predicted']].rename(columns={'predicted': 'predicted_oof_fit'})
    P = shift_model_recall(man, scores, thr_amd, shift, edges_by_cls).merge(
        P2, on=['cls', 'lo_edge'])
    P['shift_oof_fit'] = P.cls.map(shift2)
    obs = R[['cls', 'lo_edge', 'n', 'n_patients', 'recall_amdsd_thr',
             'amdsd_lo', 'amdsd_hi']].rename(
        columns={'n': 'n_aroi', 'n_patients': 'n_patients_aroi',
                 'recall_amdsd_thr': 'observed', 'amdsd_lo': 'obs_lo',
                 'amdsd_hi': 'obs_hi'})
    C = P.merge(obs, on=['cls', 'lo_edge'], how='left')
    C['residual'] = C.observed - C.predicted
    C['residual_oof_fit'] = C.observed - C.predicted_oof_fit
    for c in T:
        d = C[C.cls == c].dropna(subset=['observed', 'predicted'])
        inside = int(((d.observed >= d.pred_lo) & (d.observed <= d.pred_hi)).sum())
        print(f'  {c}: shift {shift[c]:+.2f} (OOF-fit {shift2[c]:+.2f})  '
              f'MAE {np.abs(d.residual).mean():.3f} '
              f'(OOF-fit {np.abs(d.residual_oof_fit).mean():.3f})  '
              f'{inside}/{len(d)} bins inside the predicted interval')
    return C


def oof_and_test_scores(man):
    """Per-scan score for every AMD-SD scan: out-of-fold on pool, ensembled on test.

    Mirrors the helper of the same name in derive.py; duplicated rather than
    imported so this script does not depend on that one's import side effects.
    """
    fs = sorted(glob.glob(str(S.DATA / 'amdsd_preds/preds_last4_224_f*_s0.npz')))
    D = [np.load(f, allow_pickle=True) for f in fs]
    ens = np.mean([d['test_p'] for d in D], axis=0)
    out = {}
    for i, c in enumerate(T):
        m = dict(zip(man[man.split == 'test'].file, ens[:, i]))
        for k, d in enumerate(D):
            m.update(zip(man[(man.split == 'pool') & (man.fold == k)].file,
                         d['val_p'][:, i]))
        out[c] = m
    return out


def amdsd_positive_scores(man):
    """Test-split positives with the 5-fold ensembled last-4 score, matching §4."""
    fs = sorted(glob.glob(str(S.DATA / 'amdsd_preds/preds_last4_224_f*_s0.npz')))
    D = [np.load(f, allow_pickle=True) for f in fs]
    P = np.mean([d['test_p'] for d in D], axis=0)
    te = man.query("split == 'test'").reset_index(drop=True)
    assert len(te) == len(P), 'test rows and predictions disagree'
    out = []
    for i, c in enumerate(T):
        pos = (te[f'label_{c}'] == 1).values
        out.append(pd.DataFrame(dict(
            dataset='AMD-SD', cls=c, patient=te.loc[pos, 'group'].values,
            area_patches=te.loc[pos, f'area_{c}'].values / PATCH['AMD-SD'],
            score=P[pos, i])))
    return pd.concat(out, ignore_index=True)


def aroi_positive_scores(Z):
    out = []
    for c in T:
        pos = (Z[f'label_{c}'] == 1).values
        out.append(pd.DataFrame(dict(
            dataset='AROI', cls=c, patient=Z.loc[pos, 'patient'].values,
            area_patches=Z.loc[pos, f'area_{c}'].values / PATCH['AROI'],
            score=Z[f'p_{c}'].values[pos])))
    return pd.concat(out, ignore_index=True)


def fit_interaction(d, n_boot=S.N_BOOT, seed=S.SEED):
    """logit(score) ~ log(area) + dataset + log(area) x dataset.

    Ordinary least squares, with intervals from a patient-level cluster bootstrap
    resampled within dataset so both stay represented. The dataset coefficient is
    the vertical shift at one patch; the interaction is the change in slope.
    """
    d = d[d.area_patches > 0]
    y = np.log(np.clip(d.score.values, EPS, 1 - EPS) /
               (1 - np.clip(d.score.values, EPS, 1 - EPS)))
    x = np.log(d.area_patches.values)
    D = (d.dataset.values == 'AROI').astype(float)
    X = np.column_stack([np.ones_like(x), x, D, x * D])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    rng = np.random.default_rng(seed)
    idx = {ds: {u: np.flatnonzero((d.dataset.values == ds) & (d.patient.values == u))
                for u in np.unique(d.patient.values[d.dataset.values == ds])}
           for ds in ('AMD-SD', 'AROI')}
    B = []
    for _ in range(n_boot):
        take = []
        for ds, gi in idx.items():
            ug = list(gi)
            take += [gi[v] for v in rng.choice(ug, len(ug))]
        ix = np.concatenate(take)
        try:
            B.append(np.linalg.lstsq(X[ix], y[ix], rcond=None)[0])
        except np.linalg.LinAlgError:
            continue
    B = np.array(B)
    names = ['intercept', 'log_area', 'dataset_AROI', 'log_area_x_AROI']
    return pd.DataFrame([
        dict(term=n, estimate=float(beta[i]),
             lo=float(np.percentile(B[:, i], 2.5)),
             hi=float(np.percentile(B[:, i], 97.5)),
             p=S.two_sided_p(B[:, i] - 0.0))
        for i, n in enumerate(names)])


def test2(sc):
    rows = []
    for c in T:
        d = sc[sc.cls == c]
        f = fit_interaction(d)
        n = d.groupby('dataset').size()
        rows.append(f.assign(cls=c, n_amdsd=int(n.get('AMD-SD', 0)),
                             n_aroi=int(n.get('AROI', 0))))
        inter = f[f.term == 'log_area_x_AROI'].iloc[0]
        shift = f[f.term == 'dataset_AROI'].iloc[0]
        print(f"  {c}: shift {shift.estimate:+.2f} (p {shift.p:.3f}), "
              f"interaction {inter.estimate:+.3f} (p {inter.p:.3f})")
    return pd.concat(rows, ignore_index=True)


def figure(R, sc, M):
    fig, axes = plt.subplots(2, 3, figsize=(S.WIDTH, 5.0))
    for j, c in enumerate(T):
        ax = axes[0, j]
        d = R[R.cls == c]
        ax.errorbar(d.x, d.recall_amdsd_thr,
                    yerr=[d.recall_amdsd_thr - d.amdsd_lo, d.amdsd_hi - d.recall_amdsd_thr],
                    marker='s', color='#D55E00', lw=1.4, capsize=2,
                    label="AMD-SD threshold, unchanged")
        ax.errorbar(d.x, d.recall_refit_thr,
                    yerr=[d.recall_refit_thr - d.refit_lo, d.refit_hi - d.recall_refit_thr],
                    marker='o', color='#0072B2', lw=1.4, capsize=2,
                    label="refit on AROI")
        ax.axvline(1.0, color='#999999', lw=0.8, ls=':')
        ax.set_xscale('log'); ax.set_ylim(-0.05, 1.05)
        ax.set_title(c); ax.set_xlabel('lesion area (patches)')
        if j == 0:
            ax.set_ylabel('recall, AROI')

        ax = axes[1, j]
        for ds, col in (('AMD-SD', '#999999'), ('AROI', '#0072B2')):
            e = sc[(sc.cls == c) & (sc.dataset == ds) & (sc.area_patches > 0)]
            ax.scatter(e.area_patches, e.score, s=3, alpha=0.25, color=col,
                       lw=0, label=ds)
            b = M[M.cls == c].set_index('term').estimate
            xs = np.logspace(np.log10(max(e.area_patches.min(), 1e-3)),
                             np.log10(e.area_patches.max()), 50)
            z = b['intercept'] + b['log_area'] * np.log(xs)
            if ds == 'AROI':
                z = z + b['dataset_AROI'] + b['log_area_x_AROI'] * np.log(xs)
            ax.plot(xs, 1 / (1 + np.exp(-z)), color=col, lw=2.0)
        ax.axvline(1.0, color='#999999', lw=0.8, ls=':')
        ax.set_xscale('log'); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('lesion area (patches)')
        if j == 0:
            ax.set_ylabel('score, positive scans')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.50), fontsize=8)
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.subplots_adjust(hspace=0.75)
    return fig


def figure_shift(C):
    """Predicted against observed, and both against lesion size."""
    d = C.dropna(subset=['observed', 'predicted'])
    fig, axes = plt.subplots(1, 4, figsize=(S.WIDTH, 2.1),
                             gridspec_kw=dict(width_ratios=[1.15, 1, 1, 1]))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color='#999999', lw=0.9, ls='--', zorder=1)
    for c in T:
        e = d[d.cls == c]
        ax.scatter(e.predicted, e.observed, s=22, color=S.CLASS_COLOR[c],
                   label=c, zorder=3, lw=0)
        ax.errorbar(e.predicted, e.observed,
                    yerr=[e.observed - e.obs_lo, e.obs_hi - e.observed],
                    fmt='none', ecolor=S.CLASS_COLOR[c], alpha=0.45, lw=0.9, zorder=2)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('predicted by pure shift')
    ax.set_ylabel('observed on AROI')
    ax.set_title('per bin', fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc='upper left')
    for j, c in enumerate(T):
        ax = axes[j + 1]
        e = d[d.cls == c]
        ax.plot(e.x, e.predicted, marker='o', color='#999999', lw=1.4,
                label='predicted (shift only)')
        ax.fill_between(e.x, e.pred_lo, e.pred_hi, color='#999999', alpha=0.18, lw=0)
        ax.plot(e.x, e.observed, marker='s', color=S.CLASS_COLOR[c], lw=1.4,
                label='observed AROI')
        ax.axvline(1.0, color='#999999', lw=0.8, ls=':')
        ax.set_xscale('log'); ax.set_ylim(-0.05, 1.05)
        ax.set_title(c, fontsize=9); ax.set_xlabel('lesion area (patches)')
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.62, -0.13), fontsize=8)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(
        description='Where AROI recall drops (test 1) and why (test 2).')
    ap.add_argument('--aroi', default=str(S.ROOT / 'results/aroi_zeroshot.csv'))
    a = ap.parse_args()

    Z = pd.read_csv(a.aroi)
    man = S.manifest()
    thr_amd = dict(zip(T, S.thresholds('last4_224')))
    thr_aroi = {c: float(S.youden(Z[f'label_{c}'].values, Z[f'p_{c}'].values)) for c in T}
    print('thresholds  ' + '  '.join(
        f'{c}: {thr_amd[c]:.2f} -> {thr_aroi[c]:.2f}' for c in T))

    print('\n=== test 1: recall by lesion size, two thresholds ===')
    R, edges_by_cls = test1(Z, thr_amd, thr_aroi)
    R.to_csv(S.ROOT / 'results/aroi_recall_two_thresholds.csv', index=False)
    show = ['cls', 'lo_edge', 'hi_edge', 'n', 'n_patients',
            'recall_amdsd_thr', 'recall_refit_thr', 'delta', 'delta_lo', 'delta_hi', 'p']
    print(R[show].round(3).to_string(index=False))

    print('\n=== test 2: logit(score) ~ log(area) * dataset, positives only ===')
    sc = pd.concat([amdsd_positive_scores(man), aroi_positive_scores(Z)],
                   ignore_index=True)
    sc.to_csv(S.ROOT / 'results/aroi_score_vs_size.csv', index=False)
    M = test2(sc)
    M.to_csv(S.ROOT / 'results/aroi_score_size_model.csv', index=False)
    print(M[['cls', 'term', 'estimate', 'lo', 'hi', 'p']].round(3).to_string(index=False))

    print('\n=== test 3: does a pure logit shift predict the AROI recall curve? ===')
    C = test3(man, thr_amd, M, R, edges_by_cls, Z)
    C.to_csv(S.ROOT / 'results/aroi_shift_model_recall.csv', index=False)
    print(C[['cls', 'lo_edge', 'hi_edge', 'n_amdsd', 'n_aroi', 'predicted',
             'predicted_oof_fit', 'observed', 'residual',
             'residual_oof_fit']].round(3).to_string(index=False))

    S.save(figure(R, sc, M), 'aroi_transfer', sub='report')
    S.save(figure_shift(C), 'aroi_shift_model', sub='report')
    print('\n-> results/aroi_recall_two_thresholds.csv, aroi_score_vs_size.csv, '
          'aroi_score_size_model.csv, aroi_shift_model_recall.csv')


if __name__ == '__main__':
    main()
