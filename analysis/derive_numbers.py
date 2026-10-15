import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AP, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S

T = S.CLASSES
RES = S.ROOT / 'results'
PRED = S.DATA / 'amdsd_preds'
CLS = {'IRF': 'Irf', 'SRF': 'Srf', 'PED': 'Ped'}
BIN = 'ABCDEFG'                        # index into S.PATCH_EDGES, smallest first
SEEDS = (0, 1, 2)


# ============================================================ derivations
# Numbers the paper quotes that no committed CSV held. Each writes its own CSV, so the
# macro step below reads results/ only and never recomputes an analysis.

def _reps(g):
    return S.replicates(np.asarray(g))


def _paired(y, a, b, reps):
    d = np.array([AP(y[i], a[i]) - AP(y[i], b[i]) for i in reps
                  if y[i].min() != y[i].max()])
    lo, hi = S.ci(d)
    return AP(y, a) - AP(y, b), lo, hi, S.two_sided_p(d)


def frozen_attn(c):
    """Frozen attention, 3 seeds averaged: the §6 / figure 5 protocol."""
    return np.mean([np.load(PRED / f'frozen_attn_{c}_s{s}.npz')['test_p'] for s in SEEDS],
                   axis=0)


def last4_seed_mean(arm):
    """Seed-averaged, fold-ensembled test predictions for a last-4 arm, all three seeds.

    Both last-4 arms enter §6 and C2 this way, matching the frozen arm, which averages
    its three seeds. Refuses to run on fewer seeds rather than silently comparing a
    3-seed arm with a single-seed one.
    """
    missing = [s for s in SEEDS if not glob.glob(str(PRED / f'preds_{arm}_f*_s{s}.npz'))]
    if missing:
        raise SystemExit(f'{arm}: seeds {missing} missing — §6 and C2 need all three '
                         f'on both last-4 arms (scripts/slurm/train_amdsd.sh ... attn)')
    return np.mean([S.arm_preds(arm, s)[0] for s in SEEDS], axis=0)


def derive_section6():
    """The §6 Holm family of six, and C2 itself, on one set of replicates.

    Frozen arms are the post-fix predictions written by frozen_arms.py; both last-4 arms
    are 3-seed means. Recomputing the frozen attention − mean contrast
    here rather than reading frozen_arms_tests.csv puts all eight contrasts on the same
    resamples; it reproduces that file exactly.
    """
    y = S.test_rows()
    reps = _reps(y.group.values)
    L4, L4a = last4_seed_mean('last4_224'), last4_seed_mean('last4_224_attn')
    rows = []
    for i, c in enumerate(T):
        yt = y[f'label_{c}'].values
        fa = frozen_attn(c)
        fm = np.load(PRED / f'frozen_mean_{c}_s0.npz')['test_p']
        for fam, lab, a, b in (('section6', 'frozen attention - mean', fa, fm),
                               ('section6', 'last-4 attention - mean', L4a[:, i], L4[:, i]),
                               ('c2', 'frozen+attn - last-4+mean', fa, L4[:, i]),
                               ('c2', 'frozen+attn - last-4+attn', fa, L4a[:, i])):
            d, lo, hi, p = _paired(yt, a, b, reps)
            rows.append(dict(family=fam, comparison=lab, cls=c, a=AP(yt, a),
                             b=AP(yt, b), delta=d, lo=lo, hi=hi, p=p))
    D = pd.DataFrame(rows)
    six = D.family == 'section6'
    D['p_holm'] = np.nan
    D.loc[six, 'p_holm'] = S.holm(D.loc[six, 'p'].values)
    D.to_csv(RES / 'section6_contrasts.csv', index=False)
    return D


def derive_section1():
    """§1: AUPRC per arm as the mean of per-seed AUPRC, which is what §1 says it reports,
    with a patient bootstrap of that same statistic on the shared replicates.

    encoder_grid_auprc.csv is not this. Its `mean` column scores the seed-averaged
    prediction, an ensemble, which differs from the per-seed mean where the seeds
    disagree (full FT IRF: 0.776 against 0.774). The single-seed high-resolution arms
    are computed the same way, so all five rows of the table share one method.
    """
    y = S.test_rows()
    reps = _reps(y.group.values)
    rows = []
    for arm in ('lp_224', 'lp_384', 'last4_224', 'full_224', 'full_384', 'last4_448'):
        seeds = [s for s in SEEDS if glob.glob(str(PRED / f'preds_{arm}_f*_s{s}.npz'))]
        Ps = [S.arm_preds(arm, s)[0] for s in seeds]
        Y = S.arm_preds(arm, seeds[0])[1]
        for i, c in enumerate(T):
            yt = Y[:, i]
            per = [AP(yt, P[:, i]) for P in Ps]
            bs = [np.mean([AP(yt[ix], P[ix, i]) for P in Ps]) for ix in reps
                  if yt[ix].min() != yt[ix].max()]
            lo, hi = S.ci(bs)
            rows.append(dict(arm=arm, cls=c, n_seeds=len(seeds), auprc=np.mean(per),
                             sd=np.std(per, ddof=1) if len(per) > 1 else np.nan,
                             lo=lo, hi=hi))
    D = pd.DataFrame(rows)
    D.to_csv(RES / 'section1_auprc.csv', index=False)
    return D


def derive_threshold_displacement():
    """How far each test-set Youden point sits from the one the test scale would pick.

    The right-hand columns take Youden on the held-out test set. That is a diagnostic of
    the out-of-fold-versus-ensemble scale gap, never an operating point.
    """
    P, Y, _ = S.arm_preds('last4_224', 0)
    vp, vy = S.val_preds('last4_224', 0)
    rows = []
    for i, c in enumerate(T):
        y, p = Y[:, i], P[:, i]
        r = dict(cls=c)
        for tag, t in (('used', S.youden(vy[:, i], vp[:, i])), ('diag', S.youden(y, p))):
            pr = p >= t
            rec, spec = pr[y == 1].mean(), (~pr)[y == 0].mean()
            r.update({f'{tag}_thr': t, f'{tag}_recall': rec, f'{tag}_spec': spec,
                      f'{tag}_j': rec + spec - 1})
        rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(RES / 'threshold_displacement.csv', index=False)
    return D


def _oof_and_test(c):
    """Out-of-fold on pool, fold-ensembled on test — the notebooks/contrast.py scores."""
    man = S.manifest()
    D = [np.load(f, allow_pickle=True)
         for f in sorted(glob.glob(str(PRED / 'preds_last4_224_f*_s0.npz')))]
    i = T.index(c)
    m = dict(zip(man[man.split == 'test'].file, np.mean([d['test_p'] for d in D], 0)[:, i]))
    for k, d in enumerate(D):
        m.update(zip(man[(man.split == 'pool') & (man.fold == k)].file, d['val_p'][:, i]))
    return m


def _mcfadden(X, y):
    m = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
    p = m.predict_proba(X)[:, 1]
    ll = (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).sum()
    b = y.mean()
    return 1 - ll / (y * np.log(b) + (1 - y) * np.log(1 - b)).sum()


def derive_contrast_regression():
    """§6c, replicating notebooks/contrast.py exactly.

    Unit: positive scans. Contrast enters as |area-weighted mean of the raw intensity
    difference inside − ring| over the scan's components — raw grey levels, not Weber.
    Size enters as log of the summed pixels of those same components. Outcome: detected
    at the out-of-fold Youden threshold. McFadden pseudo-R².
    """
    man = S.manifest()
    L = pd.read_csv(RES / 'lesion_contrast.csv')
    thr = dict(zip(T, S.thresholds('last4_224')))
    rows = []
    for c in T:
        s = L[L.cls == c]
        w = s.assign(cw=s.contrast * s.px).groupby('file').agg(
            total_px=('px', 'sum'), cw=('cw', 'sum'))
        w['contrast_w'] = w.cw / w.total_px
        D = w.reset_index().merge(man[['file', f'label_{c}']], on='file')
        D = D[D[f'label_{c}'] == 1]
        sc = _oof_and_test(c)
        y = (D.file.map(sc).values >= thr[c]).astype(int)
        la = np.log(D.total_px.clip(lower=1).values)
        ac = D.contrast_w.abs().values
        rows.append(dict(
            cls=c, n_lesions=len(s), weber_median=s.weber.median(),
            # per component, signed raw grey-level difference: what §6c has quoted
            corr_logpx_contrast=np.corrcoef(np.log(s.px), s.contrast)[0, 1],
            # per positive scan, the regression's own two predictors
            corr_scan=np.corrcoef(la, ac)[0, 1],
            n_scans=len(D),
            r2_size=_mcfadden(la[:, None], y),
            r2_contrast=_mcfadden(ac[:, None], y),
            r2_both=_mcfadden(np.column_stack([la, ac]), y)))
    D = pd.DataFrame(rows)
    D['r2_gain'] = D.r2_both - D.r2_size
    D.to_csv(RES / 'contrast_regression.csv', index=False)
    return D


def _half_point(b):
    """Area at which binned recall first crosses 0.5, linear in log area."""
    b = b[b.reportable].sort_values('x')
    x, r = np.log(b.x.values), b.recall.values
    for k in range(len(r) - 1):
        if r[k] < 0.5 <= r[k + 1]:
            return float(np.exp(x[k] + (0.5 - r[k]) / (r[k + 1] - r[k]) * (x[k + 1] - x[k])))
    return np.nan


def derive_half_points():
    """The §4 50%-recall points, from the same binned curve figure 3 draws.

    These were previously quoted with no script behind them.
    """
    rows = []
    for f, ds in (('fig3_recall_bins.csv', 'AMD-SD'), ('aroi_fig3_recall_bins.csv', 'AROI')):
        if not (RES / f).exists():
            continue
        B = pd.read_csv(RES / f)
        rows += [dict(dataset=ds, cls=c, half_point=_half_point(B[B.cls == c])) for c in T]
    D = pd.DataFrame(rows)
    D[D.dataset == 'AMD-SD'].to_csv(RES / 'recall_half_points.csv', index=False)
    if (D.dataset == 'AROI').any():                     # AROI-derived: gitignored name
        D[D.dataset == 'AROI'].to_csv(RES / 'aroi_recall_half_points.csv', index=False)
    return D


def derive_shrm():
    """§6c / §5: how well Weber's sign separates fluid from SHRM, and the SRF-head score
    for each tissue group. AMD-SD parts are committable; AROI parts are not."""
    C = pd.read_csv(RES / 'amdsd_shrm_contrast.csv')
    C = C[C.cls.isin(['SRF', 'SHRM'])]
    y, w = (C.cls == 'SHRM').astype(int).values, C.weber.values
    pred = w >= 0
    grid = np.unique(w)
    j = [((w >= t) & (y == 1)).sum() / y.sum() + ((w < t) & (y == 0)).sum() / (1 - y).sum() - 1
         for t in grid]
    sep = pd.DataFrame([dict(
        n_srf=int((y == 0).sum()), n_shrm=int(y.sum()), auroc=roc_auc_score(y, w),
        sensitivity=pred[y == 1].mean(), specificity=(~pred)[y == 0].mean(),
        accuracy=(pred == y).mean(), youden_cut=float(grid[int(np.argmax(j))]),
        weber_median_srf=C[C.cls == 'SRF'].weber.median(),
        weber_median_shrm=C[C.cls == 'SHRM'].weber.median(),
        bright_pct_srf=(C[C.cls == 'SRF'].weber >= 0).mean(),
        bright_pct_shrm=(C[C.cls == 'SHRM'].weber >= 0).mean(),
        median_px_srf=C[C.cls == 'SRF'].px.median(),
        median_px_shrm=C[C.cls == 'SHRM'].px.median())])
    sep.to_csv(RES / 'shrm_proxy_separation.csv', index=False)

    man = S.manifest()
    sc = _oof_and_test('SRF')
    man['p'] = man.file.map(sc).astype(float)
    thr = dict(zip(T, S.thresholds('last4_224')))['SRF']
    groups = []
    for name, d in (('amdsd_srf_present', man[man.area_SRF > 0]),
                    ('amdsd_shrm_only', man[(man.area_SHRM > 0) & (man.area_SRF == 0)]),
                    ('amdsd_neither', man[(man.area_SHRM == 0) & (man.area_SRF == 0)])):
        groups.append(dict(group=name, n=len(d), patients=d.group.nunique(),
                           mean=d.p.mean(), median=d.p.median(), p90=d.p.quantile(.9),
                           above_thr=(d.p >= thr).mean()))
    pd.DataFrame(groups).to_csv(RES / 'srf_head_score_groups.csv', index=False)

    if not ((RES / 'aroi_lesion_contrast.csv').exists()
            and (RES / 'aroi_zeroshot.csv').exists()):
        return
    import aroi_shrm_proxy as M
    L = pd.read_csv(RES / 'aroi_lesion_contrast.csv')
    Z = pd.read_csv(RES / 'aroi_zeroshot.csv').merge(M.slice_brightness(L),
                                                    on=['patient', 'slice'], how='left')
    pos = Z[(Z.label_SRF == 1) & Z.weber_wt.notna()].copy()
    pos['bright'] = (pos.weber_wt >= 0).astype(bool)
    P4, Y4, _ = S.arm_preds('last4_224', 0)
    t_ens = S.youden(Y4[:, 1], P4[:, 1])
    rows = []
    for name, d in (('aroi_bright', pos[pos.bright]), ('aroi_dark', pos[~pos.bright])):
        rows.append(dict(group=name, n=len(d), patients=d.patient.nunique(),
                         mean=d.p_SRF.mean(), median=d.p_SRF.median(),
                         p90=d.p_SRF.quantile(.9), max=d.p_SRF.max(),
                         above_thr=(d.p_SRF >= thr).mean(),
                         above_ens_thr=(d.p_SRF >= t_ens).mean(),
                         detected_ens_thr=int((d.p_SRF >= t_ens).sum())))
    for c in T:
        s = L[L.cls == c]
        rows.append(dict(group=f'aroi_{c.lower()}_components', n=len(s),
                         weber_median=s.weber.median(), bright_pct=(s.weber >= 0).mean(),
                         median_px=s.px.median()))
    pd.DataFrame(rows).to_csv(RES / 'aroi_srf_head_score_groups.csv', index=False)


def derive_within_bin_auroc(n_boot=2000):
    """Dark-SRF within-bin AUROC with patient-bootstrap intervals (AROI and AMD-SD)."""
    need = [RES / f for f in ('aroi_zeroshot.csv', 'aroi_lesion_contrast.csv')]
    if not all(f.exists() for f in need):
        return
    import aroi_shrm_proxy as M
    E = [0.03, 0.08, 0.216, 0.579, 1.554, 4.168, 30.0]
    Z = pd.read_csv(RES / 'aroi_zeroshot.csv').merge(
        M.slice_brightness(pd.read_csv(RES / 'aroi_lesion_contrast.csv')),
        on=['patient', 'slice'], how='left')
    neg_a = Z[Z.label_SRF == 0]
    pa = Z[(Z.label_SRF == 1) & Z.weber_wt.notna() & (Z.weber_wt < 0)].copy()
    pa['u'] = pa.area_SRF / S.patch_px('aroi', 224)
    man = S.manifest()
    man['p'] = man.file.map(_oof_and_test('SRF')).astype(float)
    pool = man[man.split == 'pool']
    neg_m, pm = pool[pool.label_SRF == 0], pool[pool.label_SRF == 1].copy()
    pm['u'] = pm.area_SRF / S.patch_px('amdsd', 224)

    def boot(ps, pg, ns, ng, seed):
        rng = np.random.default_rng(seed)
        s, g = np.r_[ps, ns], np.r_[pg, ng]
        y = np.r_[np.ones(len(ps)), np.zeros(len(ns))]
        ug = np.unique(g)
        gi = {u: np.flatnonzero(g == u) for u in ug}
        out = []
        for _ in range(n_boot):
            ix = np.concatenate([gi[u] for u in rng.choice(ug, len(ug))])
            if len(np.unique(y[ix])) == 2:
                out.append(roc_auc_score(y[ix], s[ix]))
        return roc_auc_score(y, s), np.array(out)

    rows = []
    for b in range(len(E) - 1):
        A = pa[(pa.u >= E[b]) & (pa.u < E[b + 1])]
        B = pm[(pm.u >= E[b]) & (pm.u < E[b + 1])]
        if not len(A) or not len(B):
            continue
        ea, ba = boot(A.p_SRF.values, A.patient.values, neg_a.p_SRF.values,
                      neg_a.patient.values, 0)
        em, bm = boot(B.p.values, B.group.values, neg_m.p.values, neg_m.group.values, 1)
        n = min(len(ba), len(bm))
        d = ba[:n] - bm[:n]
        rows.append(dict(lo=E[b], hi=E[b + 1], n_aroi=len(A), pts_aroi=A.patient.nunique(),
                         auroc_aroi=ea, aroi_lo=np.percentile(ba, 2.5),
                         aroi_hi=np.percentile(ba, 97.5), n_amdsd=len(B),
                         pts_amdsd=B.group.nunique(), auroc_amdsd=em,
                         amdsd_lo=np.percentile(bm, 2.5), amdsd_hi=np.percentile(bm, 97.5),
                         delta=ea - em, delta_lo=np.percentile(d, 2.5),
                         delta_hi=np.percentile(d, 97.5)))
    pd.DataFrame(rows).to_csv(RES / 'aroi_dark_srf_auroc_by_bin.csv', index=False)


DERIVE = {'section1': derive_section1, 'section6': derive_section6, 'threshold': derive_threshold_displacement,
          'contrast': derive_contrast_regression, 'half': derive_half_points,
          'shrm': derive_shrm, 'wbin': derive_within_bin_auroc}


# ============================================================ formatting
def num(x, nd=3):
    return f'{x:.{nd}f}'


def sgn(x, nd=3):
    """Signed, math-safe: renders as +0.017 / −0.001 in text and in math."""
    v = f'{x:+.{nd}f}'
    if v.lstrip('+-').strip('0.') == '':
        v = f'{0:.{nd}f}'
        v = ('-' if x < 0 else '+') + v
    return r'\ensuremath{' + v + '}'


def pval(p):
    return r'\ensuremath{<0.001}' if p < 0.001 else f'{p:.3f}'


def pct(x, nd=1):
    return f'{100 * x:.{nd}f}'


class Book:
    def __init__(self):
        self.rows, self.names = [], set()

    def add(self, name, value, source, note=''):
        if not re.fullmatch(r'[A-Za-z]+', name):
            raise ValueError(f'macro name must be letters only: {name!r}')
        if name in self.names:
            raise ValueError(f'duplicate macro {name!r}')
        self.names.add(name)
        self.rows.append(dict(name=name, value=value, source=source, note=note))

    def ci(self, stem, d, lo, hi, p=None, holm=None, src='', note=''):
        self.add(f'{stem}Delta', sgn(d), src, note)
        self.add(f'{stem}Lo', sgn(lo), src)
        self.add(f'{stem}Hi', sgn(hi), src)
        if p is not None:
            self.add(f'{stem}P', pval(p), src)
        if holm is not None:
            self.add(f'{stem}Holm', pval(holm), src)


def read(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


# ============================================================ macro groups
ARM = {'lp_224': 'Lp', 'lp_384': 'LpHigh', 'last4_224': 'LastFour', 'full_224': 'Full',
       'full_384': 'FullHigh', 'last4_448': 'LastFourHigh'}


def g_main(b):
    """§1 AUPRCs: mean of per-seed AUPRC with a bootstrap of that statistic."""
    D = read('section1_auprc.csv')
    for r in D.itertuples():
        note = 'single seed' if r.n_seeds == 1 else f'mean of {r.n_seeds} seeds'
        for f, v in (('Auprc', r.auprc), ('Lo', r.lo), ('Hi', r.hi)):
            b.add(f'Main{ARM[r.arm]}{CLS[r.cls]}{f}', num(v), 'section1_auprc.csv', note)


PV = {'last-4 - LP': 'DepthLastFourLp', 'full FT - last-4': 'DepthFullLastFour',
      '384 - 224, full FT': 'ResFull', '384 - 224, LP': 'ResLp',
      '384 - 224, frozen probe': 'ResProbe', '448 - 224, last-4': 'ResLastFour',
      'RETFound - MAE-IN1k, LP': 'EncMaeLp', 'RETFound - MAE-IN1k, last-4': 'EncMaeLastFour',
      'RETFound - Sup-IN21k, LP': 'EncSupLp',
      'RETFound - Sup-IN21k, last-4': 'EncSupLastFour'}


def g_pvalues(b):
    """Depth, resolution and encoder contrasts. Holm only exists for the encoder family."""
    D = read('pvalues.csv')
    for r in D.itertuples():
        holm = r.p_holm if r.family == 'Encoder' else None
        b.ci(f'{PV[r.comparison]}{CLS[r.cls]}', r.delta, r.lo, r.hi, r.p, holm,
             'pvalues.csv')
    E = read('encoder_grid_auprc.csv')
    enc = {'MAE-IN1k': 'Mae', 'Sup-IN21k': 'Sup', 'RETFound': 'Ret'}
    dep = {'lp': 'Lp', 'last4': 'LastFour', 'full': 'Full'}
    for r in E.itertuples():
        b.add(f'EncAuprc{enc[r.encoder]}{dep[r.depth]}{CLS[r.cls]}', num(r.mean),
              'encoder_grid_auprc.csv', '3-seed mean')


def g_section6(b):
    D = read('section6_contrasts.csv')
    tag = {'frozen attention - mean': 'FrozenAttnMean',
           'last-4 attention - mean': 'LastFourAttnMean',
           'frozen+attn - last-4+mean': 'Ctwo', 'frozen+attn - last-4+attn': 'CtwoAttn'}
    for r in D.itertuples():
        holm = r.p_holm if r.family == 'section6' else None
        b.ci(f'{tag[r.comparison]}{CLS[r.cls]}', r.delta, r.lo, r.hi, r.p, holm,
             'section6_contrasts.csv')
        if r.comparison == 'frozen+attn - last-4+mean':
            b.add(f'Ctwo{CLS[r.cls]}Frozen', num(r.a), 'section6_contrasts.csv',
                  'seed-averaged prediction')
            b.add(f'Ctwo{CLS[r.cls]}LastFour', num(r.b), 'section6_contrasts.csv', 'seed 0')
    F = read('frozen_arms.csv')
    for c in T:
        a = F[(F.pool == 'attn') & (F.cls == c)].auprc
        m = F[(F.pool == 'mean') & (F.cls == c)].auprc
        b.add(f'FrozenAttn{CLS[c]}Auprc', num(a.mean()), 'frozen_arms.csv',
              'mean of per-seed AUPRC')
        b.add(f'FrozenAttn{CLS[c]}Sd', num(a.std()), 'frozen_arms.csv')
        b.add(f'FrozenMean{CLS[c]}Auprc', num(m.mean()), 'frozen_arms.csv')


def g_capacity(b):
    D = read('fig5_capacity_control.csv')
    tag = {'attention - mean': 'CapAttnMean', 'attention - matched MLP': 'CapAttnMlp',
           'matched MLP - mean': 'CapMlpMean'}
    for r in D.itertuples():
        b.ci(f'{tag[r.contrast]}{CLS[r.cls]}', r.delta, r.lo, r.hi, r.p, None,
             'fig5_capacity_control.csv', 'uncorrected')


def g_subpatch(b):
    for ds, comp, scans, label, patch, gcol in (
            ('Amdsd', 'amdsd_components_per_lesion.csv', None, 'label', 'amdsd', 'file'),
            ('Aroi', 'aroi_components.csv', 'aroi_zeroshot.csv', 'label', 'aroi', None)):
        C = read(comp)
        if C is None:
            continue
        P = S.patch_px(patch, 224)
        for c in T:
            s = C[C.cls == c]
            b.add(f'SubPatch{ds}{CLS[c]}CompPct', pct((s.px < P).mean()), comp,
                  'per connected component')
            if ds == 'Amdsd':
                man = S.manifest()
                a = man[man[f'label_{c}'] == 1][f'area_{c}']
            else:
                Z = read(scans)
                a = Z[Z[f'label_{c}'] == 1][f'area_{c}']
            b.add(f'SubPatch{ds}{CLS[c]}ScanPct', pct((a < P).mean()),
                  comp if ds == 'Amdsd' else scans, 'per scan total area')


def g_recall_bins(b):
    for f, stem in (('fig3_recall_bins.csv', 'Rec'), ('aroi_fig3_recall_bins.csv', 'RecAroi')):
        D = read(f)
        if D is None:
            continue
        for r in D[D.reportable].itertuples():
            k = f'{stem}{CLS[r.cls]}Bin{BIN[int(r.bin)]}'
            b.add(f'{k}Recall', num(r.recall), f,
                  f'{r.lo_edge:.3g}-{r.hi_edge:.3g} patches')
            b.add(f'{k}Lo', num(r.lo), f)
            b.add(f'{k}Hi', num(r.hi), f)
            b.add(f'{k}N', str(int(r.n)), f)


def g_half(b):
    for f, stem in (('recall_half_points.csv', 'Half'), ('aroi_recall_half_points.csv',
                                                         'HalfAroi')):
        D = read(f)
        if D is None:
            continue
        for r in D.itertuples():
            if np.isfinite(r.half_point):
                b.add(f'{stem}{CLS[r.cls]}', num(r.half_point, 2), f, 'patches')


def g_contrast(b):
    D = read('contrast_regression.csv')
    for r in D.itertuples():
        k = f'Contrast{CLS[r.cls]}'
        b.add(f'{k}Weber', sgn(r.weber_median), 'contrast_regression.csv', 'per-lesion median')
        b.add(f'{k}Corr', sgn(r.corr_logpx_contrast), 'contrast_regression.csv',
              'per component, signed raw difference')
        b.add(f'{k}CorrScan', sgn(r.corr_scan), 'contrast_regression.csv',
              'per positive scan, the regression predictors')
        b.add(f'{k}RsqSize', num(r.r2_size), 'contrast_regression.csv', 'McFadden')
        b.add(f'{k}RsqBoth', num(r.r2_both), 'contrast_regression.csv', 'McFadden')
        b.add(f'{k}RsqGain', sgn(r.r2_gain), 'contrast_regression.csv')


def g_crossers(b):
    D = read('crossers.csv')
    for r in D[D.name == 'crossers'].itertuples():
        k = f'Cross{CLS[r.cls]}'
        b.ci(k, r.delta, r.lo, r.hi, None, None, 'crossers.csv', 'recall, 448 - 224')
        b.add(f'{k}N', str(int(r.n)), 'crossers.csv')
        b.add(f'{k}P', pval(r.mcnemar), 'crossers.csv', 'McNemar')


def g_aroi(b):
    D = read('aroi_fig7_auroc.csv')
    if D is not None:
        for r in D.itertuples():
            b.add(f'AroiAuroc{CLS[r.cls]}', num(r.auroc_aroi), 'aroi_fig7_auroc.csv')
            b.add(f'AroiAuroc{CLS[r.cls]}Lo', num(r.lo), 'aroi_fig7_auroc.csv')
            b.add(f'AroiAuroc{CLS[r.cls]}Hi', num(r.hi), 'aroi_fig7_auroc.csv')
            b.add(f'AroiAurocAmdsd{CLS[r.cls]}', num(r.auroc_amdsd), 'aroi_fig7_auroc.csv')
    D = read('aroi_recall_two_thresholds.csv')
    if D is not None:
        for c in T:
            for k, r in enumerate(D[D.cls == c].sort_values('lo_edge').itertuples()):
                s = f'Xfer{CLS[c]}Bin{BIN[k]}'
                note = f'{r.lo_edge:.3g}-{r.hi_edge:.3g} patches (merged bins)'
                b.add(f'{s}N', str(int(r.n)), 'aroi_recall_two_thresholds.csv', note)
                b.add(f'{s}AmdsdThr', num(r.recall_amdsd_thr), 'aroi_recall_two_thresholds.csv')
                b.add(f'{s}Refit', num(r.recall_refit_thr), 'aroi_recall_two_thresholds.csv')
                b.ci(s, r.delta, r.delta_lo, r.delta_hi, r.p, None,
                     'aroi_recall_two_thresholds.csv', 'refit - AMD-SD threshold')
    D = read('aroi_score_size_model.csv')
    if D is not None:
        nm = {'log_area': 'Slope', 'dataset_AROI': 'Offset', 'log_area_x_AROI': 'Inter'}
        for r in D[D.term != 'intercept'].itertuples():
            k = f'Shift{CLS[r.cls]}{nm[r.term]}'
            b.add(k, sgn(r.estimate, 2 if r.term != 'log_area_x_AROI' else 3),
                  'aroi_score_size_model.csv')
            b.add(f'{k}Lo', sgn(r.lo, 3), 'aroi_score_size_model.csv')
            b.add(f'{k}Hi', sgn(r.hi, 3), 'aroi_score_size_model.csv')
            b.add(f'{k}P', pval(r.p), 'aroi_score_size_model.csv')
    D = read('aroi_shift_model_recall.csv')
    if D is not None:
        for c in T:
            d = D[(D.cls == c) & D.observed.notna()]
            b.add(f'Shift{CLS[c]}Mae', num(d.residual.abs().mean()),
                  'aroi_shift_model_recall.csv', 'test-fitted shift')
            b.add(f'Shift{CLS[c]}MaeOof', num(d.residual_oof_fit.abs().mean()),
                  'aroi_shift_model_recall.csv', 'out-of-fold-fitted shift')


def g_shrm(b):
    D = read('shrm_proxy_separation.csv')
    if D is not None:
        r = D.iloc[0]
        for k, v, f in (('Auroc', r.auroc, num), ('Sens', r.sensitivity, num),
                        ('Spec', r.specificity, num), ('Acc', r.accuracy, num),
                        ('YoudenCut', r.youden_cut, sgn),
                        ('WeberSrf', r.weber_median_srf, sgn),
                        ('WeberShrm', r.weber_median_shrm, sgn)):
            b.add(f'ShrmProxy{k}', f(v), 'shrm_proxy_separation.csv')
        b.add('ShrmProxyBrightPctSrf', pct(r.bright_pct_srf), 'shrm_proxy_separation.csv')
        b.add('ShrmProxyBrightPctShrm', pct(r.bright_pct_shrm), 'shrm_proxy_separation.csv')
    nm = {'amdsd_srf_present': 'AmdsdSrfPresent', 'amdsd_shrm_only': 'AmdsdShrmOnly',
          'amdsd_neither': 'AmdsdNeither', 'aroi_bright': 'AroiBright', 'aroi_dark': 'AroiDark'}
    for f in ('srf_head_score_groups.csv', 'aroi_srf_head_score_groups.csv'):
        D = read(f)
        if D is None:
            continue
        for r in D.itertuples():
            if r.group in nm:
                k = f'ShrmScore{nm[r.group]}'
                b.add(f'{k}N', str(int(r.n)), f)
                b.add(f'{k}Patients', str(int(r.patients)), f)
                b.add(f'{k}Mean', num(r.mean), f)
                b.add(f'{k}Median', num(r.median), f)
                b.add(f'{k}AboveThrPct', pct(r.above_thr), f, 'at the 0.80 threshold')
                if r.group == 'aroi_bright':
                    b.add('ShrmBrightDetectedEnsThr', str(int(r.detected_ens_thr)), f,
                          'at the ensemble-scale diagnostic threshold')
            elif r.group.endswith('_components'):
                c = r.group.split('_')[1].upper()
                b.add(f'ShrmAroi{CLS[c]}BrightPct', pct(r.bright_pct), f)
                b.add(f'ShrmAroi{CLS[c]}WeberMedian', sgn(r.weber_median), f)


def g_wbin(b):
    D = read('aroi_dark_srf_auroc_by_bin.csv')
    if D is None or 'aroi_lo' not in D:
        return
    for k, r in enumerate(D.sort_values('lo').itertuples()):
        s = f'Wbin{BIN[k]}'
        f = 'aroi_dark_srf_auroc_by_bin.csv'
        b.add(f'{s}Aroi', num(r.auroc_aroi), f, f'{r.lo:.3g}-{r.hi:.3g} patches, dark SRF')
        b.add(f'{s}AroiLo', num(r.aroi_lo), f)
        b.add(f'{s}AroiHi', num(r.aroi_hi), f)
        b.add(f'{s}Amdsd', num(r.auroc_amdsd), f)
        b.ci(s, r.delta, r.delta_lo, r.delta_hi, None, None, f)
        b.add(f'{s}PatientsAroi', str(int(r.pts_aroi)), f)


def g_threshold(b):
    D = read('threshold_displacement.csv')
    for r in D.itertuples():
        k = f'Thr{CLS[r.cls]}'
        for tag, lab in (('used', 'Used'), ('diag', 'Diag')):
            note = 'diagnostic: Youden on the test set' if tag == 'diag' else 'out-of-fold Youden'
            b.add(f'{k}{lab}', num(getattr(r, f'{tag}_thr'), 2), 'threshold_displacement.csv',
                  note)
            b.add(f'{k}{lab}Recall', num(getattr(r, f'{tag}_recall')),
                  'threshold_displacement.csv')
            b.add(f'{k}{lab}Spec', num(getattr(r, f'{tag}_spec')), 'threshold_displacement.csv')
            b.add(f'{k}{lab}J', num(getattr(r, f'{tag}_j')), 'threshold_displacement.csv')


def g_leakage(b):
    D = read('leakage_inflation.csv')
    if D is not None:
        for r in D.itertuples():
            k = f'Leak{CLS[r.cls]}'
            b.add(f'{k}Patient', num(r.patient_level), 'leakage_inflation.csv')
            b.add(f'{k}Random', num(r.random_scans), 'leakage_inflation.csv')
            b.add(f'{k}Inflation', sgn(r.inflation), 'leakage_inflation.csv')
    D = read('youden_vs_default_and_size_gap.csv')
    if D is not None:
        f = 'youden_vs_default_and_size_gap.csv'
        for _, r in D.iterrows():               # column names contain '@'
            k = CLS[r['cls']]
            b.add(f'Youden{k}Thr', num(r['youden'], 2), f)
            b.add(f'Youden{k}SensHalf', num(r['sens@0.5']), f, 'sensitivity at 0.5')
            b.add(f'Youden{k}SensYouden', num(r['sens@youden']), f)
            b.add(f'Youden{k}SpecHalf', num(r['spec@0.5']), f)
            b.add(f'Youden{k}SpecYouden', num(r['spec@youden']), f)
            b.add(f'Gap{k}SubN', str(int(r['n_sub'])), f)
            b.add(f'Gap{k}SubRecall', num(r['recall_sub']), f)
            b.add(f'Gap{k}SupraN', str(int(r['n_supra'])), f)
            b.add(f'Gap{k}SupraRecall', num(r['recall_supra']), f)
            b.add(f'Gap{k}Gap', num(r['recall_gap']), f)


def g_compute(b):
    W = {1: 'One', 8: 'Eight', 32: 'ThirtyTwo'}
    D = read('bench_latency.csv')
    if D is not None:
        first = D.log.iloc[0]                       # canonical run; the second replicates it
        for r in D[D.log == first].itertuples():
            dev = 'Cpu' if r.device == 'CPU' else 'Gpu'
            k = f'Lat{dev}Batch{W[r.batch]}'
            b.add(f'{k}MsPerScan', num(r.ms_per_scan, 2), 'bench_latency.csv',
                  f'{r.pool} pooling, {r.device}')
            b.add(f'{k}ScansPerSec', num(r.scans_per_s, 1), 'bench_latency.csv')
    arms = {'last4_224': 'LastFour', 'full_224': 'Full', 'lp_224_attn': 'FrozenAttn'}
    for frag, nm in arms.items():
        p = RES / 'bench' / f'cfg_{frag}_f0_s0.json'
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        b.add(f'Bench{nm}TrainableM', num(j['trainable_M'], 1 if j['trainable_M'] > 1 else 3),
              f'bench/{p.name}')
        b.add(f'Bench{nm}SecPerEpoch', num(j['train_seconds'] / j['epochs_run'], 1),
              f'bench/{p.name}')
        b.add(f'Bench{nm}PeakGiB', num(j['peak_gpu_alloc_GiB'], 2), f'bench/{p.name}')
    p = RES / 'sacct_jobs.psv'
    if p.exists():
        d = pd.read_csv(p, sep='|')
        d.columns = [c.lower() for c in d.columns]
        for c in ('start', 'end'):
            d[c] = pd.to_datetime(d[c], errors='coerce')
        d = d[d.start.notna() & d.end.notna()].copy()
        d['secs'] = (d.end - d.start).dt.total_seconds()
        gpus = d.alloctres.fillna('').str.extract(r'gres/gpu=(\d+)')[0].astype(float).fillna(0)
        b.add('ComputeGpuHours', num((d.secs * gpus).sum() / 3600, 1), 'sacct_jobs.psv',
              f'through {d.end.max():%Y-%m-%d}')
        b.add('ComputeSerialHours', num(d.secs.sum() / 3600, 1), 'sacct_jobs.psv')


GROUPS = [g_main, g_pvalues, g_section6, g_capacity, g_subpatch, g_recall_bins, g_half,
          g_contrast, g_crossers, g_aroi, g_shrm, g_wbin, g_threshold, g_leakage, g_compute]


def build(out_tex, out_index):
    b = Book()
    for g in GROUPS:
        g(b)
    lines = ['% Generated by analysis/derive_numbers.py from results/*.csv. Do not edit.',
             '% Every number below has a source CSV listed in numbers_index.csv.',
             '%',
             '% Naming: <Group><Arm/Contrast><Class><Field>, letters only (TeX forbids digits).',
             '%   Class   Irf Srf Ped',
             '%   Arm     Lp LastFour Full = 224 px; FullHigh = full FT 384; LastFourHigh = last-4 448',
             '%   Bin     A..G = patch-unit bins smallest first (Rec*, Wbin*); Xfer* bins are the',
             '%           merged AROI bins, A..F, edges in numbers_index.csv',
             '%   Field   Delta Lo Hi P Holm | Auprc Sd | Recall N | ...Pct = percent, no % sign',
             '%   Signed values are wrapped in \\ensuremath so they work in text and in math.',
             '%   P = uncorrected bootstrap p; Holm only where a family is defined (Enc*, FrozenAttnMean*,',
             '%   LastFourAttnMean*). Cap* p-values are uncorrected by design.',
             '']
    lines += [rf'\newcommand{{\{r["name"]}}}{{{r["value"]}}}' for r in b.rows]
    Path(out_tex).write_text('\n'.join(lines) + '\n')
    pd.DataFrame(b.rows).to_csv(out_index, index=False)
    return b.rows


def main():
    ap = argparse.ArgumentParser(
        description='Write numbers.tex: one LaTeX macro per number the paper quotes, from '
                    'the committed CSVs. --derive first recomputes the CSVs for numbers '
                    'that previously had no saved source.')
    ap.add_argument('--derive', nargs='*', choices=list(DERIVE), default=None,
                    help='recompute these source CSVs first (all, if given with no names)')
    ap.add_argument('--tex', default=str(RES / 'numbers.tex'))
    ap.add_argument('--index', default=str(RES / 'numbers_index.csv'))
    a = ap.parse_args()
    if a.derive is not None:
        for k in (a.derive or list(DERIVE)):
            print(f'deriving {k}')
            DERIVE[k]()
    rows = build(a.tex, a.index)
    print(f'{len(rows)} macros -> {a.tex}\n            index -> {a.index}')


if __name__ == '__main__':
    main()
