import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
T = ['IRF', 'SRF', 'PED']
ARM = {'lp_224': 'Linear probe, 224', 'lp_384': 'Linear probe, 384',
       'last4_224': 'Last-4, 224', 'last4_448': 'Last-4, 448',
       'full_224': 'Full FT, 224', 'full_384': 'Full FT, 384'}


# ------------------------------------------------------------ formatting
def f(x, nd=3):
    return '--' if pd.isna(x) else f'{x:.{nd}f}'


def s(x, nd=3):
    if pd.isna(x):
        return '--'
    v = f'{x:+.{nd}f}'
    if v.lstrip('+-').strip('0.') == '':
        v = ('-' if x < 0 else '+') + f'{0:.{nd}f}'
    return '$' + v.replace('-', '-') + '$'


def p(x):
    return '--' if pd.isna(x) else ('$<0.001$' if x < 0.001 else f'{x:.3f}')


def ci(lo, hi, nd=3):
    return f'[{s(lo, nd)}, {s(hi, nd)}]'


def pctf(x, nd=1):
    return '--' if pd.isna(x) else f'{100 * x:.{nd}f}'


def read(name):
    q = RES / name
    return pd.read_csv(q) if q.exists() else None


def tt(text):
    """Set every *.csv filename in monospace with its underscores escaped."""
    return re.sub(r'[\w-]+\.csv', lambda m: r'\texttt{' + m.group(0).replace('_', r'\_') + '}',
                  text)


def table(n, caption, cols, rows, notes=None, mid=()):
    """One booktabs table. `mid` holds row indices after which a \\midrule goes."""
    body = []
    for i, r in enumerate(rows):
        body.append(' & '.join(str(c) for c in r) + r' \\')
        if i in mid:
            body.append(r'\midrule')
    out = [r'\begin{table}[htbp]', r'\centering', r'\small',
           rf'\caption{{{caption}}}', rf'\label{{tab:S{n}}}',
           r'\begin{tabular}{' + cols[0] + '}', r'\toprule',
           ' & '.join(cols[1]) + r' \\', r'\midrule', *body, r'\bottomrule',
           r'\end{tabular}']
    if notes:
        out += [r'\par\smallskip', r'\parbox{\linewidth}{\footnotesize ' + tt(notes) + '}']
    out.append(r'\end{table}')
    return '\n'.join(out)


def missing(n, caption, src):
    return (rf'\begin{{table}}[htbp]\centering\caption{{{caption}}}\label{{tab:S{n}}}'
            rf'\par\footnotesize Source {tt(src)} is not available on the machine that '
            r'built this file (AROI-derived results are licence-restricted).\end{table}')


# ------------------------------------------------------------ tables
def t_operating_points(n):
    D = read('operating_points.csv')
    rows = []
    for arm in ('lp_224', 'lp_384', 'last4_224', 'last4_448', 'full_224', 'full_384'):
        d = D[D.arm == arm].set_index('cls')
        rows.append([ARM[arm], int(d.n_seeds.iloc[0])] +
                    [f'{f(d.recall[c])} / {f(d.spec[c])}' for c in T])
    return table(n, 'Operating points at the Youden threshold: recall / specificity on the '
                 'test set, by arm.', ('lccc' + 'c', ['Arm', 'Seeds', 'IRF', 'SRF', 'PED']),
                 [[r[0], r[1], r[2], r[3], r[4]] for r in rows],
                 'Each seed uses its own Youden threshold, chosen on its out-of-fold validation '
                 'folds; recall and specificity are averaged over the seeds an arm has. '
                 'Test predictions are five-fold ensembles. Source: operating_points.csv.')


def t_lr(n):
    D, B = read('lr_sweep.csv'), read('lr_best_contrast.csv')
    a = []
    for arm, lab in (('full_224', 'Full FT'), ('last4_224', 'Last-4')):
        for k, lr in enumerate(sorted(D[D.arm == arm].lr.unique())):
            d = D[(D.arm == arm) & (D.lr == lr)].set_index('cls')
            a.append(' & '.join([lab if k == 0 else '', f'{lr:g}' +
                                 (' (default)' if d.default.iloc[0] else '')] +
                                [f(d.auprc[c]) for c in T]) + r' \\')
        if arm == 'full_224':
            a.append(r'\midrule')
    b0 = B.iloc[0]
    b = [' & '.join([c, f(r.last4), f(r.full), s(r.delta), ci(r.lo, r.hi), p(r.p)]) + r' \\'
         for c, r in zip(T, B.set_index('cls').loc[T].itertuples())]
    note = tt('(a) Seed 0, AUPRC on the test set. (b) Each arm at its best IRF rate, paired '
              'patient bootstrap, 5,000 resamples; the best rate is chosen on test-set IRF '
              'AUPRC, so (b) is descriptive. Sources: lr_sweep.csv, lr_best_contrast.csv.')
    return '\n'.join([
        r'\begin{table}[htbp]', r'\centering', r'\small',
        r'\caption{Fine-tuning learning-rate sweep, and each arm at its own best rate.}',
        rf'\label{{tab:S{n}}}',
        r'\textit{(a) Sweep}\par\smallskip',
        r'\begin{tabular}{llccc}', r'\toprule', r'Arm & Learning rate & IRF & SRF & PED \\',
        r'\midrule', *a, r'\bottomrule', r'\end{tabular}', r'\par\bigskip',
        rf'\textit{{(b) Last-4 at {b0.last4_lr:g} against full FT at {b0.full_lr:g}}}\par\smallskip',
        r'\begin{tabular}{lccccc}', r'\toprule',
        r'Class & Last-4 & Full FT & $\Delta$ & 95\% CI & $p$ \\', r'\midrule', *b,
        r'\bottomrule', r'\end{tabular}', r'\par\smallskip',
        r'\parbox{\linewidth}{\footnotesize ' + note + '}', r'\end{table}'])


def t_resolution(n):
    D = read('pvalues.csv')
    D = D[D.family == 'Resolution']
    lab = {'384 - 224, full FT': 'Full FT, 384 $-$ 224', '384 - 224, LP': 'Linear probe (trained), 384 $-$ 224',
           '384 - 224, frozen probe': 'Frozen logistic probe, 384 $-$ 224',
           '448 - 224, last-4': 'Last-4, 448 $-$ 224'}
    rows, k = [], 0
    for comp in lab:
        for c in T:
            r = D[(D.comparison == comp) & (D.cls == c)].iloc[0]
            rows.append([lab[comp] if c == 'IRF' else '', c, s(r.delta), ci(r.lo, r.hi), p(r.p)])
    return table(n, 'Input-resolution contrasts in test AUPRC, including the trained linear '
                 'probe and the frozen logistic probe at 384.',
                 ('llccc', ['Contrast', 'Class', r'$\Delta$ AUPRC', '95\\% CI', '$p$']), rows,
                 'Paired patient bootstrap, 5,000 resamples; uncorrected (resolution is '
                 'reported as differences with intervals, not a Holm family). The trained '
                 'linear probe and the frozen probe disagree on IRF in sign. Source: '
                 'pvalues.csv.', mid=(2, 5, 8))


def t_encoder_size(n):
    D = read('encoder_size_strat.csv')
    rows = []
    for r in D.itertuples():
        star = r'$^{*}$' if r.p_holm < 0.05 else ''
        rows.append([{'lp': 'Linear probe', 'last4': 'Last-4'}[r.depth], r.control, r.cls,
                     r.stratum, int(r.n), f(r.rec_ret), f(r.rec_ctl), s(r.delta) + star,
                     p(r.p_holm)])
    return table(n, 'Size-stratified recall, RETFound against two natural-image encoders: 20 '
                 'tests, Holm-corrected.',
                 ('lllcrcccc', ['Depth', 'Control', 'Class', 'Stratum', '$n$', 'RETFound',
                                'Control', r'$\Delta$', 'Holm $p$']), rows,
                 r'$^{*}$ Holm-adjusted $p<0.05$. Recall at each arm\textquotesingle s own Youden '
                 'threshold; three seeds per arm. Strata split at one ViT/16 patch of total '
                 'lesion area. Source: encoder_size_strat.csv.', mid=(9,))


def t_pooling(n):
    A, C, O = read('fig5_pooling_arms.csv'), read('fig5_capacity_control.csv'), read('pooling_comparison.csv')
    fr = A[A.depth == 'frozen'].set_index(['pooling', 'cls'])
    mlp = C[C.contrast == 'matched MLP - mean'].set_index('cls')
    rows = [['Mean'] + [f(fr.auprc[('mean', c)]) for c in T],
            ['Mean + matched MLP'] + [f(fr.auprc[('mean', c)] + mlp.delta[c]) for c in T],
            ['Attention'] + [f(fr.auprc[('attn', c)]) for c in T]]
    o = O.set_index(['pool', 'cls'])
    rows += [[f'{nm} (out-of-fold)'] + [f(o.oof_auprc[(k, c)]) for c in T]
             for k, nm in (('mean', 'Mean'), ('max', 'Max'), ('topk', 'Top-5\\%'))]
    return table(n, 'Pooling variants on the frozen encoder, AUPRC. Post-fix results only.',
                 ('lccc', ['Pooling', 'IRF', 'SRF', 'PED']), rows,
                 'Upper block: held-out test set, the protocol of figure 5; attention is the '
                 'corrected head at re-selected hyperparameters, three seeds averaged; the '
                 'matched MLP is mean AUPRC plus its paired difference. Lower block: '
                 'out-of-fold on the pool split, rows that never use the attention head and are '
                 'therefore unaffected by the bias-gradient fix. Out-of-fold attention is '
                 'omitted pending remeasurement. Sources: fig5_pooling_arms.csv, '
                 'fig5_capacity_control.csv, pooling_comparison.csv.', mid=(2,))


def t_attn_size(n):
    D, Q = read('pooling_size_tuned.csv'), read('pooling_size_strat_tests.csv')
    rows = []
    for r in D.itertuples():
        hr = 1 - r.rec_mean
        rows.append([r.cls, r.bin, int(r.n), f(r.rec_mean), f(r.rec_attn), s(r.delta),
                     pctf(r.delta / hr) if hr > 0 else '--', p(r.p_holm)])

    def summ(d):
        e = d['bin'].astype(str).str.split('-').str[1].astype(float)
        h = 1 - d.rec_mean
        out = []
        for m in ((e <= 0.58), (e > 0.58) & (h > 0)):
            out.append((np.average(d.delta[m], weights=d.n[m]),
                        np.average((d.delta / h)[m], weights=d.n[m]), int(d.n[m].sum())))
        return out
    st, tu = summ(Q), summ(D)
    rows.append(['', r'\textit{sub-patch, pooled}', tu[0][2], '', '', s(tu[0][0]),
                 pctf(tu[0][1], 0), ''])
    rows.append(['', r'\textit{above-patch, pooled}', tu[1][2], '', '', s(tu[1][0]),
                 pctf(tu[1][1], 0), ''])
    k = len(rows) - 3
    return table(n, 'Attention-pooling recall gain by lesion size, at the nested-CV-selected '
                 'hyperparameters.',
                 ('llrccccc', ['Class', 'Bin (patches)', '$n$', 'Mean', 'Attention',
                               r'$\Delta$', '\\% gap closed', 'Holm $p$']), rows,
                 r'\% gap closed $=\Delta/(1-\text{recall}_\text{mean})$. Pooled rows are '
                 r'scan-weighted; bins with no headroom are excluded from the above-patch '
                 f'pool. At hand-chosen 40 epochs instead: sub-patch {s(st[0][0])} '
                 f'({pctf(st[0][1], 0)}\\%), above-patch {s(st[1][0])} '
                 f'({pctf(st[1][1], 0)}\\%). Out-of-fold, three seeds, Holm across all bins. '
                 'Sources: pooling_size_tuned.csv, pooling_size_strat_tests.csv.', mid=(k,))


def t_xfer(n):
    D = read('aroi_recall_two_thresholds.csv')
    if D is None:
        return missing(n, 'AROI recall by lesion size.', 'aroi_recall_two_thresholds.csv')
    rows, mid = [], []
    for c in T:
        for i, r in enumerate(D[D.cls == c].sort_values('lo_edge').itertuples()):
            rows.append([c if i == 0 else '', f'{r.lo_edge:.2f}--{r.hi_edge:.2f}', int(r.n),
                         int(r.n_patients), f(r.recall_amdsd_thr), f(r.recall_refit_thr),
                         s(r.delta), ci(r.delta_lo, r.delta_hi), p(r.p)])
        mid.append(len(rows) - 1)
    return table(n, 'AROI recall by lesion size under the transferred AMD-SD threshold and a '
                 'threshold refitted on AROI.',
                 ('llrrccccc', ['Class', 'Patches', '$n$', 'Patients', 'AMD-SD thr.', 'Refit',
                                r'$\Delta$', '95\\% CI', '$p$']), rows,
                 'Adjacent patch-unit bins merged until each holds 20 scans and 5 patients. '
                 'Patient bootstrap, 5,000 resamples; the two thresholds score the same scans, '
                 'so $\\Delta$ is paired. Source: aroi_recall_two_thresholds.csv.',
                 mid=tuple(mid[:-1]))


def t_shift_coef(n):
    D, R = read('aroi_score_size_model.csv'), read('aroi_shift_model_recall.csv')
    if D is None:
        return missing(n, 'Score--size model.', 'aroi_score_size_model.csv')
    nm = {'log_area': r'$\log$ area (AMD-SD slope)', 'dataset_AROI': 'AROI shift',
          'log_area_x_AROI': r'$\log$ area $\times$ AROI'}
    rows = []
    for c in T:
        for t in nm:
            r = D[(D.cls == c) & (D.term == t)].iloc[0]
            rows.append([c if t == 'log_area' else '', nm[t], s(r.estimate), ci(r.lo, r.hi),
                         p(r.p)])
        d = R[(R.cls == c) & R.observed.notna()]
        rows.append(['', 'counterfactual MAE', f(d.residual.abs().mean()), '', ''])
    return table(n, r'Score--size model, $\text{logit}(\text{score}) \sim \log(\text{area}) + '
                 r'\text{dataset} + \log(\text{area})\times\text{dataset}$, positives only.',
                 ('llccc', ['Class', 'Term', 'Estimate', '95\\% CI', '$p$']), rows,
                 'AMD-SD side: test-set positives, last-4 seed 0 ensembled. Area in patch '
                 'units. OLS with a patient-level cluster bootstrap resampled within dataset, '
                 '5,000 resamples. MAE: mean absolute residual of the pure-shift counterfactual '
                 'shown bin by bin in the next table. Sources: '
                 'aroi_score_size_model.csv, aroi_shift_model_recall.csv.', mid=(3, 7))


def t_shift_resid(n):
    D = read('aroi_shift_model_recall.csv')
    if D is None:
        return missing(n, 'Shift-model counterfactual.', 'aroi_shift_model_recall.csv')
    rows, mid = [], []
    for c in T:
        for i, r in enumerate(D[D.cls == c].sort_values('lo_edge').itertuples()):
            rows.append([c if i == 0 else '', f'{r.lo_edge:.2f}--{r.hi_edge:.2f}',
                         f(r.predicted), f(r.observed), s(r.residual), f(r.predicted_oof_fit),
                         s(r.residual_oof_fit)])
        mid.append(len(rows) - 1)
    return table(n, 'Pure-shift counterfactual: predicted against observed AROI recall at the '
                 'AMD-SD threshold.',
                 ('llccccc', ['Class', 'Patches', 'Predicted', 'Observed', 'Residual',
                              r'Pred.\ (OOF fit)', r'Resid.\ (OOF fit)']), rows,
                 "AMD-SD out-of-fold scores shifted by the class's fitted AROI offset, then "
                 'thresholded at the AMD-SD Youden threshold. Main columns use the test-fitted '
                 'shift; the OOF-fit columns refit the shift with out-of-fold AMD-SD scores so '
                 'their optimism cancels. Source: aroi_shift_model_recall.csv.',
                 mid=tuple(mid[:-1]))


def t_interaction(n):
    D = read('aroi_interaction_sensitivity.csv')
    if D is None:
        return missing(n, 'Interaction sensitivity.', 'aroi_interaction_sensitivity.csv')
    rows, mid = [], []
    for c in T:
        for i, r in enumerate(D[D.cls == c].itertuples()):
            rows.append([c if i == 0 else '', r.variant, int(r.n), s(r.shift, 2), p(r.shift_p),
                         s(r.inter), ci(r.inter_lo, r.inter_hi), p(r.inter_p)])
        mid.append(len(rows) - 1)
    return table(n, 'Sensitivity of the score--size shift and interaction to logit clipping, '
                 'saturated scores and out-of-fold fitting.',
                 ('llrccccc', ['Class', 'Variant', '$n$', 'Shift', '$p$', 'Interaction',
                               '95\\% CI', '$p$']), rows,
                 'Clipping bounds the scores at $\\epsilon$ and $1-\\epsilon$ before the logit; '
                 '``dropped\'\' excludes scores within $10^{-3}$ of 0 or 1. The out-of-fold '
                 'variant has roughly five times the AMD-SD positives but an optimistic '
                 'baseline. Source: aroi_interaction_sensitivity.csv.', mid=tuple(mid[:-1]))


def t_dark(n):
    M, D = read('aroi_dark_only_model.csv'), read('aroi_dark_only_shift.csv')
    if M is None:
        return missing(n, 'Dark-only refit.', 'aroi_dark_only_model.csv')
    nm = {'log_area': r'$\log$ area', 'dataset_AROI': 'AROI shift',
          'log_area_x_AROI': r'$\log$ area $\times$ AROI'}
    rows = [[nm[r.term], s(r.all_srf), p(r.all_srf_p), s(r.dark_only), p(r.dark_only_p)]
            for r in M.itertuples()]
    rows.append([r'\textit{Counterfactual, dark only}', 'Predicted', 'Observed', 'Residual', '$n$'])
    rows += [[f'{r.lo_edge:.2f}--{r.hi_edge:.2f} patches', f(r.predicted), f(r.observed),
              s(r.residual), int(r.n)] for r in D.itertuples()]
    return table(n, 'SRF score--size model and pure-shift counterfactual refitted on dark '
                 '(fluid-like) AROI slices only.',
                 ('lcccc', ['Term', 'All SRF', '$p$', 'Dark only', '$p$']), rows,
                 'Dark: area-weighted Weber contrast of the slice\'s SRF components below 0. '
                 'AMD-SD side unchanged, since its SRF label excludes SHRM. Counterfactual MAE '
                 f'{f(D.residual.abs().mean())}. Sources: aroi_dark_only_model.csv, '
                 'aroi_dark_only_shift.csv.', mid=(2,))


def t_wbin(n):
    D = read('aroi_dark_srf_auroc_by_bin.csv')
    if D is None or 'aroi_lo' not in D:
        return missing(n, 'Within-bin AUROC.', 'aroi_dark_srf_auroc_by_bin.csv')
    rows = [[f'{r.lo:.2f}--{r.hi:.2f}', f'{int(r.n_aroi)} / {int(r.pts_aroi)}',
             f(r.auroc_aroi), ci(r.aroi_lo, r.aroi_hi),
             f'{int(r.n_amdsd)} / {int(r.pts_amdsd)}', f(r.auroc_amdsd),
             ci(r.amdsd_lo, r.amdsd_hi), s(r.delta), ci(r.delta_lo, r.delta_hi)]
            for r in D.sort_values('lo').itertuples()]
    return table(n, 'Within-bin AUROC for dark SRF: each bin\'s positives against all SRF '
                 'negatives.',
                 ('lcccccccc', ['Patches', '$n$/pts', 'AROI', '95\\% CI', '$n$/pts', 'AMD-SD',
                                '95\\% CI', r'$\Delta$', '95\\% CI']), rows,
                 'AROI: dark slices, zero-shot five-fold ensemble. AMD-SD: out-of-fold on the '
                 'pool split. Patient bootstrap, 2,000 resamples per dataset; $\\Delta$ '
                 'interval from independent draws. Source: aroi_dark_srf_auroc_by_bin.csv.')


def t_shrm(n):
    S_, W = read('shrm_proxy_separation.csv'), read('aroi_weber_cut_sensitivity.csv')
    r = S_.iloc[0]
    head = (f'AMD-SD components: {int(r.n_srf):,} SRF, {int(r.n_shrm):,} SHRM. Weber contrast '
            f'separates them with AUROC {f(r.auroc)}; at a cut of 0, sensitivity {f(r.sensitivity)}, '
            f'specificity {f(r.specificity)}; Youden-optimal cut {s(r.youden_cut)}. Median Weber '
            f'SRF {s(r.weber_median_srf)}, SHRM {s(r.weber_median_shrm)}.')
    if W is None:
        W = read('weber_cut_sensitivity.csv')
    rows = []
    for w in W.itertuples():
        row = [s(w.cut, 2), f(w.amdsd_sens), f(w.amdsd_spec)]
        if 'aroi_bright_n' in W:
            row += [f'{int(w.aroi_bright_n)} / {int(w.aroi_bright_patients)}',
                    f(w.bright_recall_thr), f(w.dark_recall_thr), f(w.bright_recall_ens),
                    f(w.dark_recall_ens)]
        rows.append(row)
    cols = ['Cut', 'Sens.', 'Spec.']
    if 'aroi_bright_n' in W:
        cols += ['Bright $n$/pts', 'Bright @0.80', 'Dark @0.80', 'Bright @0.24', 'Dark @0.24']
    return table(n, 'SHRM proxy: calibration on AMD-SD, where SHRM is annotated, and '
                 'sensitivity of the AROI split to the Weber cut-off.',
                 ('c' * len(cols), cols), rows,
                 head + ' Sens./spec.: AMD-SD SHRM against SRF components at each cut. Bright '
                 '$n$/pts: AROI SRF slices at or above the cut (area-weighted per slice). '
                 'Recall at the transferred 0.80 threshold and at the 0.24 ensemble-scale '
                 'diagnostic threshold (Youden on the AMD-SD test set; not an operating point). '
                 'Sources: shrm_proxy_separation.csv, aroi_weber_cut_sensitivity.csv.')


def t_threshold(n):
    D = read('threshold_displacement.csv')
    rows = [[r.cls, f(r.used_thr, 2), f(r.used_recall), f(r.used_spec), f(r.used_j),
             f(r.diag_thr, 2), f(r.diag_recall), f(r.diag_spec), f(r.diag_j)]
            for r in D.itertuples()]
    return table(n, 'Threshold displacement on the AMD-SD test set: the out-of-fold Youden '
                 'threshold used throughout, against the threshold that maximises $J$ on the '
                 'test scores.',
                 ('lcccccccc', ['Class', 'Thr.', 'Recall', 'Spec.', '$J$', 'Thr.', 'Recall',
                                'Spec.', '$J$']), rows,
                 'Left: Youden on the concatenated out-of-fold validation folds, applied to '
                 'ensembled test scores. Right: Youden on the test set itself, a diagnostic of '
                 'the out-of-fold-versus-ensemble scale gap and not a legitimate operating '
                 'point. Last-4, 224, seed 0. Source: threshold_displacement.csv.')


def t_norm(n):
    D = read('norm_sensitivity.csv')
    A, C = D[D.kind == 'auprc'], D[D.kind == 'contrast']
    rows = []
    for c in T:
        g = lambda tok, pool: A[(A.tokens == tok) & (A.pool == pool) & (A.cls == c)].auprc.mean()
        rows.append([c, f(g('parameter-free', 'attn')), f(g('pretrained', 'attn')),
                     f(g('parameter-free', 'mean')), f(g('pretrained', 'mean'))])
    rows2 = []
    for lab, tok in (('attention $-$ mean, parameter-free', 'parameter-free'),
                     ('attention $-$ mean, pretrained', 'pretrained'),
                     ('pretrained $-$ param.-free, attention', None),
                     ('pretrained $-$ param.-free, mean', None)):
        cells = []
        for c in T:
            if tok:
                r = C[(C.contrast == 'attention - mean') & (C.tokens == tok) & (C.cls == c)].iloc[0]
            else:
                arm = 'attn' if 'attention' in lab.split(',')[1] else 'mean'
                r = C[(C.contrast == f'pretrained - parameter-free ({arm})') & (C.cls == c)].iloc[0]
            cells.append(f'{s(r.delta)} ({p(r.p)})')
        rows2.append([lab] + cells)
    allrows = rows + [[r'\textit{Contrast} ($\Delta$, $p$)', 'IRF', 'SRF', 'PED', '']] + \
        [r + [''] for r in rows2]
    return table(n, 'Sensitivity of the frozen arms to RETFound\'s pretrained final norm.',
                 ('lcccc', ['Class', 'Attn, param.-free', 'Attn, pretrained',
                            'Mean, param.-free', 'Mean, pretrained']), allrows,
                 'Run on the corrected attention head at the hyperparameters selected '
                 '\\emph{before} the bias-gradient fix (IRF $L$=64, lr $10^{-3}$, 5 epochs; '
                 'SRF $L$=32, $3\\times10^{-3}$, 55; PED $L$=32, $3\\times10^{-3}$, 30), not at '
                 'the re-selected configuration of the main text. The pretrained norm is added '
                 'per token before pooling; the parameter-free post-pooling layernorm is kept. '
                 'Uncorrected; sensitivity analysis. Source: norm_sensitivity.csv.', mid=(2,))


def t_leakage(n):
    D = read('leakage_inflation.csv')
    rows = [[r.cls, f(r.patient_level), f(r.random_scans), s(r.inflation)] for r in D.itertuples()]
    return table(n, 'Patient-level against scan-level splitting: out-of-fold AUPRC of the same '
                 'frozen probe.',
                 ('lccc', ['Class', 'Patient-level', 'Scans at random', 'Inflation']), rows,
                 'Frozen RETFound features, logistic probe with $C=0.01$, pool split, five '
                 'folds; only the fold assignment differs. Scan-level splitting lets slices of '
                 'the same eye fall on both sides. Source: leakage_inflation.csv.')


ORDER = [('Operating points at Youden', t_operating_points),
         ('Learning-rate sweep and best-rate comparison', t_lr),
         ('Resolution contrasts', t_resolution),
         ('Size-stratified encoder recall, 20 tests, Holm', t_encoder_size),
         ('Pooling variants, post-fix', t_pooling),
         ('Attention size-stratified recall gain', t_attn_size),
         ('Transfer: per-bin recall', t_xfer),
         ('Transfer: score-size model coefficients', t_shift_coef),
         ('Transfer: shift-model counterfactual residuals', t_shift_resid),
         ('Transfer: interaction sensitivity', t_interaction),
         ('Transfer: dark-only refit', t_dark),
         ('Transfer: within-bin AUROC', t_wbin),
         ('SHRM proxy calibration and Weber cut-off sensitivity', t_shrm),
         ('Threshold displacement', t_threshold),
         ('Norm sensitivity', t_norm),
         ('Leakage', t_leakage)]


def main():
    ap = argparse.ArgumentParser(description='Write supplementary.tex: one numbered table per '
                                 'supplementary analysis, from results/*.csv.')
    ap.add_argument('--out', default=str(RES / 'supplementary.tex'))
    a = ap.parse_args()
    parts = ['% Generated by analysis/build_supplementary.py from results/*.csv. Do not edit.',
             '% Needs \\usepackage{booktabs,amsmath}. Labels are tab:S1, tab:S2, ...', '']
    index = []
    for i, (name, fn) in enumerate(ORDER, 1):
        parts.append(fn(i))
        index.append((i, name))
    Path(a.out).write_text('\n\n'.join(parts) + '\n')
    for i, name in index:
        print(f'  S{i:<3} tab:S{i:<4} {name}')
    print(f'-> {a.out}')


if __name__ == '__main__':
    main()
