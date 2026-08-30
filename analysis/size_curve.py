#!/usr/bin/env python3
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score as AP, roc_auc_score

import style as S

FULL = 118
SIZES = [15, 30, 60, FULL]
SEEDS = [0, 1, 2]
ARMS = ['frozen_attn', 'last4_mean', 'frozen_mean']
LABEL = {'frozen_attn': 'Frozen + attention (0.07 M)',
         'last4_mean': 'Last-4 + mean pooling (50 M)',
         'frozen_mean': 'Frozen + mean pooling (0.003 M)'}
COLOR = {'frozen_attn': '#0072B2', 'last4_mean': '#D55E00', 'frozen_mean': '#999999'}
STYLE = {'frozen_attn': dict(ls='-', marker='o', zorder=3),
         'last4_mean': dict(ls='-', marker='s', zorder=3),
         'frozen_mean': dict(ls='--', marker='^', lw=1.0, zorder=2)}


def last4_arm(n):
    """Prefer an explicit size-curve run; at the full pool fall back to the existing
    untagged runs, whose config the step-matched schedule reproduces exactly."""
    arm = f'last4_224_n{n}'
    if glob.glob(str(S.DATA / f'amdsd_preds/preds_{arm}_f*.npz')):
        return arm
    if n == FULL:
        return 'last4_224'
    return arm


def load_last4():
    """Fold-ensembled test predictions per (n, seed, class), thresholds from val."""
    rows, P = [], {}
    for n in SIZES:
        arm = last4_arm(n)
        for s in SEEDS:
            if not glob.glob(str(S.DATA / f'amdsd_preds/preds_{arm}_f*_s{s}.npz')):
                print(f'  missing: {arm} seed {s} — skipped')
                continue
            p, y, g = S.arm_preds(arm, s)
            vp, vy = S.val_preds(arm, s)
            for i, c in enumerate(S.CLASSES):
                thr = S.youden(vy[:, i], vp[:, i])
                pred = p[:, i] >= thr
                P[('last4_mean', n, s, c)] = p[:, i]
                rows.append(dict(arm='last4_mean', n_patients=n, seed=s, cls=c,
                                 auprc=AP(y[:, i], p[:, i]),
                                 auroc=roc_auc_score(y[:, i], p[:, i]),
                                 recall=(pred & (y[:, i] == 1)).sum() / max((y[:, i] == 1).sum(), 1),
                                 spec=((~pred) & (y[:, i] == 0)).sum() / max((y[:, i] == 0).sum(), 1),
                                 thr=thr))
    return pd.DataFrame(rows), P


def load_frozen(path):
    R = pd.read_csv(path)
    P = {}
    for _, r in R.iterrows():
        d = np.load(S.DATA / f"amdsd_preds/sizecurve_frozen_{r['pool']}_{r['cls']}"
                             f"_n{int(r['n_patients'])}_s{int(r['seed'])}.npz")
        P[(r['arm'], int(r['n_patients']), int(r['seed']), r['cls'])] = d['test_p']
    return R, P


def curve(ax, R, cls):
    for arm in ARMS:
        d = R[(R.arm == arm) & (R.cls == cls)]
        if d.empty:
            continue
        g = d.groupby('n_patients').auprc.agg(['mean', 'std', 'count']).sort_index()
        x, m, sd = g.index.values, g['mean'].values, np.nan_to_num(g['std'].values)
        ax.plot(x, m, color=COLOR[arm], label=LABEL[arm], **STYLE[arm])
        ax.fill_between(x, m - sd, m + sd, color=COLOR[arm], alpha=0.15, lw=0)
    ax.set_xscale('log')
    ax.set_xticks(SIZES); ax.set_xticklabels(SIZES)
    ax.minorticks_off()
    ax.set_title(cls)
    ax.set_xlabel('training patients')


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='Learning curves for frozen+attention against last-4 fine-tuning.')
    ap.add_argument('--frozen', default=str(S.ROOT / 'results/size_curve_frozen.csv'))
    ap.add_argument('--out', default=str(S.ROOT / 'results/size_curve.csv'))
    a = ap.parse_args()

    F, Pf = load_frozen(a.frozen)
    L, Pl = load_last4()
    R = pd.concat([F[['arm', 'n_patients', 'seed', 'cls', 'auprc', 'auroc',
                      'recall', 'spec', 'thr']], L], ignore_index=True)
    R.to_csv(a.out, index=False)
    P = {**Pf, **Pl}

    print('\n=== AUPRC, mean over seeds ===')
    print(R.pivot_table(index=['cls', 'arm'], columns='n_patients',
                        values='auprc').round(3).to_string())

    # the question the experiment asks: is the gap largest when data is scarcest?
    y_all = S.test_rows()
    reps = S.replicates(y_all.group.values)
    print('\n=== frozen+attention − last-4+mean, paired patient bootstrap ===')
    print(f"{'cls':<5}{'n':>5}{'attn':>8}{'last4':>8}{'delta':>9}{'95% CI':>20}{'p':>8}")
    tests = []
    for c in S.CLASSES:
        yt = y_all[f'label_{c}'].values
        for n in SIZES:
            A = [P[('frozen_attn', n, s, c)] for s in SEEDS if ('frozen_attn', n, s, c) in P]
            B = [P[('last4_mean', n, s, c)] for s in SEEDS if ('last4_mean', n, s, c) in P]
            if not A or not B:
                continue
            A, B = np.mean(A, 0), np.mean(B, 0)
            d = np.array([AP(yt[i], A[i]) - AP(yt[i], B[i]) for i in reps
                          if yt[i].min() != yt[i].max()])
            lo, hi = S.ci(d)
            p = S.two_sided_p(d)
            tests.append(dict(cls=c, n_patients=n, attn=AP(yt, A), last4=AP(yt, B),
                              delta=AP(yt, A) - AP(yt, B), lo=lo, hi=hi, p=p))
            print(f'{c:<5}{n:>5}{tests[-1]["attn"]:>8.3f}{tests[-1]["last4"]:>8.3f}'
                  f'{tests[-1]["delta"]:>+9.3f}{f"[{lo:+.3f},{hi:+.3f}]":>20}{p:>8.3f}')
    T = pd.DataFrame(tests)
    if len(T):
        T['holm'] = S.holm(T.p.values)
        T.to_csv(S.ROOT / 'results/size_curve_tests.csv', index=False)

    fig, axes = plt.subplots(1, 3, figsize=(S.WIDTH, 2.6), sharey=False)
    for ax, c in zip(axes, S.CLASSES):
        curve(ax, R, c)
    axes[0].set_ylabel('AUPRC (test set)')
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.055), handlelength=2.0, columnspacing=1.4)
    fig.text(0.5, 0.005, 'Test set fixed at 20 patients / 440 scans throughout; '
             'bands are ±1 SD over 3 seeds', ha='center', fontsize=7.5,
             color='#666666')
    S.save(fig, 'size_curve', sub='report')
    print(f'\n-> {a.out}, size_curve_tests.csv')


if __name__ == '__main__':
    main()
