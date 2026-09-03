#!/usr/bin/env python3
import glob
from pathlib import Path

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
            fs = glob.glob(str(S.DATA / f'amdsd_preds/preds_{arm}_f*_s{s}.npz'))
            if not fs:
                print(f'  missing: {arm} seed {s} — skipped')
                continue
            # a partly-failed array task would otherwise be averaged silently, and a
            # 3-fold ensemble is not comparable with the 5-fold ones it is plotted against
            if len(fs) != 5:
                raise SystemExit(
                    f'{arm} seed {s}: found {len(fs)} folds, expected 5 — '
                    f'rerun the missing folds before analysing\n  ' +
                    '\n  '.join(sorted(Path(f).name for f in fs)))
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


def load_frozen_full_pool():
    """The n=118 point of the frozen arm, reused from the existing frozen_arms.py run.

    At the full pool the subsample is the whole of `split == 'pool'` and the step
    budget scales by 118/118 = 1, so frozen_size_curve.py at n=118 refits exactly what
    frozen_arms.py already fitted: same mask, same modal hyperparameters, same
    per-fold models, same Youden threshold. Recomputing it burns ~3 h of CPU for
    identical numbers.
    """
    csv = S.ROOT / 'results/frozen_arms.csv'
    if not csv.exists():
        return None, {}
    A = pd.read_csv(csv)
    rows, P = [], {}
    for how in ('attn', 'mean'):
        avail = sorted(A[A.pool == how].seed.unique())
        if not avail:
            return None, {}
        for c in S.CLASSES:
            for s in SEEDS:
                # mean pooling is a deterministic logistic fit and the full-pool
                # subset does not depend on the seed, so its single run stands for
                # every seed; the attention arm genuinely varies and is read per seed
                src = s if (how == 'attn' and s in avail) else avail[0]
                f = S.DATA / f'amdsd_preds/frozen_{how}_{c}_s{src}.npz'
                r = A[(A.pool == how) & (A.cls == c) & (A.seed == src)]
                if not f.exists() or r.empty:
                    return None, {}
                r = r.iloc[0]
                P[(f'frozen_{how}', FULL, s, c)] = np.load(f)['test_p']
                rows.append(dict(arm=f'frozen_{how}', pool=how, n_patients=FULL,
                                 seed=s, cls=c, auprc=r.auprc, auroc=r.auroc,
                                 recall=r.recall, spec=r.spec, thr=r.thr))
    return pd.DataFrame(rows), P


def load_frozen(pattern):
    """One CSV per array task, or a single CSV if the arm was run in one go."""
    fs = sorted(glob.glob(pattern))
    if not fs:
        raise SystemExit(f'no frozen results matching {pattern!r} — '
                         f'run scripts/features/frozen_size_curve.py first')
    print(f'  frozen: {len(fs)} file(s) — ' + ', '.join(Path(f).name for f in fs))
    R = (pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
           .drop_duplicates(['arm', 'n_patients', 'seed', 'cls'], keep='last'))
    # a size is usable only if every cell ran: 2 pools x 3 classes x 3 seeds
    want = 2 * len(S.CLASSES) * len(SEEDS)
    have = R.groupby('n_patients').size()
    for n, k in have.items():
        if k < want:
            print(f'  n={n}: {k}/{want} rows — incomplete, dropped')
    R = R[R.n_patients.isin(have[have == want].index)]

    P = {}
    for _, r in R.iterrows():
        d = np.load(S.DATA / f"amdsd_preds/sizecurve_frozen_{r['pool']}_{r['cls']}"
                             f"_n{int(r['n_patients'])}_s{int(r['seed'])}.npz")
        P[(r['arm'], int(r['n_patients']), int(r['seed']), r['cls'])] = d['test_p']

    if FULL not in set(R.n_patients):
        Rf, Pf = load_frozen_full_pool()
        if Rf is None:
            print(f'  n={FULL}: not in the size-curve run and frozen_arms.py output '
                  f'not found — the frozen curve will stop short of the full pool')
        else:
            print(f'  n={FULL}: reused from results/frozen_arms.csv '
                  f'(identical protocol at the full pool)')
            R = pd.concat([R, Rf], ignore_index=True)
            P.update(Pf)
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
    ap.add_argument('--frozen', default=str(S.ROOT / 'results/size_curve_frozen*.csv'),
                    help='glob; the slurm array writes one CSV per training-set size')
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
    print('    (difference taken within each data draw, then averaged over seeds)')
    print(f"{'cls':<5}{'n':>5}{'attn':>8}{'last4':>8}{'delta':>9}{'95% CI':>20}{'p':>8}")
    tests = []
    for c in S.CLASSES:
        yt = y_all[f'label_{c}'].values
        for n in SIZES:
            sd = [s for s in SEEDS if ('frozen_attn', n, s, c) in P
                  and ('last4_mean', n, s, c) in P]
            if not sd:
                continue
            A = [P[('frozen_attn', n, s, c)] for s in sd]
            B = [P[('last4_mean', n, s, c)] for s in sd]
            # Difference within a seed, then average over seeds. Averaging the
            # predictions first would pool three *different* patient draws at n<118
            # and score a 3-draw ensemble as though it were one model trained on n
            # patients — at n=15 that turns IRF 0.626 into 0.837. Both arms share the
            # same subsample at a given seed, so the within-seed difference is paired.
            d = np.array([np.mean([AP(yt[i], a[i]) - AP(yt[i], b[i])
                                   for a, b in zip(A, B)])
                          for i in reps if yt[i].min() != yt[i].max()])
            lo, hi = S.ci(d)
            p = S.two_sided_p(d)
            ma = float(np.mean([AP(yt, a) for a in A]))
            mb = float(np.mean([AP(yt, b) for b in B]))
            tests.append(dict(cls=c, n_patients=n, attn=ma, last4=mb,
                              delta=float(np.mean([AP(yt, a) - AP(yt, b)
                                                   for a, b in zip(A, B)])),
                              lo=lo, hi=hi, p=p, n_seeds=len(sd)))
            print(f'{c:<5}{n:>5}{ma:>8.3f}{mb:>8.3f}'
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
