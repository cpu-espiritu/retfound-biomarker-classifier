#!/usr/bin/env python3
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score as AP, precision_recall_curve,
                             f1_score, roc_auc_score)

import style as S
from style import CLASSES as T


# ------------------------------------------------------------------ shared setup
def load_all(n_boot):
    d = {}
    d['man'] = S.manifest()
    d['te'] = S.test_rows()
    d['P'] = {a: S.arm_preds(a)[0] for a in S.ARMS}
    _, d['Y'], d['G'] = S.arm_preds('last4_224')
    d['thr'] = {a: S.thresholds(a) for a in S.ARMS}
    d['reps'] = S.replicates(d['G'], n=n_boot)
    return d


def linear_probes(man):
    """Frozen-feature logistic probes, one per encoder. C picked by OOF AP on pool."""
    pool = (man.split == 'pool').values
    test = (man.split == 'test').values
    out = {}
    for enc in S.ENCODERS:
        X = S.features(enc)
        pred = np.zeros((test.sum(), len(T)))
        for i, c in enumerate(T):
            y = man[f'label_{c}'].values
            best_C, best_s = None, -np.inf
            for C in (0.001, 0.01, 0.1, 1.0):
                oof = np.full(len(man), np.nan)
                for k in range(5):
                    tr, va = pool & (man.fold != k).values, pool & (man.fold == k).values
                    sc = StandardScaler().fit(X[tr])
                    lr = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
                    oof[va] = lr.predict_proba(sc.transform(X[va]))[:, 1]
                s = AP(y[pool], oof[pool])
                if s > best_s:
                    best_C, best_s = C, s
            sc = StandardScaler().fit(X[pool])
            lr = LogisticRegression(C=best_C, max_iter=3000).fit(sc.transform(X[pool]), y[pool])
            pred[:, i] = lr.predict_proba(sc.transform(X[test]))[:, 1]
        out[enc] = pred
        print(f'    linear probe fitted: {enc}')
    return out


# ------------------------------------------------------------------------- fig 1
def fig1(d):
    """Findings 1, 2, 4 — paired differences in AUPRC."""
    M = dict(d['P'])
    for enc, p in linear_probes(d['man']).items():
        M[f'LP:{enc}'] = p
    Y, reps = d['Y'], d['reps']

    apb = {m: {c: np.full(len(reps), np.nan) for c in T} for m in M}
    for b, idx in enumerate(reps):
        for i, c in enumerate(T):
            yy = Y[idx, i]
            if yy.min() == yy.max():
                continue
            for m in M:
                apb[m][c][b] = AP(yy, M[m][idx, i])

    comps = [('last4_224', 'lp_224', 'Last-4 $-$ linear probe', 'adaptation depth'),
             ('full_224', 'last4_224', 'Full FT $-$ last-4', 'adaptation depth'),
             ('full_384', 'full_224', '384 $-$ 224', 'input resolution, full FT'),
             ('LP:RETFound', 'LP:MAE-IN1k', 'RETFound $-$ MAE-IN1k', 'linear probe only'),
             ('LP:RETFound', 'LP:Sup-IN21k', 'RETFound $-$ Sup-IN21k', 'linear probe only')]

    rows = []
    for a, b_, label, note in comps:
        for i, c in enumerate(T):
            dd = apb[a][c] - apb[b_][c]
            lo, hi = S.ci(dd)
            rows.append(dict(group=label, note=note, cls=c,
                             delta=AP(Y[:, i], M[a][:, i]) - AP(Y[:, i], M[b_][:, i]),
                             lo=lo, hi=hi, p=S.two_sided_p(dd)))
    fx = pd.DataFrame(rows)
    fx['p_holm'] = S.holm(fx.p.values)
    fx['state'] = [S.sig_state(r.p, r.p_holm) for r in fx.itertuples()]

    # ---- layout: each group gets a header row of its own, then its class rows
    slots = []                                    # top-to-bottom
    for _, _, label, note in comps:
        slots.append(('header', label, note))
        for c in T:
            slots.append(('data', label, c))
    ypos = {k: len(slots) - 1 - n for n, k in enumerate(slots)}

    fig, ax = plt.subplots(figsize=(S.WIDTH, 6.4))
    fig.subplots_adjust(left=0.24, right=0.63, top=0.94, bottom=0.20)

    ax.axvspan(-S.EQUIV, S.EQUIV, color='#4a7fb5', alpha=0.10, lw=0, zorder=0)
    ax.axvline(0, c='k', lw=1, zorder=1)

    COL_D, COL_P = 1.03, 1.60
    for _, r in fx.iterrows():
        k = ypos[('data', r['group'], r['cls'])]
        st = S.SIG[r['state']]
        ax.plot([r['lo'], r['hi']], [k, k], c=st['color'], lw=1.6,
                solid_capstyle='round', zorder=2)
        ax.plot(r['delta'], k, ls='none', ms=5.5, mew=1.3, zorder=3,
                marker=st['marker'], mfc=st['mfc'], mec=st['mec'])
        pt = '<0.001' if r['p'] < 0.001 else f"{r['p']:.3f}"
        ax.text(COL_D, k, f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]",
                transform=ax.get_yaxis_transform(), va='center', ha='left',
                fontsize=6.2, family='monospace')
        ax.text(COL_P, k, pt, transform=ax.get_yaxis_transform(), va='center',
                ha='left', fontsize=6.2, family='monospace',
                color='k' if r['state'] == 'holm' else '#666666')

    top = len(slots) - 0.4
    ax.text(COL_D, top, 'Δ  [95% CI]', transform=ax.get_yaxis_transform(),
            fontsize=7, ha='left', style='italic')
    ax.text(COL_P, top, 'p', transform=ax.get_yaxis_transform(),
            fontsize=7, ha='left', style='italic')

    # class ticks on data rows only
    data_slots = [s for s in slots if s[0] == 'data']
    ax.set_yticks([ypos[s] for s in data_slots])
    ax.set_yticklabels([s[2] for s in data_slots])
    for lbl, s in zip(ax.get_yticklabels(), data_slots):
        lbl.set_color(S.CLASS_COLOR[s[2]])

    # group headers on their own rows, in the left margin
    for n, (_, _, label, note) in enumerate(comps):
        k = ypos[('header', label, note)]
        ax.text(-0.24, k + 0.05, label, transform=ax.get_yaxis_transform(),
                va='center', ha='left', fontsize=8.5, fontweight='bold')
        ax.text(-0.24, k - 0.55, note, transform=ax.get_yaxis_transform(),
                va='center', ha='left', fontsize=6.6, style='italic', color='#666666')
        if n:
            ax.axhline(k + 0.75, c='lightgray', lw=0.7, zorder=0)

    ax.set_ylim(-0.8, len(slots) - 0.2)
    ax.set_xlabel('Δ AUPRC', labelpad=2)
    ax.grid(axis='y', visible=False)
    ax.annotate('← favours subtracted', xy=(0, -0.135), xycoords='axes fraction',
                ha='left', fontsize=6.5, color='#444444', annotation_clip=False)
    ax.annotate('favours leading →', xy=(1, -0.135), xycoords='axes fraction',
                ha='right', fontsize=6.5, color='#444444', annotation_clip=False)

    h = [plt.Line2D([], [], ls='none', ms=5.5, mew=1.3, label=l,
                    **{k: v for k, v in S.SIG[s].items() if k != 'color'})
         for s, l in [('holm', 'survives Holm correction'),
                      ('nominal', 'nominal p<0.05 only'),
                      ('null', 'no evidence of difference')]]
    h.append(plt.Rectangle((0, 0), 1, 1, fc='#4a7fb5', alpha=0.10,
                           label=f'equivalence margin ±{S.EQUIV}'))
    ax.legend(handles=h, loc='upper left', bbox_to_anchor=(0, -0.175), ncol=2,
              frameon=False, fontsize=6.8, handletextpad=0.5, columnspacing=1.2)
    ax.set_title('Paired differences in AUPRC', loc='left', pad=8)
    S.save(fig, 'fig1_paired_delta')
    return fx


# ------------------------------------------------------------------------- fig 2
def fig2(d):
    """Finding 3 — recall vs lesion size in absolute patch units."""
    te, Y, reps = d['te'], d['Y'], d['reps']
    patch = S.patch_px('amdsd', 224)
    E = S.PATCH_EDGES

    fig, ax = plt.subplots(figsize=(S.WIDTH, 3.4))
    dropped = []
    for i, c in enumerate(T):
        y = Y[:, i]
        pos = y == 1
        u = te[f'area_{c}'].values[pos] / patch
        hit = (d['P']['last4_224'][pos, i] >= d['thr']['last4_224'][i]).astype(int)
        g = d['G'][pos]

        xs, rec, lo_, hi_ = [], [], [], []
        for b in range(len(E) - 1):
            m = (u >= E[b]) & (u < E[b + 1])
            if m.sum() < S.MIN_SCANS or len(np.unique(g[m])) < S.MIN_PATIENTS:
                dropped.append((c, f'{E[b]:.2f}-{E[b+1]:.2f}', int(m.sum()),
                                len(np.unique(g[m]))))
                continue
            h, gg = hit[m], g[m]
            vals = S.boot_stat(lambda idx, h=h, gg=gg: h[idx].mean(),
                               S.replicates(gg, n=len(reps)))
            l, hh = S.ci(vals)
            xs.append(np.sqrt(E[b] * E[b + 1])); rec.append(h.mean())
            lo_.append(l); hi_.append(hh)
        rec = np.array(rec)
        ax.errorbar(xs, rec, yerr=[rec - np.array(lo_), np.array(hi_) - rec],
                    marker='o', ms=4, lw=1.6, capsize=2, color=S.CLASS_COLOR[c],
                    label=f'{c}  (n={int(pos.sum())})')

    ax.axvline(1.0, color='k', ls='--', lw=1.2)
    ax.annotate('1 patch', xy=(1.0, 0.03), xytext=(4, 0),
                textcoords='offset points', fontsize=7.5)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('scan lesion area (multiples of one ViT-L/16 patch, log scale)')
    ax.set_ylabel('recall')
    ax.legend(frameon=False, loc='lower right')
    ax.set_title('Recall is set by patch-relative lesion size', loc='left')
    fig.tight_layout()
    S.save(fig, 'fig2_recall_by_size', 'report')
    if dropped:
        print(f'    bins dropped (<{S.MIN_SCANS} scans or <{S.MIN_PATIENTS} patients):')
        for c, b, n, npat in dropped:
            print(f'      {c} {b}  n={n} patients={npat}')
    print(f'    1 patch = {patch:.0f} px at 224 input')


# ------------------------------------------------------------------------- fig 3
def fig3(d, arm='last4_224'):
    Y, reps = d['Y'], d['reps']
    vp, vy = S.val_preds(arm)
    ts = np.linspace(0.02, 0.98, 97)

    fig, axes = plt.subplots(1, len(T), figsize=(S.WIDTH, 3.0), sharey=True)
    w = 0.35
    for ax, (i, c) in zip(axes, enumerate(T)):
        y, p = Y[:, i], d['P'][arm][:, i]
        t_f1 = ts[int(np.argmax([f1_score(vy[:, i], vp[:, i] >= t, zero_division=0)
                                 for t in ts]))]
        t_j = S.youden(vy[:, i], vp[:, i])

        for k, (nm, t, col) in enumerate([('F1', t_f1, '#999999'),
                                          ("Youden's J", t_j, S.CLASS_COLOR[c])]):
            pred = p >= t
            est = [pred[y == 1].mean(), (~pred[y == 0]).mean()]
            err = []
            for metric in ('rec', 'spec'):
                vals = S.boot_stat(
                    lambda idx, m=metric: (pred[idx][y[idx] == 1].mean() if m == 'rec'
                                           else (~pred[idx][y[idx] == 0]).mean()), reps)
                err.append(S.ci(vals))
            x = np.arange(2) + (k - 0.5) * w
            ax.bar(x, est, w, color=col, label=f'{nm}  (t={t:.2f})',
                   edgecolor='white', linewidth=0.4)
            ax.errorbar(x, est, fmt='none', ecolor='#333333', elinewidth=0.8, capsize=1.5,
                        yerr=[[est[j] - err[j][0] for j in range(2)],
                              [err[j][1] - est[j] for j in range(2)]])

        ax.set_xticks(range(2))
        ax.set_xticklabels(['recall', 'specificity'])
        ax.set_title(f'{c}   prevalence {y.mean():.2f}\nAUROC {roc_auc_score(y, p):.3f}',
                     color=S.CLASS_COLOR[c], fontsize=8.5)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='x', visible=False)
        ax.legend(frameon=False, fontsize=6.5, loc='lower left')

    axes[0].set_ylabel('rate')
    fig.suptitle(f'Threshold rule changes specificity, not ranking — {S.ARM_LABEL[arm]}',
                 fontsize=9.5, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    S.save(fig, 'fig3_threshold_rule')


# ------------------------------------------------------------------------- fig 4
def fig4(d):
    te, Y, reps = d['te'], d['Y'], d['reps']
    grid = np.linspace(1, 4.4, 160)
    fig, axes = plt.subplots(1, len(T), figsize=(S.WIDTH, 3.0), sharey=True)
    summary = []

    for ax, (i, c) in zip(axes, enumerate(T)):
        y = Y[:, i]
        pos = y == 1
        la = np.log10(np.clip(te[f'area_{c}'].values, 1, None))

        for a in S.ARMS:
            det = (d['P'][a][:, i] >= d['thr'][a][i]).astype(int)
            if det[pos].min() == det[pos].max():
                continue
            lr = LogisticRegression().fit(la[pos].reshape(-1, 1), det[pos])
            ax.plot(grid, lr.predict_proba(grid.reshape(-1, 1))[:, 1],
                    color=S.ARM_COLOR[a], label=S.ARM_LABEL[a])
            w0, b0 = lr.coef_[0][0], lr.intercept_[0]
            if w0 <= 0:
                continue
            x50 = -b0 / w0

            def one(idx, det=det, la=la):
                m = idx[y[idx] == 1]
                if det[m].min() == det[m].max():
                    return np.nan
                f = LogisticRegression().fit(la[m].reshape(-1, 1), det[m])
                return -f.intercept_[0] / f.coef_[0][0] if f.coef_[0][0] > 0 else np.nan

            lo, hi = S.ci(S.boot_stat(one, reps))
            summary.append(dict(cls=c, arm=S.ARM_LABEL[a], px=10 ** x50,
                                lo=10 ** lo, hi=10 ** hi))
            ax.plot([x50], [0.5], marker='o', ms=4, color=S.ARM_COLOR[a], zorder=4)
            ax.hlines(0.5, lo, hi, color=S.ARM_COLOR[a], lw=3, alpha=0.35, zorder=3)

        ax.axhline(0.5, ls=':', c='gray', lw=0.9)
        ax.set_title(f'{c}  (n={int(pos.sum())})', color=S.CLASS_COLOR[c])
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['10', '100', '1k', '10k'])
        ax.set_xlabel('lesion area (px)')
        ax.set_ylim(0, 1.0)

    axes[0].set_ylabel('P(detected)')
    axes[0].legend(frameon=False, fontsize=6.5, loc='upper left')
    fig.suptitle('Detection probability vs lesion size; dot and bar = 50% detection size',
                 fontsize=9.5, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    S.save(fig, 'fig4_detection_size')
    return pd.DataFrame(summary)


# ------------------------------------------------------------------------- fig 5
def fig5(d):
    Y, reps = d['Y'], d['reps']
    fig, axes = plt.subplots(1, len(T), figsize=(S.WIDTH, 2.9), sharex=True)

    for ax, (i, c) in zip(axes, enumerate(T)):
        y = Y[:, i]
        for k, a in enumerate(S.ARMS):
            p = d['P'][a][:, i]
            est = AP(y, p)
            lo, hi = S.ci(S.boot_stat(
                lambda idx: (AP(y[idx], p[idx])
                             if y[idx].min() != y[idx].max() else np.nan), reps))
            ax.errorbar(est, k, xerr=[[est - lo], [hi - est]], fmt='o', ms=5,
                        color=S.ARM_COLOR[a], ecolor=S.ARM_COLOR[a],
                        elinewidth=1.5, capsize=2)
        ax.axvline(y.mean(), ls='--', c='#888888', lw=1)
        ax.text(y.mean(), -0.75, f'prevalence {y.mean():.2f}', fontsize=6.5,
                ha='center', color='#666666')
        ax.set_yticks(range(len(S.ARMS)))
        ax.set_yticklabels([S.ARM_LABEL[a] for a in S.ARMS] if i == 0 else [])
        ax.set_ylim(-1.1, len(S.ARMS) - 0.5)
        ax.set_xlim(0, 1.0)                     # never truncate
        ax.set_title(c, color=S.CLASS_COLOR[c])
        ax.set_xlabel('AUPRC')
        ax.grid(axis='y', visible=False)

    fig.tight_layout()
    S.save(fig, 'fig5_main_result')


# ------------------------------------------------------------------------ fig S1
def figS1(d):
    man = d['man']
    fig, ax = plt.subplots(figsize=(S.WIDTH, 2.8))
    bins = np.logspace(1, 5, 55)
    for c in T:
        a = man.loc[man[f'area_{c}'] > 0, f'area_{c}']
        ax.hist(a, bins=bins, histtype='step', lw=1.6,
                color=S.CLASS_COLOR[c], label=f'{c}  (n={len(a)})')
    ax.set_xscale('log')
    ax.set_xlabel('lesion area (pixels, positives only)')
    ax.set_ylabel('B-scans')
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.set_title('No bimodal gap: any minimum-area threshold would be invented',
                 loc='left')
    fig.tight_layout()
    S.save(fig, 'figS1_lesion_area')


# ---------------------------------------------------------------------- driver
FIGS = {'fig1': fig1, 'fig2': fig2, 'fig3': fig3,
        'fig4': fig4, 'fig5': fig5, 'figS1': figS1}

# every figure maps to a numbered claim; a figure with no claim does not belong here
FINDING = {
    'fig1':  'Findings 1, 2, 4 - paired Delta AUPRC across depth, resolution, pretraining',
    'fig2':  'Finding 3 - recall vs lesion size in absolute patch units',
    'fig3':  'Finding 5 - F1 vs Youden threshold rule',
    'fig4':  'Finding 3 - detection probability vs lesion size',
    'fig5':  'Main result - AUPRC per arm against prevalence',
    'figS1': 'Labels - lesion areas, no minimum-area threshold',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--boot', type=int, default=S.N_BOOT)
    ap.add_argument('--only', nargs='*', choices=list(FIGS))
    a = ap.parse_args()

    print(f'loading results  (bootstrap {a.boot:,} resamples)')
    d = load_all(a.boot)
    print(f'  {len(d["te"])} test scans, {len(np.unique(d["G"]))} patients\n')

    for name in (a.only or FIGS):
        print(f'{name}: {FINDING[name]}')
        out = FIGS[name](d)
        if isinstance(out, pd.DataFrame):
            print(out.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
        print()


if __name__ == '__main__':
    main()
