#!/usr/bin/env python3
"""Figure 3 — the size effect: recall against lesion area, and the score behind it.

Panel a is recall per patch-unit bin with a bootstrap band and a count strip that
shows which bins were too small to report. Panel b is the continuous score, so a
missed small lesion is visibly scored low rather than scored negative.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import S, T, table, panel_letters, save

BAND_ALPHA = 0.18
STRIP_FRAC = 0.20


def main():
    B = table('fig3_recall_bins')
    A = table('aroi_fig3_recall_bins', required=False)
    Sc = table('fig3_scores')
    TH = table('fig3_thresholds').set_index('cls')
    if A is not None:
        B = pd.concat([B, A], ignore_index=True)
    datasets = list(dict.fromkeys(B.dataset))
    edges = np.unique(np.concatenate([B.lo_edge.values, B.hi_edge.values]))
    log_e = np.log10(edges)

    fig = plt.figure(figsize=(S.WIDTH, 6.6))
    gs = fig.add_gridspec(2, len(datasets), height_ratios=[1.0, STRIP_FRAC],
                          hspace=0.07, wspace=0.24, left=0.09, right=0.99,
                          top=0.955, bottom=0.545)
    gsb = fig.add_gridspec(1, len(T), wspace=0.24, left=0.09, right=0.99,
                           top=0.435, bottom=0.075)

    axa = []
    for j, name in enumerate(datasets):
        ax = fig.add_subplot(gs[0, j])
        st = fig.add_subplot(gs[1, j], sharex=ax)
        axa.append(ax)
        for k, c in enumerate(T):
            d = B[(B.dataset == name) & (B.cls == c)]
            r = d[d.reportable]
            if len(r):
                ax.fill_between(r.x, r.lo, r.hi, color=S.CLASS_COLOR[c],
                                alpha=BAND_ALPHA, lw=0)
                ax.plot(r.x, r.recall, '-o', ms=3.0, lw=1.5, color=S.CLASS_COLOR[c],
                        label=f'{c} (n={int(d.n_positive.iloc[0])})')
            w = (log_e[1:] - log_e[:-1]) / len(T)
            left = log_e[:-1] + k * w
            cols = [to_rgba(S.CLASS_COLOR[c], 0.85 if o else 0.25) for o in d.reportable]
            st.bar(10 ** (left + w / 2), d.n.values,
                   width=10 ** (left + w * 0.9) - 10 ** left, color=cols, lw=0)
        ax.axvline(1.0, color='k', ls='--', lw=1.1)
        st.axvline(1.0, color='k', ls='--', lw=1.1)
        ax.set_xscale('log'); ax.set_ylim(0, 1.0)
        ax.set_title(name, loc='left', fontsize=8.5)
        ax.tick_params(labelbottom=False)
        top = max(st.get_ylim()[1], 1)
        st.set_ylim(0, top)
        st.set_yticks([top]); st.set_yticklabels([f'{int(top)}'], fontsize=5.5)
        st.grid(False)
        st.set_xlabel('lesion area (patches)', fontsize=8)
        for sp in ('right', 'top'):
            st.spines[sp].set_visible(False)
        if j == 0:
            ax.set_ylabel('recall')
            ax.legend(fontsize=6.5, loc='upper left', frameon=True, facecolor='white',
                      edgecolor='none', framealpha=0.9, borderpad=0.5)
            ax.annotate('1 patch', xy=(1.0, 0.02), xytext=(3, 0),
                        textcoords='offset points', fontsize=6.5)
            st.set_ylabel('scans\nper bin', fontsize=6, rotation=0, ha='right',
                          va='center', labelpad=12)

    axb = []
    for j, c in enumerate(T):
        ax = fig.add_subplot(gsb[0, j])
        axb.append(ax)
        d = Sc[Sc.cls == c]
        neg, pos = d[d.label == 0], d[d.label == 1]
        rng = np.random.default_rng(0)
        ax.scatter(0.012 * np.exp(rng.normal(0, .12, len(neg))), neg.score,
                   s=4, alpha=.22, color='#999999', label='negative')
        ax.scatter(pos.area_patches, pos.score, s=5, alpha=.42,
                   color=S.CLASS_COLOR[c], label='positive')
        med = B[(B.dataset == 'AMD-SD') & (B.cls == c) & B.reportable]
        xs, ms = [], []
        for r in med.itertuples():
            m = pos[(pos.area_patches >= r.lo_edge) & (pos.area_patches < r.hi_edge)]
            if len(m) >= 10:
                xs.append(r.x); ms.append(m.score.median())
        ax.plot(xs, ms, '-o', ms=3.5, lw=1.7, color='k', zorder=5, label='median')
        thr = TH.threshold[c]
        ax.axhline(thr, ls='--', lw=1.1, color='#b02a2a')
        ax.text(0.985, thr, f' {thr:.2f} ', transform=ax.get_yaxis_transform(),
                ha='right', va='bottom', fontsize=6.5, color='#b02a2a')
        ax.set_xscale('log'); ax.set_ylim(0, 1.02)
        ax.set_xlabel('lesion area (patches)', fontsize=8)
        ax.set_title(f'{c}   $\\rho$={TH.spearman_rho[c]:.2f}',
                     color=S.CLASS_COLOR[c], fontsize=8.5)
    axb[0].set_ylabel('model score')
    axb[0].legend(frameon=False, fontsize=6, loc='center left', markerscale=1.8)

    panel_letters(fig, [axa[0], axb[0]], 'ab')
    save(fig, 'figure3_size_effect')


if __name__ == '__main__':
    main()
