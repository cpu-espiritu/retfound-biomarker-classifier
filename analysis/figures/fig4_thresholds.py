#!/usr/bin/env python3
"""Figure 4 — threshold behaviour on sub-patch lesions, and what transfers.

Panel a sweeps the decision threshold on AMD-SD and reads sub-patch recall
against specificity. Panel b applies the model to AROI at AMD-SD's thresholds
and at thresholds refit on AROI.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import S, T, table, panel_letters, save

SHOW_ALL = True          # dotted all-lesion sweep behind each sub-patch curve


def main():
    W = table('fig4_threshold_sweep')
    Z = table('aroi_fig4_operating_points', required=False)
    ncol = 1 if Z is None else 2

    fig, axes = plt.subplots(1, ncol, figsize=(S.WIDTH, 3.0), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.0, 1.15][:ncol],
                                              wspace=0.32))
    axes = axes[0]

    ax = axes[0]
    for c in T:
        d = W[W.cls == c].sort_values('threshold')
        if SHOW_ALL:
            ax.plot(d.specificity, d.recall_all, lw=1.0, ls=':',
                    color=S.CLASS_COLOR[c], alpha=0.55)
        ax.plot(d.specificity, d.recall_subpatch, lw=1.7, color=S.CLASS_COLOR[c],
                label=f'{c}  (n={int(d.n_subpatch.iloc[0])} sub-patch)')
        t = d.youden_threshold.iloc[0]
        k = int(np.argmin(np.abs(d.threshold.values - t)))
        ax.plot(d.specificity.values[k], d.recall_subpatch.values[k], 'o', ms=5.5,
                mfc='white', mew=1.6, color=S.CLASS_COLOR[c], zorder=5)
        ax.annotate(f'{t:.2f}', (d.specificity.values[k], d.recall_subpatch.values[k]),
                    textcoords='offset points', xytext=(5, -7), fontsize=6.5,
                    color=S.CLASS_COLOR[c])
    ax.plot([], [], lw=1.0, ls=':', color='#666666', label='all lesions')
    ax.plot([], [], 'o', ms=5.5, mfc='white', mew=1.6, color='#666666',
            label='Youden point')
    ax.set_xlim(1.02, -0.02)                 # specificity falls left to right
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('specificity')
    ax.set_ylabel('recall, sub-patch lesions')
    ax.set_title('AMD-SD test, threshold sweep', loc='left', fontsize=8.5)
    ax.legend(fontsize=6.2, loc='lower right', frameon=True, facecolor='white',
              edgecolor='none', framealpha=0.9)

    if Z is not None:
        ax = axes[1]
        w = 0.36
        srcs = list(dict.fromkeys(Z.threshold_source))
        for i, c in enumerate(T):
            for j, src in enumerate(srcs):
                r = Z[(Z.cls == c) & (Z.threshold_source == src)].iloc[0]
                ax.bar(i + (j - 0.5) * w, r.recall, width=w * 0.92,
                       facecolor=S.CLASS_COLOR[c] if j == 0 else 'white',
                       edgecolor=S.CLASS_COLOR[c], lw=1.1,
                       hatch='' if j == 0 else '////')
                ax.errorbar(i + (j - 0.5) * w, r.recall,
                            yerr=[[r.recall - r.lo], [r.hi - r.recall]], fmt='none',
                            ecolor='#444444', elinewidth=1.0, capsize=2)
                ax.text(i + (j - 0.5) * w, r.hi + 0.03,
                        f't={r.threshold:.2f}\nspec {r.specificity:.2f}', ha='center',
                        va='bottom', fontsize=5.8, color='#555555', linespacing=1.35)
        n = Z.drop_duplicates('cls').set_index('cls')
        ax.set_xticks(range(len(T)))
        ax.set_xticklabels([f'{c}\n{int(n.n_positive[c])}+ / {int(n.n_negative[c])}-'
                            for c in T], fontsize=7)
        ax.set_ylim(0, 1.32)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylabel('recall')
        ax.set_title('AROI zero-shot', loc='left', fontsize=8.5)
        ax.grid(axis='x', visible=False)
        ax.legend(handles=[Patch(facecolor='#777777', edgecolor='#777777',
                                 label=srcs[0]),
                           Patch(facecolor='white', edgecolor='#777777', hatch='////',
                                 label=srcs[1])],
                  frameon=False, fontsize=6.5, loc='upper left')

    fig.subplots_adjust(left=0.10, right=0.985, top=0.88, bottom=0.20)
    panel_letters(fig, axes, 'ab'[:ncol])
    save(fig, 'figure4_thresholds')


if __name__ == '__main__':
    main()
