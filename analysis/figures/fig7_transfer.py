#!/usr/bin/env python3
"""Figure 7 — what transfers to AROI and what does not.

Discrimination survives the scanner change; the decision threshold does not.
Both panels need the AROI tables, which the dataset licence keeps out of this
repository.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import S, T, table, panel_letters, save

SHOW_AMDSD = True        # AMD-SD AUROC as a reference tick in panel a


def main():
    A = table('aroi_fig7_auroc', required=False)
    R = table('aroi_fig7_recall', required=False)
    if A is None or R is None:
        print('    figure 7 needs the AROI tables — skipped')
        return
    A = A.set_index('cls')

    fig, axes = plt.subplots(1, 2, figsize=(S.WIDTH, 3.1),
                             gridspec_kw=dict(width_ratios=[1.0, 1.15], wspace=0.32))

    ax = axes[0]
    for k, c in enumerate(T):
        r = A.loc[c]
        ax.errorbar(r.auroc_aroi, -k, xerr=[[r.auroc_aroi - r.lo],
                                            [r.hi - r.auroc_aroi]],
                    marker='o', ms=5, lw=1.6, capsize=2.5, color=S.CLASS_COLOR[c],
                    ecolor=S.CLASS_COLOR[c])
        ax.annotate(f'{r.auroc_aroi:.3f}', (r.auroc_aroi, -k),
                    textcoords='offset points', xytext=(0, 7), ha='center',
                    fontsize=6.5, color=S.CLASS_COLOR[c])
        if SHOW_AMDSD:
            ax.plot(r.auroc_amdsd, -k, marker='|', ms=9, mew=1.4, color='#888888')
    ax.set_yticks([-k for k in range(len(T))])
    ax.set_yticklabels(T, fontsize=8)
    for t_, c in zip(ax.get_yticklabels(), T):
        t_.set_color(S.CLASS_COLOR[c])
    ax.set_ylim(-len(T) + 0.4, 0.75)
    lo = min(A.lo.min(), A.auroc_amdsd.min()) - 0.02
    ax.set_xlim(min(lo, 0.85), 1.005)
    ax.set_xlabel('AUROC, AROI zero-shot', fontsize=8)
    ax.set_title('discrimination transfers', loc='left', fontsize=8.5)
    ax.grid(axis='y', visible=False)
    if SHOW_AMDSD:
        ax.annotate('|  AMD-SD test', xy=(0.02, -len(T) + 0.65),
                    xycoords=('axes fraction', 'data'), fontsize=6, color='#888888')

    ax = axes[1]
    arms = list(dict.fromkeys(R.arm))
    aroi_arms = [a for a in arms if a.startswith('AROI')]
    w = 0.36
    thr = {}
    for i, c in enumerate(T):
        d = R[R.cls == c].set_index('arm')
        for j, a in enumerate(aroi_arms):
            v = d.loc[a]
            ax.bar(i + (j - 0.5) * w, v.recall_std, width=w * 0.92,
                   facecolor=S.CLASS_COLOR[c] if j == 0 else 'white',
                   edgecolor=S.CLASS_COLOR[c], lw=1.1, hatch='' if j == 0 else '////')
            ax.text(i + (j - 0.5) * w, v.recall_std + 0.02, f'{v.recall_std:.3f}',
                    ha='center', va='bottom', fontsize=6.5, color=S.CLASS_COLOR[c])
        thr[c] = (d.loc[aroi_arms[0]].threshold, d.loc[aroi_arms[1]].threshold)
    ax.set_xticks(range(len(T)))
    ax.set_xticklabels([f'{c}\n{thr[c][0]:.2f} → {thr[c][1]:.2f}' for c in T],
                       fontsize=7)
    for t_, c in zip(ax.get_xticklabels(), T):
        t_.set_color(S.CLASS_COLOR[c])
    ax.set_xlabel('Youden threshold, AMD-SD → refit on AROI', fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('sub-patch recall\n(standardised)', fontsize=8)
    ax.set_title('calibration does not', loc='left', fontsize=8.5)
    ax.grid(axis='x', visible=False)
    ax.legend(handles=[Patch(facecolor='#777777', edgecolor='#777777',
                             label=aroi_arms[0]),
                       Patch(facecolor='white', edgecolor='#777777', hatch='////',
                             label=aroi_arms[1])],
              frameon=False, fontsize=6.5, loc='upper left')

    fig.subplots_adjust(left=0.10, right=0.985, top=0.88, bottom=0.22)
    panel_letters(fig, axes, 'ab')
    save(fig, 'figure7_transfer')


if __name__ == '__main__':
    main()
