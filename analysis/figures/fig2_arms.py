#!/usr/bin/env python3
"""Figure 2 — adaptation arms against AUPRC, including last-4 at 448.

Depth sets the hue, input resolution sets the marker fill, so no arm colour
reuses a class colour and the panel titles never contradict the markers.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import S, T, table, save

LABEL = {'lp_224': 'Linear probe, 224', 'last4_224': 'Last-4 blocks, 224',
         'last4_448': 'Last-4 blocks, 448', 'full_224': 'Full FT, 224',
         'full_384': 'Full FT, 384'}
COLOR = {'linear probe': '#999999', 'last-4': '#5D3A9B', 'full FT': '#882255'}
ORDER = ['lp_224', 'last4_224', 'last4_448', 'full_224', 'full_384']


def main():
    D = table('fig2_arms_auprc')
    arms = [a for a in ORDER if a in set(D.arm)]
    base = D.groupby('depth').input_size.min().to_dict()   # 224 is the filled marker

    fig, axes = plt.subplots(1, len(T), figsize=(S.WIDTH, 3.0), sharex=True)
    for ax, c in zip(axes, T):
        d = D[D.cls == c].set_index('arm')
        for k, a in enumerate(arms):
            r = d.loc[a]
            col = COLOR[r.depth]
            ax.errorbar(r.auprc, k, xerr=[[r.auprc - r.lo], [r.hi - r.auprc]],
                        fmt='o', ms=5, color=col, ecolor=col,
                        mfc=col if r.input_size == base[r.depth] else 'white',
                        mew=1.4, elinewidth=1.5, capsize=2)
        prev = D.loc[D.cls == c, 'prevalence'].iloc[0] if 'prevalence' in D else None
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([f'{LABEL[a]}  ({d.loc[a].trainable_M:g} M)' for a in arms]
                           if c == T[0] else [], fontsize=7)
        ax.set_ylim(-0.7, len(arms) - 0.4)
        ax.set_xlim(0, 1.0)                     # never truncate a bounded metric
        ax.set_title(c, color=S.CLASS_COLOR[c])
        ax.set_xlabel('AUPRC')
        ax.grid(axis='y', visible=False)
        if prev is not None:
            ax.axvline(prev, ls='--', c='#888888', lw=1)
            # white patch behind the label so the line never crosses the text
            ax.text(prev, -0.45, f'prevalence {prev:.2f}', fontsize=6.5, ha='center',
                    color='#666666', bbox=dict(fc='white', ec='none', pad=1.2))
    fig.tight_layout()
    save(fig, 'figure2_arms')


if __name__ == '__main__':
    main()
