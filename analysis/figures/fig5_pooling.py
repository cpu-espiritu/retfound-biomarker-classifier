#!/usr/bin/env python3
"""Figure 5 — pooling x adaptation depth: the interaction, its control, its cost.

Attention pooling lifts a frozen encoder and does nothing to a fine-tuned one.
Panel b shows the gain survives a matched-parameter control, panel c what the
parameters buy.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import S, T, table, panel_letters, save

POOL_COLOR = {'mean': '#666666', 'attn': '#7A5195'}
POOL_LABEL = {'mean': 'mean pooling', 'attn': 'attention pooling'}
DEPTHS = ['frozen', 'last-4']


def main():
    A = table('fig5_pooling_arms')
    C = table('fig5_capacity_control')

    fig = plt.figure(figsize=(S.WIDTH, 6.4))
    gs = fig.add_gridspec(1, len(T), wspace=0.32, left=0.085, right=0.985,
                          top=0.94, bottom=0.60)
    gs2 = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.25], wspace=0.42,
                           left=0.085, right=0.985, top=0.45, bottom=0.085)

    axa = []
    for j, c in enumerate(T):
        ax = fig.add_subplot(gs[0, j])
        axa.append(ax)
        d = A[A.cls == c].set_index(['depth', 'pooling'])
        for pl in ('mean', 'attn'):
            e = [d.loc[(dp, pl)] for dp in DEPTHS]
            xs = np.arange(len(DEPTHS)) + (0.04 if pl == 'attn' else -0.04)
            ax.errorbar(xs, [r.auprc for r in e],
                        yerr=[[r.auprc - r.lo for r in e], [r.hi - r.auprc for r in e]],
                        marker='o', ms=4.5, lw=1.6, capsize=2.5,
                        color=POOL_COLOR[pl], ecolor=POOL_COLOR[pl],
                        label=POOL_LABEL[pl])
        ax.set_xticks(range(len(DEPTHS))); ax.set_xticklabels(DEPTHS)
        ax.set_xlim(-0.35, len(DEPTHS) - 0.65)
        ax.set_title(c, color=S.CLASS_COLOR[c], fontsize=9)
        ax.grid(axis='x', visible=False)
        # each class on its own scale: shared limits flatten SRF and PED against
        # IRF's much wider intervals, and the interaction shape is the point
        lo, hi = d.lo.min(), d.hi.max()
        pad = 0.10 * (hi - lo)
        ax.set_ylim(lo - pad, min(hi + pad, 1.005))
        ax.tick_params(labelsize=7)
    axa[0].set_ylabel('AUPRC')
    axa[0].legend(frameon=False, fontsize=6.5, loc='lower right')
    axa[0].set_xlabel('adaptation depth', fontsize=8)

    axc = fig.add_subplot(gs2[0, 0])
    contrasts = list(dict.fromkeys(C.contrast))
    for k, lab in enumerate(contrasts):
        for j, c in enumerate(T):
            r = C[(C.contrast == lab) & (C.cls == c)].iloc[0]
            pos = k * 3.6 + j
            axc.errorbar(r.delta, pos, xerr=[[r.delta - r.lo], [r.hi - r.delta]],
                         marker='o', ms=4.5, lw=1.5, capsize=2.5,
                         color=S.CLASS_COLOR[c], ecolor=S.CLASS_COLOR[c])
            axc.annotate(c, (r.delta, pos), textcoords='offset points', xytext=(0, 6),
                         ha='center', fontsize=6, color=S.CLASS_COLOR[c])
    axc.axvline(0, color='k', lw=1.1)
    axc.set_yticks([k * 3.6 + 1 for k in range(len(contrasts))])
    axc.set_yticklabels(contrasts, fontsize=7.5)
    axc.set_ylim(-1.0, (len(contrasts) - 1) * 3.6 + 3.0)
    axc.set_xlabel('Δ AUPRC (paired patient bootstrap)', fontsize=8)
    axc.set_title('capacity control, frozen encoder', loc='left', fontsize=8.5)
    axc.grid(axis='y', visible=False)

    axb = fig.add_subplot(gs2[0, 1])
    irf = A[A.cls == 'IRF'].set_index(['depth', 'pooling'])
    off = {('frozen', 'mean'): (6, -14, 'left'), ('frozen', 'attn'): (6, 6, 'left'),
           ('last-4', 'mean'): (8, -16, 'left'), ('last-4', 'attn'): (8, 6, 'left')}
    for dp in DEPTHS:
        for pl in ('mean', 'attn'):
            r = irf.loc[(dp, pl)]
            axb.plot(r.trainable_M, r.auprc, marker='o' if pl == 'mean' else 'D',
                     ms=5.5, color=POOL_COLOR[pl])
            dx, dy, ha = off[(dp, pl)]
            axb.annotate(f'{dp} + {"mean" if pl == "mean" else "attention"}\n'
                         f'{r.trainable_M:g} M   AUPRC {r.auprc:.3f}',
                         (r.trainable_M, r.auprc), textcoords='offset points',
                         xytext=(dx, dy), ha=ha, fontsize=5.8,
                         color=POOL_COLOR[pl], linespacing=1.3)
    axb.set_xscale('log')
    axb.set_xlim(irf.trainable_M.min() / 3, irf.trainable_M.max() * 50)
    lo, hi = irf.auprc.min(), irf.auprc.max()
    axb.set_ylim(lo - 0.06, hi + 0.06)
    axb.set_xlabel('trainable parameters (M, log)', fontsize=8)
    axb.set_ylabel('IRF AUPRC')
    axb.set_title('what the parameters buy', loc='left', fontsize=8.5)

    fig.text(0.005, 0.985, 'a', fontsize=11, fontweight='bold', va='top')
    panel_letters(fig, [axc, axb], 'bc')
    save(fig, 'figure5_pooling')


if __name__ == '__main__':
    main()
