#!/usr/bin/env python3
"""Figure 6 — one forest plot for depth, resolution, encoder, patch crossing.

No significance stars: three of the four panels carry equivalence claims, which a
multiplicity correction reads backwards. Zero is marked and the intervals state
the precision. The one mark is the Holm survivors inside the encoder panel.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import S, T, table, panel_letters, save

GAP = 0.9                    # blank rows between contrast groups
MARKED_PANEL = 'Encoder'


def layout(d):
    """Row positions, top down, with a gap between groups."""
    pos, y = [], 0.0
    for k, g in enumerate(d.contrast):
        if k and g != d.contrast.iloc[k - 1]:
            y += GAP
        pos.append(-y)
        y += 1.0
    return np.array(pos)


def main():
    F = table('fig6_forest')
    panels = list(dict.fromkeys(F.panel))
    heights = [len(F[F.panel == p]) + GAP * (F[F.panel == p].contrast.nunique() - 1)
               + 1.6 for p in panels]
    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(S.WIDTH, 8.0),
                             gridspec_kw=dict(height_ratios=heights, hspace=0.20))

    for ax, name in zip(axes, panels):
        d = F[F.panel == name].reset_index(drop=True)
        pos = layout(d)
        for k, r in enumerate(d.itertuples()):
            marked = name == MARKED_PANEL and r.p_holm < 0.05
            ax.errorbar(r.delta, pos[k], xerr=[[r.delta - r.lo], [r.hi - r.delta]],
                        marker='o', ms=4.5, lw=1.5, capsize=2.5,
                        color=S.CLASS_COLOR[r.cls], ecolor=S.CLASS_COLOR[r.cls],
                        mfc=S.CLASS_COLOR[r.cls] if (marked or name != MARKED_PANEL)
                        else 'white', mew=1.4)
            if k == 0 or d.contrast[k - 1] != r.contrast:
                ax.annotate(r.contrast, xy=(0.008, pos[k] + 0.62),
                            xycoords=('axes fraction', 'data'), fontsize=6.8,
                            color='#333333', va='center')
            if np.isfinite(r.n):
                ax.annotate(f'n={int(r.n)}', xy=(0.995, pos[k]),
                            xycoords=('axes fraction', 'data'), ha='right',
                            va='center', fontsize=6, color='#666666')
        ax.axvline(0, color='k', lw=1.1, zorder=0)
        ax.set_yticks(pos)
        ax.set_yticklabels(d.cls, fontsize=6.5)
        ax.set_ylim(pos.min() - 0.7, pos.max() + 1.3)
        ax.grid(axis='y', visible=False)
        ax.set_title(name, loc='left', fontsize=8.5)
        if name == MARKED_PANEL:
            ax.annotate('filled marker: survives Holm within this panel',
                        xy=(0.995, pos.max() + 0.75),
                        xycoords=('axes fraction', 'data'), ha='right', va='center',
                        fontsize=6, color='#666666')

    metrics = F.groupby('panel').metric.first()
    letters = dict(zip(panels, 'abcdefgh'))
    units = ', '.join(f'{metrics[p]} in {letters[p]}' for p in panels)
    axes[-1].set_xlabel(f'Δ  ({units}).  Paired patient bootstrap, '
                        f'{S.N_BOOT:,} resamples', fontsize=8)
    panel_letters(fig, axes, 'abcd')
    save(fig, 'figure6_forest')


if __name__ == '__main__':
    main()
