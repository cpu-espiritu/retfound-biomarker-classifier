#!/usr/bin/env python3
"""Figure 1 — example B-scans with masks, and lesion size in patch units.

Panel a needs the source images, which the AMD-SD release distributes and this
repository does not; without them the figure falls back to panel b alone.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import S, T, table, panel_letters, save

IMAGES = S.ROOT.parent / 'data/amdsd/images'
MASKS = S.ROOT.parent / 'data/amdsd/masks'
MASK_IDX = {'SRF': 1, 'IRF': 2, 'PED': 3}       # as recovered in prep_amdsd.py
ALPHA = 0.5
PATCH_H, PATCH_W = 380 / 14, 570 / 14           # one ViT-L/16 patch, original px


def main():
    ex = table('fig1_examples', required=False)
    have_images = ex is not None and IMAGES.exists() and MASKS.exists()

    A = table('amdsd_components_per_lesion')
    AM = table('amdsd_MERGED_per_lesion', required=False)
    C = table('aroi_components', required=False)
    patch = S.patch_px('amdsd', 224)

    panels = [('AMD-SD', A.assign(u=A.px / patch),
               None if AM is None else AM.assign(u=AM.px / patch))]
    if C is not None:
        panels.append(('AROI', C.assign(u=C.mm2 / S.patch_mm2(224)), None))

    allu = ([p['u'] for _, p, _ in panels]
            + [x['u'] for _, _, x in panels if x is not None])
    bins = np.logspace(np.log10(min(u[u > 0].min() for u in allu)),
                       np.log10(max(u.max() for u in allu)), 55)

    nrow = len(panels)
    fig = plt.figure(figsize=(S.WIDTH, (1.05 if have_images else 0) + 2.5 * nrow))
    if have_images:
        gs = fig.add_gridspec(1, 3, left=0.09, right=0.995, top=0.965,
                              bottom=1 - 0.16, wspace=0.05)
        gsb = fig.add_gridspec(nrow, 1, left=0.09, right=0.995, top=1 - 0.24,
                               bottom=0.075, hspace=0.30)
    else:
        gsb = fig.add_gridspec(nrow, 1, left=0.09, right=0.995, top=0.95,
                               bottom=0.09, hspace=0.30)

    axes_a = []
    if have_images:
        for j, c in enumerate(T):
            r = ex.set_index('cls').loc[c]
            img = np.array(Image.open(IMAGES / r.file).convert('L'))
            m = np.array(Image.open(MASKS / r.file).convert('L')) == MASK_IDX[c]
            r0, h = int(r.crop_top), int(r.crop_h)
            img, m = img[r0:r0 + h], m[r0:r0 + h]

            ax = fig.add_subplot(gs[0, j])
            axes_a.append(ax)
            ax.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.imshow(np.ma.masked_where(~m, m), cmap=ListedColormap([S.CLASS_COLOR[c]]),
                      alpha=ALPHA, interpolation='nearest')
            ax.contour(m, levels=[0.5], colors=[S.CLASS_COLOR[c]], linewidths=0.5)
            ax.add_patch(plt.Rectangle((6, 6), PATCH_W, PATCH_H, fill=False,
                                       ec='w', lw=0.9))
            ax.text(6 + PATCH_W + 4, 6 + PATCH_H / 2, '1 patch', color='w',
                    fontsize=6, va='center')
            ax.set_title(c, color=S.CLASS_COLOR[c], fontsize=9.5, pad=3)
            ax.set_xlabel(f'{int(r.n_components)} lesions,  '
                          f'{r.min_patches:.2f}-{r.max_patches:.1f} patches',
                          fontsize=6.5, labelpad=2)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    else:
        print('    source B-scans absent — panel b only')

    axb = []
    for i, (name, D_, extra) in enumerate(panels):
        ax = fig.add_subplot(gsb[i, 0], sharex=axb[0] if axb else None)
        axb.append(ax)
        for c in T:
            v = D_.loc[D_.cls == c, 'u']
            if not len(v):
                continue
            ax.hist(v, bins=bins, histtype='step', lw=1.7, color=S.CLASS_COLOR[c],
                    label=f'{c}  n={len(v)},  {(v < 1).mean():.0%} sub-patch')
        if extra is not None:
            v = extra['u']
            ax.hist(v, bins=bins, histtype='step', lw=1.5, ls='--',
                    color=S.CLASS_COLOR['SRF'],
                    label=f'SRF+SHRM merged  n={len(v)},  '
                          f'{(v < 1).mean():.0%} sub-patch')
        ax.axvline(1.0, color='k', ls='--', lw=1.2)
        ax.set_xscale('log')
        ax.set_ylabel('lesions')
        ax.set_ylim(bottom=0)
        ax.set_title(name, loc='left', fontsize=9.5)
        ax.legend(frameon=False, loc='upper left', fontsize=7)
        if i < len(panels) - 1:
            ax.tick_params(labelbottom=False)
    axb[0].annotate('1 patch', xy=(1.0, axb[0].get_ylim()[1]), xytext=(4, -4),
                    textcoords='offset points', va='top', fontsize=7.5)
    axb[-1].set_xlabel('lesion area (multiples of one ViT-L/16 patch, log scale)')

    panel_letters(fig, ([axes_a[0]] if axes_a else []) + [axb[0]],
                  'ab' if axes_a else 'b')
    save(fig, 'figure1_lesion_scale')


if __name__ == '__main__':
    main()
