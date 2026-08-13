#!/usr/bin/env python3
"""Per-scan connected-component statistics for AMD-SD masks.

Emits one row per (scan, class) so recall can be regressed on fragmentation
(component count) while controlling for size (total area). The manifest only carries
total pixel area, which cannot separate "one big lesion" from "six small ones".

    python explore/components_amdsd.py \
        --masks $PROJECT/retfound/data/amdsd/masks \
        --manifest $PROJECT/retfound/data/amdsd_splits/manifest.csv \
        --out amdsd_components.csv

Writes numbers only. Roughly 3,000 masks, about a minute.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

CLASS_MAP = {1: 'SRF', 2: 'IRF', 3: 'PED'}      # as recovered in prep_amdsd.py
CONN = np.ones((3, 3), int)                     # 8-connectivity: diagonals joined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--masks', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', default='amdsd_components.csv')
    ap.add_argument('--min-px', type=int, default=1,
                    help='drop components smaller than this (default: keep all)')
    a = ap.parse_args()

    df = pd.read_csv(a.manifest)
    mdir = Path(a.masks)
    rows, missing = [], 0

    for n, fname in enumerate(df['file'], 1):
        p = mdir / fname
        if not p.exists():
            missing += 1
            continue
        m = np.array(Image.open(p).convert('L'))
        for k, cls in CLASS_MAP.items():
            lab, ncomp = ndimage.label(m == k, structure=CONN)
            if ncomp == 0:
                rows.append(dict(file=fname, cls=cls, n_comp=0, total_px=0,
                                 median_comp_px=0, max_comp_px=0, mean_comp_px=0))
                continue
            sizes = ndimage.sum(np.ones_like(lab), lab, range(1, ncomp + 1))
            sizes = np.asarray(sizes)[np.asarray(sizes) >= a.min_px]
            if sizes.size == 0:
                rows.append(dict(file=fname, cls=cls, n_comp=0, total_px=0,
                                 median_comp_px=0, max_comp_px=0, mean_comp_px=0))
                continue
            rows.append(dict(file=fname, cls=cls, n_comp=int(sizes.size),
                             total_px=int(sizes.sum()),
                             median_comp_px=float(np.median(sizes)),
                             max_comp_px=int(sizes.max()),
                             mean_comp_px=float(sizes.mean())))
        if n % 500 == 0:
            print(f'  {n}/{len(df)}', flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    if missing:
        print(f'[warn] {missing} manifest rows had no mask on disk')

    print(f'\n{len(out)} rows -> {a.out}')
    print(f"\n{'class':<6}{'scans+':>8}{'comps':>8}{'comp/scan':>11}"
          f"{'med comp px':>13}{'med total px':>14}")
    for cls in CLASS_MAP.values():
        s = out[(out.cls == cls) & (out.n_comp > 0)]
        print(f'{cls:<6}{len(s):>8}{int(s.n_comp.sum()):>8}{s.n_comp.mean():>11.2f}'
              f'{s.median_comp_px.median():>13.0f}{s.total_px.median():>14.0f}')

    # cross-check against the manifest's own area columns
    print('\nagreement with manifest area_* (should be exact):')
    for cls in CLASS_MAP.values():
        j = out[out.cls == cls].merge(df[['file', f'area_{cls}']], on='file')
        bad = int((j.total_px != j[f'area_{cls}']).sum())
        print(f'  {cls}: {len(j) - bad}/{len(j)} match' +
              ('' if bad == 0 else f'   [{bad} MISMATCH — investigate]'))


if __name__ == '__main__':
    main()
