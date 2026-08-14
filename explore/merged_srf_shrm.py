#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

# AMD-SD mask indices, as recovered in prep_amdsd.py
IDX_SRF, IDX_SHRM = 1, 4
CONN = np.ones((3, 3), int)          # 8-connectivity, matching the AROI analysis

# AMD-SD frames are 570x380 and resize to 224x224, so one ViT/16 patch covers:
AMDSD_FRAME = (380, 570)             # (rows, cols)
PATCH_TOKENS = 224 / 16


def patch_px(frame):
    rows, cols = frame
    return (rows / PATCH_TOKENS) * (cols / PATCH_TOKENS)


def main():
    ap = argparse.ArgumentParser(
        description='AMD-SD SRF+SHRM merged into one class, for comparison against '
                    "AROI's merged SRF. Writes its own file; never mix these numbers "
                    'into the IRF/SRF/PED classification tables.')
    ap.add_argument('--masks', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', default='amdsd_SRF_SHRM_MERGED_components.csv')
    a = ap.parse_args()

    if 'MERGED' not in Path(a.out).name.upper():
        raise SystemExit('--out must contain "MERGED" so this file cannot be '
                         'mistaken for the per-class results')

    df = pd.read_csv(a.manifest)
    mdir = Path(a.masks)
    ppx = patch_px(AMDSD_FRAME)
    rows, missing = [], 0

    for n, fname in enumerate(df['file'], 1):
        p = mdir / fname
        if not p.exists():
            missing += 1
            continue
        m = np.array(Image.open(p).convert('L'))
        # merge BEFORE labelling: touching SRF and SHRM become one lesion,
        # which is what AROI's combined annotation would produce
        merged = (m == IDX_SRF) | (m == IDX_SHRM)
        lab, ncomp = ndimage.label(merged, structure=CONN)
        srf_only = int((m == IDX_SRF).sum())
        shrm_only = int((m == IDX_SHRM).sum())
        if ncomp == 0:
            rows.append(dict(file=fname, cls='SRF+SHRM', n_comp=0, total_px=0,
                             median_comp_px=0, max_comp_px=0,
                             median_comp_patches=0.0,
                             srf_px=srf_only, shrm_px=shrm_only, n_comp_srf_only=0))
            continue
        sizes = np.asarray(ndimage.sum(np.ones_like(lab), lab, range(1, ncomp + 1)))
        # how many components would SRF alone have given? difference = merging effect
        _, ncomp_srf = ndimage.label(m == IDX_SRF, structure=CONN)
        rows.append(dict(file=fname, cls='SRF+SHRM', n_comp=int(sizes.size),
                         total_px=int(sizes.sum()),
                         median_comp_px=float(np.median(sizes)),
                         max_comp_px=int(sizes.max()),
                         median_comp_patches=float(np.median(sizes)) / ppx,
                         srf_px=srf_only, shrm_px=shrm_only,
                         n_comp_srf_only=int(ncomp_srf)))
        if n % 500 == 0:
            print(f'  {n}/{len(df)}', flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    if missing:
        print(f'[warn] {missing} manifest rows had no mask on disk')

    pos = out[out.n_comp > 0]
    print(f'\n{len(out)} rows -> {a.out}')
    print(f'  one patch = {ppx:.0f} px at {AMDSD_FRAME[0]}x{AMDSD_FRAME[1]} -> 224\n')
    print(f"{'':<12}{'scans+':>8}{'comps':>8}{'comp/scan':>11}{'med comp px':>13}"
          f"{'med comp patches':>18}")
    print(f"{'SRF+SHRM':<12}{len(pos):>8}{int(pos.n_comp.sum()):>8}"
          f"{pos.n_comp.mean():>11.2f}{pos.median_comp_px.median():>13.0f}"
          f"{pos.median_comp_patches.median():>18.2f}")

    with_shrm = pos[pos.shrm_px > 0]
    print(f'\n  scans where SHRM contributes: {len(with_shrm)}/{len(pos)} '
          f'({len(with_shrm) / max(len(pos), 1):.1%})')
    fused = with_shrm[with_shrm.n_comp < with_shrm.n_comp_srf_only +
                      (with_shrm.shrm_px > 0).astype(int)]
    print(f'  merging fuses SRF and SHRM into one component in {len(fused)} scans')
    print(f'  SHRM share of merged area: '
          f'{pos.shrm_px.sum() / max(pos.total_px.sum(), 1):.1%}')

    print('\nDo not merge this file into the IRF/SRF/PED tables. '
          'SRF here is not the SRF reported elsewhere.')


if __name__ == '__main__':
    main()
