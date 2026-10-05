import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
# AMD-SD mask indices, as recovered in prep_amdsd.py. SHRM has its own index here;
# AROI's scheme has no SHRM class at all, which is what this script exists to exploit.
AMDSD_IDX = {'SRF': 1, 'IRF': 2, 'PED': 3, 'SHRM': 4}
ALL_LESION = tuple(AMDSD_IDX.values())
CONN = np.ones((3, 3), int)


def per_component(images, masks, files, want, ring=8):
    """Weber contrast of every component of the wanted classes, against an 8 px ring
    that excludes all other annotated lesion pixels. Same recipe as notebooks/contrast.py,
    extended to SHRM so fluid and hyperreflective material can be compared directly."""
    st = np.ones((2 * ring + 1, 2 * ring + 1), int)
    rows = []
    for j, f in enumerate(files):
        g = np.array(Image.open(Path(images) / f).convert('L')).astype(np.float32)
        m = np.array(Image.open(Path(masks) / f).convert('L'))
        lesion_any = np.isin(m, ALL_LESION)
        for c in want:
            lab, n = ndimage.label(m == AMDSD_IDX[c], structure=CONN)
            if not n:
                continue
            objs = ndimage.find_objects(lab)
            for cid in range(1, n + 1):
                sl = objs[cid - 1]
                pad = tuple(slice(max(0, s.start - ring - 1), min(d, s.stop + ring + 1))
                            for s, d in zip(sl, m.shape))
                sub = lab[pad] == cid
                ring_m = (ndimage.binary_dilation(sub, structure=st)
                          & ~sub & ~lesion_any[pad])
                if ring_m.sum() < 20:
                    continue
                gi = g[pad]
                inside, around = float(gi[sub].mean()), float(gi[ring_m].mean())
                rows.append(dict(file=f, cls=c, px=int(sub.sum()), inside=inside,
                                 around=around, contrast=inside - around,
                                 weber=(inside - around) / max(around, 1.0),
                                 ring_px=int(ring_m.sum())))
        if (j + 1) % 500 == 0:
            print(f'  {j + 1}/{len(files)} scans', flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description='Weber contrast of AMD-SD SRF and SHRM components. SHRM is '
                    'annotated separately in AMD-SD but not in AROI, so these two '
                    'distributions calibrate what "fluid-like" and "SHRM-like" mean '
                    'before that cut is applied to AROI SRF.')
    ap.add_argument('--manifest', default=str(ROOT / 'data/amdsd_splits/manifest.csv'))
    ap.add_argument('--images', default=str(ROOT.parent / 'data/amdsd/images'))
    ap.add_argument('--masks', default=str(ROOT.parent / 'data/amdsd/masks'))
    ap.add_argument('--classes', default='SRF,SHRM')
    ap.add_argument('--ring', type=int, default=8)
    ap.add_argument('--out', default=str(ROOT / 'results/amdsd_shrm_contrast.csv'))
    a = ap.parse_args()

    man = pd.read_csv(a.manifest)
    want = a.classes.split(',')
    L = per_component(a.images, a.masks, man.file.tolist(), want, a.ring)
    L.to_csv(a.out, index=False)

    print(f'\n{len(L)} components with a usable ring, ring {a.ring} px\n')
    print(f"{'cls':<6}{'n':>7}{'median w':>10}{'mean w':>9}{'p10':>8}{'p90':>8}"
          f"{'median px':>11}{'% w>=0':>9}")
    for c in want:
        d = L[L.cls == c]
        print(f'{c:<6}{len(d):>7}{d.weber.median():>10.3f}{d.weber.mean():>9.3f}'
              f'{d.weber.quantile(.10):>8.3f}{d.weber.quantile(.90):>8.3f}'
              f'{d.px.median():>11.0f}{100 * (d.weber >= 0).mean():>8.1f}%')
    print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
