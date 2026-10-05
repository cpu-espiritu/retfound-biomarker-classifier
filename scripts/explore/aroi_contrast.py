import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

# AROI mask indices, as used by scripts/finetune/infer_aroi.py. Note there is no
# SHRM class: anything hyperreflective in the subretinal space is annotated as SRF,
# which is the asymmetry against AMD-SD this script is here to measure.
AROI_IDX = {'IRF': 7, 'SRF': 6, 'PED': 5}
ALL_LESION = tuple(AROI_IDX.values())
CONN = np.ones((3, 3), int)


def per_component(root, want, ring=8):
    st = np.ones((2 * ring + 1, 2 * ring + 1), int)
    pats = sorted([p for p in Path(root).iterdir()
                   if p.is_dir() and p.name.startswith('patient')],
                  key=lambda p: int(re.sub(r'\D', '', p.name)))
    rows, n_slices = [], 0
    for p in pats:
        for mf in sorted((p / 'mask' / 'number').glob('*.png')):
            rf = p / 'raw' / 'ALL' / mf.name
            if not rf.exists():
                print(f'[warn] no raw slice for {mf.name}')
                continue
            g = np.array(Image.open(rf).convert('L')).astype(np.float32)
            m = np.array(Image.open(mf).convert('L'))
            if g.shape != m.shape:
                print(f'[warn] shape mismatch {mf.name}: {g.shape} vs {m.shape}')
                continue
            n_slices += 1
            lesion_any = np.isin(m, ALL_LESION)
            sl_no = int(re.search(r'raw(\d+)', mf.stem).group(1))
            for c in want:
                lab, n = ndimage.label(m == AROI_IDX[c], structure=CONN)
                if not n:
                    continue
                objs = ndimage.find_objects(lab)
                for cid in range(1, n + 1):
                    s_ = objs[cid - 1]
                    pad = tuple(slice(max(0, x.start - ring - 1),
                                      min(d, x.stop + ring + 1))
                                for x, d in zip(s_, m.shape))
                    sub = lab[pad] == cid
                    ring_m = (ndimage.binary_dilation(sub, structure=st)
                              & ~sub & ~lesion_any[pad])
                    if ring_m.sum() < 20:
                        continue
                    gi = g[pad]
                    inside, around = float(gi[sub].mean()), float(gi[ring_m].mean())
                    rows.append(dict(patient=p.name, slice=sl_no, file=mf.name, cls=c,
                                     px=int(sub.sum()), inside=inside, around=around,
                                     contrast=inside - around,
                                     weber=(inside - around) / max(around, 1.0),
                                     ring_px=int(ring_m.sum())))
    print(f'{n_slices} slices, {len(pats)} patients')
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description='Weber contrast of every AROI lesion component, same 8 px ring '
                    'recipe as notebooks/contrast.py. Writes no image data, only '
                    'per-component intensity summaries.')
    ap.add_argument('--aroi', required=True, help='the "24 patient" directory')
    ap.add_argument('--classes', default='SRF,IRF,PED')
    ap.add_argument('--ring', type=int, default=8)
    ap.add_argument('--out', default='results/aroi_lesion_contrast.csv')
    a = ap.parse_args()

    want = a.classes.split(',')
    L = per_component(a.aroi, want, a.ring)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    L.to_csv(a.out, index=False)

    print(f'\n{len(L)} components with a usable ring, ring {a.ring} px\n')
    print(f"{'cls':<6}{'n':>7}{'median w':>10}{'mean w':>9}{'p10':>8}{'p90':>8}"
          f"{'median px':>11}{'% w>=0':>9}")
    for c in want:
        d = L[L.cls == c]
        if not len(d):
            continue
        print(f'{c:<6}{len(d):>7}{d.weber.median():>10.3f}{d.weber.mean():>9.3f}'
              f'{d.weber.quantile(.10):>8.3f}{d.weber.quantile(.90):>8.3f}'
              f'{d.px.median():>11.0f}{100 * (d.weber >= 0).mean():>8.1f}%')
    print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
