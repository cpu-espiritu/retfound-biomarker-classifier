#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

man, mdir = Path(sys.argv[1]), Path(sys.argv[2])
d = pd.read_csv(man); d['h'] = d.crop_bot - d.crop_top
print(d[['crop_top', 'crop_bot', 'h']].describe().round(1))
print("no-op rows:", int((d.h == 380).sum()), "/", len(d))

tot = inside = 0; miss = []
for _, r in d.iterrows():
    m = np.array(Image.open(mdir / r['file']).convert('L'))
    ys = np.nonzero(m == 5)[0]
    if not ys.size: continue
    k = int(((ys >= r.crop_top) & (ys < r.crop_bot)).sum())
    tot += ys.size; inside += k
    if k < ys.size: miss.append(r['file'])
print(f"\nIS/OS pixels inside crop: {inside/tot:.4f}")
print(f"scans clipping IS/OS: {len(miss)}  e.g. {miss[:5]}")
