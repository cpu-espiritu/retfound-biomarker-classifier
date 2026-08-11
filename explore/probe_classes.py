#!/usr/bin/env python3
import argparse, random
from pathlib import Path
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--masks', required=True)
ap.add_argument('--sample', type=int, default=400)
a = ap.parse_args()

paths = sorted(Path(a.masks).glob('*.png'))
random.seed(0)
paths = random.sample(paths, min(a.sample, len(paths)))

stats = {k: {'n': 0, 'area': [], 'cy': [], 'h': [], 'w': [], 'rel': []}
         for k in range(1, 6)}

for p in paths:
    m = np.array(Image.open(p).convert('L'))
    H, W = m.shape
    cys = {}
    for k in range(1, 6):
        ys, xs = np.nonzero(m == k)
        if ys.size == 0: continue
        cys[k] = ys.mean() / H
        s = stats[k]
        s['n'] += 1
        s['area'].append(ys.size)
        s['cy'].append(ys.mean() / H)
        s['h'].append((ys.max() - ys.min() + 1))
        s['w'].append((xs.max() - xs.min() + 1))
    ref = cys.get(5)
    if ref is not None:
        for k, v in cys.items():
            if k != 5: stats[k]['rel'].append(v - ref)

print(f"{'idx':>3} {'#masks':>7} {'medArea':>9} {'cy(0=top)':>10} "
      f"{'bboxH':>7} {'bboxW':>7} {'fill%':>7} {'cy - cy(cls5)':>14}")
print('-' * 76)
for k in range(1, 6):
    s = stats[k]
    if s['n'] == 0: print(f"{k:>3} {'absent':>7}"); continue
    fill = np.median(s['area']) / (np.median(s['h']) * np.median(s['w'])) * 100
    rel = f"{np.median(s['rel']):+.3f}" if s['rel'] else "n/a"
    print(f"{k:>3} {s['n']:>7} {np.median(s['area']):>9.0f} "
          f"{np.median(s['cy']):>10.3f} {np.median(s['h']):>7.0f} "
          f"{np.median(s['w']):>7.0f} {fill:>7.1f} {rel:>14}")
