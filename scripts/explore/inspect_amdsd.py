#!/usr/bin/env python3
import argparse, os, random
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

IMG_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def walk_tree(root, max_depth=3):
    root = Path(root)
    print("=" * 70); print("DIRECTORY TREE"); print("=" * 70)
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        depth = len(d.relative_to(root).parts)
        if depth > max_depth:
            dirnames[:] = []; continue
        dirnames.sort()
        imgs = sorted(f for f in filenames if Path(f).suffix.lower() in IMG_EXT)
        other = sorted(f for f in filenames if Path(f).suffix.lower() not in IMG_EXT)
        print(f"{'  ' * depth}{d.name if depth else root}/  "
              f"[{len(dirnames)} dirs | {len(imgs)} imgs | {len(other)} other]")
        if other: print(f"{'  ' * depth}   non-image: {other[:6]}")
        if imgs:  print(f"{'  ' * depth}   e.g. {imgs[:3]}")

def collect(root):
    return sorted(p for p in Path(root).rglob('*') if p.suffix.lower() in IMG_EXT)

def report(paths, label, n_sample, palette=False):
    print("\n" + "=" * 70); print(f"{label}  (n={len(paths)})"); print("=" * 70)
    if not paths: return
    random.seed(0)
    sample = random.sample(paths, min(n_sample, len(paths)))
    modes, sizes = Counter(), Counter()
    colour_px, colour_files = Counter(), Counter()
    for p in sample:
        im = Image.open(p)
        modes[im.mode] += 1; sizes[im.size] += 1
        if palette:
            a = np.array(im.convert('RGB')).reshape(-1, 3)
            cols, cnts = np.unique(a, axis=0, return_counts=True)
            for c, n in zip(cols, cnts):
                t = tuple(int(x) for x in c)
                colour_px[t] += int(n); colour_files[t] += 1
    print(f"modes: {dict(modes)}")
    print("sizes (top 5):")
    for s, n in sizes.most_common(5): print(f"   {s[0]}x{s[1]}  ({n}/{len(sample)})")
    print(f"paths (relative-ish):")
    for p in sample[:5]: print(f"   {'/'.join(p.parts[-3:])}")
    if palette:
        print(f"\nUNIQUE COLOURS across {len(sample)} masks "
              f"(colour | total px | # masks containing it):")
        for c, n in colour_px.most_common(20):
            print(f"   {str(c):18s} {n:12,d}   {colour_files[c]}/{len(sample)}")

def tokens(paths, n=400):
    print("\n" + "=" * 70); print("FILENAME TOKENS (candidate patient/eye/scan IDs)"); print("=" * 70)
    import re
    pats = Counter()
    for p in paths[:n]:
        pats[re.sub(r'\d+', '#', p.stem)] += 1
    for pat, n_ in pats.most_common(10):
        print(f"   {pat:40s}  x{n_}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--sample', type=int, default=200)
    a = ap.parse_args()

    walk_tree(a.root)
    allp = collect(a.root)
    print(f"\nTOTAL images found: {len(allp)}")

    masky = [p for p in allp if any(k in str(p).lower()
             for k in ('mask', 'label', 'gt', 'seg', 'annot'))]
    imgy = [p for p in allp if p not in set(masky)]

    report(imgy,  "LIKELY B-SCANS", a.sample, palette=False)
    report(masky, "LIKELY MASKS",   a.sample, palette=True)
    if not masky:
        print("\n!! No path contained mask/label/gt/seg — sampling ALL for palette:")
        report(allp, "ALL (palette probe)", 40, palette=True)
    tokens(allp)
