#!/usr/bin/env python3
"""
AMD-SD prep: group-safe splits + crop + multi-label derivation -> manifest.csv

Class indices confirmed against the AMD-SD paper palette:
  1=SRF  2=IRF  3=PED  4=SHRM  5=ISOS  0=background
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

CLASS_MAP = {1: 'SRF', 2: 'IRF', 3: 'PED', 4: 'SHRM', 5: 'ISOS'}
TARGETS = ['IRF', 'SRF', 'PED']


def retina_band(gray, margin, frac=0.30, guard=4):
    """Longest contiguous bright run = retina. Robust to edge artefacts."""
    H = gray.shape[0]
    prof = np.convolve(gray.astype(float).mean(axis=1), np.ones(9) / 9, mode='same')
    core = prof[guard:H - guard]
    lo, hi = np.percentile(core, 5), np.percentile(core, 99)
    on = core > lo + frac * (hi - lo)
    if not on.any():
        return 0, H
    best, start = (0, 0), None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        if start is not None and (not v or i == len(on) - 1):
            end = i + 1 if v else i
            if end - start > best[1] - best[0]:
                best = (start, end)
            start = None
    top = max(0, best[0] + guard - margin)
    bot = min(H, best[1] + guard + margin)
    return int(top), int(bot)


def parse_min_area(s):
    if s is None:
        return {c: 0 for c in TARGETS}
    s = s.strip()
    if '=' not in s:
        return {c: int(s) for c in TARGETS}
    d = {c: 0 for c in TARGETS}
    for kv in s.split(','):
        k, v = kv.split('=')
        k = k.strip().upper()
        if k not in d:
            sys.exit(f"unknown class in --min-area: {k}")
        d[k] = int(v)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='dir containing images/ and masks/')
    ap.add_argument('--out', required=True)
    ap.add_argument('--demographics', default=None,
                    help='CSV with patient ID and eye ID columns')
    ap.add_argument('--crop', action='store_true',
                    help='derive labels from the CROPPED mask')
    ap.add_argument('--crop-margin', type=int, default=40)
    ap.add_argument('--min-area', default='0',
                    help="0 = any-pixel. Or 'IRF=30,SRF=20,PED=50'")
    ap.add_argument('--test-frac', type=float, default=0.15)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    root = Path(a.root)
    idir, mdir = root / 'images', root / 'masks'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    min_area = parse_min_area(a.min_area)

    imgs = sorted(p for p in idir.glob('*.png'))
    if not imgs:
        sys.exit(f"no PNGs in {idir}")
    print(f"[scan] {len(imgs)} B-scans")

    rows, n_crop_changed, missing = [], 0, 0
    for n, p in enumerate(imgs):
        mp = mdir / p.name
        if not mp.exists():
            missing += 1; continue
        m = np.array(Image.open(mp).convert('L'))
        g = np.array(Image.open(p).convert('L'))
        if m.shape != g.shape:
            sys.exit(f"shape mismatch {p.name}: img {g.shape} mask {m.shape}")

        top, bot = (retina_band(g, a.crop_margin) if a.crop else (0, m.shape[0]))
        m_use = m[top:bot] if a.crop else m

        full = {c: int((m == k).sum()) for k, c in CLASS_MAP.items()}
        area = {c: int((m_use == k).sum()) for k, c in CLASS_MAP.items()}
        if a.crop and any((full[c] > 0) != (area[c] > 0) for c in TARGETS):
            n_crop_changed += 1

        eye, scan = p.stem.split('_')
        r = {'file': p.name, 'eye_id': int(eye), 'scan_idx': int(scan),
             'crop_top': top, 'crop_bot': bot}
        for c in CLASS_MAP.values():
            r[f'area_{c}'] = area[c]
        for c in TARGETS:
            T, A = min_area[c], area[c]
            r[f'label_{c}'] = int(A >= T) if T > 0 else int(A > 0)
            r[f'valid_{c}'] = int(not (T > 0 and 0 < A < T))
        rows.append(r)
        if (n + 1) % 500 == 0:
            print(f"  ...{n + 1}/{len(imgs)}")

    df = pd.DataFrame(rows)
    if missing:
        print(f"[warn] {missing} images had no matching mask")
    if a.crop:
        pct = 100 * n_crop_changed / len(df)
        print(f"[crop] labels changed by cropping: {n_crop_changed} ({pct:.2f}%)")
        if pct > 1.0:
            print("[warn] >1% — crop is too tight, raise --crop-margin")

    # ---- group key
    if a.demographics:
        dp = Path(a.demographics)
        d = (pd.read_excel(dp) if dp.suffix.lower() in ('.xlsx', '.xls')
             else pd.read_csv(dp))
        print(f"[demog] columns: {list(d.columns)}")
        cols = {c.lower().replace(' ', '').replace('_', ''): c for c in d.columns}
        pk = next((cols[k] for k in cols if 'patient' in k), d.columns[0])
        ek = next((cols[k] for k in cols if 'eye' in k and 'categ' not in k),
                  d.columns[1])
        d = d[[pk, ek]].rename(columns={pk: 'patient_id', ek: 'eye_id'})
        d['eye_id'] = d['eye_id'].astype(int)
        df = df.merge(d.drop_duplicates('eye_id'), on='eye_id', how='left')
        if df['patient_id'].isna().any():
            n = df['patient_id'].isna().sum()
            sys.exit(f"{n} scans have no patient_id — check the demographics file")
        df['group'] = df['patient_id'].astype(str)
        print(f"[group] PATIENT-level: {df['group'].nunique()} patients, "
              f"{df['eye_id'].nunique()} eyes")
    else:
        df['patient_id'] = pd.NA
        df['group'] = 'eye' + df['eye_id'].astype(str)
        print(f"[group] EYE-level: {df['group'].nunique()} eyes")
        print("[warn] no demographics — fellow eyes may split across folds. "
              "State this as a limitation.")

    # ---- stratification key (label powerset, rare strata merged)
    key = df[[f'label_{c}' for c in TARGETS]].astype(str).agg(''.join, axis=1)
    grp_key = key.groupby(df['group']).agg(lambda s: s.mode()[0])
    counts = grp_key.value_counts()
    rare = counts[counts < a.folds + 1].index
    grp_key = grp_key.replace({r: 'rare' for r in rare})
    df['strat'] = df['group'].map(grp_key)

    # ---- held-out test, by group, STRATIFIED
    n_out = max(2, round(1 / a.test_frac))
    sgkf0 = StratifiedGroupKFold(n_splits=n_out, shuffle=True, random_state=a.seed)
    _, test_i = next(sgkf0.split(df, df['strat'], df['group']))
    df['split'] = 'pool'; df.loc[df.index[test_i], 'split'] = 'test'
    df['fold'] = -1

    pool = df[df['split'] == 'pool']
    sgkf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
    for k, (_, va) in enumerate(sgkf.split(pool, pool['strat'], pool['group'])):
        df.loc[pool.index[va], 'fold'] = k

    # ---- gates
    gp, gt = set(df[df.split == 'pool'].group), set(df[df.split == 'test'].group)
    assert not (gp & gt), f"LEAK pool/test: {gp & gt}"
    for k in range(a.folds):
        tr = set(df[(df.split == 'pool') & (df.fold != k)].group)
        va = set(df[(df.split == 'pool') & (df.fold == k)].group)
        assert not (tr & va), f"LEAK fold {k}: {tr & va}"
    assert (df[df.split == 'pool'].fold >= 0).all(), "unassigned pool rows"
    print("[gate] leakage assertions passed")

    df.to_csv(out / 'manifest.csv', index=False)

    print(f"\n{'':<10}{'scans':>8}{'groups':>8}" +
          ''.join(f"{c:>8}" for c in TARGETS))
    for name, sub in [('test', df[df.split == 'test']),
                      *[(f'fold{k}', df[df.fold == k]) for k in range(a.folds)]]:
        prev = ''.join(f"{sub[f'label_{c}'].mean():>8.3f}" for c in TARGETS)
        print(f"{name:<10}{len(sub):>8}{sub.group.nunique():>8}{prev}")

    print("\nArea quantiles (px, positives only):")
    for c in TARGETS:
        v = df.loc[df[f'area_{c}'] > 0, f'area_{c}']
        q = np.percentile(v, [1, 5, 25, 50, 75, 95])
        print(f"  {c:<5} n={len(v):>5}  "
              f"p1={q[0]:.0f} p5={q[1]:.0f} p25={q[2]:.0f} "
              f"p50={q[3]:.0f} p75={q[4]:.0f} p95={q[5]:.0f}")
    print(f"\n-> {out / 'manifest.csv'}")


if __name__ == '__main__':
    main()
