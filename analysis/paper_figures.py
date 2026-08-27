#!/usr/bin/env python3
import argparse
import glob
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S
from style import CLASSES as T

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
PATCH_AMDSD = S.patch_px('amdsd', 224)
PATCH_AROI = S.patch_px('aroi', 224)
EDGES = S.PATCH_EDGES


def youden(y, p, ts=np.linspace(0.02, 0.98, 97)):
    a_, b_ = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    return ts[int(np.argmax([((p >= t) & (y == 1)).sum() / a_ +
                             ((p < t) & (y == 0)).sum() / b_ - 1 for t in ts]))]


def amdsd_predictions(df):
    """Out-of-fold for pool scans, fold ensemble for test scans."""
    D = [np.load(f, allow_pickle=True)
         for f in sorted(glob.glob(str(S.DATA / 'amdsd_preds/preds_last4_224_f*_s0.npz')))]
    if not D:
        raise SystemExit('no predictions found; see data/README.md')
    pred, thr = {}, {}
    for i, c in enumerate(T):
        vp = np.concatenate([d['val_p'] for d in D])[:, i]
        vy = np.concatenate([d['val_y'] for d in D])[:, i]
        thr[c] = youden(vy, vp)
        m = dict(zip(df[df.split == 'test'].file,
                     np.mean([x['test_p'] for x in D], axis=0)[:, i]))
        for k, x in enumerate(D):
            m.update(zip(df[(df.split == 'pool') & (df.fold == k)].file, x['val_p'][:, i]))
        pred[c] = m
    return pred, thr


def recall_bins(u, hit, g, n_boot=2000):
    rng = np.random.default_rng(0)
    out = []
    for b in range(len(EDGES) - 1):
        m = (u >= EDGES[b]) & (u < EDGES[b + 1])
        if m.sum() < S.MIN_SCANS or len(np.unique(g[m])) < S.MIN_PATIENTS:
            continue
        h, gg = hit[m], g[m]
        ug = np.unique(gg)
        gi = {v: np.flatnonzero(gg == v) for v in ug}
        bs = [h[np.concatenate([gi[v] for v in rng.choice(ug, len(ug))])].mean()
              for _ in range(n_boot)]
        out.append(dict(x=np.sqrt(EDGES[b] * EDGES[b + 1]), r=h.mean(),
                        lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    return pd.DataFrame(out)


def figure1(df, pred, thr):
    Z = pd.read_csv(RESULTS / 'aroi_zeroshot.csv')
    thr_aroi = {c: youden(Z[f'label_{c}'].values, Z[f'p_{c}'].values) for c in T}

    fig, axes = plt.subplots(2, 3, figsize=(S.WIDTH, 5.4))
    panels = [
        ('AMD-SD', df, PATCH_AMDSD, 'group',
         {c: df.file.map(pred[c]).values for c in T}, thr),
        ('AROI, transferred thresholds', Z, PATCH_AROI, 'patient',
         {c: Z[f'p_{c}'].values for c in T}, thr),
        ('AROI, recalibrated', Z, PATCH_AROI, 'patient',
         {c: Z[f'p_{c}'].values for c in T}, thr_aroi),
    ]
    for ax, (name, D_, patch, gcol, probs, th) in zip(axes[0], panels):
        for c in T:
            pos = (D_[f'label_{c}'] == 1).values
            u = (D_.loc[pos, f'area_{c}'] / patch).values
            hit = (probs[c][pos] >= th[c]).astype(int)
            r = recall_bins(u, hit, D_.loc[pos, gcol].values)
            if not len(r):
                continue
            ax.errorbar(r.x, r.r, yerr=[r.r - r.lo, r.hi - r.r], marker='o', ms=3.5,
                        lw=1.5, capsize=2, color=S.CLASS_COLOR[c],
                        label=f'{c} (n={int(pos.sum())})')
        ax.axvline(1.0, color='k', ls='--', lw=1.1)
        ax.set_xscale('log'); ax.set_ylim(0, 1.0)
        ax.set_title(name, loc='left', fontsize=8.5)
        ax.set_xlabel('lesion area (patches)', fontsize=8)
    axes[0][0].set_ylabel('recall')
    axes[0][0].legend(frameon=False, fontsize=6.5, loc='lower right')
    axes[0][0].annotate('1 patch', xy=(1.0, 0.02), xytext=(3, 0),
                        textcoords='offset points', fontsize=6.5)

    te = df[df.split == 'test'].reset_index(drop=True)
    D = [np.load(f, allow_pickle=True)
         for f in sorted(glob.glob(str(S.DATA / 'amdsd_preds/preds_last4_224_f*_s0.npz')))]
    Pt, Yt = np.mean([d['test_p'] for d in D], axis=0), D[0]['test_y']
    rng = np.random.default_rng(0)
    for ax, (i, c) in zip(axes[1], enumerate(T)):
        y, p = Yt[:, i], Pt[:, i]
        u = te[f'area_{c}'].values / PATCH_AMDSD
        ax.scatter(0.012 * np.exp(rng.normal(0, .12, (y == 0).sum())), p[y == 0],
                   s=4, alpha=.22, color='#999999', label='negative')
        ax.scatter(u[y == 1], p[y == 1], s=5, alpha=.42,
                   color=S.CLASS_COLOR[c], label='positive')
        xs, ms = [], []
        for b in range(len(EDGES) - 1):
            m = (y == 1) & (u >= EDGES[b]) & (u < EDGES[b + 1])
            if m.sum() >= 10:
                xs.append(np.sqrt(EDGES[b] * EDGES[b + 1])); ms.append(np.median(p[m]))
        ax.plot(xs, ms, '-o', ms=3.5, lw=1.7, color='k', zorder=5, label='median')
        ax.axhline(thr[c], ls='--', lw=1.1, color='#b02a2a')
        ax.text(0.985, thr[c], f' {thr[c]:.2f} ', transform=ax.get_yaxis_transform(),
                ha='right', va='bottom', fontsize=6.5, color='#b02a2a')
        rho, _ = spearmanr(u[y == 1], p[y == 1])
        ax.set_xscale('log'); ax.set_ylim(0, 1.02)
        ax.set_xlabel('lesion area (patches)', fontsize=8)
        ax.set_title(f'{c}   $\\rho$={rho:.2f}', color=S.CLASS_COLOR[c], fontsize=8.5)
    axes[1][0].set_ylabel('model score')
    axes[1][0].legend(frameon=False, fontsize=6, loc='center left', markerscale=1.8)

    fig.text(0.005, 0.985, 'a', fontsize=11, fontweight='bold', va='top')
    fig.text(0.005, 0.505, 'b', fontsize=11, fontweight='bold', va='top')
    fig.tight_layout(rect=[0.02, 0, 1, 1])
    S.save(fig, 'figure1_effect', 'paper')


def main():
    ap = argparse.ArgumentParser(description='Compose the paper figure set.')
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    df = S.manifest()
    pred, thr = amdsd_predictions(df)
    if not a.only or 'figure1' in a.only:
        figure1(df, pred, thr)


if __name__ == '__main__':
    main()
