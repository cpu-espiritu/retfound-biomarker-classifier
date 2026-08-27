#!/usr/bin/env python3
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score as AP

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUT = ROOT / 'analysis' / 'output'

# ---------------------------------------------------------------- fixed orders
CLASSES = ['IRF', 'SRF', 'PED']                      # never reorder
ARMS = ['lp_224', 'last4_224', 'full_224', 'full_384']
ENCODERS = ['RETFound', 'MAE-IN1k', 'Sup-IN21k']

ARM_LABEL = {'lp_224': 'Linear probe', 'last4_224': 'Last-4 blocks',
             'full_224': 'Full FT, 224', 'full_384': 'Full FT, 384'}
ARM_TRAINABLE = {'lp_224': '0.003 M', 'last4_224': '50 M',
                 'full_224': '303 M', 'full_384': '303 M'}

# ---------------------------------------------------------------- fixed colours
# Okabe-Ito derived: colour-blind safe and distinguishable in greyscale.
CLASS_COLOR = {'IRF': '#0072B2', 'SRF': '#D55E00', 'PED': '#009E73'}
ARM_COLOR = {'lp_224': '#999999', 'last4_224': '#0072B2',
             'full_224': '#D55E00', 'full_384': '#CC79A7'}
ENCODER_COLOR = {'RETFound': '#0072B2', 'MAE-IN1k': '#999999', 'Sup-IN21k': '#E69F00'}

SIG = {'holm':    dict(marker='o', mfc='#b02a2a', mec='#b02a2a', color='#b02a2a'),
       'nominal': dict(marker='o', mfc='white',   mec='#b02a2a', color='#b02a2a'),
       'null':    dict(marker='D', mfc='#8a8a8a', mec='#8a8a8a', color='#8a8a8a')}

# ---------------------------------------------------------------- print geometry
WIDTH = 6.5                 # inches; the printed text width. Never scale a figure.

# ---- patch geometry. One ViT/16 patch covers (rows/G) x (cols/G) original px,
# where G = input_size/16. Absolute patch-unit bins are the only basis on which
# lesion size compares across datasets: quartiles mean something different in each.
FRAME = {'amdsd': (380, 570), 'aroi': (1024, 512)}      # (rows, cols)
AROI_UM_PER_PX = (1.96, 11.74)                          # axial, lateral


def patch_px(dataset, input_size=224):
    rows, cols = FRAME[dataset]
    g = input_size / 16
    return (rows / g) * (cols / g)


def patch_mm2(input_size=224):
    rows, cols = FRAME['aroi']
    g = input_size / 16
    a, l = AROI_UM_PER_PX
    return (rows / g * a / 1000) * (cols / g * l / 1000)


PATCH_EDGES = np.logspace(np.log10(0.03), np.log10(30), 8)   # multiples of one patch
MIN_SCANS, MIN_PATIENTS = 15, 5      # a bin needs both, or it carries no honest CI
EQUIV = 0.02                # pre-specified equivalence margin = fold-to-fold SD
N_BOOT = 5000
SEED = 0

mpl.rcParams.update({
    'figure.figsize': (WIDTH, 4.0),
    'figure.dpi': 110,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,     # embed as TrueType so text stays selectable/editable
    'ps.fonttype': 42,
    'font.size': 9,         # 8-10pt at final size
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.6,
    'lines.linewidth': 1.6,
})


SUBDIRS = ('report', 'exploratory', 'archive')


def save(fig, name, sub='report'):
    """Write a PDF at final printed size into figures/<sub>/."""
    if sub not in SUBDIRS:
        raise ValueError(f'sub must be one of {SUBDIRS}, got {sub!r}')
    d = OUT / sub
    d.mkdir(parents=True, exist_ok=True)
    p = d / f'{name}.pdf'
    fig.savefig(p)
    print(f'  -> {p.relative_to(ROOT)}')
    return p


# ---------------------------------------------------------------- data access
def manifest():
    return pd.read_csv(DATA / 'amdsd_splits/manifest.csv')


def test_rows():
    return manifest().query("split == 'test'").reset_index(drop=True)


def arm_preds(arm, seed=0):
    """Fold-ensembled test predictions for one arm -> (P, Y, groups)."""
    fs = sorted(glob.glob(str(DATA / f'amdsd_preds/preds_{arm}_f*_s{seed}.npz')))
    if not fs:
        raise FileNotFoundError(f'no predictions for arm {arm!r}')
    D = [np.load(f, allow_pickle=True) for f in fs]
    for d in D[1:]:
        assert np.array_equal(d['test_y'], D[0]['test_y']), f'{arm}: folds disagree'
    return (np.mean([d['test_p'] for d in D], axis=0), D[0]['test_y'], D[0]['test_g'])


def val_preds(arm, seed=0):
    fs = sorted(glob.glob(str(DATA / f'amdsd_preds/preds_{arm}_f*_s{seed}.npz')))
    D = [np.load(f, allow_pickle=True) for f in fs]
    return (np.concatenate([d['val_p'] for d in D]),
            np.concatenate([d['val_y'] for d in D]))


def features(encoder):
    fn = {'RETFound': 'features_RETFound_mae_natureOCT_224.npy',
          'MAE-IN1k': 'features_mae_in1k_224.npy',
          'Sup-IN21k': 'features_sup_in21k_224.npy'}[encoder]
    return np.load(DATA / 'amdsd_features' / fn)


# ---------------------------------------------------------------- statistics
def youden(y, p, ts=np.linspace(0.02, 0.98, 97)):
    npos, nneg = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    j = [((p >= t) & (y == 1)).sum() / npos + ((p < t) & (y == 0)).sum() / nneg - 1
         for t in ts]
    return ts[int(np.argmax(j))]


def thresholds(arm):
    """Youden thresholds per class, chosen on validation folds only."""
    vp, vy = val_preds(arm)
    return [youden(vy[:, i], vp[:, i]) for i in range(len(CLASSES))]


def replicates(groups, n=N_BOOT, seed=SEED):
    """Patient-level cluster bootstrap index sets.

    Shared across every model so comparisons are paired: within a replicate the
    same patients are scored under all arms, and patient difficulty cancels.
    """
    ug = np.unique(groups)
    gi = {u: np.nonzero(groups == u)[0] for u in ug}
    rng = np.random.default_rng(seed)
    return [np.concatenate([gi[u] for u in rng.choice(ug, len(ug))]) for _ in range(n)]


def ci(values, lo=2.5, hi=97.5):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return (np.percentile(v, lo), np.percentile(v, hi)) if len(v) else (np.nan, np.nan)


def boot_stat(fn, reps):
    """Apply fn(idx) over replicates, returning the finite values."""
    out = np.array([fn(idx) for idx in reps], dtype=float)
    return out[np.isfinite(out)]


def two_sided_p(deltas):
    d = np.asarray(deltas, float)
    d = d[np.isfinite(d)]
    B = len(d)
    return min(2 * min((1 + (d <= 0).sum()) / (B + 1),
                       (1 + (d >= 0).sum()) / (B + 1)), 1.0)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, same order as input."""
    p = np.asarray(pvals, float)
    m = len(p)
    adj, run = np.empty(m), 0.0
    for rank, j in enumerate(p.argsort()):
        run = max(run, (m - rank) * p[j])
        adj[j] = min(run, 1.0)
    return adj


def sig_state(p_raw, p_adj):
    return 'holm' if p_adj < 0.05 else ('nominal' if p_raw < 0.05 else 'null')
