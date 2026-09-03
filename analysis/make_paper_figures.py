#!/usr/bin/env python3
"""Build the paper figure set.

Each figure is one script in analysis/figures/, reading only CSVs from results/.
Regenerate those CSVs with `python analysis/derive.py` on a machine that has the
predictions, features and masks; the figures then rebuild anywhere.
"""
import argparse
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'figures'))

FIGURES = [
    ('1', 'fig1_lesion_scale', 'example B-scans with masks, lesion size in patches'),
    ('2', 'fig2_arms', 'adaptation arms against AUPRC, including 448'),
    ('3', 'fig3_effect', 'recall against lesion size, and the score behind it'),
    ('4', 'fig4_thresholds', 'threshold behaviour on sub-patch lesions'),
    ('5', 'fig5_pooling', 'pooling x depth: interaction, control, cost'),
    ('6', 'fig6_forest', 'depth, resolution, encoder, patch crossing'),
    ('7', 'fig7_transfer', 'what transfers to AROI and what does not'),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only', nargs='*', help='figure numbers, e.g. --only 2 5')
    a = ap.parse_args()
    want = set(a.only or [n for n, _, _ in FIGURES])
    for num, mod, blurb in FIGURES:
        if num not in want:
            continue
        print(f'figure {num} — {blurb}')
        m = importlib.import_module(mod)
        importlib.reload(m)
        m.main()


if __name__ == '__main__':
    main()
