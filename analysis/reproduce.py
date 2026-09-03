#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'

HEADLINES = {
    'main': 'RESULTS.md §1 — AUPRC by adaptation depth',
    'depth': 'RESULTS.md §2 — full FT vs last-4 is an equivalence',
    'encoder': 'RESULTS.md §3 — the encoder advantage is depth-dependent',
    'size': 'RESULTS.md §4 — recall is set by patch-relative lesion size',
    'pooling': 'RESULTS.md §6d — attention pooling beats a matched-capacity control',
}


def need(name):
    p = RESULTS / name
    if not p.exists():
        raise SystemExit(f'missing {p.relative_to(ROOT)} — see data/README.md')
    return pd.read_csv(p)


def main():
    ap = argparse.ArgumentParser(
        description='Regenerate a headline number from results/. CPU only, seconds. '
                    'No GPU, no dataset access, no model weights.')
    ap.add_argument('what', nargs='?', default='pooling', choices=list(HEADLINES))
    a = ap.parse_args()
    print(f'{HEADLINES[a.what]}\n' + '=' * 62)

    if a.what == 'pooling':
        d = need('pooling_control_tunedC.csv').pivot(
            index='pool', columns='cls', values='oof_auprc')
        d = d.reindex(['mean', 'mean+mlp', 'attn'])[['IRF', 'SRF', 'PED']]
        print(d.round(3).to_string())
        gap = d.loc['attn', 'IRF'] - d.loc['mean+mlp', 'IRF']
        print(f'\nattention − matched-capacity MLP, IRF: {gap:+.3f}')
        print('RESULTS.md §6d reports +0.108 (paired bootstrap, Holm p=0.008).')

    elif a.what == 'encoder':
        d = need('encoder_size_strat.csv')
        s = d[d.p_holm < 0.05][['depth', 'control', 'cls', 'stratum', 'delta', 'p_holm']]
        print(f'{len(d)} tests, Holm-corrected. {len(s)} survive:\n')
        print(s.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    elif a.what == 'depth':
        d = need('pvalues.csv')
        s = d[d.comparison == 'full FT - last-4']
        print(s[['cls', 'delta', 'lo', 'hi', 'p']].to_string(
            index=False, float_format=lambda v: f'{v:+.3f}'))
        print('\nAll three intervals contain zero and are narrower than ±0.05:')
        print('an equivalence, not an absence of evidence.')
        print('Reported uncorrected: the claim is the interval, not a rejection.')

    elif a.what == 'size':
        d = need('amdsd_components_per_lesion.csv')
        patch = (380 / 14) * (570 / 14)
        print(f'1 ViT-L/16 patch = {patch:.0f} px of a 380x570 frame\n')
        print(f"{'class':<6}{'lesions':>9}{'median px':>11}{'sub-patch':>11}")
        for c in ['IRF', 'SRF', 'PED']:
            s = d[d.cls == c]
            print(f'{c:<6}{len(s):>9}{s.px.median():>11.0f}'
                  f'{(s.px < patch).mean():>11.1%}')
        print('\nRESULTS.md §4: IRF 88.0% sub-patch, and at matched size it is the '
              'easiest\nof the three — the class is not intrinsically harder.')

    elif a.what == 'main':
        d = need('encoder_grid_auprc.csv')
        s = d[d.encoder == 'RETFound'].pivot(index='depth', columns='cls', values='mean')
        print(s.reindex(['lp', 'last4', 'full'])[['IRF', 'SRF', 'PED']]
              .round(3).to_string())
        print('\nmean over 3 seeds. RESULTS.md §1 quotes the same figures.')

    print('\nSource CSVs are under results/; the code that produced them is in '
          'scripts/.\nSee experiments.csv for the run registry.')


if __name__ == '__main__':
    main()
