import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S

T = S.CLASSES
PATCH_AROI = S.patch_px('aroi', 224)
# Calibrated on AMD-SD, where SHRM is annotated separately: fluid SRF sits at Weber
# -0.468 with 0.7% of components at or above zero, SHRM at +0.238 with 94.9% above.
# The sign of Weber contrast is therefore a near-clean fluid/SHRM separator, and it is
# the only one available on AROI, whose scheme has no SHRM class.
SHRM_CUT = 0.0


def slice_brightness(L):
    """One brightness summary per slice, area-weighted over its SRF components."""
    d = L[L.cls == 'SRF'].copy()
    d['w_px'] = d.weber * d.px
    g = d.groupby(['patient', 'slice'])
    out = g.agg(srf_px=('px', 'sum'), n_comp=('px', 'size'),
                w_sum=('w_px', 'sum'), w_max_comp=('weber', 'max')).reset_index()
    out['weber_wt'] = out.w_sum / out.srf_px
    big = d.loc[g.px.idxmax().values, ['patient', 'slice', 'weber', 'px']].rename(
        columns={'weber': 'weber_largest', 'px': 'px_largest'})
    out = out.merge(big, on=['patient', 'slice'], how='left')
    out['shrm_like'] = out.weber_wt >= SHRM_CUT
    return out.drop(columns='w_sum')


def dark_only(man, Z, pos, C):
    """Test 2 and the shift model refitted on dark (fluid-like) AROI SRF only.

    AMD-SD's SRF label already excludes SHRM — it is annotated as its own class — so
    only the AROI side needs filtering. If the bright slices were driving SRF's
    anomalies, three things follow: the shift shrinks, the interaction goes, and the
    shift model fits dark SRF as well as it fits PED.
    """
    from aroi_transfer import (fit_interaction, shift_model_recall,
                               amdsd_positive_scores, oof_and_test_scores, PATCH)

    amd = amdsd_positive_scores(man)
    amd = amd[amd.cls == 'SRF']
    dark = pos[~pos.shrm_like]
    aroi = pd.DataFrame(dict(dataset='AROI', cls='SRF', patient=dark.patient.values,
                             area_patches=dark.area_SRF.values / PATCH_AROI,
                             score=dark.p_SRF.values))
    sc = pd.concat([amd, aroi], ignore_index=True)

    print('\n=== prediction 1 and 2: logit(score) ~ log(area) * dataset, dark SRF only ===')
    f = fit_interaction(sc)
    allf = pd.read_csv(S.ROOT / 'results/aroi_score_size_model.csv')
    old = allf[allf.cls == 'SRF'].set_index('term')
    print(f"{'term':<18}{'all SRF':>10}{'p':>8}{'dark only':>12}{'p':>8}")
    for t_ in ('log_area', 'dataset_AROI', 'log_area_x_AROI'):
        n_ = f[f.term == t_].iloc[0]
        print(f'{t_:<18}{old.loc[t_, "estimate"]:>10.3f}{old.loc[t_, "p"]:>8.3f}'
              f'{n_.estimate:>12.3f}{n_.p:>8.3f}')
    shift = float(f[f.term == 'dataset_AROI'].estimate.iloc[0])

    print('\n=== prediction 3: shift model against dark SRF ===')
    edges = sorted(set(C[C.cls == 'SRF'].lo_edge) | set(C[C.cls == 'SRF'].hi_edge))
    scores = oof_and_test_scores(man)
    P = shift_model_recall(man, scores, {'SRF': dict(zip(T, S.thresholds('last4_224')))['SRF'],
                                         'IRF': 0.5, 'PED': 0.5},
                           {'SRF': shift, 'IRF': 0.0, 'PED': 0.0},
                           {'SRF': edges, 'IRF': edges, 'PED': edges})
    P = P[P.cls == 'SRF'].set_index('lo_edge')
    dark['u'] = dark.area_SRF / PATCH_AROI
    print(f"{'bin':>14}{'n':>6}{'predicted':>11}{'observed':>10}{'residual':>10}")
    res = []
    for b in range(len(edges) - 1):
        m = (dark.u >= edges[b]) & (dark.u < edges[b + 1])
        if not m.sum():
            continue
        obs = dark[m].hit.mean(); pr = float(P.loc[edges[b], 'predicted'])
        res.append(obs - pr)
        print(f"{f'{edges[b]:.2f}-{edges[b+1]:.2f}':>14}{int(m.sum()):>6}"
              f"{pr:>11.3f}{obs:>10.3f}{obs - pr:>+10.3f}")
    print(f'\n  MAE dark-only SRF {np.mean(np.abs(res)):.3f}   '
          f'(all SRF 0.173, PED 0.072, IRF 0.107)')
    return f, np.mean(np.abs(res))


def main():
    ap = argparse.ArgumentParser(
        description='Does AROI SRF contain SHRM, and does that explain where the '
                    'shift model fails? Needs aroi_lesion_contrast.csv from '
                    'scripts/explore/aroi_contrast.py, which must run where the '
                    'AROI masks are.')
    ap.add_argument('--contrast', default=str(S.ROOT / 'results/aroi_lesion_contrast.csv'))
    ap.add_argument('--aroi', default=str(S.ROOT / 'results/aroi_zeroshot.csv'))
    ap.add_argument('--shift', default=str(S.ROOT / 'results/aroi_shift_model_recall.csv'))
    a = ap.parse_args()

    for p in (a.contrast, a.aroi, a.shift):
        if not Path(p).exists():
            raise SystemExit(f'missing {p}')
    L = pd.read_csv(a.contrast)
    Z = pd.read_csv(a.aroi)
    C = pd.read_csv(a.shift)

    srf = L[L.cls == 'SRF']
    print(f'=== AROI SRF components: {len(srf)} ===')
    print(f'  median Weber {srf.weber.median():+.3f}   '
          f'at or above zero: {100 * (srf.weber >= SHRM_CUT).mean():.1f}% '
          f'(AMD-SD SRF: 0.7%, AMD-SD SHRM: 94.9%)')
    print(f'  median px: dark {srf[srf.weber < 0].px.median():.0f}, '
          f'bright {srf[srf.weber >= 0].px.median():.0f}')

    B = slice_brightness(L)
    thr = dict(zip(T, S.thresholds('last4_224')))['SRF']
    Z = Z.merge(B, on=['patient', 'slice'], how='left')
    pos = Z[(Z.label_SRF == 1) & Z.weber_wt.notna()].copy()
    # recomputed, not carried through the merge: a left join leaves NaN in the
    # unmatched rows, which demotes the boolean column to object dtype, and `~` on
    # object dtype negates bitwise (~True == -2) instead of logically
    pos['shrm_like'] = (pos.weber_wt >= SHRM_CUT).astype(bool)
    pos['hit'] = (pos.p_SRF >= thr).astype(int)
    pos['u'] = pos.area_SRF / PATCH_AROI
    print(f'\n  {len(pos)} SRF-positive slices with contrast, '
          f'{100 * pos.shrm_like.mean():.1f}% SHRM-like by area-weighted Weber')

    edges = sorted(set(C[C.cls == 'SRF'].lo_edge) | set(C[C.cls == 'SRF'].hi_edge))
    pred = C[C.cls == 'SRF'].set_index('lo_edge').predicted
    print('\n=== observed recall at the AMD-SD threshold, split by brightness ===')
    print(f"{'bin':>14}{'pred':>7}{'dark n':>8}{'dark':>7}{'resid':>8}"
          f"{'brt n':>7}{'bright':>8}{'resid':>8}")
    rows = []
    for b in range(len(edges) - 1):
        m = (pos.u >= edges[b]) & (pos.u < edges[b + 1])
        d, br = pos[m & ~pos.shrm_like], pos[m & pos.shrm_like]
        pr = float(pred.get(edges[b], np.nan))
        r = dict(lo_edge=edges[b], hi_edge=edges[b + 1], predicted=pr,
                 n_dark=len(d), n_bright=len(br),
                 recall_dark=d.hit.mean() if len(d) else np.nan,
                 recall_bright=br.hit.mean() if len(br) else np.nan)
        r['resid_dark'] = r['recall_dark'] - pr
        r['resid_bright'] = r['recall_bright'] - pr
        rows.append(r)
        f = lambda v: '    —' if not np.isfinite(v) else f'{v:>7.3f}'
        print(f"{f'{edges[b]:.2f}-{edges[b+1]:.2f}':>14}{pr:>7.3f}{len(d):>8}"
              f"{f(r['recall_dark'])}{f(r['resid_dark'])}{len(br):>7}"
              f"{f(r['recall_bright'])}{f(r['resid_bright'])}")
    R = pd.DataFrame(rows)
    R.to_csv(S.ROOT / 'results/aroi_shrm_residuals.csv', index=False)

    ok = R.dropna(subset=['resid_dark', 'resid_bright'])
    if len(ok):
        print(f'\nmean residual, bins with both groups ({len(ok)}): '
              f'dark {ok.resid_dark.mean():+.3f}   bright {ok.resid_bright.mean():+.3f}')
        print('The hypothesis predicts the shortfall sits in the bright group, i.e. a '
              'bright residual much more negative than the dark one.')
    dark_only(S.manifest(), Z, pos, C)
    print(f'\n-> results/aroi_shrm_residuals.csv')


if __name__ == '__main__':
    main()
