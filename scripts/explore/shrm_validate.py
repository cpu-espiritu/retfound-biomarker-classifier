import sys; sys.path.insert(0,'analysis')
import numpy as np, pandas as pd
import style as S
from sklearn.metrics import roc_auc_score
from aroi_transfer import oof_and_test_scores
import aroi_shrm_proxy as M

CUT = 0.0
L = pd.read_csv('results/amdsd_shrm_contrast.csv')
d = L[L.cls.isin(['SRF','SHRM'])]
y = (d.cls=='SHRM').astype(int).values; w = d.weber.values

print('=== 1. Weber separates AMD-SD SRF from SHRM, per component ===')
pred = (w >= CUT).astype(int)
tp,fn = int(((pred==1)&(y==1)).sum()), int(((pred==0)&(y==1)).sum())
fp,tn = int(((pred==1)&(y==0)).sum()), int(((pred==0)&(y==0)).sum())
print(f'  n = {len(d)}  ({y.sum()} SHRM, {(1-y).sum()} SRF)   cut at Weber >= {CUT}')
print(f'                 pred SHRM   pred SRF')
print(f'  true SHRM   {tp:>10}{fn:>11}    sensitivity {tp/(tp+fn):.3f}')
print(f'  true SRF    {fp:>10}{tn:>11}    specificity {tn/(tn+fp):.3f}')
print(f'  accuracy {(tp+tn)/len(d):.3f}   AUROC {roc_auc_score(y,w):.4f}')
best = S.youden(y, (w - w.min())/(w.max()-w.min()))
print(f'  Youden-optimal cut on Weber: {best*(w.max()-w.min())+w.min():+.3f} '
      f'(zero is {"near-" if abs(best*(w.max()-w.min())+w.min())<0.1 else ""}optimal)')

print('\n=== 2. OOF SRF-head score: AMD-SD SHRM-only vs AROI bright ===')
man = S.manifest(); sc = oof_and_test_scores(man)
man['p_SRF'] = man.file.map(sc['SRF']).astype(float)
thr = dict(zip(S.CLASSES, S.thresholds('last4_224')))['SRF']
shrm_only = man[(man.area_SHRM>0) & (man.area_SRF==0)]
srf_pos   = man[man.area_SRF>0]
neither   = man[(man.area_SHRM==0) & (man.area_SRF==0)]

Z = pd.read_csv('results/aroi_zeroshot.csv')
B = M.slice_brightness(pd.read_csv('results/aroi_lesion_contrast.csv'))
Z = Z.merge(B, on=['patient','slice'], how='left')
pos = Z[(Z.label_SRF==1) & Z.weber_wt.notna()].copy()
pos['bright'] = (pos.weber_wt >= CUT).astype(bool)

print(f"{'group':<34}{'n':>6}{'pts':>5}{'mean':>8}{'median':>8}{'p90':>8}{'>=thr':>8}")
rows = [('AMD-SD SHRM only (SRF absent)', shrm_only.p_SRF, shrm_only.group),
        ('AMD-SD SRF present',            srf_pos.p_SRF,   srf_pos.group),
        ('AMD-SD neither',                neither.p_SRF,   neither.group),
        ('AROI bright SRF slices',        pos[pos.bright].p_SRF,  pos[pos.bright].patient),
        ('AROI dark SRF slices',          pos[~pos.bright].p_SRF, pos[~pos.bright].patient)]
for name, v, g in rows:
    print(f'{name:<34}{len(v):>6}{g.nunique():>5}{v.mean():>8.3f}{v.median():>8.3f}'
          f'{v.quantile(.9):>8.3f}{100*(v>=thr).mean():>7.1f}%')
print(f'\n  AMD-SD threshold {thr:.2f}')
