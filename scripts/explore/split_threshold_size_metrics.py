import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'analysis'))
import style as S
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score as AP

T = S.CLASSES
man = S.manifest()
PATCH = S.patch_px('amdsd', 224)

# ---------------------------------------------------------------- 1. leakage
# Frozen RETFound features + logistic probe. Only the fold assignment changes:
# patient-level (the repo's splits) vs scans shuffled at random, which lets the
# same eye appear on both sides. Same features, same classifier, same C.
X = S.features('RETFound')
pool = (man.split == 'pool').values
Xp, mp = X[pool], man[pool].reset_index(drop=True)
rng = np.random.default_rng(0)
rand_fold = rng.integers(0, 5, len(mp))          # leaky: ignores patient entirely

rows = []
for name, folds in [('patient-level (correct)', mp.fold.values),
                    ('random scans (leaky)', rand_fold)]:
    for c in T:
        y = mp[f'label_{c}'].values
        oof = np.full(len(mp), np.nan)
        for k in range(5):
            tr, va = folds != k, folds == k
            sc = StandardScaler().fit(Xp[tr])
            lr = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(Xp[tr]), y[tr])
            oof[va] = lr.predict_proba(sc.transform(Xp[va]))[:, 1]
        rows.append((name, c, AP(y, oof)))
L = pd.DataFrame(rows, columns=['split', 'cls', 'auprc'])
piv = L.pivot(index='cls', columns='split', values='auprc').loc[T]
piv['inflation'] = piv['random scans (leaky)'] - piv['patient-level (correct)']
piv.rename(columns={'patient-level (correct)': 'patient_level',
                    'random scans (leaky)': 'random_scans'}).reset_index().to_csv(
    S.ROOT / 'results/leakage_inflation.csv', index=False)
print('=== 1. AUPRC, leaky vs patient-level split (frozen probe, pool split, C=0.01)')
print(piv.round(4).to_string(), '\n')

# ---------------------------------------------------------------- 2 & 3
P, Y, G = S.arm_preds('last4_224')
te = S.test_rows()
thr = S.thresholds('last4_224')                  # Youden, chosen on validation folds
rows = []
for i, c in enumerate(T):
    y, p = Y[:, i], P[:, i]
    u = te[f'area_{c}'].values / PATCH
    pos = y == 1
    sens_half = (p[pos] >= 0.5).mean()
    sens_youd = (p[pos] >= thr[i]).mean()
    spec_half = (p[~pos] < 0.5).mean()
    spec_youd = (p[~pos] < thr[i]).mean()
    small, large = pos & (u < 1), pos & (u >= 1)
    rows.append((c, thr[i], sens_half, sens_youd, spec_half, spec_youd,
                 (p[small] >= thr[i]).mean(), small.sum(),
                 (p[large] >= thr[i]).mean(), large.sum()))
D = pd.DataFrame(rows, columns=['cls', 'youden', 'sens@0.5', 'sens@youden',
                                'spec@0.5', 'spec@youden', 'recall_sub', 'n_sub',
                                'recall_supra', 'n_supra'])
D['sens_gain'] = D['sens@youden'] - D['sens@0.5']
D['spec_cost'] = D['spec@0.5'] - D['spec@youden']
D['recall_gap'] = D['recall_supra'] - D['recall_sub']
D.to_csv(S.ROOT / 'results/youden_vs_default_and_size_gap.csv', index=False)
print('=== 2. Sensitivity at the default 0.5 vs at Youden (last-4, 224, test set)')
print(D[['cls', 'youden', 'sens@0.5', 'sens@youden', 'sens_gain',
         'spec@0.5', 'spec@youden', 'spec_cost']].round(3).to_string(index=False), '\n')
print('=== 3. Recall, sub-patch (<1 patch) vs >=1 patch, at the Youden threshold')
print(D[['cls', 'n_sub', 'recall_sub', 'n_supra', 'recall_supra',
         'recall_gap']].round(3).to_string(index=False))
