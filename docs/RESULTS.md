# Wet AMD Biomarker Classification using RETFound: Stage 1 Results

## Task

Multi-label B-scan classification of three wet AMD fluid biomarkers (**IRF, SRF, PED**) as
Stage 1 of a programme predicting anti-VEGF treatment response. Currently only 3 biomarkers, with the possibility of extending to 4 with SHRM.

- **Encoder:** RETFound-MAE-OCT (`RETFound_mae_natureOCT`), ViT-L/16
- **Head:** 3 independent sigmoids, BCE loss with per-class validity masking
- **Data:** AMD-SD: 3,049 B-scans, 156 eyes, 138 patients, all wet AMD
- **Labels:** derived from pixel masks, any-pixel classifies as yes (finding 5)
- **Splits:** **patient-level**, stratified. 20 patients held out (~14% testing), 5-fold CV on 118 (23 each-fold, ~68% training, ~17% validation)
- **Metrics:** AUPRC, per-class recall, specificity; **patient-level cluster bootstrap** 95% CIs

---

## Main result: adaptation depth (test set, 5-fold cross validation, Youden thresholds)

| Arm                    | Trainable | IRF AUPRC               | SRF AUPRC               | PED AUPRC               |
| ---------------------- | --------- | ----------------------- | ----------------------- | ----------------------- |
| Linear probe, 224      | 0.003 M   | 0.683 [0.440–0.868]     | 0.972 [0.927–0.992]     | 0.889 [0.763–0.967]     |
| **Last-4 blocks, 224** | **50 M**  | **0.791 [0.577–0.898]** | **0.985 [0.959–0.996]** | **0.960 [0.906–0.987]** |
| Full FT, 224           | 303 M     | 0.785 [0.555–0.901]     | 0.984 [0.956–0.996]     | 0.952 [0.888–0.986]     |
| Full FT, 384           | 303 M     | 0.799 [0.574–0.913]     | 0.979 [0.943–0.995]     | 0.942 [0.865–0.987]     |

Class prevalence in test: IRF 0.268, SRF 0.675, PED 0.682.

---

## Findings

**1. Last-4 matches full fine-tuning at 1/6 the cost.**
Last-4 blocks is best or joint-best on all three classes and has the highest specificity
in every class (IRF 0.901, SRF 0.986, PED 0.900). Full FT does not improve anything at n=138 patients.

**2. Frozen features under-serve the hardest class**
LP to last-4 lifts IRF AUPRC by +0.11 and PED by +0.07. The information is present in the
encoder but is not linearly separable. Relevant to the critique that MAE pixel
reconstruction may under-represent small structures.

**3. Lesion size dominates detectability, fine-tuning only partly fixes it.**

| Recall by lesion area | small | medium    | large |
| --------------------- | ----- | --------- | ----- |
| IRF: linear probe     | 0.400 | 0.517     | 1.000 |
| IRF: last-4           | 0.433 | **0.810** | 1.000 |
| SRF: linear probe     | 0.392 | 0.858     | 0.973 |
| SRF: last-4           | 0.446 | 0.932     | 0.987 |
| PED: linear probe     | 0.400 | 0.827     | 0.893 |
| PED: last-4           | 0.547 | 0.853     | 0.960 |

Gains concentrate in **medium** lesions. Small-lesion recall stays ≤ 0.55 in every arm and
every class. Monotonic in all arms.

**4. Input resolution is not the bottleneck, tested at two adaptation depths.**
384 fails at linear probe (all three classes down) and at full fine-tuning (IRF +0.014,
inside CI, SRF and PED down). Small-lesion recall shows no consistent benefit.
Retinal-band cropping also decreased metrics at LP, likely from variable vertical scaling before resize.

**5. Decision thresholds are a first-order effect, not a detail.**
Optimal thresholds span **0.16 to 0.85**, none is near 0.5. F1-optimal thresholds degenerate
at high prevalence, PED specificity 0.271 -> 0.686 switching to Youden's J with the same
AUROC. Consistent with reports that F1 is the least robust metric in this setting.
Also: AMD-SD lesion areas are smooth over three orders of magnitude with no gap, so
no minimum-area threshold is justified, size-stratified recall is reported instead.

---

## Limitations

- **20 test patients.** CIs are wide (IRF AUPRC spans 0.44–0.90). Fold-to-fold SD (±0.02)
  understates true uncertainty by an order of magnitude.
- **No true negatives.** All AMD-SD eyes are wet AMD, so prevalences are within-disease.
- **Single scanner, single centre.** Cross-vendor generalisation untested; RETOUCH planned.
- **Index→class mapping** was undocumented in the distributed masks; recovered from the
  published palette and confirmed by clinical review.
- Fine-tuning arms are single-seed.

## Next

Cross-scanner transfer, eye-level aggregation, ImageNet ViT-L comparison, occlusion-based localisation check against the pixel masks, multi-seed for the LP to fine-tune gap.
