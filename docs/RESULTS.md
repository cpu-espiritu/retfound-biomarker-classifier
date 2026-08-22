# Wet AMD Biomarker Classification using RETFound: Stage 1 Results

## Task

Multi-label B-scan classification of three wet AMD fluid biomarkers (**IRF, SRF, PED**) as
Stage 1 of a programme predicting anti-VEGF treatment response.

- **Encoder:** RETFound-MAE-OCT (`RETFound_mae_natureOCT`), ViT-L/16
- **Head:** 3 independent sigmoids, BCE loss with per-class validity masking
- **Data:** AMD-SD: 3,049 B-scans, 156 eyes, 138 patients, all wet AMD
- **External:** AROI: 1,136 annotated B-scans, 24 patients (evaluation only, never trained on)
- **Labels:** derived from pixel masks, any-pixel classifies as yes
- **Splits:** **patient-level**, stratified. 20 patients held out (~14% test), 5-fold CV on 118
- **Metrics:** AUPRC, per-class recall, specificity; **patient-level cluster bootstrap** 95% CIs
- **Seeds:** 3 for every RETFound arm and every control at linear probe

---

## 1. Main result: adaptation depth

Test set, 5-fold ensembled, Youden thresholds, mean over 3 seeds.

| Arm                    | Trainable | IRF AUPRC               | SRF AUPRC               | PED AUPRC               |
| ---------------------- | --------- | ----------------------- | ----------------------- | ----------------------- |
| Linear probe, 224      | 0.003 M   | 0.678 [0.440–0.868]     | 0.972 [0.927–0.992]     | 0.891 [0.763–0.967]     |
| **Last-4 blocks, 224** | **50 M**  | **0.784 [0.577–0.898]** | **0.984 [0.959–0.996]** | **0.964 [0.906–0.987]** |
| Full FT, 224           | 303 M     | 0.774 [0.555–0.901]     | 0.985 [0.956–0.996]     | 0.955 [0.888–0.986]     |
| Full FT, 384           | 303 M     | 0.799 [0.574–0.913]     | 0.979 [0.943–0.995]     | 0.942 [0.865–0.987]     |
| Last-4, 448            | 50 M      | 0.806                   | 0.977                   | 0.914                   |

Class prevalence in test: IRF 0.268, SRF 0.675, PED 0.682.

**Seed-to-seed SD is negligible: ≤ 0.012 across all 21 measured cells** (max: RETFound
last-4 IRF, 0.012). Every single-seed number previously reported was reliable.

448 raises IRF AUPRC to 0.806 but costs PED (0.964 → 0.914) and recall across the board
(IRF 0.763 → 0.661). It is not an improvement overall, and §6a shows it does not move the
lesions the patch argument predicts it should.

Operating-point metrics at the Youden threshold, mean over available seeds:

| Arm | IRF rec / spec | SRF rec / spec | PED rec / spec |
| --- | --- | --- | --- |
| Linear probe, 224 | 0.605 / 0.920 | 0.780 / 0.937 | 0.720 / 0.745 |
| **Last-4 blocks, 224** | 0.763 / **0.871** | 0.816 / **0.984** | 0.810 / **0.898** |
| Full FT, 224 | 0.734 / 0.841 | 0.870 / 0.984 | 0.786 / 0.869 |
| Full FT, 384 | 0.771 / 0.863 | 0.879 / 0.965 | 0.803 / 0.793 |
| Last-4, 448 | 0.661 / 0.950 | 0.848 / 0.958 | 0.773 / 0.736 |

Last-4 is not uniformly the most specific arm — 448 is more specific on IRF (0.950), at the
cost of 10 points of recall. The earlier claim that last-4 had the highest specificity in
every class was measured on a single seed and does not hold.

---

## 2. Paired differences, not marginal intervals

Marginal CIs overlap almost completely and establish nothing. The paired patient bootstrap
(5,000 resamples, all arms scored on identical resamples) is the correct comparison.

| Comparison | Class | Δ AUPRC | 95% CI | p | Holm |
| --- | --- | --- | --- | --- | --- |
| last-4 − LP | IRF | +0.108 | [−0.032, +0.271] | 0.125 | 0.998 |
| last-4 − LP | SRF | +0.012 | [+0.000, +0.040] | 0.044 | 0.392 |
| last-4 − LP | PED | +0.071 | [+0.009, +0.165] | 0.012 | 0.136 |
| full − last-4 | IRF | −0.005 | [−0.044, +0.030] | 0.913 | 1.000 |
| full − last-4 | SRF | −0.000 | [−0.008, +0.006] | 0.962 | 1.000 |
| full − last-4 | PED | −0.008 | [−0.029, +0.007] | 0.348 | 1.000 |
| 384 − 224 | IRF | +0.014 | [−0.023, +0.034] | 0.326 | 1.000 |
| 384 − 224 | SRF | −0.006 | [−0.025, +0.004] | 0.246 | 1.000 |
| 384 − 224 | PED | −0.010 | [−0.036, +0.014] | 0.426 | 1.000 |

**Full FT vs last-4 is a demonstrated equivalence, not an absence of evidence.** The
intervals are narrow (±0.03 or better), so this is a positive claim about the null.

**The equivalence is not an artefact of the chosen learning rates.** Three-point sweep,
seed 0, AUPRC:

| arm | lr | IRF | SRF | PED |
| --- | --- | --- | --- | --- |
| full FT | 1e-5 | 0.779 | 0.984 | 0.947 |
| full FT | **2e-5** (default) | **0.785** | 0.984 | 0.952 |
| full FT | 5e-5 | 0.777 | 0.982 | 0.953 |
| last-4 | 3e-5 | 0.734 | 0.980 | 0.950 |
| last-4 | **1e-4** (default) | 0.791 | 0.985 | 0.960 |
| last-4 | **3e-4** | **0.806** | 0.984 | 0.960 |

Full FT is flat across an order of magnitude — 0.777–0.785 on IRF — so 2e-5 was not an
unlucky choice. Last-4 improves at 3e-4 (IRF 0.791 → 0.806). Comparing each arm at its own
best rate, last-4 is still equal or ahead:

| Class | last-4 @ 3e-4 | full FT @ 2e-5 | Δ | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| IRF | 0.806 | 0.785 | +0.020 | [−0.028, +0.048] | 0.437 |
| SRF | 0.984 | 0.984 | −0.000 | [−0.007, +0.009] | 0.793 |
| PED | 0.960 | 0.952 | +0.008 | [−0.006, +0.032] | 0.332 |

Tuning therefore strengthens rather than weakens the conclusion. One caveat: last-4's
optimum sits at the top of the tested range, so a higher rate might do better still.

**The headline IRF gain does not clear zero.** last-4 − LP on IRF is +0.108
[−0.032, +0.271], and does not survive correction. Direction is well-supported
(P(Δ>0) = 0.938) but 10 IRF-positive test patients cannot establish it at 95%.

---

## 3. Encoder comparison: the advantage is depth-dependent

Frozen-feature comparison (linear probe, 3 seeds, paired bootstrap):

| Comparison | IRF | SRF | PED |
| --- | --- | --- | --- |
| RETFound − MAE-IN1k | **+0.189** (Holm 0.029) | **+0.054** (Holm 0.006) | **+0.077** (Holm 0.026) |
| RETFound − Sup-IN21k | +0.017 (p 0.625) | **+0.066** (Holm 0.022) | +0.107 (p 0.024) |

At **last-4** depth, where the controls are given the same adaptation budget (3 seeds each):

| Comparison | IRF | SRF | PED |
| --- | --- | --- | --- |
| RETFound − Sup-IN21k | +0.034 (p 0.482) | +0.053 (p <0.001) | +0.075 (p 0.031) |
| RETFound − MAE-IN1k | +0.055 (p 0.203) | +0.036 (p <0.001) | +0.079 (p 0.012) |

**Size-stratified, 3 seeds per arm, Holm across 20 tests.** Six survive:

| depth | control | class | stratum | Δ recall | p_holm |
| --- | --- | --- | --- | --- | --- |
| last-4 | Sup-IN21k | SRF | **sub-patch** | **+0.277** | 0.042 |
| last-4 | Sup-IN21k | SRF | above-patch | +0.140 | 0.008 |
| last-4 | MAE-IN1k | PED | above-patch | +0.114 | 0.008 |
| last-4 | Sup-IN21k | PED | above-patch | +0.079 | 0.034 |
| LP | Sup-IN21k | SRF | above-patch | +0.136 | 0.034 |
| LP | MAE-IN1k | SRF | above-patch | +0.123 | 0.022 |

Sub-patch IRF at last-4 reaches nominal significance against both controls
(+0.152 p 0.009, +0.177 p 0.012) but does not survive correction.

**At LP the sub-patch differences run against RETFound** — Sup-IN21k is better on sub-patch
IRF (−0.177) and PED (−0.148), neither significant. The frozen-feature advantage is an
above-patch phenomenon; it is only after fine-tuning that RETFound gains on small lesions.

An earlier version of this comparison used 3 seeds for RETFound against 1 for the controls
and found no surviving sub-patch result. The symmetric comparison changes that conclusion.

**MAE-IN1k's IRF deficit collapses under fine-tuning**: AUPRC 0.406 → 0.736, a +0.330 gain,
versus RETFound's +0.106. Poor frozen features, perfectly adequate initialisation.

**On IRF there is no significant encoder advantage at last-4** against either control.
RETFound's value is concentrated in what it provides *without* fine-tuning.

**Sup-IN21k matches RETFound on IRF at both depths** (+0.017 at LP, +0.034 at last-4, both
null). Supervised ImageNet pretraining is competitive on the hardest class.

---

## 4. Lesion size determines recall, and it replicates externally

Lesion areas are expressed in **patch units** — multiples of one ViT-L/16 patch footprint,
which is 1105 px in AMD-SD (380×570 → 224) and 2675 px = 0.0616 mm² in AROI (1024×512).
Quartiles cannot be compared across datasets; patch units can.

| | AMD-SD sub-patch | AROI sub-patch |
| --- | --- | --- |
| IRF | 88.0% | 95.4% |
| SRF | 53.5% | 52.4% |
| PED | 56.4% | 56.7% |

Recall by patch-unit bin, AMD-SD (out-of-fold, all 3,049 scans):

| patches | IRF | SRF | PED |
| --- | --- | --- | --- |
| 0.03–0.08 | 0.318 | 0.188 | 0.302 |
| 0.08–0.22 | 0.549 | 0.333 | 0.357 |
| 0.22–0.58 | 0.719 | 0.655 | 0.607 |
| 0.58–1.55 | 0.878 | 0.878 | 0.762 |
| 1.55–4.17 | 0.938 | 0.971 | 0.914 |
| 4.17–11.2 | 1.000 | 0.984 | 0.992 |

AROI is the only dataset with a published pixel scale, so absolute physical sizes come
from it alone (1 px = 0.000023 mm², 1 patch = 0.0616 mm²), measured per connected component:

| class | components | median mm² | p95 mm² | median as fraction of a patch |
| --- | --- | --- | --- | --- |
| IRF | 1,399 | **0.0036** | 0.054 | **1/17** |
| SRF | 885 | 0.0470 | 0.682 | 3/4 |
| PED | 1,616 | 0.0348 | 0.540 | 4/7 |

A median IRF lesion is 0.0036 mm² — smaller than a single ViT-L/16 patch by a factor of 17.

**The three classes lie on one curve.** At matched patch-relative size IRF is the *best* of
the three in four of six bins. Its poor headline recall is a consequence of where its
lesions sit on this axis, not of the class being intrinsically harder.

**Small lesions are scored low, not scored as negative.** The recall figures are binary;
the underlying score is continuous and rises monotonically with area. Spearman ρ between
lesion area and score, among positives: IRF 0.71, SRF 0.70, PED 0.53 (all p < 1e-19).

Median score by patch-unit bin, against each class's Youden threshold:

| | 0.08–0.22 | 0.22–0.58 | 0.58–1.55 | 1.55–4.17 | threshold | negative median |
| --- | --- | --- | --- | --- | --- | --- |
| IRF | 0.11 | 0.51 | 0.83 | 1.00 | 0.16 | 0.012 |
| SRF | **0.59** | 0.88 | 0.97 | 1.00 | **0.80** | 0.032 |
| PED | **0.83** | 0.90 | 0.99 | 0.99 | **0.83** | 0.384 |

SRF lesions of 0.08–0.22 patches score a median 0.59 — far above the negative median of
0.032, but below the 0.80 cutoff, so they are recorded as missed. The encoder sees them;
the operating point discards them.

This links three findings that were previously separate: the size dependence in this
section, the high thresholds in §7, and the AROI calibration failure in §5 are the same
mechanism. It also implies a cheap intervention — a size-conditional or simply lower
threshold would recover much of the small-lesion loss at a specificity cost, testable from
the existing predictions with no retraining.

**The 50% point is ~0.2 patches, not 1.0** (IRF 0.11, SRF 0.22, PED 0.23; AROI SRF 0.20,
PED 0.21). Sub-patch lesions are detected at *reduced* rate, not missed. The transition is
graded; one patch is not a cliff.

**PED replicates across datasets to within 0.3 points** (56.4% vs 56.7% sub-patch). SRF
appears to replicate (53.5% vs 52.4%) but that comparison is not like-for-like: AROI's SRF
includes SHRM. Matching the definition, AMD-SD SRF+SHRM is 40.6% sub-patch against AROI's
52.4% — a 12-point gap. **PED is the honest replication.**

---

## 5. External validation: discrimination transfers, calibration does not

AMD-SD-trained last-4 model, 5 folds ensembled, applied zero-shot to AROI:

| | AUROC | mean p (positive) | mean p (negative) |
| --- | --- | --- | --- |
| IRF | **0.951** | 0.422 | 0.029 |
| SRF | **0.901** | 0.561 | 0.068 |
| PED | **0.967** | 0.799 | 0.139 |

Applying AMD-SD's Youden thresholds unchanged collapses recall on SRF and PED. Refitting
thresholds on AROI restores it:

| | AMD-SD sub-patch recall | AROI, AMD-SD thr | AROI, refit thr |
| --- | --- | --- | --- |
| IRF | 0.605 | 0.504 | 0.797 |
| SRF | 0.530 | **0.011** | 0.555 |
| PED | 0.517 | **0.122** | 0.558 |

(composition-standardised to AMD-SD's bin distribution)

AROI's optimal thresholds are far lower — SRF 0.80 → 0.11, PED 0.83 → 0.35.

**Deployment implication: a new scanner needs threshold recalibration on a small labelled
sample, not retraining.** That is a substantially cheaper claim than "requires fine-tuning".

---

## 6. Mechanism: what explains the size dependence

Four candidate mechanisms were tested. Three failed.

### 6a. Patch tokenisation — falsified

Retraining at 448 quarters the patch footprint (1105 → 276 px), moving 194 test scans from
sub-patch to above-patch. Recall on **exactly those scans**:

| class | n | recall 224 → 448 | Δ | 95% CI |
| --- | --- | --- | --- | --- |
| IRF | 55 | 0.800 → 0.727 | −0.073 | [−0.205, +0.041] |
| SRF | 56 | 0.679 → 0.607 | −0.071 | [−0.244, +0.067] |
| PED | 83 | 0.663 → 0.614 | −0.048 | [−0.185, +0.101] |

All three point the wrong way. Scans sub-patch at *both* resolutions were the only group to
improve (SRF +0.188 [+0.067, +0.296]), which is a resolution effect, not a patch-crossing
effect. **The patch grid is not the causal mechanism**; patch units remain a valid
comparison unit.

### 6b. Fragmentation — null

IRF fragments into 3.21 components per scan versus 1.40 for PED in AMD-SD, and the pattern
replicates in AROI (6.14 vs 1.59 per slice). With total area in the
model, component count adds 0.011 pseudo-R² and its CI crosses zero
(+0.506 [−0.210, +1.128]). `log(total_area)` is the best single predictor
(pseudo-R² 0.226) ahead of `log(max_component)` (0.188) and `log(n_components)` (0.094).
More components predicts *better* detection, because it proxies for more total fluid.

### 6c. Contrast — independent and real

Per-lesion contrast against an 8 px ring, excluding all other annotated lesion pixels:

| class | Weber contrast | corr(log area, contrast) | size-only R² | size+contrast R² |
| --- | --- | --- | --- | --- |
| IRF | −0.456 | 0.068 | 0.230 | **0.378** |
| SRF | −0.468 | −0.217 | 0.323 | **0.417** |
| PED | −0.072 | −0.543 | 0.222 | 0.227 |

Size and contrast are **near-orthogonal** for IRF (r = 0.068) and explain different things.
Contrast adds +0.148 pseudo-R² on IRF, versus fragmentation's +0.011.

**PED is nearly isointense** (Weber −0.072) and contrast adds nothing to it. PED is detected
by geometry — RPE elevation — not by intensity. That is why it behaves unlike the other two
throughout.

**IRF is not low-contrast** (−0.456, as dark as SRF). Small and faint are separate problems
that co-occur in the same class.

### 6d. Global mean pooling — supported

RETFound averages all 196 patch tokens into one vector. Swapping the pooling, frozen
encoder, head-only training, out-of-fold on 2,609 pool scans:

| pooling | IRF | SRF | PED | seeds |
| --- | --- | --- | --- | --- |
| mean, `C` tuned by inner CV | 0.759 | 0.939 | 0.907 | deterministic |
| mean (RETFound default, fixed `C`) | 0.756 | 0.944 | 0.911 | deterministic |
| max | 0.672 | 0.943 | 0.917 | deterministic |
| top-5% | 0.677 | 0.925 | 0.911 | deterministic |
| mean + matched-capacity MLP | 0.714 ± 0.014 | 0.942 ± 0.005 | 0.905 ± 0.008 | 3 |
| **attention** | **0.819 ± 0.007** | **0.953 ± 0.015** | **0.948 ± 0.007** | 3 |

Over 3 seeds: attention − mean = +0.064 / +0.010 / +0.037, attention − matched MLP =
**+0.105** / +0.012 / +0.043. Seed SD is 0.007–0.015, well inside the gaps.

**The baseline was not handicapped.** Selecting the logistic head's `C` by inner CV, on the
same grid used for the encoder probes, moves mean-pooled IRF from 0.756 to 0.759. The fixed
`C = 0.01` was already near-optimal, so attention's margin is not an artefact of an
under-fitted comparator.

Paired patient bootstrap, Holm across nine tests:

| test | IRF | SRF | PED |
| --- | --- | --- | --- |
| attn − mean | +0.056 (Holm 0.160) | +0.019 (0.096) | +0.039 (**0.008**) |
| **attn − mean+MLP** | **+0.108 (Holm 0.008)** | +0.024 (0.130) | **+0.039 (0.002)** |
| mean+MLP − mean | −0.052 (0.192) | −0.005 (0.760) | +0.000 (0.960) |

**The gain is selectivity, not capacity.** A matched-parameter MLP (66,691 vs attention's
66,625) on mean-pooled features buys nothing and makes IRF *worse*. Attention beats that
control by more than it beats plain mean.

**Dilution alone is falsified.** `max` has zero dilution and is 0.084 *worse* than mean on
IRF; naive selection is high-variance. Attention keeps the averaging but concentrates it.

This is the only mechanism that survives a proper control, and it implies a concrete fix:
`global_pool=True` discards localisation the encoder already has, recoverable with a
~66k-parameter head.

**The gain is not a small-lesion fix.** Δrecall by patch-unit bin, 3 seeds, Holm across
18 tests:

| | raw Δ recall | % of remaining gap closed |
| --- | --- | --- |
| sub-patch (n=992) | +0.120 | 24% |
| above-patch (n=2,729) | +0.077 | **55%** |

Raw Δ is larger on small lesions only because more headroom exists there. Normalised by
available headroom, attention closes more than twice as much of the gap on large lesions
(`corr(size rank, % gap closed)` = +0.86 SRF, +0.93 PED). In the smallest bin it actively
hurts (IRF −0.046, PED −0.133): too little signal for the weights to concentrate on.

Bins surviving Holm are all PED and SRF, none IRF:

| class | bin (patches) | Δ recall | p_holm |
| --- | --- | --- | --- |
| PED | 0.08–0.22 | +0.240 | 0.038 |
| PED | 0.22–0.58 | +0.189 | 0.008 |
| PED | 0.58–1.55 | +0.205 | 0.008 |
| PED | 1.55–4.17 | +0.087 | 0.014 |
| SRF | 0.22–0.58 | +0.130 | 0.014 |

Note this reverses the AUPRC ordering, where IRF showed the largest gain. AUPRC measures
ranking across all scans; this measures recall at a fixed threshold within a size band.
Both hold; the claim must state which.

**Attention pooling is therefore a better pooling operator, not a fix for the size
bottleneck.** It cannot be framed as addressing §4.

---

## 7. Decision thresholds are a first-order effect

F1-optimal versus Youden's J at last-4, identical AUROC:

| PED | F1 | Youden |
| --- | --- | --- |
| threshold | 0.22 | 0.83 |
| recall | 0.970 | 0.803 |
| **specificity** | **0.386** | **0.900** |
| AUROC | 0.910 | 0.910 |

Optimal thresholds span 0.16–0.85; none is near 0.5. Threshold choice is a modelling
decision, not a detail — and §5 shows it is also the part that fails to transfer.

---

## 8. Eye-level aggregation

Eye-level prevalence: IRF 50.0% (78/156), SRF 73.1% (114/156), PED 94.9% (148/156).
**PED is unusable as an eye-level target** — all 22 test eyes are positive.

Eight aggregators over 134 pool eyes, out-of-fold:

| method | IRF | SRF |
| --- | --- | --- |
| **decision-max** | **0.829** | **0.957** |
| noisy-OR | 0.789 | 0.927 |
| top-3 mean | 0.717 | 0.894 |
| mean | 0.673 | 0.897 |
| max | 0.648 | 0.912 |
| moment | 0.646 | 0.895 |
| ABMIL attention | 0.628 | 0.888 |
| concat | 0.626 | 0.896 |

**Decision-level beats every feature-level method by ~0.15 AUPRC on IRF.** IRF appears in a
minority of an eye's scans (median burden 0.40), so classifying each scan and taking the
maximum preserves signal that pooling 1024-d vectors destroys.

Note this is the opposite of the within-scan result (§6d), and both make sense: within a
scan a lesion spans adjacent patches, so weighted averaging helps; across an eye the signal
is in a few slices, so max over decisions helps.

---

## 9. Error structure

28 IRF false negatives on the test set at last-4. They cluster by eye, not by size:

- **Eye 143 contributes 8**, all 637–1317 px — comfortably mid-sized, with predictions
  0.03–0.15 against a 0.16 threshold. The largest missed lesion in the whole set (1317 px)
  drew the model's *most* confident rejection.
- The remaining 20 span 7 eyes and are 22–512 px, consistent with the size story.

32 false positives, also clustered: **eye 122 scans 14–21 and eye 64 scans 1–9** are
contiguous runs called positive with 0.50–0.99 confidence against empty masks. Consecutive-
run errors of that kind are the signature of annotation gaps rather than random failure.
Clinical review of both is pending.

---

## Limitations

- **20 test patients.** CIs are wide (IRF AUPRC spans 0.58–0.90 at last-4). Multi-seed
  confirms stability (SD ≤ 0.012) but cannot narrow sampling uncertainty.
- **No true negatives.** All AMD-SD eyes are wet AMD; prevalences are within-disease.
- **Selection bias in the source annotation.** AMD-SD excluded predominantly-normal B-scans
  before annotation, so every prevalence here is conditional on a slice already being
  pathological (IRF 23.8%, SRF 58.7%, PED 68.4% at scan level). Thresholds are tuned to that
  inflated base rate and will be miscalibrated on an unselected scan stream — consistent
  with the threshold transfer failure in §5. AROI's slice selection is content-driven for
  the same reason (§ Limitations, below), so it does not correct the bias.
- **Controls at last-4 are single-seed.** RETFound has 3; the comparison is asymmetric.
- **Most hyperparameters are fixed defaults, applied identically across arms.** Only the
  logistic-head regularisation `C` is selected (grid `[0.001, 0.01, 0.1, 1.0]`, by inner CV
  on the training folds) — for the encoder probes and, since this revision, for the
  mean-pooling baseline in §6d. Fixed throughout: 20 epochs, batch 32, warmup 2, weight
  decay 0.05, `drop_path` 0.1 for full FT and 0 elsewhere. Applying them identically makes
  the arms comparable but leaves open whether any arm would benefit from its own tuning.
- **Fine-tuning learning rates were chosen, not selected**: 1e-3 (LP), 1e-4 (last-4),
  2e-5 (full FT). A three-point sweep (§2) shows full FT is flat across 1e-5–5e-5 and
  last-4 improves at 3e-4, so the equivalence is not an lr artefact — but last-4's optimum
  is at the edge of the tested range and the sweep is single-seed. No layer-wise lr decay
  is used, whereas the upstream RETFound recipe applies 0.65 — full FT is the arm that
  recipe most protects, and that remains untested.
- **Attention head hyperparameters are untuned.** Epochs (40), L (64), lr and weight decay
  were chosen. Changing epochs 40 → 120 moves IRF AUPRC by 0.008. Nested-CV selection with
  test-set confirmation is running.
- **Pooling results are out-of-fold on the pool split**, not the held-out test set, and are
  not comparable to the AUPRC figures in §1–3.
- **AROI SRF includes SHRM; AMD-SD SRF does not.** Only PED is a like-for-like
  cross-dataset comparison.
- **AROI slice selection is content-driven** — 11 of 24 patients have a fully contiguous
  annotated block, and annotated fractions range 14.8%–80.5%. Volume estimates by
  area × spacing would be biased.
- **Index→class mapping** was undocumented in both datasets and was recovered, then
  confirmed geometrically (AROI: IRF 0.23 → SRF 0.50 → PED 0.72 depth against the RPE–BM
  band, across all 1,136 masks).

## Next

Inner-CV selection of the attention head, then attention pooling at last-4 depth on the
test set; three seeds for the controls at last-4; clinical review of eyes 143, 122 and 64;
composing within-scan attention with across-slice decision-max for the eye-level model.
