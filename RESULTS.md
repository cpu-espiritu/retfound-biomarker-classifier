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
| Last-4, 448            | 50 M      | 0.806 [0.531–0.917]     | 0.977 [0.933–0.995]     | 0.914 [0.801–0.976]     |

Class prevalence in test: IRF 0.268, SRF 0.675, PED 0.682.

**Seed-to-seed SD is negligible: ≤ 0.012 across all 21 measured cells** (max: RETFound
last-4 IRF, 0.012). Every single-seed number previously reported was reliable.

448's intervals are a single seed (only `preds_last4_448_f*_s0` exists), on the same
5,000 patient-level replicates as every other arm.

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

AROI prevalence differs from AMD-SD, which matters for reading these numbers:

| | AROI slices | AROI patients | AMD-SD slices |
| --- | --- | --- | --- |
| IRF | 228 (20.1%) | 13/24 (54%) | 23.8% |
| SRF | 648 (57.0%) | 21/24 (88%) | 58.7% |
| PED | 1014 (**89.3%**) | **24/24 (100%)** | 68.4% |

IRF and SRF are comparable across datasets. PED is not: 89.3% of AROI slices are positive,
leaving only 122 negatives, so its AUROC below is measured against a much easier negative
set than AMD-SD's. IRF is present in only 13 of 24 patients, so the AROI IRF curves rest on
about half the cohort.

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

**Repeated at the selected hyperparameters, the conclusion is unchanged.** The table above
used 40 epochs, chosen by hand; nested CV later selected 5 for IRF, 55 for SRF, 30 for PED.
Rerunning with those, three seeds each and the tuned baseline:

| | epochs=40, hand-chosen | nested-CV selected |
| --- | --- | --- |
| sub-patch Δ recall | +0.120 (24% of available gap) | +0.120 (24%) |
| above-patch Δ recall | +0.077 (55%) | +0.068 (56%) |

Five of 20 bins survive Holm under both, though which ones shifts: SRF's 0.08–0.22 bin now
clears correction (+0.269, p_holm 0.008) where PED's smallest no longer does. The
size-independence of the gain is therefore a property of attention pooling, not of the
epoch count.

### Confirmed on the held-out test set

Everything above is out-of-fold on the pool split with hyperparameters I chose. This is the
clean version: L, learning rate, weight decay and epochs selected by nested CV inside the
training folds, the baseline's `C` selected on the same grid, and the test set touched once.

| class | attention | mean pooling | Δ AUPRC | 95% CI | p | Holm |
| --- | --- | --- | --- | --- | --- | --- |
| **IRF** | **0.831** | 0.699 | **+0.131** | [+0.052, +0.201] | 0.0016 | **0.0048** |
| SRF | 0.980 | 0.967 | +0.012 | [−0.004, +0.041] | 0.125 | 0.125 |
| **PED** | **0.953** | 0.902 | **+0.051** | [+0.011, +0.129] | 0.0032 | **0.0064** |

IRF and PED survive correction. The IRF effect is **larger** on the test set than
out-of-fold (+0.131 vs +0.105), so it is not selection-optimistic.

**A frozen encoder with a 66k-parameter attention head beats fine-tuning 50M parameters on
IRF**, on the same 20 test patients:

| | frozen + attention | last-4 fine-tuned | Δ |
| --- | --- | --- | --- |
| IRF | **0.831** | 0.784 | **+0.047** |
| SRF | 0.980 | 0.984 | −0.005 |
| PED | 0.953 | 0.964 | −0.011 |

That comparison is not perfectly matched — the attention head's hyperparameters were
selected by nested CV while the fine-tuning arms used fixed defaults (§ Limitations) — but
it reframes §1 and §2: on the hardest class, the gain from fine-tuning is available more
cheaply by changing how the encoder's output is pooled.

### Frozen encoder with attention pooling matches fine-tuning

All rows below are on the same held-out test set (440 scans, 20 patients) under the same
protocol as §1: one model per fold, test predictions ensembled, thresholds from the
held-out validation fold. Attention arms are the mean of 3 seeds; the logistic head on
mean-pooled features is deterministic and runs once.

| | IRF | SRF | PED | trainable |
| --- | --- | --- | --- | --- |
| **frozen + attention** | **0.807** ± 0.005 | 0.984 ± 0.001 | 0.963 ± 0.004 | **0.07 M** |
| frozen + mean pooling | 0.707 | 0.971 | 0.919 | 0.003 M |
| last-4 + mean pooling | 0.791 | 0.985 | 0.960 | 50 M |
| last-4 + attention | 0.786 | 0.983 | 0.963 | 50 M |

**Attention pooling helps a frozen encoder and does nothing to a fine-tuned one.** Both
contrasts on the same 5,000 patient resamples, Holm across all six:

| depth | class | attention | mean | Δ AUPRC | 95% CI | p | Holm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **frozen** | IRF | 0.807 | 0.707 | **+0.100** | [+0.024, +0.170] | 0.004 | **0.026** |
| **frozen** | SRF | 0.984 | 0.971 | **+0.013** | [+0.002, +0.040] | 0.012 | **0.046** |
| **frozen** | PED | 0.963 | 0.919 | **+0.044** | [+0.008, +0.112] | 0.006 | **0.030** |
| last-4 | IRF | 0.786 | 0.791 | −0.005 | [−0.020, +0.023] | 0.619 | 0.619 |
| last-4 | SRF | 0.983 | 0.985 | −0.002 | [−0.007, +0.000] | 0.111 | 0.257 |
| last-4 | PED | 0.963 | 0.960 | +0.004 | [−0.000, +0.012] | 0.086 | 0.257 |

All three frozen contrasts survive correction; none of the fine-tuned ones does. The frozen
attention arm is the mean of 3 seeds against a deterministic baseline; the last-4 rows are
single-seed on both sides.

**But it matches fine-tuning rather than beating it.** Paired against the last-4 arms,
nothing survives Holm across six tests:

| comparison | IRF | SRF | PED |
| --- | --- | --- | --- |
| frozen+attn − last-4+mean | +0.016 [−0.061, +0.056] p 0.544 | −0.001 p 0.901 | +0.003 p 0.644 |
| frozen+attn − last-4+attn | +0.021 [−0.050, +0.049] p 0.452 | +0.001 p 0.562 | −0.000 p 0.981 |

**Adding attention to a fine-tuned encoder gains nothing**: within last-4, attention − mean
is −0.005 [−0.020, +0.023] on IRF, −0.002 SRF, +0.004 PED.

**The claim.** A frozen encoder with a 0.07 M-parameter pooling head reaches the same
performance as fine-tuning 50 M parameters — an equivalence, not a win. The two
interventions address the same deficiency: a frozen encoder carries localisation
information that mean pooling discards and attention recovers; a fine-tuned encoder
reorganises its features so the mean already carries it. Applying both gains nothing,
which is what a substitution predicts.

_An earlier version of this section reported frozen + attention at 0.831 and claimed it beat
fine-tuning by 0.045. That figure came from a single model fitted on all pool data, while
the fine-tuned arms were 5-fold ensembles — the protocols were not matched, and the
mismatch favoured the frozen arm because its single model saw 2,609 training scans against
2,087 for each ensemble member. Under the matched protocol the difference is +0.016 with an
interval spanning zero._

**Selected hyperparameters were unstable in epochs.** Five folds chose five different
configurations; epochs ranged 5–80 for IRF and 5–60 for SRF, and the modal choices were
IRF (L=64, lr=1e-3, wd=1e-3, 5 epochs), SRF (L=32, 3e-3, 1e-3, 55), PED (L=32, 3e-3,
1e-3, 30). `L` and learning rate were stable; the stopping point was not, which means the
inner-validation curve is flat and the epoch count is close to arbitrary within that range.
The result holds despite that, but a reader should know the head is not tightly identified
by 118 IRF-positive patients.

### Sample efficiency: does the equivalence hold when data is scarce? — running

The section above establishes an equivalence at the full 118-patient pool. It does not say
which arm gets there on less data, and that is the claim that matters for anyone deciding
where to spend an annotation budget. Pre-registered protocol, results to follow:

**Design.** Both arms trained at 15, 30, 60 and 118 pool patients, three seeds each,
AUPRC on the fixed held-out test set (20 patients, 440 scans) plotted against training-set
size.

- **Subsampling is by patient**, so every scan of a sampled patient is kept and no eye
  straddles the boundary. Subsets **nest** within a seed (15 subset of 30 subset of 60 subset of 118),
  so the curve is a curve and not four unrelated draws; each subset holds the five preset
  folds in proportion and preserves the patient-level IRF rate, the scarcest label. At 118
  every fold is taken whole, so that point *is* the existing run rather than a re-run.
  `scripts/prep/make_size_subsets.py`, deterministic from the manifest.
- **Both arms consume the identical patient set** at each (size, seed) — the same file
  drives `frozen_size_curve.py` and `train_amdsd.py --n-train-patients`.
- **The whole pool split shrinks, training and validation alike.** Best-epoch selection and
  the Youden threshold get noisier at small n; that is a real cost of having fewer
  patients, and it is paid equally by both arms.
- **Hyperparameters are fixed at the values tuned at 118** for both arms. Re-tuning per
  size would confound sample efficiency with tuning budget.
- **The step budget is held constant, not the epoch count** (`--epoch-budget steps`,
  epochs and warmup scaled by 118/n). A fixed 20 epochs would give the 15-patient arm 160
  gradient steps against 1,300 at full size, and the curve would then measure
  undertraining as much as data scarcity. Under the step budget every fold-job is
  1,131–1,264 steps. Overfitting at small n is *not* corrected for — that is the
  phenomenon being measured, and best-epoch-on-validation already bounds it.
- **The test set is never touched by the subsampling.**

**Cost.** Both arms run on BlueBEAR; the token cache and the images are only there.
The frozen arm is head-only on cached tokens but not free: holding the step budget makes
every size cost what the full pool costs, so the grid is ~4x the existing
`frozen_arms.py` run — about 5.5 CPU-hours single-threaded, one batch job.
(`--epoch-budget fixed` would be ~2.7 h, since the large sizes dominate either way.)
The fine-tuned arm is 45 new A100 fold-jobs (3 sizes x 3 seeds x 5 folds), each ~1,200
gradient steps, i.e. roughly the cost of an existing full-pool run; the 15 jobs at n=118
already exist as `last4_224_f*_s{0,1,2}`.

**What would falsify the interesting reading.** If the two curves stay parallel, attention
pooling is a cheaper route to the same place and nothing more — the §6 equivalence just
extends downward. The claim worth making is the one where the gap *widens* as n falls, and
IRF is where to look for it.

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

**How much small-lesion recall a lower threshold buys, and what it costs.** Following from
§4, the threshold decides the fate of sub-patch lesions almost exclusively: above-patch
recall is flat (IRF 0.949 across thresholds 0.10–0.50).

| class | operating point | sub-patch recall | specificity |
| --- | --- | --- | --- |
| SRF | Youden 0.80 | 0.500 | 0.986 |
| **SRF** | **0.20** | **0.784** | **0.937** |
| IRF | Youden 0.16 | 0.671 | 0.901 |
| IRF | 0.07 | 0.800 | 0.783 |
| PED | Youden 0.83 | 0.597 | 0.900 |
| PED | 0.50 | 0.807 | 0.593 |

SRF is the favourable case: **+0.284 sub-patch recall for −0.049 specificity**. IRF costs
roughly as much as it gains. PED is expensive — its negatives have a median score of 0.384,
so lowering the cutoff sweeps them in, costing 31 points of specificity for 21 of recall.

A **size-conditional** threshold is not deployable: lesion area is unknown at inference, so
only a single global cutoff can be chosen. This is a trade-off curve to be selected on
clinical cost, not a free improvement.

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
- **Attention head hyperparameters are now selected, not chosen** (§6d), but the epoch
  count is not identifiable — five folds chose values spanning 5–80. The fine-tuning arms
  in §1–2 remain on fixed defaults, so the frozen+attention vs last-4 comparison gives the
  attention arm a tuning advantage the fine-tuned arms did not get.
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

The sample-efficiency curve in §6 is specified and queued; the frozen arm runs on CPU and
the 45 fine-tuning fold-jobs are submitted with `scripts/slurm/size_curve.sh`.
