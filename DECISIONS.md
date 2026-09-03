# Design Decisions — AMD-SD Stage 1

Each entry: what was chosen, why, and the evidence. Reverse chronological within sections.

---

## Task framing

**Multi-label classification, not segmentation.**
Three independent sigmoids (IRF / SRF / PED) at B-scan level. RETFound is an image-level
encoder with no decoder; segmentation would require bolting on a UNet/UPerNet head, which is
a different project. Clinical decisions depend on presence and burden, not pixel boundaries.
Masks are still used — for label derivation, size-stratified reporting, and (planned)
localisation checks.

**Fluid burden proxy for Stage 2.**
Fraction of positive B-scans per eye gives a quantitative measure without a decoder. This is
the intended bridge to the treatment-response model: a per-eye, per-visit 3-vector.

---

## Data

**Patient-level splits, not eye-level or scan-level.**
Adjacent B-scans from one eye are near-duplicates; scan-level splitting leaks badly. Eye-level
still leaks the ~18 patients contributing both eyes (fellow eyes correlate in wet AMD).
Patient IDs come from `demographics.xlsx`, obtained from the Figshare release — the Kaggle
mirror had dropped it. 138 patients / 156 eyes / 3,049 B-scans.

**Official train/val split ignored.**
The AMD-SD release includes 80:20 txt files, but the split is at image level, so B-scans from
the same eye appear on both sides. Not usable. Own splits: 20 patients held out for test,
5-fold StratifiedGroupKFold on the remaining 118.

**Test set stratified, not random.**
First implementation used plain `GroupShuffleSplit` for the held-out test set, which ignores
labels. With only 156 groups, IRF prevalence drifted to 0.093 in test vs 0.18–0.32 across
folds — roughly 41 positives from ~4 eyes. Switched to `StratifiedGroupKFold` for the test
carve-out too. Test prevalence now sits inside the fold range on all three classes.

**Mask class mapping recovered, not assumed.**
Distributed masks are integer-indexed 1–5 with no documentation in this copy. Geometry and
reflectivity probes plus the published palette gave: 1=SRF, 2=IRF, 3=PED, 4=SHRM, 5=IS/OS.
Confirmed by clinical review. SHRM and IS/OS dropped from the target set.

---

## Labels

**Any-pixel presence rule; no minimum-area threshold.**
Lesion areas are smooth over three orders of magnitude with no bimodal gap:

| class | p1  | p5  | p25 | p50  | p75  | p95   |
| ----- | --- | --- | --- | ---- | ---- | ----- |
| IRF   | 23  | 59  | 286 | 823  | 2438 | 5909  |
| SRF   | 49  | 153 | 714 | 2025 | 4369 | 9911  |
| PED   | 66  | 152 | 662 | 1700 | 4138 | 11749 |

A threshold would be an invented parameter. Instead, report **recall stratified by lesion
size**, which turns the assumption into a measured result. (This produced the size-gradient
finding.) The machinery for per-class thresholds with validity masking is implemented
(`--min-area 'IRF=50,...'`) and reserved for a sensitivity analysis.

**Per-class validity masking, not row dropping.**
An image can be borderline for one class and clear for the others. If a threshold is ever
applied, the loss is masked per class rather than dropping the whole row, preserving the two
good labels.

---

## Preprocessing

**Retinal-band cropping: tried, rejected.**
Method: longest contiguous bright row-run + 60px margin. Median retained height 242/380.
Validated (IS/OS containment 0.9989, zero label changes, zero PED pixel loss).

Result — worse on all three classes at linear probe:

| AUROC | no crop | crop  |
| ----- | ------- | ----- |
| IRF   | 0.836   | 0.805 |
| SRF   | 0.942   | 0.915 |
| PED   | 0.839   | 0.745 |

Likely cause: crop heights vary (192–380 px) but everything resizes to a fixed 224×224, so
the same lesion is stretched differently across scans. Added nuisance variation without
meaningful gain, since the retinal band already dominates these images.

Side benefit of dropping it: no-crop transports to external datasets with no parameters to
re-tune.

_Note on the first implementation:_ taking the first and last row above threshold made the
detector fragile — one bright edge row dragged the crop to full height in 86% of scans. The
retina is one connected band; noise is scattered. Longest-run selection uses that.

**Anisotropic resize, no centre crop.**
570×380 → 224×224 squashes horizontally. Deliberate: a centre crop would discard
lesion-bearing periphery.

---

## Model and training

**RETFound-MAE-OCT (`RETFound_mae_natureOCT`) as encoder.**
Justified empirically against two ViT-L controls at linear probe (see below), not assumed.

**Last-4 blocks as the recommended adaptation depth.**
Best or joint-best AUPRC on all three classes and highest specificity in every class, at 50M
trainable params vs 303M for full fine-tuning. Full FT buys nothing at n=138 patients.

| AUPRC | LP (0.003M) | last-4 (50M) | full (303M) |
| ----- | ----------- | ------------ | ----------- |
| IRF   | 0.683       | **0.791**    | 0.785       |
| SRF   | 0.972       | **0.985**    | 0.984       |
| PED   | 0.889       | **0.960**    | 0.952       |

**384 input resolution: tested at two depths, rejected.**
Fails at linear probe (all three classes down) _and_ at full fine-tuning (IRF +0.014, inside
CI; SRF −0.005, PED −0.018). No consistent small-lesion benefit. Testing at both depths was
necessary — at LP alone the result is confounded by distribution shift (MAE learned patch
statistics at 224, and position embeddings are interpolated), so only the fine-tuned arm
separates "resolution doesn't help" from "the frozen encoder can't adapt".

**Feature caching before any head experiments.**
One GPU pass per encoder/resolution config produces a 12 MB `.npy`. Every downstream
experiment — regularisation sweeps, thresholds, aggregation, encoder comparison — is then
CPU work in seconds. This is what made the ablation grid affordable.

---

## Evaluation

**Youden's J for threshold selection, not F1.**
F1 ignores true negatives. At high prevalence the cheapest way to score well is to predict
positive almost always. Observed directly on PED (68% prevalence):

| PED         | F1-optimal | Youden |
| ----------- | ---------- | ------ |
| threshold   | 0.22       | 0.83   |
| recall      | 0.970      | 0.803  |
| specificity | 0.386      | 0.900  |
| AUROC       | 0.910      | 0.910  |

_Revised: these were recomputed at last-4 over three seeds. The earlier figures
(threshold 0.28, specificity 0.271) came from a single-seed run._

Identical AUROC — same model, same ranking, only the cutoff moved. Final thresholds span
0.16–0.85; none is near 0.5. Threshold choice is a first-order modelling decision here.

**AUPRC reported against prevalence, not in isolation.**
AUPRC's floor is the class prevalence. IRF 0.791 against a 0.268 floor is a stronger result
than PED 0.960 against 0.682. Prevalence is quoted alongside every AUPRC.

**Patient-level cluster bootstrap for CIs.**
Resampling B-scans would treat near-duplicate slices as independent and produce fiction.
Resampling patients gives honest intervals — and they are wide (IRF AUPRC [0.577, 0.898] at
last-4). Fold-to-fold SD (±0.02) understates true uncertainty by roughly an order of
magnitude and is not quoted as an uncertainty estimate.

**Holm correction inside the encoder family only.**
Multiplicity correction is applied within the family defined by the claim it supports, and
only where the claim is a rejection. The encoder comparison (2 controls x 2 depths x 3
classes = 12 tests) is Holm-corrected. Depth and resolution are reported as differences with
intervals, uncorrected: depth carries an equivalence claim (full FT == last-4) and resolution
a null, and a multiplicity correction serves neither — inflating p-values makes an
equivalence claim look stronger, which is backwards. The interval is the evidence there.

_Revised: an earlier `pvalues.csv` Holm-corrected all 15 depth, resolution and encoder tests
as one family, and its encoder rows could not be regenerated from the saved predictions at
either depth or any seed count. Both that file and `encoder_grid_auprc.csv` were orphans with
no generator in the repo; `analysis/derive_tables.py` now produces both and is the only
source. The stale encoder row (RETFound - MAE-IN1k, IRF +0.189, Holm 0.029) does not survive:
the regenerated value is +0.271 at LP with Holm 0.149, and +0.041 at last-4._

**Per-class recall and specificity always reported.**
Accuracy and macro-F1 mask class collapse. Specificity was added after PED's inflated recall
was traced to the F1 threshold problem.

---

## Controls

**ImageNet ViT-L controls at linear probe — RETFound wins all classes.**

| AUPRC | RETFound-MAE-OCT | MAE-IN1k | Sup-IN21k |
| ----- | ---------------- | -------- | --------- |
| IRF   | **0.699**        | 0.510    | 0.682     |
| SRF   | **0.976**        | 0.922    | 0.910     |
| PED   | **0.925**        | 0.847    | 0.818     |

Same architecture (ViT-L/16), same patch size, same 1024-dim pooled output; only pretraining
differs. Note the two controls disagree with each other on IRF (0.682 vs 0.510) — MAE on
natural images gives the _worst_ features here, so the gain is not "SSL is good" but
specifically MAE **paired with retinal data**.

_Superseded._ That claim used size quartiles from a single seed. Repeated with absolute
patch-unit strata, three seeds per arm, and Holm correction across 20 tests (RESULTS.md §3),
the picture is different: at **linear probe** the surviving differences are all
**above-patch**, and the sub-patch differences run *against* RETFound (IRF −0.177,
PED −0.148 vs Sup-IN21k, neither significant). Only after fine-tuning to last-4 does
RETFound gain on small lesions, and only SRF survives correction there (+0.277, p_holm
0.042).

_Resolved._ The controls were run at last-4 with three seeds each (RESULTS.md §3). The
advantage shrinks substantially: MAE-IN1k's IRF deficit falls from +0.189 at LP to +0.055
at last-4 (p 0.203), and Sup-IN21k matches RETFound on IRF at both depths. SRF and PED
retain smaller but significant advantages. RETFound's value is concentrated in what it
provides without fine-tuning.

---

## Known limitations

- **20 test patients.** CIs are wide and no further experimentation narrows them.
- **No true negatives.** All AMD-SD eyes are wet AMD; prevalences are within-disease.
- **Single scanner, single centre.** Cross-vendor generalisation untested (RETOUCH planned).
- **Fine-tuning arms are single-seed.** Multi-seed queued for the headline LP → last-4 gap.
- **Index→class mapping** was undocumented in the distributed masks.

---

## Open questions for the UCL team

1. Citable list of OCT datasets in the 1.6M pretraining corpus (Kermany confirmed in; AMD-SD,
   NEH, OCTDL, RETOUCH unknown).
2. `RETFound_mae_meh` modality — OCT, CFP, or mixed? Better for wet AMD fluid biomarkers?
3. If MEH-MIDAS dominates OCT pretraining and Stage 2 evaluates on Moorfields data, how should
   that be framed? Not label leakage, but not clean OOD either.
4. With `global_pool=True`, `models_vit.py` deletes `norm` and creates `fc_norm`, but the
   checkpoint has no `fc_norm` weights — loads as identity LayerNorm. Intended, or should
   `norm` be remapped? Affects every frozen-feature result here.
5. Has RETFound been applied to fluid-compartment biomarkers rather than global disease
   labels? Does the MAE objective preserve that level of detail?
6. MEH-MIDAS access for Stage 2 — process, timeline, what makes an application succeed.
