# Paper figure set

Regenerate the whole set:

```bash
python analysis/make_figures.py     # report figures
python analysis/paper_figures.py    # composes figures 1 and 3, adopts the rest
```

Output lands in `analysis/output/paper/`, which is gitignored.

| file | shows | claim |
| --- | --- | --- |
| `figure1_effect.pdf` | **a** recall vs lesion size — AMD-SD, AROI with transferred thresholds, AROI recalibrated. **b** continuous model score vs size, against each class's threshold. | RESULTS §4, §5. The effect, its external replication, and why small lesions are missed: they are scored low, not scored negative. |
| `figure2_eliminations.pdf` | paired Δ AUPRC — depth, resolution, pretraining, Holm-corrected | RESULTS §2, §3. Two-thirds of the elimination chain. Patch scale is carried by figure 3, not here. |
| `figure3_patch_scale.pdf` | lesion size distributions in patch units, both datasets, plus AMD-SD SRF+SHRM | RESULTS §4. Why size matters, and why patch units are the only valid cross-dataset unit. |
| `figure4_main_result.pdf` | AUPRC per arm against the prevalence floor | RESULTS §1 |
| `figure5_thresholds.pdf` | F1 vs Youden, recall and specificity | RESULTS §7 |
| `figureS1_lesion_areas.pdf` | lesion area distribution, log scale | Supplement. Justifies the any-pixel rule: no bimodal gap, so no minimum-area cut is defensible. |

## Decisions

- **`recall_vs_area_patch_units` archived.** A strict subset of figure 1a.
- **Figure 2 left at three comparisons, not extended to five.** The patch-scale
  elimination is a different kind of evidence — a paired crossing test on 194 scans, not a
  Δ AUPRC — so it does not share an axis with the others. Figure 3 carries it.
- **Figure 3 degrades gracefully.** Its AROI panel needs `results/aroi_components.csv`,
  which is gitignored because its lesion areas derive from the restricted AROI masks. From
  a clean clone the figure builds with the AMD-SD panel only and says so.
