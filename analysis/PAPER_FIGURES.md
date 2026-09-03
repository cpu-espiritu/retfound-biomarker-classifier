# Paper figure set

Two stages. `derive.py` recomputes every number the figures draw and writes it to
`results/*.csv`; the scripts in `figures/` read only those CSVs and write PDFs. So a
machine with the restricted inputs runs stage 1 once, and the figures rebuild anywhere.

```bash
python analysis/derive.py                  # needs predictions, features, masks
python analysis/make_paper_figures.py      # needs only results/*.csv
python analysis/make_paper_figures.py --only 2 5     # one or two of them
```

Output lands in `analysis/output/paper/`, which is gitignored. No figure script contains a
measured number: if a value is not in a CSV, it belongs in `derive.py`.

| figure | script | reads | shows | claim |
| --- | --- | --- | --- | --- |
| `figure1_lesion_scale.pdf` | `fig1_lesion_scale.py` | `fig1_examples`, `amdsd_components_per_lesion`, `amdsd_MERGED_per_lesion`, `aroi_components` | **a** one B-scan per class with its mask and a one-patch square. **b** lesion size distributions in patch units, both datasets | RESULTS §4. Why size matters, and why patch units are the only valid cross-dataset unit. |
| `figure2_arms.pdf` | `fig2_arms.py` | `fig2_arms_auprc` | AUPRC per adaptation arm against the prevalence floor, including last-4 at 448 | RESULTS §1 |
| `figure3_size_effect.pdf` | `fig3_effect.py` | `fig3_recall_bins`, `aroi_fig3_recall_bins`, `fig3_scores`, `fig3_thresholds` | **a** recall vs lesion size with bootstrap bands and a per-bin count strip. **b** the continuous score behind it | RESULTS §4, §5. Small lesions are scored low, not scored negative. |
| `figure4_thresholds.pdf` | `fig4_thresholds.py` | `fig4_threshold_sweep`, `aroi_fig4_operating_points` | **a** sub-patch recall against specificity as the threshold sweeps. **b** AROI recall at AMD-SD vs refit thresholds | RESULTS §7. Threshold choice decides the fate of sub-patch lesions. |
| `figure5_pooling.pdf` | `fig5_pooling.py` | `fig5_pooling_arms`, `fig5_capacity_control` | **a** pooling x depth interaction. **b** matched-capacity control. **c** what the parameters buy | RESULTS §6d. The gain is selectivity, not capacity, and it is a substitute for fine-tuning. |
| `figure6_forest.pdf` | `fig6_forest.py` | `fig6_forest` | paired Δ for depth, resolution, encoder, patch crossing, on one axis | RESULTS §2, §3, §6a. The elimination chain. |
| `figure7_transfer.pdf` | `fig7_transfer.py` | `aroi_fig7_auroc`, `aroi_fig7_recall` | **a** AROI zero-shot AUROC. **b** standardised sub-patch recall, transferred vs refit thresholds | RESULTS §5. Discrimination transfers, calibration does not. |

## What is committed and what is not

`results/*.csv` is tracked, with one exception: `.gitignore` blocks `results/aroi_*.csv`,
because those measurements come from masks the AROI licence forbids redistributing. Every
figure that wants one degrades: figure 1 drops its AROI panel, figure 3 drops its AROI
column, figure 4 drops panel b, and figure 7 skips entirely, each saying so on stdout.

Figure 1a additionally needs the source B-scans in `data/amdsd/`, which come from the
AMD-SD release rather than this repository. Without them figure 1 builds panel b alone.

## Decisions

- **No significance stars in figure 6.** Three of its four panels carry equivalence claims,
  which a multiplicity correction reads backwards — failing to reject gets easier, not
  harder. Zero is marked and the intervals state the precision. The one exception is the
  encoder panel, where Holm runs across its own twelve tests and survivors are drawn filled.
- **Arm colours never reuse a class colour.** Figure 2 sets hue by adaptation depth and
  marker fill by input resolution, so a marker can never contradict the class in the panel
  title.
- **Panel letters are drawn from each axes' own position**, not hardcoded figure
  coordinates, and are kept out of the axes titles.
- **Figure 5's matched-MLP control is refit** from cached mean-pooled features, since no
  predictions were saved for it. Refitting the mean arm the same way reproduces the saved
  one to ~0.001 AUPRC, which is the check that the substitution is sound.
- **`paper_figures.py` and `make_figures.py` are superseded** by this set. They still build
  the older report figures and are kept for reference, not for the paper.
