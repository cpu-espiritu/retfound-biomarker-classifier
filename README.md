# RETFound / AMD-SD — Stage 1

Multi-label B-scan classification of wet AMD fluid biomarkers (IRF, SRF, PED) with a
RETFound-MAE-OCT encoder. Results: [docs/RESULTS.md](docs/RESULTS.md). Rationale for every
choice: [docs/DECISIONS.md](docs/DECISIONS.md).

## Layout

Directories follow the order the pipeline runs in.

| Directory   | Stage                                                              |
| ----------- | ------------------------------------------------------------------ |
| `explore/`  | One-off dataset forensics. Run before writing any prep.             |
| `prep/`     | Raw dataset → splits (`manifest.csv`, or ImageFolder symlink trees).|
| `features/` | Cache frozen encoder features, then probe them. CPU-cheap arm.      |
| `finetune/` | Fine-tune the encoder end-to-end, then analyse its predictions.     |
| `slurm/`    | Cluster wrappers for AMD-SD. Thin — parameters live in the Python.  |
| `slurm/diagnosis/` | Wrappers for the secondary CNV/DRUSEN/NORMAL track.          |
| `docs/`     | Results and the decision log.                                       |
| `data/`     | Committed prediction artefacts (`amdsd_preds/*.npz`, `*.json`).     |

`features/` and `finetune/` are the two experimental arms compared in RESULTS.md: linear
probe on cached features vs. last-4 / full fine-tuning.

## AMD-SD pipeline (the main line of work)

```
explore/inspect_amdsd.py    --root RAW                        # inventory: sizes, modes, mask palette
explore/probe_classes.py    --masks RAW/masks                 # recover mask index → class by geometry
        ↓
prep/prep_amdsd.py          --root RAW --out SPLITS \         # → SPLITS/manifest.csv
                            --demographics demographics.xlsx  #   patient-level, stratified, leak-gated
        ↓
   ┌────────────────────────────────────┬─────────────────────────────────────┐
   │ frozen-feature arm                 │ fine-tuning arm                     │
   │                                    │                                     │
   │ slurm/extract_amdsd.sh SPLITS SIZE │ slurm/train_amdsd.sh MODE SIZE      │
   │   → features/extract_features.py   │   → finetune/train_amdsd.py         │
   │   → features_{tag}.npy (12 MB)     │   (sbatch --array=0-4, one per fold)│
   │                                    │   → data/amdsd_preds/preds_{tag}.npz│
   │ features/probe_heads.py            │                                     │
   │   --features … --manifest …        │ finetune/analyse_preds.py           │
   │   → per-class AUPRC, Youden thr,   │   --preds … --manifest … --arm …    │
   │     size-stratified recall, CIs    │   → ensembles folds, same metrics   │
   └────────────────────────────────────┴─────────────────────────────────────┘
```

Encoder controls (ImageNet ViT-L, `mae_in1k` / `sup_in21k`) go through
`slurm/extract_control.sh` → `features/extract_control.py`, producing features in the same
format so `features/probe_heads.py` reads them unchanged.

`explore/check_crop.py MANIFEST MASKDIR` validates a `--crop` manifest (IS/OS containment,
crop-height distribution). Cropping was tested and rejected — see DECISIONS.md — so this is
only needed if the crop path is revisited.

## Diagnosis track: UCSD / NEH (secondary)

3-class CNV/DRUSEN/NORMAL diagnosis, not fluid biomarkers. This track predates the AMD-SD
work and shares none of its code: every job runs the upstream RETFound repo's
`main_finetune.py` rather than `finetune/train_amdsd.py`, so `slurm/diagnosis/` scripts are
the whole interface and the settings live in the `torchrun` flags.

Data prep:

- `prep/make_val_split.py` — carves a patient-level val set out of the Kermany/UCSD `train/`.
- `prep/prep_neh.py` — leakage-safe patient-grouped NEH fold trees (`fold0…fold4`).
- `prep/make_neh_sets.py` — three NEH eval sets: `A_scan_labels`, `B_patient_labels`,
  `C_worstcase`. A vs B isolates the cost of patient-level labelling on identical images.

Jobs, all under `slurm/diagnosis/`:

| Script | Trains / evaluates on | Notes |
| --- | --- | --- |
| `train_ucsd.sh` | UCSD (`$OCT`) | baseline linear probe, 100 epochs |
| `train_ucsd_seed.sh` | UCSD | needs `SEED` exported; task tagged `_s$SEED` |
| `train_ucsd_weighted.sh` | UCSD | adds `--weighted_loss`; task tagged `_w_s$SEED` |
| `train_neh_lp.sh` | NEH folds | `--array=0-4`, one fold per task |
| `eval_diag.sh` | UCSD test | in-domain; needs `SRC` (checkpoint task name) |
| `eval_neh.sh` | `neh_sets/$NEHSET` | external validation; needs `SRC` and `NEHSET` |

The `_seed` / `_weighted` / `eval` scripts read `SEED`, `SRC`, and `NEHSET` from the
environment, so submit them as `SEED=1 sbatch slurm/diagnosis/train_ucsd_seed.sh`. Unset
variables fail at argparse rather than silently defaulting.

## Dependencies

`features/extract_features.py` and `finetune/train_amdsd.py` import `models_vit` and
`util.pos_embed` from the **RETFound_MAE repo**, which is not vendored here. The SLURM
wrappers handle this by `cd`-ing into it and exporting `PYTHONPATH`; running locally requires
the same. Weights are pulled from `YukunZhou/RETFound_mae_natureOCT` on the HF Hub
(`HF_HUB_OFFLINE=1` on compute nodes — cache them on the login node first).

Everything else is numpy / pandas / scikit-learn / torch / timm / Pillow.
