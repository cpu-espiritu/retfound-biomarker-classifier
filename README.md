# Wet AMD fluid biomarker classification with RETFound

Multi-label B-scan classification of three wet AMD fluid biomarkers (IRF, SRF, PED),
Stage 1 of a programme predicting anti-VEGF treatment response.

- **Findings and numbers:** [RESULTS.md](RESULTS.md)
- **Why each choice was made:** [DECISIONS.md](DECISIONS.md)

## Reproduce a headline number in under a minute

No GPU, no dataset access, no model weights — the measurements are in `results/`.

```bash
git clone https://github.com/cpu-espiritu/retfound-biomarker-classifier.git
cd retfound-biomarker-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python analysis/reproduce.py pooling
```

Expected output:

```
cls         IRF    SRF    PED
mean      0.759  0.939  0.907
mean+mlp  0.704  0.939  0.911
attn      0.811  0.962  0.950

attention − matched-capacity MLP, IRF: +0.108
```

That is the central mechanistic result: replacing RETFound's mean pooling over patch
tokens with attention pooling gains +0.108 AUPRC on IRF out-of-fold, and the gain survives
a matched-parameter control (66,625 vs 66,691 parameters). On the held-out test set under
the protocol of RESULTS.md §1 the gain is +0.100 [+0.024, +0.170], which brings a frozen
encoder level with fine-tuning 50M parameters — see RESULTS.md §6d.

Other headlines: `main`, `depth`, `encoder`, `size`.

## Reproduce the figures

```bash
python analysis/make_figures.py          # the six report figures
python analysis/paper_figures.py         # the composed paper figures
```

These need `data/amdsd_preds/` and `data/amdsd_splits/manifest.csv`, which are not
redistributable. See [data/README.md](data/README.md).

## Rerun the experiments

Needs a GPU, the source datasets, and the RETFound weights.

```bash
pip install torch torchvision timm huggingface-hub    # see requirements.txt for versions
python scripts/prep/prep_amdsd.py --root <AMD-SD> --out data/amdsd_splits \
    --demographics <AMD-SD>/demographics.xlsx
sbatch scripts/slurm/train_amdsd.sh last4 224 retfound 0
```

Every arm reported has a config in `configs/`, carrying its exact arguments and the
submit line that produced it.

## Layout

```
data/splits/       patient-level fold and test assignments (factual; redistributable)
configs/           one file per experimental arm
scripts/prep/      raw dataset -> manifest and splits
scripts/features/  frozen-feature and token extraction, pooling experiments
scripts/finetune/  training, evaluation, external inference
scripts/explore/   dataset forensics: class maps, lesion components, contrast
scripts/slurm/     cluster wrappers
results/           the CSVs every figure and table is built from
analysis/          results -> figures and tables
experiments.csv    run registry: one row per fold, with parameters and output path
```

## Provenance

`experiments.csv` registers all 135 training runs with their parameters and outputs.
`data/splits/manifest.sha256` pins the exact manifest every reported number used.

Two honest gaps: runs completed before 2026-08-27 predate git-SHA recording, so their
`git_sha` is blank — the trainer records it from that date onward. And 115 of the 135
rows have parameters reconstructed from the output filename rather than read from a
saved config, because early runs were pulled from the cluster without their `cfg_*.json`.
The `params_from` column marks which is which.
