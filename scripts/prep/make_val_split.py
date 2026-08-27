#!/usr/bin/env python3
import os, random, shutil
from collections import defaultdict

OCT = os.environ["OCT"]
VAL_FRAC = 0.15
SEED = 0
CLASSES = ["CNV", "DRUSEN", "NORMAL"]

random.seed(SEED)

val_root = os.path.join(OCT, "val")
if os.path.isdir(val_root) and any(os.scandir(val_root)):
    raise SystemExit(f"{val_root} exists and is non-empty. Delete it to rebuild.")

for cls in CLASSES:
    train_dir = os.path.join(OCT, "train", cls)
    val_dir = os.path.join(OCT, "val", cls)
    os.makedirs(val_dir, exist_ok=True)

    by_patient = defaultdict(list)
    for fname in os.listdir(train_dir):
        if not fname.lower().endswith(".jpeg"):
            continue
        parts = fname.split("-")
        if len(parts) < 3:
            print(f"  skipping unexpected filename: {fname}")
            continue
        by_patient[parts[1]].append(fname)

    patients = sorted(by_patient.keys())
    random.shuffle(patients)
    n_val = int(len(patients) * VAL_FRAC)
    val_patients = patients[:n_val]

    moved = 0
    for pid in val_patients:
        for fname in by_patient[pid]:
            shutil.move(os.path.join(train_dir, fname), os.path.join(val_dir, fname))
            moved += 1

    print(f"{cls}: {len(patients)} patients -> {n_val} to val "
          f"({moved} scans moved, {len(patients)-n_val} patients left in train)")
