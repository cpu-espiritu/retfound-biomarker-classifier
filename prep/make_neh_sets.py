#!/usr/bin/env python3
import os, csv, sys
from collections import Counter

RAW  = os.environ["PROJECT"] + "/retfound/data/neh_raw"
CSV  = RAW + "/data_information.csv"
DEST = os.environ["PROJECT"] + "/retfound/data/neh_sets"
CLASSES = ["CNV", "DRUSEN", "NORMAL"]

# locate the dir containing CNV/ DRUSEN/ NORMAL
IMG_ROOT = None
for dp, dn, _ in os.walk(RAW):
    if all(c in dn for c in CLASSES):
        IMG_ROOT = dp; break
if IMG_ROOT is None:
    sys.exit("Could not find CNV/DRUSEN/NORMAL dirs under " + RAW)
print("image root:", IMG_ROOT)

rows = list(csv.DictReader(open(CSV)))
print("csv rows:", len(rows))

sets = {"A_scan_labels": [], "B_patient_labels": [], "C_worstcase": []}
missing = 0
for r in rows:
    src = os.path.join(IMG_ROOT, r["Directory"])
    if not os.path.isfile(src):
        missing += 1; continue
    cls, lab = r["Class"].strip().upper(), r["Label"].strip().upper()
    name = f'{cls}_p{r["Patient ID"]}_{r["Eye"]}_b{r["B-scan"]}.jpg'
    sets["A_scan_labels"].append((src, lab, name))
    sets["B_patient_labels"].append((src, cls, name))
    if cls == lab:
        sets["C_worstcase"].append((src, lab, name))
if missing:
    print(f"WARNING: {missing} files listed in CSV not found on disk")

for sname, items in sets.items():
    base = os.path.join(DEST, sname)
    # main_finetune builds train/val/test even with --eval, so all three must exist
    for split in ["train", "val", "test"]:
        for c in CLASSES:
            os.makedirs(os.path.join(base, split, c), exist_ok=True)
    for src, lab, name in items:
        d = os.path.join(base, "test", lab, name)
        if not os.path.islink(d):
            os.symlink(src, d)
    # dummy train/val (never used in eval mode, but ImageFolder needs non-empty dirs)
    for split in ["train", "val"]:
        for c in CLASSES:
            ex = [i for i in items if i[1] == c][:2]
            for src, _, name in ex:
                d = os.path.join(base, split, c, name)
                if not os.path.islink(d):
                    os.symlink(src, d)
    cnt = Counter(l for _, l, _ in items)
    print(f"{sname}: {len(items)} images  " + "  ".join(f"{c}={cnt[c]}" for c in CLASSES))
