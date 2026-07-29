#!/usr/bin/env python3
"""
Prepare the Noor Eye Hospital (NEH) OCT dataset for the RETFound pipeline.

Leakage-safe, DRIL-protocol split as ImageFolder symlink trees so your existing
build_dataset()/main_finetune.py run UNCHANGED — you only swap --data_path per fold.

  * Worst-case (Option 2) filter: keep scans where Class == Label.
  * Group by PATIENT: key = "<Class>_<PatientID>" (Class prefix REQUIRED because
    PatientID resets within each class). Patient grouping closes the fellow-eye
    correlation confound.
  * 15% held-out test by group, touched once. 5-fold StratifiedGroupKFold on 85%.

Output (data_path per fold = OUT/fold{k}):
    OUT/_shared_test/<CLASS>/<flat>.jpg
    OUT/fold0/train/<CLASS>/<flat>.jpg
    OUT/fold0/val/<CLASS>/<flat>.jpg
    OUT/fold0/test -> ../_shared_test
    OUT/split_manifest.csv

ImageFolder indexes classes alphabetically: CNV=0, DRUSEN=1, NORMAL=2.
"""

import argparse
import os
import sys
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def parse_args():
    p = argparse.ArgumentParser(description="Prepare NEH OCT splits (DRIL protocol).")
    p.add_argument("--csv", required=True, help="Path to data_information.csv")
    p.add_argument("--image-root", required=True,
                   help="Root such that image_root + Directory is a real file.")
    p.add_argument("--out", required=True, help="Output dir for fold symlink trees.")
    p.add_argument("--seed", type=int, default=42, help="Seed for test split (DRIL=42).")
    p.add_argument("--test-frac", type=float, default=0.15, help="Fraction of GROUPS as test.")
    p.add_argument("--folds", type=int, default=5, help="CV folds on the 85%% pool.")
    return p.parse_args()


def build_frame(csv_path):
    df = pd.read_csv(csv_path)
    required = {"Patient ID", "Class", "Eye", "B-scan", "Label", "Directory"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"CSV missing columns: {missing}")

    before = len(df)
    df = df[df["Class"] == df["Label"]].copy()
    print(f"Worst-case filter: {before} -> {len(df)} scans")

    df["group"] = df["Class"].astype(str) + "_" + df["Patient ID"].astype(str)
    df["cls"] = df["Label"].astype(str).str.upper()
    df["flat"] = df["Directory"].str.replace("/", "__", regex=False)
    return df


def summarise(df, label):
    print(f"  [{label}] scans={len(df)}  patients={df['group'].nunique()}")
    print(f"           scans/class={df['cls'].value_counts().to_dict()}")
    print(f"           patients/class={df.groupby('cls')['group'].nunique().to_dict()}")


def make_symlinks(rows, image_root, dst_dir):
    made = 0
    for _, r in rows.iterrows():
        src = os.path.abspath(os.path.join(image_root, r["Directory"]))
        cls_dir = os.path.join(dst_dir, r["cls"])
        os.makedirs(cls_dir, exist_ok=True)
        dst = os.path.join(cls_dir, r["flat"])
        if not os.path.lexists(dst):
            os.symlink(src, dst)
            made += 1
    return made


def main():
    args = parse_args()
    df = build_frame(args.csv)
    summarise(df, "all worst-case")

    groups = df["group"].values
    y = df["cls"].values

    # 1) Hold out 15% of GROUPS as shared test (once).
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    pool_idx, test_idx = next(gss.split(df, y, groups))
    df_pool = df.iloc[pool_idx].copy()
    df_test = df.iloc[test_idx].copy()
    assert set(df_pool["group"]) & set(df_test["group"]) == set(), \
        "LEAKAGE: patient overlap between pool and test"
    print("\nHeld-out test carved (by patient):")
    summarise(df_test, "test")
    summarise(df_pool, "pool (85%)")

    # 2) StratifiedGroupKFold on the pool.
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    pool_groups = df_pool["group"].values
    pool_y = df_pool["cls"].values

    manifest = [(r["group"], r["cls"], "test", -1) for _, r in df_test.iterrows()]

    # 3) Shared test tree once.
    out = os.path.abspath(args.out)
    shared_test = os.path.join(out, "_shared_test")
    os.makedirs(shared_test, exist_ok=True)
    n = make_symlinks(df_test, args.image_root, shared_test)
    print(f"\nShared test: {n} symlinks -> {shared_test}")

    # 4) Per-fold train/val trees + test-> _shared_test link.
    seen_val_groups = set()
    for k, (tr_idx, va_idx) in enumerate(sgkf.split(df_pool, pool_y, pool_groups)):
        df_tr = df_pool.iloc[tr_idx]
        df_va = df_pool.iloc[va_idx]
        assert set(df_tr["group"]) & set(df_va["group"]) == set(), \
            f"LEAKAGE: train/val patient overlap in fold {k}"

        fold_dir = os.path.join(out, f"fold{k}")
        n_tr = make_symlinks(df_tr, args.image_root, os.path.join(fold_dir, "train"))
        n_va = make_symlinks(df_va, args.image_root, os.path.join(fold_dir, "val"))

        test_link = os.path.join(fold_dir, "test")
        if not os.path.lexists(test_link):
            os.symlink(shared_test, test_link, target_is_directory=True)

        seen_val_groups |= set(df_va["group"])
        manifest += [(r["group"], r["cls"], f"fold{k}", k) for _, r in df_va.iterrows()]
        print(f"fold{k}: train={n_tr} val={n_va} "
              f"(train patients={df_tr['group'].nunique()}, "
              f"val patients={df_va['group'].nunique()})")

    assert seen_val_groups == set(df_pool["group"]), \
        "CV COVERAGE: some pool patients never used as validation"
    print("\nCV coverage OK: every pool patient is validation exactly once.")

    # 5) Manifest.
    man = pd.DataFrame(manifest, columns=["group", "cls", "split", "fold"])
    man_path = os.path.join(out, "split_manifest.csv")
    man.to_csv(man_path, index=False)
    print(f"Manifest -> {man_path}")
    print("\nDone. Per fold, point --data_path at OUT/fold{k}.")


if __name__ == "__main__":
    main()
