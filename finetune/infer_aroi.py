#!/usr/bin/env python3
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import models_vit as models

T = ['IRF', 'SRF', 'PED']          # class order the AMD-SD head was trained in
AROI_IDX = {'IRF': 7, 'SRF': 6, 'PED': 5}      # AROI mask indices


class Slices(Dataset):
    def __init__(self, paths, size):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.Resize((size, size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert('RGB')), i


def build(ckpt, size, device):
    m = models.RETFound_mae(img_size=size, num_classes=3, drop_path_rate=0.0,
                            global_pool=True)
    sd = torch.load(ckpt, map_location='cpu')
    sd = sd['model'] if 'model' in sd else sd
    msg = m.load_state_dict(sd, strict=False)
    assert not msg.missing_keys, f'missing weights: {msg.missing_keys[:8]}'
    return m.eval().to(device)


def main():
    ap = argparse.ArgumentParser(
        description='Zero-shot: run the AMD-SD-trained model over AROI annotated '
                    'slices. Writes predictions and mask areas; no image data.')
    ap.add_argument('--aroi', required=True, help='the "24 patient" directory')
    ap.add_argument('--checkpoints', required=True,
                    help="glob, e.g. '.../model_last4_224_f*_s0.pth'")
    ap.add_argument('--out', default='aroi_zeroshot.csv')
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    root = Path(a.aroi)
    pats = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith('patient')],
                  key=lambda p: int(re.sub(r'\D', '', p.name)))

    rows, raw_paths = [], []
    for p in pats:
        for mf in sorted((p / 'mask' / 'number').glob('*.png')):
            rf = p / 'raw' / 'ALL' / mf.name
            if not rf.exists():
                print(f'[warn] no raw slice for {mf.name}')
                continue
            m = np.array(Image.open(mf).convert('L'))
            r = dict(patient=p.name,
                     slice=int(re.search(r'raw(\d+)', mf.stem).group(1)),
                     file=mf.name)
            for c in T:
                r[f'area_{c}'] = int((m == AROI_IDX[c]).sum())
                r[f'label_{c}'] = int(r[f'area_{c}'] > 0)
            rows.append(r)
            raw_paths.append(rf)
    df = pd.DataFrame(rows)
    print(f'{len(df)} annotated slices, {df.patient.nunique()} patients')
    print('  positives: ' + '  '.join(f'{c} {int(df[f"label_{c}"].sum())}' for c in T))

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl = DataLoader(Slices(raw_paths, a.input_size), batch_size=a.batch_size,
                    shuffle=False, num_workers=a.workers, pin_memory=True)

    ckpts = sorted(glob.glob(a.checkpoints))
    assert ckpts, f'no checkpoints matched {a.checkpoints}'
    print(f'{len(ckpts)} folds on {dev}')

    per_fold = []
    for ci, ck in enumerate(ckpts):
        model = build(ck, a.input_size, dev)
        out, order = [], []
        with torch.no_grad():
            for x, idx in dl:
                with torch.cuda.amp.autocast(enabled=dev.type == 'cuda'):
                    out.append(torch.sigmoid(model(x.to(dev))).float().cpu().numpy())
                order.append(idx.numpy())
        assert np.array_equal(np.concatenate(order), np.arange(len(df))), 'order broke'
        per_fold.append(np.concatenate(out))
        print(f'  fold {ci} done')
        del model
        torch.cuda.empty_cache()

    P = np.mean(per_fold, axis=0)
    for i, c in enumerate(T):
        df[f'p_{c}'] = P[:, i]
        for f_ in range(len(per_fold)):
            df[f'p_{c}_f{f_}'] = per_fold[f_][:, i]

    df.to_csv(a.out, index=False)
    print(f'\n-> {a.out}')
    print(f"\n{'':6}{'positives':>11}{'mean p (pos)':>14}{'mean p (neg)':>14}")
    for c in T:
        pos = df[df[f'label_{c}'] == 1][f'p_{c}']
        neg = df[df[f'label_{c}'] == 0][f'p_{c}']
        print(f'{c:<6}{len(pos):>11}{pos.mean():>14.3f}'
              f'{(neg.mean() if len(neg) else float("nan")):>14.3f}')
    print('\nAROI SRF includes SHRM; AMD-SD SRF does not. Not a like-for-like class.')


if __name__ == '__main__':
    main()
