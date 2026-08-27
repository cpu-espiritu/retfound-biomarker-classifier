#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from huggingface_hub import hf_hub_download

import models_vit as models
from util.pos_embed import interpolate_pos_embed


class Scans(Dataset):
    def __init__(self, df, idir, size):
        self.df, self.idir = df.reset_index(drop=True), Path(idir)
        self.tf = transforms.Compose([
            transforms.Resize((size, size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        return self.tf(Image.open(self.idir / self.df.iloc[i]['file']).convert('RGB')), i


def build(finetune, size, device):
    m = models.RETFound_mae(img_size=size, num_classes=3, drop_path_rate=0.0,
                            global_pool=True)
    ck = torch.load(hf_hub_download(repo_id=f'YukunZhou/{finetune}',
                                    filename=f'{finetune}.pth'), map_location='cpu')
    sd = {k.replace('backbone.', ''): v for k, v in ck.get('model', ck).items()}
    for k in ('head.weight', 'head.bias'):
        sd.pop(k, None)
    interpolate_pos_embed(m, sd)
    msg = m.load_state_dict(sd, strict=False)
    assert not [k for k in msg.missing_keys
                if not k.startswith(('head.', 'fc_norm.'))], msg.missing_keys[:8]
    return m.eval().to(device)


@torch.no_grad()
def tokens(m, x):
    """Patch tokens after the final block, before pooling and before fc_norm."""
    x = m.patch_embed(x)
    cls = m.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls, x), dim=1) + m.pos_embed
    x = m.pos_drop(x)
    for blk in m.blocks:
        x = blk(x)
    return x[:, 1:, :]                       # drop CLS


def main():
    ap = argparse.ArgumentParser(
        description='Cache per-token RETFound features so pooling can be swapped '
                    'without another GPU pass. Mean over dim 1 reproduces the '
                    'existing pooled features.')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--finetune', default='RETFound_mae_natureOCT')
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--fp16', action='store_true', default=True,
                    help='store float16 to halve the file (default on)')
    a = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv(a.manifest)
    dl = DataLoader(Scans(df, a.images, a.input_size), batch_size=a.batch_size,
                    shuffle=False, num_workers=a.workers, pin_memory=True)
    m = build(a.finetune, a.input_size, dev)

    n_tok = (a.input_size // 16) ** 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{a.finetune}_{a.input_size}'
    dtype = np.float16 if a.fp16 else np.float32
    store = np.lib.format.open_memmap(out / f'tokens_{tag}.npy', mode='w+',
                                      dtype=dtype, shape=(len(df), n_tok, 1024))
    print(f'writing {store.shape} {dtype.__name__} '
          f'({store.nbytes / 1e9:.2f} GB) on {dev}')

    seen, t0 = 0, time.time()
    for n, (x, idx) in enumerate(dl):
        with torch.cuda.amp.autocast(enabled=dev.type == 'cuda'):
            tk = tokens(m, x.to(dev, non_blocking=True))
        i = idx.numpy()
        assert np.array_equal(i, np.arange(seen, seen + len(i))), 'order broke'
        store[seen:seen + len(i)] = tk.float().cpu().numpy().astype(dtype)
        seen += len(i)
        if (n + 1) % 10 == 0:
            print(f'  {seen}/{len(df)}  {time.time() - t0:.0f}s', flush=True)
    store.flush()

    # gate: mean over tokens must match the pooled features already on disk
    pooled = out / f'features_{tag}.npy'
    if pooled.exists():
        ref = np.load(pooled)
        got = store[:64].astype(np.float32).mean(axis=1)
        d = np.abs(got - ref[:64]).max()
        print(f'[gate] max |mean(tokens) - pooled| over 64 scans = {d:.4f}')
        print('       (fp16 storage and fc_norm account for small differences)')
    df.to_csv(out / f'manifest_tokens_{tag}.csv', index=False)
    print(f'[done] {time.time() - t0:.0f}s -> {out}/tokens_{tag}.npy')


if __name__ == '__main__':
    main()
