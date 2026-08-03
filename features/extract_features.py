#!/usr/bin/env python3
"""Cache frozen RETFound features for AMD-SD, driven by manifest.csv."""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from huggingface_hub import hf_hub_download

import models_vit as models
from util.pos_embed import interpolate_pos_embed


class Scans(Dataset):
    def __init__(self, df, idir, size, crop):
        self.df, self.idir, self.crop = df.reset_index(drop=True), Path(idir), crop
        self.tf = transforms.Compose([
            transforms.Resize((size, size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(self.idir / r['file']).convert('RGB')
        if self.crop:
            im = im.crop((0, int(r['crop_top']), im.width, int(r['crop_bot'])))
        return self.tf(im), i


def build(finetune, size, device):
    m = models.RETFound_mae(img_size=size, num_classes=3,
                            drop_path_rate=0.0, global_pool=True)
    ck = torch.load(hf_hub_download(repo_id=f"YukunZhou/{finetune}",
                                    filename=f"{finetune}.pth"), map_location='cpu')
    sd = ck['model'] if 'model' in ck else ck
    sd = {k.replace('backbone.', ''): v for k, v in sd.items()}
    for k in ('head.weight', 'head.bias'):
        sd.pop(k, None)
    interpolate_pos_embed(m, sd)
    msg = m.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    assert not [k for k in msg.missing_keys if not k.startswith(('head.', 'fc_norm.'))], \
        f"backbone weights missing: {msg.missing_keys[:8]}"
    return m.eval().to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--finetune', default='RETFound_mae_natureOCT')
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--crop', action='store_true')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[dev] {dev}")

    df = pd.read_csv(a.manifest)
    if a.crop and 'crop_top' not in df.columns:
        raise SystemExit("manifest has no crop columns — use the --crop manifest")
    dl = DataLoader(Scans(df, a.images, a.input_size, a.crop),
                    batch_size=a.batch_size, shuffle=False,
                    num_workers=a.workers, pin_memory=True)

    model = build(a.finetune, a.input_size, dev)
    feats, order, t0 = [], [], time.time()
    with torch.no_grad():
        for n, (x, idx) in enumerate(dl):
            with torch.cuda.amp.autocast():
                f = model.forward_features(x.to(dev, non_blocking=True))
            feats.append(f.squeeze(1).float().cpu().numpy())
            order.append(idx.numpy())
            if (n + 1) % 10 == 0:
                print(f"  {(n+1)*a.batch_size}/{len(df)}  {time.time()-t0:.0f}s", flush=True)

    F = np.concatenate(feats)
    assert np.array_equal(np.concatenate(order), np.arange(len(df))), "order broke"
    assert F.shape == (len(df), 1024), F.shape
    assert np.isfinite(F).all(), "non-finite features"

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.finetune}_{a.input_size}{'_crop' if a.crop else ''}"
    np.save(out / f"features_{tag}.npy", F.astype(np.float32))
    df.to_csv(out / f"manifest_{tag}.csv", index=False)
    print(f"[done] {F.shape} in {time.time()-t0:.0f}s -> {out}/features_{tag}.npy")
    print(f"       norm mean={np.linalg.norm(F,axis=1).mean():.1f} std={F.std():.3f}")


if __name__ == '__main__':
    main()
