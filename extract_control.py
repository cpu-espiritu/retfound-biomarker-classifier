#!/usr/bin/env python3
"""Cache frozen features from non-RETFound ViT-L controls."""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, timm, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

ARCH = {
    'mae_in1k':  'vit_large_patch16_224.mae',
    'sup_in21k': 'vit_large_patch16_224.augreg_in21k_ft_in1k',
}


class Scans(Dataset):
    def __init__(self, df, idir, size):
        self.df, self.idir = df.reset_index(drop=True), Path(idir)
        self.tf = transforms.Compose([
            transforms.Resize((size, size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        return self.tf(Image.open(self.idir / self.df.iloc[i]['file']).convert('RGB')), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--encoder', required=True, choices=list(ARCH))
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv(a.manifest)
    dl = DataLoader(Scans(df, a.images, a.input_size), batch_size=a.batch_size,
                    shuffle=False, num_workers=a.workers, pin_memory=True)

    m = timm.create_model(ARCH[a.encoder], pretrained=True,
                          img_size=a.input_size, num_classes=0).eval().to(dev)
    print(f"[model] {ARCH[a.encoder]}  params {sum(p.numel() for p in m.parameters())/1e6:.0f}M")

    F, order, t0 = [], [], time.time()
    with torch.no_grad():
        for n, (x, idx) in enumerate(dl):
            with torch.cuda.amp.autocast():
                tok = m.forward_features(x.to(dev, non_blocking=True))
            # mean over patch tokens, matching RETFound global_pool
            f = tok[:, m.num_prefix_tokens:, :].mean(dim=1)
            F.append(f.float().cpu().numpy()); order.append(idx.numpy())
            if (n + 1) % 10 == 0:
                print(f"  {(n+1)*a.batch_size}/{len(df)}  {time.time()-t0:.0f}s", flush=True)

    F = np.concatenate(F)
    assert np.array_equal(np.concatenate(order), np.arange(len(df))), "order broke"
    assert F.shape == (len(df), 1024), F.shape
    assert np.isfinite(F).all()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.encoder}_{a.input_size}"
    np.save(out / f"features_{tag}.npy", F.astype(np.float32))
    df.to_csv(out / f"manifest_{tag}.csv", index=False)
    print(f"[done] {F.shape} in {time.time()-t0:.0f}s -> features_{tag}.npy")
    print(f"       norm mean={np.linalg.norm(F,axis=1).mean():.1f} std={F.std():.3f}")


if __name__ == '__main__':
    main()
