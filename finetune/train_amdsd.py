#!/usr/bin/env python3
import argparse, json, math, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from sklearn.metrics import average_precision_score
from huggingface_hub import hf_hub_download

import timm
import models_vit as models
from util.pos_embed import interpolate_pos_embed

T = ['IRF', 'SRF', 'PED']


class DS(Dataset):
    def __init__(self, df, idir, size, train):
        self.df, self.idir = df.reset_index(drop=True), Path(idir)
        aug = [transforms.RandomHorizontalFlip(),
               transforms.RandomAffine(0, translate=(0.02, 0.02), scale=(0.95, 1.05))] if train else []
        self.tf = transforms.Compose(
            [transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC)]
            + aug +
            [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = self.tf(Image.open(self.idir / r['file']).convert('RGB'))
        y = torch.tensor([r[f'label_{c}'] for c in T], dtype=torch.float32)
        v = torch.tensor([r[f'valid_{c}'] for c in T], dtype=torch.float32)
        return x, y, v


TIMM_ARCH = {'mae_in1k': 'vit_large_patch16_224.mae',
             'sup_in21k': 'vit_large_patch16_224.augreg_in21k_ft_in1k'}


def build(finetune, size, mode, device, encoder='retfound'):
    if encoder != 'retfound':
        m = timm.create_model(TIMM_ARCH[encoder], pretrained=True, img_size=size,
                              num_classes=3, global_pool='avg',
                              drop_path_rate=0.1 if mode == 'full' else 0.0)
        if mode == 'lp':
            keep = lambda n: n.startswith('head.')
        elif mode == 'last4':
            keep = lambda n: (n.startswith(('head.', 'fc_norm.', 'norm.'))
                              or any(n.startswith(f'blocks.{i}.') for i in range(20, 24)))
        else:
            keep = lambda n: True
        for n, prm in m.named_parameters():
            prm.requires_grad = keep(n)
        print(f"[{encoder}/{mode}] trainable "
              f"{sum(x.numel() for x in m.parameters() if x.requires_grad)/1e6:.1f}M")
        return m.to(device)
    return _build_retfound(finetune, size, mode, device)


def _build_retfound(finetune, size, mode, device):
    m = models.RETFound_mae(img_size=size, num_classes=3,
                            drop_path_rate=0.1 if mode == 'full' else 0.0,
                            global_pool=True)
    ck = torch.load(hf_hub_download(repo_id=f"YukunZhou/{finetune}",
                                    filename=f"{finetune}.pth"), map_location='cpu')
    sd = {k.replace('backbone.', ''): v for k, v in (ck.get('model', ck)).items()}
    for k in ('head.weight', 'head.bias'): sd.pop(k, None)
    interpolate_pos_embed(m, sd)
    msg = m.load_state_dict(sd, strict=False)
    assert not [k for k in msg.missing_keys
                if not k.startswith(('head.', 'fc_norm.'))], msg.missing_keys[:8]

    if mode == 'lp':
        keep = lambda n: n.startswith('head.')
    elif mode == 'last4':
        keep = lambda n: (n.startswith(('head.', 'fc_norm.'))
                          or any(n.startswith(f'blocks.{i}.') for i in range(20, 24)))
    else:
        keep = lambda n: True
    for n, p in m.named_parameters():
        p.requires_grad = keep(n)
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"[{mode}] trainable {n_tr/1e6:.1f}M")
    return m.to(device)


@torch.no_grad()
def infer(m, dl, dev):
    m.eval(); P, Y = [], []
    for x, y, _ in dl:
        with torch.cuda.amp.autocast():
            P.append(torch.sigmoid(m(x.to(dev, non_blocking=True))).float().cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--mode', choices=['lp', 'last4', 'full'], required=True)
    ap.add_argument('--encoder', default='retfound',
                    choices=['retfound', 'mae_in1k', 'sup_in21k'])
    ap.add_argument('--finetune', default='RETFound_mae_natureOCT')
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-model', action='store_true')
    a = ap.parse_args()

    lr = a.lr or {'lp': 1e-3, 'last4': 1e-4, 'full': 2e-5}[a.mode]
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device('cuda')

    df = pd.read_csv(a.manifest)
    tr = df[(df.split == 'pool') & (df.fold != a.fold)]
    va = df[(df.split == 'pool') & (df.fold == a.fold)]
    te = df[df.split == 'test']
    print(f"fold {a.fold} | train {len(tr)} val {len(va)} test {len(te)}")

    mk = lambda d, t: DataLoader(DS(d, a.images, a.input_size, t),
                                 batch_size=a.batch_size, shuffle=t,
                                 num_workers=8, pin_memory=True, drop_last=t)
    dl_tr, dl_va, dl_te = mk(tr, True), mk(va, False), mk(te, False)

    m = build(a.finetune, a.input_size, a.mode, dev, a.encoder)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler()
    crit = nn.BCEWithLogitsLoss(reduction='none')

    best, best_ep, best_state = -1, -1, None
    for ep in range(a.epochs):
        # cosine with warmup
        if ep < a.warmup: cur = lr * (ep + 1) / a.warmup
        else: cur = 1e-6 + (lr - 1e-6) * 0.5 * (
            1 + math.cos(math.pi * (ep - a.warmup) / max(a.epochs - a.warmup, 1)))
        for g in opt.param_groups: g['lr'] = cur

        m.train(); tot, n = 0.0, 0
        for x, y, v in dl_tr:
            x, y, v = x.to(dev, non_blocking=True), y.to(dev), v.to(dev)
            with torch.cuda.amp.autocast():
                loss = (crit(m(x), y) * v).sum() / v.sum().clamp(min=1)
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item() * len(x); n += len(x)

        P, Y = infer(m, dl_va, dev)
        aps = [average_precision_score(Y[:, i], P[:, i]) for i in range(3)]
        mAP = float(np.mean(aps))
        print(f"ep{ep:>2} lr {cur:.2e} loss {tot/n:.4f} | val mAP {mAP:.4f} "
              f"({' '.join(f'{c}={v:.3f}' for c, v in zip(T, aps))})", flush=True)
        if mAP > best:
            best, best_ep = mAP, ep
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    print(f"[best] epoch {best_ep} val mAP {best:.4f}")
    m.load_state_dict(best_state)
    Pv, Yv = infer(m, dl_va, dev)
    Pt, Yt = infer(m, dl_te, dev)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pre = "" if a.encoder == "retfound" else f"{a.encoder}_"
    tag = f"{pre}{a.mode}_{a.input_size}_f{a.fold}_s{a.seed}"
    np.savez(out / f"preds_{tag}.npz",
             val_p=Pv, val_y=Yv, val_g=va.group.values,
             test_p=Pt, test_y=Yt, test_g=te.group.values,
             classes=np.array(T), best_epoch=best_ep, val_mAP=best)
    (out / f"cfg_{tag}.json").write_text(json.dumps(vars(a) | {'lr': lr}, indent=2))
    if a.save_model:
        torch.save({"model": best_state, "epoch": best_ep, "cfg": vars(a) | {"lr": lr}},
                   out / f"model_{tag}.pth")
        print(f"-> {out}/model_{tag}.pth")
    print(f"-> {out}/preds_{tag}.npz")


if __name__ == '__main__':
    main()
