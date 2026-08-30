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
FULL_POOL_PATIENTS = 118


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


class AttnPoolViT(torch.nn.Module):
    """RETFound trunk with attention pooling replacing the mean over patch tokens.

        e_t = w . tanh(V h_t)      a = softmax(e)      z = sum_t a_t h_t

    The trunk is the unmodified ViT; only the pooling step differs, so this is a
    like-for-like swap against global_pool=True.
    """

    def __init__(self, vit, L=64):
        super().__init__()
        self.vit = vit
        d = vit.pos_embed.shape[-1]
        self.V = torch.nn.Linear(d, L, bias=False)
        self.w = torch.nn.Linear(L, 1, bias=False)
        torch.nn.init.normal_(self.V.weight, std=0.02)
        torch.nn.init.normal_(self.w.weight, std=0.02)

    def forward(self, x):
        v = self.vit
        x = v.patch_embed(x)
        cls = v.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1) + v.pos_embed
        x = v.pos_drop(x)
        for blk in v.blocks:
            x = blk(x)
        tok = x[:, 1:, :]                                   # (B, N, D)
        a = torch.softmax(self.w(torch.tanh(self.V(tok))).squeeze(-1), dim=1)
        z = torch.einsum('bn,bnd->bd', a, tok)
        return v.head(v.fc_norm(z))

    def attention(self, x):
        """Per-token weights, for localisation checks."""
        v = self.vit
        h = v.pos_drop(torch.cat((v.cls_token.expand(x.shape[0], -1, -1),
                                  v.patch_embed(x)), dim=1) + v.pos_embed)
        for blk in v.blocks:
            h = blk(h)
        return torch.softmax(
            self.w(torch.tanh(self.V(h[:, 1:, :]))).squeeze(-1), dim=1)


def build(finetune, size, mode, device, encoder='retfound', pool='mean'):
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
    return _build_retfound(finetune, size, mode, device, pool)


def _build_retfound(finetune, size, mode, device, pool='mean'):
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
    if pool == 'attn':
        # freezing is applied to the trunk first, then the pooling head is added
        # trainable on top, so `mode` keeps its usual meaning
        m = AttnPoolViT(m)
        for prm in list(m.V.parameters()) + list(m.w.parameters()):
            prm.requires_grad = True
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"[{mode}/{pool}] trainable {n_tr/1e6:.1f}M")
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
    ap.add_argument('--pool', default='mean', choices=['mean', 'attn'],
                    help="'attn' replaces global mean pooling over patch tokens "
                         "with learned attention pooling (RETFound encoder only)")
    ap.add_argument('--encoder', default='retfound',
                    choices=['retfound', 'mae_in1k', 'sup_in21k'])
    ap.add_argument('--finetune', default='RETFound_mae_natureOCT')
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-train-patients', type=int, default=None,
                    help='restrict the pool split to this many patients, sampled by '
                         'patient so every scan of a sampled patient is kept; the test '
                         'split is never touched')
    ap.add_argument('--subsets', default=None,
                    help='size_subsets.csv from scripts/prep/make_size_subsets.py')
    ap.add_argument('--subset-seed', type=int, default=None,
                    help='which subsample draw to use (default: --seed)')
    ap.add_argument('--epoch-budget', choices=['steps', 'fixed'], default='steps',
                    help="'steps' scales epochs and warmup by 118/n so every training "
                         "set size gets a comparable number of gradient steps; 'fixed' "
                         "keeps --epochs, so small sizes take proportionally fewer")
    ap.add_argument('--save-model', action='store_true')
    a = ap.parse_args()

    lr = a.lr or {'lp': 1e-3, 'last4': 1e-4, 'full': 2e-5}[a.mode]
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device('cuda')

    df = pd.read_csv(a.manifest)
    pool = df.split == 'pool'
    epochs, warmup = a.epochs, a.warmup
    if a.n_train_patients is not None:
        if not a.subsets:
            raise SystemExit('--n-train-patients needs --subsets')
        S = pd.read_csv(a.subsets)
        ss = a.seed if a.subset_seed is None else a.subset_seed
        keep = S[(S.n_patients == a.n_train_patients) & (S.seed == ss)]
        if keep.empty:
            raise SystemExit(f'no subset for n={a.n_train_patients} seed={ss} '
                             f'in {a.subsets}')
        pool &= df.group.isin(keep.group.values)
        if a.epoch_budget == 'steps':
            # the tuned recipe is 20 epochs at the full 118-patient pool; hold the
            # gradient-step count roughly there so a small-n arm is not simply
            # undertrained, and keep the warmup fraction of the schedule intact
            f = FULL_POOL_PATIENTS / a.n_train_patients
            epochs, warmup = max(1, round(a.epochs * f)), max(1, round(a.warmup * f))
    tr = df[pool & (df.fold != a.fold)]
    va = df[pool & (df.fold == a.fold)]
    te = df[df.split == 'test']
    print(f"fold {a.fold} | train {len(tr)} val {len(va)} test {len(te)} | "
          f"{tr.group.nunique()} train patients | {epochs} epochs warmup {warmup}")
    if not len(tr) or not len(va):
        raise SystemExit('empty train or val split after subsampling')

    mk = lambda d, t: DataLoader(DS(d, a.images, a.input_size, t),
                                 batch_size=a.batch_size, shuffle=t,
                                 num_workers=8, pin_memory=True, drop_last=t)
    dl_tr, dl_va, dl_te = mk(tr, True), mk(va, False), mk(te, False)

    if a.pool == 'attn' and a.encoder != 'retfound':
        raise SystemExit('--pool attn is implemented for the RETFound trunk only')
    m = build(a.finetune, a.input_size, a.mode, dev, a.encoder, a.pool)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler()
    crit = nn.BCEWithLogitsLoss(reduction='none')

    best, best_ep, best_state = -1, -1, None
    for ep in range(epochs):
        # cosine with warmup
        if ep < warmup: cur = lr * (ep + 1) / warmup
        else: cur = 1e-6 + (lr - 1e-6) * 0.5 * (
            1 + math.cos(math.pi * (ep - warmup) / max(epochs - warmup, 1)))
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
    # an explicit --lr gets its own tag, so a sweep cannot overwrite the default run
    lrtag = "" if a.lr is None else f"_lr{a.lr:g}"
    pooltag = "" if a.pool == "mean" else f"_{a.pool}"
    # a size-curve run always carries its own tag, so it can never overwrite the
    # full-pool runs that back the numbers in RESULTS.md — not even at n=118, where
    # the config happens to be identical
    ntag = "" if a.n_train_patients is None else f"_n{a.n_train_patients}"
    tag = f"{pre}{a.mode}_{a.input_size}{pooltag}{ntag}{lrtag}_f{a.fold}_s{a.seed}"
    np.savez(out / f"preds_{tag}.npz",
             val_p=Pv, val_y=Yv, val_g=va.group.values,
             test_p=Pt, test_y=Yt, test_g=te.group.values,
             classes=np.array(T), best_epoch=best_ep, val_mAP=best)
    # record the exact code version, so a run can always be traced to a commit
    try:
        import subprocess
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'],
                             cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain'],
                                    cwd=Path(__file__).resolve().parent,
                                    capture_output=True, text=True,
                                    timeout=10).stdout.strip())
    except Exception:
        sha, dirty = '', None
    (out / f"cfg_{tag}.json").write_text(json.dumps(
        vars(a) | {'lr': lr, 'epochs_run': epochs, 'warmup_run': warmup,
                   'n_train_patients_actual': int(tr.group.nunique()),
                   'git_sha': sha, 'git_dirty': dirty}, indent=2))
    if a.save_model:
        torch.save({"model": best_state, "epoch": best_ep, "cfg": vars(a) | {"lr": lr}},
                   out / f"model_{tag}.pth")
        print(f"-> {out}/model_{tag}.pth")
    print(f"-> {out}/preds_{tag}.npz")


if __name__ == '__main__':
    main()
