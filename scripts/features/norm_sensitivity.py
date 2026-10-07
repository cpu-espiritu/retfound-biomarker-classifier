import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frozen_arms import fold_models, youden

ROOT = Path(__file__).resolve().parents[2]
T = ['IRF', 'SRF', 'PED']
EPS = 1e-6


def load_norm(finetune, norm_file):
    """RETFound's trained final LayerNorm, as (weight, bias)."""
    if norm_file:
        z = np.load(norm_file)
        return z['weight'].astype(np.float32), z['bias'].astype(np.float32)
    import torch
    from huggingface_hub import hf_hub_download
    ck = torch.load(hf_hub_download(repo_id=f'YukunZhou/{finetune}',
                                    filename=f'{finetune}.pth'), map_location='cpu')
    sd = {k.replace('backbone.', ''): v for k, v in (ck.get('model', ck)).items()}
    miss = [k for k in ('norm.weight', 'norm.bias') if k not in sd]
    if miss:
        raise SystemExit(f'checkpoint has no {miss}; keys like: '
                         f'{[k for k in sd if "norm" in k][:8]}')
    return (sd['norm.weight'].numpy().astype(np.float32),
            sd['norm.bias'].numpy().astype(np.float32))


def apply_norm(src, dst, w, b, chunk=128):
    """Per-token LayerNorm with the pretrained affine, written to a memmap.

    This is the operation the encoder itself applies to the token sequence before
    pooling; the cache in extract_tokens.py deliberately stops short of it. Applying
    it changes the attention weights as well as the pooled vector, since the weights
    are computed from the tokens.
    """
    X = np.load(src, mmap_mode='r')
    out = np.lib.format.open_memmap(dst, mode='w+', dtype=np.float32, shape=X.shape)
    for i in range(0, len(X), chunk):
        z = np.asarray(X[i:i + chunk], np.float32)
        mu = z.mean(-1, keepdims=True)
        sd = z.std(-1, keepdims=True)
        out[i:i + chunk] = (z - mu) / (sd + EPS) * w + b
    out.flush()
    return dst


def sanity(images, manifest, finetune, size, normed, n=5):
    """The normed cache must match the encoder run with its own `norm` intact."""
    import torch
    from PIL import Image
    from torchvision import transforms
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from huggingface_hub import hf_hub_download
    import models_vit as models
    from util.pos_embed import interpolate_pos_embed

    man = pd.read_csv(manifest).head(n)
    m = models.RETFound_mae(img_size=size, num_classes=3, global_pool=True)
    ck = torch.load(hf_hub_download(repo_id=f'YukunZhou/{finetune}',
                                    filename=f'{finetune}.pth'), map_location='cpu')
    sd = {k.replace('backbone.', ''): v for k, v in (ck.get('model', ck)).items()}
    for k in ('head.weight', 'head.bias'):
        sd.pop(k, None)
    interpolate_pos_embed(m, sd)
    m.load_state_dict(sd, strict=False)
    ln = torch.nn.LayerNorm(sd['norm.weight'].shape[0], eps=EPS)
    ln.weight.data, ln.bias.data = sd['norm.weight'], sd['norm.bias']
    m.eval()

    tf = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
    X = np.load(normed, mmap_mode='r')
    worst = 0.0
    with torch.no_grad():
        for i, f in enumerate(man.file):
            x = tf(Image.open(Path(images) / f).convert('RGB'))[None]
            h = m.patch_embed(x)
            h = torch.cat((m.cls_token.expand(1, -1, -1), h), dim=1) + m.pos_embed
            h = m.pos_drop(h)
            for blk in m.blocks:
                h = blk(h)
            ref = ln(h[:, 1:, :])[0].numpy()
            worst = max(worst, float(np.abs(ref - np.asarray(X[i], np.float32)).max()))
    print(f'[sanity] max |encoder-with-norm - normed cache| over {n} scans = {worst:.5f}')
    return worst


def run_arms(X, df, pool, test, modal, seeds):
    """Both frozen arms under one token representation. Returns per-class preds."""
    P, rows = {}, []
    G = df.loc[test, 'group'].values
    for how in ('mean', 'attn'):
        for c in T:
            y = df[f'label_{c}'].values
            for s in (seeds if how == 'attn' else [seeds[0]]):
                tp, vp, vy = fold_models(X, df, pool, test, y, how, modal[c], s)
                ens = tp.mean(0)
                P[(how, c, s)] = ens
                rows.append(dict(pool=how, cls=c, seed=s, auprc=AP(y[test], ens),
                                 thr=youden(vy, vp)))
                print(f'    {how:<5}{c:<5}s{s}  AUPRC {rows[-1]["auprc"]:.4f}', flush=True)
    return P, pd.DataFrame(rows), G


def paired(y, A, B, reps):
    d = np.array([AP(y[i], A[i]) - AP(y[i], B[i]) for i in reps
                  if y[i].min() != y[i].max()])
    lo, hi = np.percentile(d, [2.5, 97.5])
    n = len(d)
    p = min(2 * min((1 + (d <= 0).sum()) / (n + 1), (1 + (d >= 0).sum()) / (n + 1)), 1.0)
    return AP(y, A) - AP(y, B), lo, hi, p


def main():
    ap = argparse.ArgumentParser(
        description="Sensitivity of the frozen arms to RETFound's pretrained final "
                    "norm. The cached tokens stop before it and both arms normalise "
                    "only after pooling; this adds the per-token pretrained norm and "
                    "re-measures. Sensitivity analysis only — the parameter-free "
                    "setting stays primary.")
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--selected', required=True, help='attn_tuned.csv')
    ap.add_argument('--finetune', default='RETFound_mae_natureOCT')
    ap.add_argument('--norm-file', default=None,
                    help='npz with weight/bias, if the checkpoint is not reachable')
    ap.add_argument('--work', default=None, help='where to write the normed cache')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--boot', type=int, default=5000)
    ap.add_argument('--sanity', type=int, default=5, help='0 to skip')
    ap.add_argument('--images', default=None)
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--out', default=str(ROOT / 'results/norm_sensitivity.csv'))
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(',')]
    df = pd.read_csv(a.manifest)
    sel = pd.read_csv(a.selected)
    modal = {c: {k: sel[sel.cls == c][k].mode().iloc[0]
                 for k in ('L', 'lr', 'wd', 'epochs')} for c in T}
    pool = (df.split == 'pool').values
    test = (df.split == 'test').values

    w, b = load_norm(a.finetune, a.norm_file)
    print(f'pretrained norm: weight mean {w.mean():.3f} sd {w.std():.3f} '
          f'range [{w.min():.3f}, {w.max():.3f}]; bias mean {b.mean():.3f}')

    work = Path(a.work or Path(a.tokens).parent) / 'tokens_prenormed.npy'
    if not work.exists():
        print(f'writing {work} ...', flush=True)
        apply_norm(a.tokens, str(work), w, b)
    else:
        print(f'reusing {work}')

    if a.sanity and a.images:
        sanity(a.images, a.manifest, a.finetune, a.input_size, str(work), a.sanity)
    elif a.sanity:
        print('[sanity] skipped: --images not given')

    out, preds = [], {}
    for name, path in (('parameter-free', a.tokens), ('pretrained', str(work))):
        print(f'\n=== {name} ===', flush=True)
        X = np.load(path, mmap_mode='r')
        P, R, G = run_arms(X, df, pool, test, modal, seeds)
        preds[name] = P
        out.append(R.assign(tokens=name))
        del X
    R = pd.concat(out, ignore_index=True)

    ug = np.unique(G)
    gi = {u: np.flatnonzero(G == u) for u in ug}
    rng = np.random.default_rng(0)
    reps = [np.concatenate([gi[u] for u in rng.choice(ug, len(ug))])
            for _ in range(a.boot)]

    tests = []
    for c in T:
        y = df[f'label_{c}'].values[test]
        for name in ('parameter-free', 'pretrained'):
            A = np.mean([preds[name][('attn', c, s)] for s in seeds], axis=0)
            B = preds[name][('mean', c, seeds[0])]
            d, lo, hi, p = paired(y, A, B, reps)
            tests.append(dict(contrast='attention - mean', tokens=name, cls=c,
                              delta=d, lo=lo, hi=hi, p=p))
        for how in ('attn', 'mean'):
            A = (np.mean([preds['pretrained'][(how, c, s)] for s in seeds], axis=0)
                 if how == 'attn' else preds['pretrained'][(how, c, seeds[0])])
            B = (np.mean([preds['parameter-free'][(how, c, s)] for s in seeds], axis=0)
                 if how == 'attn' else preds['parameter-free'][(how, c, seeds[0])])
            d, lo, hi, p = paired(y, A, B, reps)
            tests.append(dict(contrast=f'pretrained - parameter-free ({how})',
                              tokens='both', cls=c, delta=d, lo=lo, hi=hi, p=p))
    Tst = pd.DataFrame(tests)

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([R.assign(kind='auprc'), Tst.assign(kind='contrast')],
              ignore_index=True).to_csv(outp, index=False)

    print('\n=== AUPRC, mean over seeds ===')
    print(R.pivot_table(index='cls', columns=['tokens', 'pool'],
                        values='auprc').round(4).to_string())
    print('\n=== contrasts (no Holm; sensitivity analysis) ===')
    print(Tst[['contrast', 'tokens', 'cls', 'delta', 'lo', 'hi', 'p']]
          .round(4).to_string(index=False))
    print(f'\n-> {outp}')


if __name__ == '__main__':
    main()
