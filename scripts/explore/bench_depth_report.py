#!/usr/bin/env python3
import argparse, json
from pathlib import Path

# label -> filename tag fragment, as train_amdsd.py builds it
ARMS = {'last-4': 'last4_{size}', 'full FT': 'full_{size}',
        'frozen+attn': 'lp_{size}_attn'}
BASELINE = 'full FT'
FIELDS = [('trainable_M', 'trainable params (M)', '{:.3g}'),
          ('seconds_per_epoch', 'seconds per epoch', '{:.1f}'),
          ('train_seconds', 'training wall-clock (s)', '{:.0f}'),
          ('total_seconds', 'total job wall-clock (s)', '{:.0f}'),
          ('peak_gpu_alloc_GiB', 'peak GPU allocated (GiB)', '{:.2f}'),
          ('peak_gpu_reserved_GiB', 'peak GPU reserved (GiB)', '{:.2f}'),
          ('weights_gpu_GiB', 'weights at build (GiB)', '{:.2f}')]


def main():
    ap = argparse.ArgumentParser(
        description='Compare the cost fields of two runs written by train_amdsd.py.')
    ap.add_argument('--bench', required=True, help='directory holding the cfg_*.json')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--input-size', type=int, default=224)
    a = ap.parse_args()

    cfg = {}
    for arm, frag in ARMS.items():
        p = (Path(a.bench)
             / f'cfg_{frag.format(size=a.input_size)}_f{a.fold}_s{a.seed}.json')
        if not p.exists():
            print(f'skipping {arm}: no {p.name}')
            continue
        cfg[arm] = json.loads(p.read_text())
    if BASELINE not in cfg:
        raise SystemExit(f'need the {BASELINE} run as the baseline')

    gpus = {c.get('gpu_name') for c in cfg.values()}
    if len(gpus) > 1:
        print(f'!! different GPUs: {gpus} — the wall-clock comparison is not clean\n')
    else:
        b = cfg[BASELINE]
        print(f'{gpus.pop()}, batch {b["batch_size_run"]}, {b["epochs_run"]} epochs, '
              f'fold {a.fold} seed {a.seed}\n')

    arms = list(cfg)
    head = ''.join(f'{x:>14}' for x in arms)
    print(f'{"":<26}{head}      (change vs {BASELINE})')
    for key, label, fmt in FIELDS:
        base = cfg[BASELINE].get(key)
        cells = ''
        for arm in arms:
            v = cfg[arm].get(key)
            if v is None:
                cells += f'{"-":>14}'
            elif arm == BASELINE or not base:
                cells += f'{fmt.format(v):>14}'
            else:
                # signed change in the quantity: a reduction reads as negative
                cells += f'{fmt.format(v) + f" ({v / base - 1:+.0%})":>14}'
        print(f'{label:<26}{cells}')


if __name__ == '__main__':
    main()
