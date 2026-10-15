#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'finetune'))
from train_amdsd import build                                    # noqa: E402


def timed(m, dev, size, batch, reps, warmup):
    """Median seconds per forward pass over `reps`, after discarding `warmup`."""
    x = torch.randn(batch, 3, size, size, device=dev)
    amp = dev.type == 'cuda'
    ts = []
    with torch.no_grad():
        for i in range(warmup + reps):
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=amp):
                m(x)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            if i >= warmup:
                ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser(
        description='Inference latency per scan. Random tensors, so no dataset or '
                    'trained weights are needed — architecture sets the cost.')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--mode', default='full', choices=['lp', 'last4', 'full'],
                    help='only affects requires_grad, not inference cost')
    ap.add_argument('--pool', default='mean', choices=['mean', 'attn'])
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--batches', type=int, nargs='*', default=[1, 8, 32])
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--finetune', default='RETFound_mae_natureOCT')
    a = ap.parse_args()

    dev = torch.device(a.device)
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('no CUDA here — pass --device cpu')
    m = build(a.finetune, a.input_size, a.mode, dev, 'retfound', a.pool).eval()

    name = (torch.cuda.get_device_name(dev) if a.device == 'cuda'
            else f'CPU, {torch.get_num_threads()} threads')
    print(f'{name} | {a.pool} pooling | {a.input_size}px | '
          f'median of {a.reps} passes after {a.warmup} warmup\n')
    print(f'{"batch":>6}{"per pass (ms)":>16}{"per scan (ms)":>16}{"scans/s":>11}')
    for b in a.batches:
        s = timed(m, dev, a.input_size, b, a.reps, a.warmup)
        print(f'{b:>6}{s * 1e3:>16.1f}{s / b * 1e3:>16.2f}{b / s:>11.1f}')


if __name__ == '__main__':
    main()
