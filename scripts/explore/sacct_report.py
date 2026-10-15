#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

OK = 'COMPLETED'


def main():
    ap = argparse.ArgumentParser(
        description='Wall-clock span and GPU-hours from sacct --parsable2 output.')
    ap.add_argument('--file', default='results/sacct_jobs.psv')
    a = ap.parse_args()
    p = Path(a.file)
    if not p.exists():
        raise SystemExit(f'missing {p} — see the ssh one-liner in the results doc')

    d = pd.read_csv(p, sep='|')
    d.columns = [c.lower() for c in d.columns]
    for c in ('start', 'end'):
        d[c] = pd.to_datetime(d[c], errors='coerce')
    d = d[d.start.notna() & d.end.notna()].copy()
    d['secs'] = (d.end - d.start).dt.total_seconds()
    d['base'] = d.jobid.astype(str).str.split('_').str[0]
    d['gpus'] = d.alloctres.fillna('').str.extract(r'gres/gpu=(\d+)')[0].astype(float)
    d['gpus'] = d.gpus.fillna(0)

    span = (d.end.max() - d.start.min()).total_seconds()
    serial = d.secs.sum()
    gpu_h = (d.secs * d.gpus).sum() / 3600
    done = d.state.str.startswith(OK)

    print(f'{len(d)} tasks over {d.base.nunique()} submissions, '
          f'{d.start.min():%Y-%m-%d} to {d.end.max():%Y-%m-%d}\n')
    print(f'serial compute       {serial / 3600:8.1f} h')
    print(f'calendar span        {span / 3600:8.1f} h  ({span / 86400:.1f} days)')
    print(f'GPU-hours            {gpu_h:8.1f} h')
    print(f'wasted (not {OK}) {d.loc[~done, "secs"].sum() / 3600:8.1f} h '
          f'({d.loc[~done, "secs"].sum() / max(serial, 1):.0%})\n')

    # Per-array span: last end minus first start, against the serial sum. This is
    # the real speedup — tasks queue, so they are not all concurrent.
    g = d[d.jobid.astype(str).str.contains('_')].groupby('base')
    arr = pd.DataFrame({'n': g.size(),
                        'serial_h': g.secs.sum() / 3600,
                        'span_h': (g.end.max() - g.start.min()).dt.total_seconds() / 3600,
                        'name': g.jobname.first()})
    arr = arr[arr.n >= 5].copy()
    arr['speedup'] = (arr.serial_h / arr.span_h.clip(lower=1 / 3600)).round(1)
    print('--- arrays of 5+ tasks, serial sum vs actual span ---')
    print(arr.sort_values('n', ascending=False).head(15).round(2).to_string())
    print(f'\nall array tasks: serial {arr.serial_h.sum():.1f} h, '
          f'summed spans {arr.span_h.sum():.1f} h, '
          f'overall {arr.serial_h.sum() / max(arr.span_h.sum(), 0.01):.1f}x')


if __name__ == '__main__':
    main()
