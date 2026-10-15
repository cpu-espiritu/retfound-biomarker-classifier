import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ROW = re.compile(r'^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$')


def parse(path):
    """One bench_latency.sh log -> rows of (device, pool, batch, ms/pass, ms/scan, scans/s).

    The script only prints, so this is how its numbers reach results/. The device and
    pooling are read from each block's own header rather than assumed from the job.
    """
    rows, dev, pool, threads = [], None, None, None
    for line in Path(path).read_text().splitlines():
        if 'A100' in line or line.startswith('NVIDIA'):
            dev, threads = line.split('|')[0].strip(), None
            pool = line.split('|')[1].strip().split()[0]
        elif line.startswith('CPU,'):
            dev = 'CPU'
            threads = int(re.search(r'(\d+) threads', line).group(1))
            pool = line.split('|')[1].strip().split()[0]
        else:
            m = ROW.match(line)
            if m and dev:
                rows.append(dict(log=Path(path).name, device=dev, threads=threads,
                                 pool=pool, batch=int(m.group(1)),
                                 ms_per_pass=float(m.group(2)),
                                 ms_per_scan=float(m.group(3)),
                                 scans_per_s=float(m.group(4))))
    return rows


def main():
    ap = argparse.ArgumentParser(description='Collect bench_latency.sh logs into one CSV.')
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--out', default=str(ROOT / 'results/bench_latency.csv'))
    a = ap.parse_args()
    D = pd.DataFrame([r for f in a.logs for r in parse(f)])
    D.to_csv(a.out, index=False)
    print(D.to_string(index=False))
    print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
