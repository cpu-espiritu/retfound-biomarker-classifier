"""Loading and drawing helpers shared by the figure scripts.

Every figure script reads CSVs from results/ and writes one PDF into
analysis/output/paper/. Nothing here recomputes a number: if a value is not in a
CSV, it belongs in analysis/derive.py instead.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import style as S                                          # noqa: E402

RESULTS = S.ROOT / 'results'
T = S.CLASSES


class Missing(Exception):
    """A CSV the figure needs is not on this machine."""


def table(name, required=True):
    """Read results/<name>.csv. AROI tables are gitignored under the dataset
    licence, so a figure that wants one must cope with its absence."""
    p = RESULTS / f'{name}.csv'
    if p.exists():
        return pd.read_csv(p)
    msg = (f'results/{name}.csv absent — run `python analysis/derive.py` on a '
           f'machine with the source data (see data/README.md)')
    if required:
        raise Missing(msg)
    print(f'    {msg}')
    return None


def panel_letters(fig, axes, letters=None, dx=0.075, dy=0.03):
    """Bold panel letters in the margin, placed from each axes' own box so they
    stay put when the layout changes. Kept separate from the axes titles."""
    for ax, letter in zip(axes, letters or 'abcdefgh'):
        box = ax.get_position()
        fig.text(max(box.x0 - dx, 0.004), box.y1 + dy, letter, fontsize=11,
                 fontweight='bold', va='bottom')


def save(fig, name):
    return S.save(fig, name, 'paper')
