"""Shared publication figure style for the paper figure set (out/paper/).

Usage:
    import paperfig as pf
    pf.setup()                       # once per script (rcParams)
    fig, ax = pf.figure(width="single")   # or "double"; returns (fig, ax)
    ...
    pf.save(fig, "fig_name")         # writes out/paper/fig_name.png and .pdf

Conventions: serif text, 8-9 pt, IEEE column widths (single 3.4 in, double
7.0 in), consistent colors: DATA black, CLASSICAL orange (tab:orange dashed),
MODEL blue (tab:blue), stuck/free fills red/green at 30%/20% alpha.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "out" / "paper"
COL = dict(data="black", classical="tab:orange", model="tab:blue",
           stuck="tab:red", free="tab:green", cav="0.35")
LS = dict(data="-", classical="--", model="-")
WIDTH = dict(single=3.4, double=7.0)


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5,
        "axes.titlesize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "lines.linewidth": 1.2, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.5, "legend.frameon": False,
        "figure.dpi": 100, "savefig.dpi": 300, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def figure(width="single", height=None, nrows=1, ncols=1, **kw):
    w = WIDTH[width]
    h = height if height is not None else (0.62 * w if nrows == 1 else 0.5 * w * nrows)
    return plt.subplots(nrows, ncols, figsize=(w, h), **kw)


def save(fig, name, tight=True):
    OUT.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return OUT / f"{name}.png"
