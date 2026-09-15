"""E12 figures and summary tables (paperfig style, saved into out/e12/).

Figures
  fig_e12_heat_q2500_A1-5 / _A6-10   rows A, cols SUMO | classical | C&R winner
  fig_e12_heat_q2000_A1-5 / _A6-10   same, zero-refit transfer
  fig_e12_data_heat                  SUMO 5-run mean only, 10 A x {q2500, q2000}
  fig_e12_kappa_vs_A                 kappa_c, kappa_r, ratio, W1 vs A
  fig_e12_hypothesis                 nested shared-parameter comparison
  fig_e12_profiles                   A = 1, 5, 10 at t = 500 / 700 / 850 s
Summary: summary.md (tables used by E12_results.md).

Used through e12_assertiveness.py --figures / --summary (or --smoke).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import e12_assertiveness as e12
import paperfig as pf
from loader import CELL_LEN, N_CELL

X_UP_KM = (np.arange(N_CELL) + 1) * CELL_LEN / 1000.0
DATA_LABEL = "SUMO (5-run mean)"
CL_LABEL = "classical LWR+MB"
CR_LABEL = "catch & release"
PROFILE_T = (500.0, 700.0, 850.0)
PROFILE_A = (1, 5, 10)
KAPPA_FLOOR = 1e-6           # log-axis floor for zero-event kappas
KEY_FIT = e12.scen_key(e12.UC, e12.QIN_FIT)
KEY_TR = e12.scen_key(e12.UC, e12.QIN_TRANSFER)


def _save(fig, name, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return out / f"{name}.png"


def load(S):
    out = Path(S["out"])
    ladder = json.loads((out / "ladder.json").read_text())
    hyp_path = out / "hypothesis.json"
    hyp = json.loads(hyp_path.read_text()) if hyp_path.exists() else None
    fields = {A: dict(np.load(out / f"fields_A{A:g}.npz"))
              for A in S["A_levels"]}
    return ladder, hyp, fields


def load_extras(S):
    p = Path(S["out"]) / "extras.json"
    return json.loads(p.read_text()) if p.exists() else None


def _chunks(seq, n=5):
    seq = list(seq)
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def _winner_eval(blk, key):
    return blk["fits"][blk["winner"]]["eval"][key]


# ---------------------------------------------------------------------------
# heatmaps
# ---------------------------------------------------------------------------

def _heat(ax, rho, xc, tt, label, title=None):
    im = ax.imshow(np.asarray(rho, float), origin="lower", aspect="auto",
                   cmap="turbo", vmin=0.0, vmax=90.0,
                   extent=[0.0, N_CELL * CELL_LEN / 1000.0, 0.0, tt[-1]],
                   interpolation="nearest")
    ax.plot(np.asarray(xc, float) / 1000.0, tt, color="white", lw=0.8)
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(0.0, tt[-1])
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 250, 500, 750, 1000])
    ax.grid(False)
    if label:
        ax.text(0.03, 0.96, label, transform=ax.transAxes, ha="left",
                va="top", fontsize=6.5, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45,
                          lw=0))
    if title:
        ax.set_title(title, fontsize=8)
    return im


def fig_heat_grid(S, ladder, fields, qin, A_list, name):
    key = e12.scen_key(e12.UC, qin)
    n = len(A_list)
    fig, axes = plt.subplots(n, 3, figsize=(7.0, 1.55 * n + 0.5),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    im = None
    for i, A in enumerate(A_list):
        fd, blk = fields[A], ladder[f"A{A:g}"]
        tt = fd["tt"]
        w1_cl = blk["classical"][key]["W1"]
        w1_cr = _winner_eval(blk, key)["W1"]
        panels = [(fd[f"rho_mean_{key}"], fd[f"xcav_data_{key}"],
                   f"A = {A:g}", DATA_LABEL),
                  (fd[f"classical_rho_{key}"], fd[f"classical_xcav_{key}"],
                   f"A = {A:g}, W1 = {w1_cl:.1f}", CL_LABEL),
                  (fd[f"winner_rho_{key}"], fd[f"winner_xcav_{key}"],
                   f"A = {A:g} ({blk['winner']}), W1 = {w1_cr:.1f}",
                   CR_LABEL)]
        for j, (rho, xc, label, title) in enumerate(panels):
            im = _heat(axes[i, j], rho, xc, tt, label,
                       title if i == 0 else None)
            if i == n - 1:
                axes[i, j].set_xlabel("x [km]")
            if j == 0:
                axes[i, j].set_ylabel("t [s]")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, pad=0.01)
    cb.set_label(r"$\rho$ [veh/km]")
    fig.suptitle(rf"$u_\xi$ = {e12.UC * 3.6:.0f} km/h, $q_{{in}}$ = {qin:g} veh/h"
                 + ("  (fit scenario)" if qin == e12.QIN_FIT
                    else "  (zero-refit transfer)"), fontsize=8.5)
    return _save(fig, name, S["out"])


def fig_data_heat(S, fields, name="fig_e12_data_heat"):
    rows = []
    for qin in (e12.QIN_FIT, e12.QIN_TRANSFER):
        for chunk in _chunks(S["A_levels"]):
            rows.append((qin, chunk))
    ncol = max(len(c) for _, c in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(7.0, 1.6 * len(rows) + 0.4),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    im = None
    for i, (qin, chunk) in enumerate(rows):
        key = e12.scen_key(e12.UC, qin)
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(chunk):
                ax.axis("off")
                continue
            A = chunk[j]
            fd = fields[A]
            im = _heat(ax, fd[f"rho_mean_{key}"], fd[f"xcav_data_{key}"],
                       fd["tt"], None, rf"A = {A:g}, $q_{{in}}$ = {qin:g}")
            if i == len(rows) - 1:
                ax.set_xlabel("x [km]")
            if j == 0:
                ax.set_ylabel("t [s]")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, pad=0.01)
    cb.set_label(r"$\rho$ [veh/km]")
    fig.suptitle(f"{DATA_LABEL}, $u_\\xi$ = {e12.UC * 3.6:.0f} km/h",
                 fontsize=8.5)
    return _save(fig, name, S["out"])


# ---------------------------------------------------------------------------
# kappa vs A, W1 vs A
# ---------------------------------------------------------------------------

def _ladders(S, ladder):
    A = np.array(list(S["A_levels"]), float)
    keys = [f"A{a:g}" for a in S["A_levels"]]
    c1 = np.array([[ladder[k]["fits"]["C1"]["kappa_c"],
                    ladder[k]["fits"]["C1"]["kappa_r"]] for k in keys])
    win = np.array([[ladder[k]["fits"][ladder[k]["winner"]]["kappa_c"],
                     ladder[k]["fits"][ladder[k]["winner"]]["kappa_r"]]
                    for k in keys])
    return A, keys, c1, win


def _event_arrays(hyp, keys, skey, which):
    rows = hyp["events"].get(skey, {}) if hyp else {}
    A, hat, lo, hi = [], [], [], []
    for k in keys:
        if k in rows:
            v = rows[k][which]
            A.append(float(k[1:]))
            hat.append(v[0]); lo.append(v[1]); hi.append(v[2])
    return (np.array(A), np.maximum(np.array(hat), KAPPA_FLOOR),
            np.maximum(np.array(lo), KAPPA_FLOOR),
            np.maximum(np.array(hi), KAPPA_FLOOR))


def fig_kappa_vs_A(S, ladder, hyp, name="fig_e12_kappa_vs_A"):
    A, keys, c1, win = _ladders(S, ladder)
    fig, axes = pf.figure("double", height=5.0, nrows=2, ncols=2)
    ev_style = {KEY_FIT: dict(color="0.15", marker="^", ls="none", ms=3.5,
                              label=r"event MLE, $q_{in}$ = 2500"),
                KEY_TR: dict(color="0.55", marker="v", ls="none", ms=3.5,
                             label=r"event MLE, $q_{in}$ = 2000")}
    for ax, col, lab in ((axes[0, 0], 0, r"$\kappa_c$ [veh$^{-1}$]"),
                         (axes[0, 1], 1, r"$\kappa_r$ [veh$^{-1}$]")):
        ax.plot(A, c1[:, col], color=pf.COL["model"], marker="o", ms=3.5,
                label="field fit C1")
        ax.plot(A, win[:, col], color=pf.COL["model"], marker="s", ms=3.5,
                mfc="white", ls="--", label="field fit, per-A winner")
        which = "kappa_c" if col == 0 else "kappa_r"
        for skey, st in ev_style.items():
            a, hat, lo, hi = _event_arrays(hyp, keys, skey, which)
            if a.size:
                ax.errorbar(a, hat, yerr=[hat - lo, hi - hat], capsize=1.5,
                            elinewidth=0.6, **st)
        ax.set_yscale("log")
        ax.set_ylabel(lab)
    ax = axes[1, 0]
    ax.plot(A, c1[:, 0] / c1[:, 1], color=pf.COL["model"], marker="o", ms=3.5,
            label="field fit C1")
    ax.plot(A, win[:, 0] / win[:, 1], color=pf.COL["model"], marker="s",
            ms=3.5, mfc="white", ls="--", label="field fit, per-A winner")
    for skey, st in ev_style.items():
        a, kc, _, _ = _event_arrays(hyp, keys, skey, "kappa_c")
        a2, kr, _, _ = _event_arrays(hyp, keys, skey, "kappa_r")
        if a.size:
            ax.plot(a, kc / kr, **st)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\kappa_c / \kappa_r$")
    ax = axes[1, 1]
    w1 = {}
    for key, ls_cl, ls_cr, mfc, tag in ((KEY_FIT, "--", "-", None, "2500 (fit)"),
                                        (KEY_TR, ":", "-.", "white", "2000 (transfer)")):
        cl = np.array([ladder[k]["classical"][key]["W1"] for k in keys])
        cr = np.array([_winner_eval(ladder[k], key)["W1"] for k in keys])
        w1[key] = (cl, cr)
        ax.plot(A, cl, color=pf.COL["classical"], ls=ls_cl, marker="s", ms=3.5,
                mfc=mfc or pf.COL["classical"], label=f"{CL_LABEL}, $q_{{in}}$ = {tag}")
        ax.plot(A, cr, color=pf.COL["model"], ls=ls_cr, marker="o", ms=3.5,
                mfc=mfc or pf.COL["model"], label=f"{CR_LABEL}, $q_{{in}}$ = {tag}")
        worse = [(a, r) for a, c, r in zip(A, cl, cr) if r > c]
        if worse:
            ax.plot(*zip(*worse), ls="none", marker="x", ms=7, mew=1.2,
                    color="tab:red", label=("C&R worse than classical"
                                            if key == KEY_FIT or not w1.get("_lbl")
                                            else None))
            w1["_lbl"] = True
    extras = load_extras(S)
    if extras and extras.get("events_field"):
        efk = [k for k in keys if k in extras["events_field"]]
        ax.plot([float(k[1:]) for k in efk],
                [extras["events_field"][k]["W1_dt05"] for k in efk],
                color="0.4", ls="-", marker="^", ms=3.5, lw=0.9,
                label=r"C&R with event-MLE $\kappa$ (no field fit), $q_{in}$ = 2500")
    ax.set_ylabel("W1 [veh km]")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, handlelength=2.2)
    for ax, letter in zip(axes.ravel(), "abcd"):
        ax.set_xlabel("assertiveness A")
        ax.set_xticks(list(S["A_levels"]))
        ax.set_title(f"({letter})", loc="left", fontsize=8)
    hs, ls = axes[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="upper center", ncol=4, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.2)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return _save(fig, name, S["out"])


def fig_hypothesis(S, ladder, hyp, name="fig_e12_hypothesis"):
    A = np.array(list(S["A_levels"]), float)
    keys = [f"A{a:g}" for a in S["A_levels"]]
    from matplotlib.ticker import NullFormatter
    from matplotlib.patches import Patch
    fig, axes = pf.figure("double", height=2.5, nrows=1, ncols=3,
                          constrained_layout=True)
    ax = axes[0]
    names = ["M_none", "M_c", "M_r", "M_both"]
    labels = ["shared\n(2)", r"$\kappa_c(A)$" + "\n(11)",
              r"$\kappa_r(A)$" + "\n(11)", "both\n(20)"]
    tot = [hyp["totals_dt1"][n] for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x, tot, width=0.55, color=["0.6", pf.COL["model"],
                                              pf.COL["classical"], "0.3"])
    for b, v in zip(bars, tot):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center",
                va="bottom", fontsize=6.5)
    for i, n in ((0, "M_none_ws"), (1, "M_c_ws"), (2, "M_r_ws")):
        v = hyp[n]["total_dt1"]
        ax.bar(x[i] + 0.3, v, width=0.2, color="white",
               edgecolor=bars[i].get_facecolor(), hatch="////", lw=0.6)
        ax.text(x[i] + 0.3, v, f"{v:.0f}", ha="center", va="bottom",
                fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"$\sum_A$ W1 [veh km]  (dt = 1 s)")
    lo = min(tot + [hyp[n]["total_dt1"] for n in ("M_none_ws", "M_c_ws", "M_r_ws")])
    ax.set_ylim(0.9 * lo, 1.06 * max(tot))
    ax.legend(handles=[Patch(fc="white", ec="0.3", hatch="////",
                             label="same, $w_s$ = 0.6 w")],
              fontsize=6, loc="upper right")
    ax = axes[1]
    ax.plot(A, [hyp["M_c"]["ladder"][k] for k in keys], color=pf.COL["model"],
            marker="o", ms=3.5, label=r"$M_c$: $\kappa_c(A)$, $\kappa_r$ shared")
    ax.plot(A, [hyp["M_both"]["ladder_kc"][k] for k in keys], color="0.3",
            marker="s", ms=3.0, mfc="white", ls="--", label=r"$M_{both}$ (C1)")
    ax.axhline(hyp["M_r"]["shared"]["kc"], color=pf.COL["classical"], lw=1.0,
               ls=":", label=r"$M_r$: shared $\kappa_c$")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\kappa_c$ [veh$^{-1}$]")
    ax = axes[2]
    ax.plot(A, [hyp["M_r"]["ladder"][k] for k in keys], color=pf.COL["classical"],
            marker="o", ms=3.5, label=r"$M_r$: $\kappa_r(A)$, $\kappa_c$ shared")
    ax.plot(A, [hyp["M_both"]["ladder_kr"][k] for k in keys], color="0.3",
            marker="s", ms=3.0, mfc="white", ls="--", label=r"$M_{both}$ (C1)")
    ax.axhline(hyp["M_c"]["shared"]["kr"], color=pf.COL["model"], lw=1.0,
               ls=":", label=r"$M_c$: shared $\kappa_r$")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\kappa_r$ [veh$^{-1}$]")
    for ax in axes[1:]:
        ax.set_xlabel("assertiveness A")
        ax.set_xticks(list(S["A_levels"]))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.legend(fontsize=5.5, loc="best")
    for ax, letter in zip(axes, "abc"):
        ax.set_title(f"({letter})", loc="left", fontsize=8)
    return _save(fig, name, S["out"])


def fig_profiles(S, ladder, fields, name="fig_e12_profiles"):
    A_list = [a for a in PROFILE_A if a in S["A_levels"]]
    fig, axes = pf.figure("double", height=1.75 * len(PROFILE_T) + 0.3,
                          nrows=len(PROFILE_T), ncols=len(A_list),
                          sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    if axes.shape != (len(PROFILE_T), len(A_list)):
        axes = axes.reshape(len(PROFILE_T), len(A_list))
    row_max = np.zeros(len(PROFILE_T))
    for j, A in enumerate(A_list):
        fd, blk = fields[A], ladder[f"A{A:g}"]
        tt = fd["tt"]
        for i, t_snap in enumerate(PROFILE_T):
            ax = axes[i, j]
            k = int(np.argmin(np.abs(tt - t_snap)))
            s, f = fd[f"winner_s_{KEY_FIT}"][k], fd[f"winner_f_{KEY_FIT}"][k]
            ax.fill_between(X_UP_KM, 0.0, s, color=pf.COL["stuck"], alpha=0.3,
                            lw=0, label="stuck class $s$")
            ax.fill_between(X_UP_KM, s, s + f, color=pf.COL["free"], alpha=0.2,
                            lw=0, label="free class $f$")
            ax.plot(X_UP_KM, fd[f"rho_mean_{KEY_FIT}"][k], color=pf.COL["data"],
                    lw=1.3, label=DATA_LABEL)
            ax.plot(X_UP_KM, fd[f"classical_rho_{KEY_FIT}"][k],
                    color=pf.COL["classical"], ls="--", lw=1.1, label=CL_LABEL)
            ax.plot(X_UP_KM, fd[f"winner_rho_{KEY_FIT}"][k], color=pf.COL["model"],
                    lw=1.2, label=CR_LABEL + r" $\rho$")
            ax.axhline(e7a_rho_crit(), color="0.5", ls="--", lw=0.7)
            xc = fd[f"winner_xcav_{KEY_FIT}"][k] / 1000.0
            if np.isfinite(xc):
                ax.axvline(xc, color=pf.COL["cav"], ls=":", lw=0.8)
            sel = X_UP_KM <= 16.0
            row_max[i] = max(row_max[i], float(fd[f"rho_mean_{KEY_FIT}"][k][sel].max()),
                             float(fd[f"classical_rho_{KEY_FIT}"][k][sel].max()),
                             float(fd[f"winner_rho_{KEY_FIT}"][k][sel].max()))
            ax.set_xlim(0.0, 16.0)
            ax.text(0.02, 0.95, f"A = {A:g} ({blk['winner']}), t = {t_snap:.0f} s"
                    + ("  (post-release)" if t_snap > 750.0 else ""),
                    transform=ax.transAxes, ha="left", va="top", fontsize=7)
            if j == 0:
                ax.set_ylabel(r"$\rho$ [veh/km]")
            if i == len(PROFILE_T) - 1:
                ax.set_xlabel("x [km]")
    for i in range(len(PROFILE_T)):
        axes[i, 0].set_ylim(0.0, 5.0 * np.ceil(1.08 * row_max[i] / 5.0))
    hs, ls = axes[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=7, handlelength=1.8,
               columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return _save(fig, name, S["out"])


def fig_ridge(S, extras, name="fig_e12_ridge"):
    """W1 vs kappa_c/kappa_r per A, one line per kappa_r magnitude
    (identifiability: flat-in-magnitude = fast-equilibrium ridge)."""
    keys = [f"A{a:g}" for a in S["A_levels"]]
    ratios = np.asarray(extras["_meta"]["ratios"])
    mags = extras["_meta"]["magnitudes"]
    n = len(keys)
    ncol = 5 if n > 5 else n
    nrow = int(np.ceil(n / ncol))
    fig, axes = pf.figure("double", height=1.9 * nrow + 0.5, nrows=nrow,
                          ncols=ncol, sharex=True, sharey=False)
    axes = np.atleast_1d(axes).reshape(nrow, ncol)
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(keys):
        ax = axes[i // ncol, i % ncol]
        W = np.asarray(extras["ridge"][k]["W1"])
        for m, mag in enumerate(mags):
            ax.plot(ratios, W[m], color=cmap(m / max(1, len(mags) - 1)),
                    lw=1.0, marker=".", ms=2.5,
                    label=rf"$\kappa_r$ = {mag:g}")
        ax.set_xscale("log")
        ax.set_title(f"A = {k[1:]}", fontsize=8)
        ax.grid(alpha=0.25)
        if i % ncol == 0:
            ax.set_ylabel("W1 [veh km]")
        if i // ncol == nrow - 1:
            ax.set_xlabel(r"$\kappa_c/\kappa_r$")
    for j in range(n, nrow * ncol):
        axes[j // ncol, j % ncol].axis("off")
    hs, ls = axes[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="upper center", ncol=len(mags), fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, name, S["out"])


def e7a_rho_crit():
    import e7_ablation as e7a
    return e7a.RHO_CRIT


def make_figures(S):
    pf.setup()
    ladder, hyp, fields = load(S)
    made = []
    for qin in (e12.QIN_FIT, e12.QIN_TRANSFER):
        for chunk in _chunks(S["A_levels"]):
            made.append(fig_heat_grid(S, ladder, fields, qin, chunk,
                                      f"fig_e12_heat_q{qin:g}_A{chunk[0]}-{chunk[-1]}"))
    made.append(fig_data_heat(S, fields))
    made.append(fig_kappa_vs_A(S, ladder, hyp))
    if hyp is not None:
        made.append(fig_hypothesis(S, ladder, hyp))
    made.append(fig_profiles(S, ladder, fields))
    extras = load_extras(S)
    if extras is not None:
        made.append(fig_ridge(S, extras))
    for p in made:
        print("wrote", p)
    return made


# ---------------------------------------------------------------------------
# summary tables
# ---------------------------------------------------------------------------

def _pct(v):
    return "n/a" if v is None else f"{100 * v:+.1f}"


def write_summary(S):
    ladder, hyp, _ = load(S)
    keys = [f"A{a:g}" for a in S["A_levels"]]
    keys_all = [e12.scen_key(*s) for s in e12.SCENARIOS]
    L = []
    L.append("# E12 summary — pure E7 catch & release, A = "
             f"{S['A_levels'][0]}..{S['A_levels'][-1]}\n")
    L.append(f"Structure: {ladder['_meta']['structure']}. Forbidden knobs: "
             f"{ladder['_meta']['forbidden_knobs']}. Classical: "
             f"{ladder['_meta']['classical']}. Fit scenario u15/q2500; "
             "everything else zero-refit. Objective: "
             f"{ladder['_meta']['objective']}.\n")

    # (i) per-A winner table
    L.append("## 1. Per-A winner (E7 rule) and W1 on all four scenarios\n")
    L.append("| A | winner | kappa_c | kappa_r | gamma | w_s/w | "
             + " | ".join(f"W1 {k} cl / CR" for k in keys_all) + " |")
    L.append("|" + "---|" * (6 + len(keys_all)))
    for k in keys:
        b = ladder[k]
        f = b["fits"][b["winner"]]
        cells = [k[1:], b["winner"], f"{f['kappa_c']:.3e}", f"{f['kappa_r']:.3e}",
                 "-" if f["gamma"] is None else f"{f['gamma']:.3f}",
                 "-" if f["ws_frac"] is None else f"{f['ws_frac']:.3f}"]
        for sk in keys_all:
            cl, cr = b["classical"][sk]["W1"], f["eval"][sk]["W1"]
            mark = " **(worse)**" if cr > cl else ""
            cells.append(f"{cl:.1f} / {cr:.1f}{mark}")
        L.append("| " + " | ".join(cells) + " |")

    L.append("\n## 2. Per-A event metrics at u15 (classical / catch & release winner)\n")
    L.append("| A | e_s q2500 [%] | e_s q2000 [%] | omega err q2500 [%] | "
             "omega err q2000 [%] | wake q2500 (data) | wake q2000 (data) | "
             "N_s MAE q2500 | ds stuck 700 s (CR) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        b = ladder[k]
        f = b["fits"][b["winner"]]
        cl1, cl2 = b["classical"][KEY_FIT], b["classical"][KEY_TR]
        cr1, cr2 = f["eval"][KEY_FIT], f["eval"][KEY_TR]
        ds = cr1.get("ds_stuck_700")
        L.append(f"| {k[1:]} | {_pct(cl1['e_s'])} / {_pct(cr1['e_s'])} | "
                 f"{_pct(cl2['e_s'])} / {_pct(cr2['e_s'])} | "
                 f"{_pct(cl1['omega_err'])} / {_pct(cr1['omega_err'])} | "
                 f"{_pct(cl2['omega_err'])} / {_pct(cr2['omega_err'])} | "
                 f"{cl1['wake']:.1f} / {cr1['wake']:.1f} ({b['data'][KEY_FIT]['wake']:.1f}) | "
                 f"{cl2['wake']:.1f} / {cr2['wake']:.1f} ({b['data'][KEY_TR]['wake']:.1f}) | "
                 f"{cl1['Ns_mae']:.1f} / {cr1['Ns_mae']:.1f} | "
                 f"{'n/a' if ds is None else f'{100 * ds:.1f} %'} |")

    # (ii) config comparison
    L.append("\n## 3. Configuration comparison, W1 at q2500 (fit) / q2000 (transfer)\n")
    names = [n for n, _ in e12.CONFIGS]
    L.append("| A | classical | " + " | ".join(names) + " |")
    L.append("|---|---|" + "---|" * len(names))
    for k in keys:
        b = ladder[k]
        cells = [k[1:], f"{b['classical'][KEY_FIT]['W1']:.1f} / "
                        f"{b['classical'][KEY_TR]['W1']:.1f}"]
        for n in names:
            e = b["fits"][n]["eval"]
            star = "**" if n == b["winner"] else ""
            cells.append(f"{star}{e[KEY_FIT]['W1']:.1f} / {e[KEY_TR]['W1']:.1f}{star}")
        L.append("| " + " | ".join(cells) + " |")

    # (iii) C1 ladders + spearman
    L.append("\n## 4. C1 (kappa_c, kappa_r) ladder and rank correlations with A\n")
    L.append("| A | kappa_c | kappa_r | kappa_c/kappa_r | W1 q2500 | W1 q2000 | "
             "event kappa_c q2500 | event kappa_r q2500 |")
    L.append("|---|---|---|---|---|---|---|---|")
    ev = hyp["events"].get(KEY_FIT, {}) if hyp else {}
    for k in keys:
        c = ladder[k]["fits"]["C1"]
        e = ev.get(k)
        L.append(f"| {k[1:]} | {c['kappa_c']:.3e} | {c['kappa_r']:.3e} | "
                 f"{c['kappa_c'] / c['kappa_r']:.2f} | "
                 f"{c['eval'][KEY_FIT]['W1']:.1f} | {c['eval'][KEY_TR]['W1']:.1f} | "
                 + ("n/a | n/a |" if e is None else
                    f"{e['kappa_c'][0]:.3e} [{e['kappa_c'][1]:.1e}, {e['kappa_c'][2]:.1e}] | "
                    f"{e['kappa_r'][0]:.3e} [{e['kappa_r'][1]:.1e}, {e['kappa_r'][2]:.1e}] |"))
    if hyp:
        L.append("\nSpearman rank correlation with A (rho, p):\n")
        L.append("| ladder | kappa_c | kappa_r | ratio |")
        L.append("|---|---|---|---|")
        for name, t in hyp["trends"].items():
            cells = [name]
            for q in ("kappa_c", "kappa_r", "ratio"):
                if q in t:
                    cells.append(f"{t[q]['rho']:+.2f} (p = {t[q]['p']:.3f})")
                else:
                    cells.append("-")
            L.append("| " + " | ".join(cells) + " |")

    # (iv) hypothesis
    if hyp:
        L.append("\n## 5. Nested shared-parameter models (objective = sum over A "
                 "of W1 at u15/q2500)\n")
        L.append("| model | free per A | shared | n params | sum W1 dt=1 | "
                 "sum W1 dt=0.5 | monotone? |")
        L.append("|---|---|---|---|---|---|---|")
        rows = (("M_none", "none", hyp["M_none"]["shared"], None),
                ("M_c", "kappa_c", hyp["M_c"]["shared"], hyp["M_c"]),
                ("M_r", "kappa_r", hyp["M_r"]["shared"], hyp["M_r"]),
                ("M_both", "both", {}, None),
                ("M_none_ws", "none (w_s = 0.6 w)", hyp["M_none_ws"]["shared"], None),
                ("M_c_ws", "kappa_c (w_s = 0.6 w)", hyp["M_c_ws"]["shared"], hyp["M_c_ws"]),
                ("M_r_ws", "kappa_r (w_s = 0.6 w)", hyp["M_r_ws"]["shared"], hyp["M_r_ws"]))
        for n, free, shared, m in rows:
            sh = ", ".join(f"{a} = {v:.3e}" for a, v in shared.items()) or "-"
            mono = ("-" if m is None else
                    f"{m['monotone']}: {'yes' if m['monotone_ok'] else 'no'}")
            L.append(f"| {n} | {free} | {sh} | {hyp[n]['n_params']} | "
                     f"{hyp[n]['total_dt1']:.1f} | {hyp[n]['total_dt05']:.1f} | {mono} |")
        L.append(f"\nVerdict: {hyp['verdict']}. Nestedness flags: {hyp['nested']}.\n")
        L.append("Per-A ladders under the single-carrier models:\n")
        L.append("| A | M_c kappa_c | M_c W1 | M_r kappa_r | M_r W1 | M_both W1 | M_none W1 |")
        L.append("|---|---|---|---|---|---|---|")
        for k in keys:
            L.append(f"| {k[1:]} | {hyp['M_c']['ladder'][k]:.3e} | "
                     f"{hyp['M_c']['per_A_W1_dt1'][k]:.1f} | "
                     f"{hyp['M_r']['ladder'][k]:.3e} | {hyp['M_r']['per_A_W1_dt1'][k]:.1f} | "
                     f"{hyp['M_both']['per_A_W1_dt1'][k]:.1f} | "
                     f"{hyp['M_none']['per_A_W1_dt1'][k]:.1f} |")

    # (v) identifiability
    L.append("\n## 6. Identifiability of the C1 fit (6x6 grid, near-optimal set "
             f"<= {100 * e12.NEAR_OPT_TOL:.0f} % above the grid minimum)\n")
    L.append("| A | grid min W1 | n near-opt | extent log10 kappa_c | "
             "extent log10 kappa_r | extent log10 ratio |")
    L.append("|---|---|---|---|---|---|")
    for k in keys:
        g = ladder[k]["grid_C1"]
        no = g["near_opt"]
        L.append(f"| {k[1:]} | {g['min']:.1f} | {no['n']} | {no['extent_log_kc']:.1f} | "
                 f"{no['extent_log_kr']:.1f} | {no['extent_log_ratio']:.1f} |")

    # (vi) where CR loses
    L.append("\n## 7. Where catch & release (winner) is worse than the classical model\n")
    for sk in keys_all:
        lose = [(k[1:], _winner_eval(ladder[k], sk)["W1"] - ladder[k]["classical"][sk]["W1"])
                for k in keys if _winner_eval(ladder[k], sk)["W1"] > ladder[k]["classical"][sk]["W1"]]
        if lose:
            L.append(f"- {sk}: " + ", ".join(f"A = {a} (+{d:.2f} veh km)" for a, d in lose))
        else:
            L.append(f"- {sk}: none (catch & release better at every A)")
    # (viii) extras: event kappas in the field, ridge scan
    extras = load_extras(S)
    if extras is not None:
        L.append("\n## 8. Event-based (E-V3/E4) kappas evaluated in the field "
                 "without fitting, u15/q2500 (dt = 0.5)\n")
        L.append("| A | event kappa_c | event kappa_r | W1 C1 | W1 C1 + w_s = 0.6 w | "
                 "W1 classical | W1 field-fit winner |")
        L.append("|---|---|---|---|---|---|---|")
        for k in keys:
            e = extras["events_field"].get(k)
            if e is None:
                continue
            L.append(f"| {k[1:]} | {e['kappa_c']:.3e} | {e['kappa_r']:.3e} | "
                     f"{e['W1_dt05']:.1f} | {e['W1_ws_dt05']:.1f} | "
                     f"{ladder[k]['classical'][KEY_FIT]['W1']:.1f} | "
                     f"{_winner_eval(ladder[k], KEY_FIT)['W1']:.1f} |")
        L.append("\n## 9. Identifiability scan: W1(kappa_c/kappa_r, kappa_r) "
                 "at u15/q2500 (dt = 1)\n")
        mags = extras["_meta"]["magnitudes"]
        L.append("| A | best (ratio, kappa_r) | min W1 | "
                 + " | ".join(f"best ratio @ kappa_r={m:g} (W1)" for m in mags)
                 + " | slow kinetics better? | flat for kappa_r >= 0.1? |")
        L.append("|" + "---|" * (5 + len(mags)))
        for k in keys:
            r = extras["ridge"][k]
            cells = [k[1:], f"({r['argmin']['ratio']:.1f}, {r['argmin']['magnitude']:g})",
                     f"{r['min']:.1f}"]
            cells += [f"{b['ratio']:.1f} ({b['W1']:.1f})"
                      for b in r["best_ratio_per_magnitude"]]
            cells.append("yes" if r["slow_kinetics_better"] else "no")
            cells.append("yes" if r["ridge_flat_above_0p1"] else "no")
            L.append("| " + " | ".join(cells) + " |")
    text = "\n".join(L) + "\n"
    (Path(S["out"]) / "summary.md").write_text(text)
    return text
