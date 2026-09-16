"""E13 v2 figures and summary (paperfig style, out/e13/).

  fig_e13_hull            reachable set (MAE_N, eps_omega) per A with the box,
                          the classical point and the E12 point (stage p)
  fig_e13_tradeoff        summed trade-off with the classical null line,
                          lambda_max shading, lambda*; per-A W1, MAE_N,
                          |eps_omega|, eps_R vs lambda
  fig_e13_gap_vs_lambda   nested totals, M_c - M_r gap (jackknife bars),
                          eta_c / eta_r, ridge residual vs lambda; lam* ladders
  fig_e13_kappa_vs_A      M_both ladders at lam = 0 and lam* vs event MLE,
                          with the +-0.5 dex agreement band
  fig_e13_metrics_vs_A    W1, MAE_N, eps_omega, eps_R, tau, wake per A:
                          classical / E12 (lam = 0) / E13 (lam*)
  fig_e13_heat_q2500_*, fig_e13_heat_q2000_*, fig_e13_profiles, fig_e13_ridge
  summary13.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import e12_assertiveness as e12
import e12_figures as f12
import e13_joint as e13
import paperfig as pf

KEY_FIT, KEY_TR = e13.KEY_FIT, e13.KEY_TR
CL, CR = f12.CL_LABEL, f12.CR_LABEL
LBL_E12 = "catch & release, W1 only ($\\lambda$ = 0)"
LBL_E13 = "catch & release, joint ($\\lambda^*$)"
LBL_EV = "event MLE (E-V3/E4)"


def load(S):
    out = Path(S["out"])
    ladder = json.loads((out / "ladder13.json").read_text())
    sel = json.loads((out / "lambda_star.json").read_text())
    opt = {}
    for name in ("hypothesis13", "extras13", "final13", "prehull13"):
        p = out / f"{name}.json"
        opt[name] = json.loads(p.read_text()) if p.exists() else None
    fields = {A: dict(np.load(out / f"fields13_A{A:g}.npz"))
              for A in S["A_levels"] if (out / f"fields13_A{A:g}.npz").exists()}
    return ladder, sel, opt["hypothesis13"], opt["extras13"], opt["final13"], opt["prehull13"], fields


def _winner(ladder, k, lam):
    return e13._winner(ladder, k, lam)


def _lam_axis(ax, lams):
    pos = [l for l in lams if l > 0]
    ax.set_xscale("symlog", linthresh=min(pos) if pos else 1.0)
    ax.set_xticks(list(lams))
    ax.set_xticklabels([f"{l:g}" for l in lams], fontsize=6, rotation=45)
    ax.set_xlabel(r"$\lambda$")


# ---------------------------------------------------------------------------

def fig_hull(S, pre, ladder, name="fig_e13_hull"):
    keys = [f"A{a:g}" for a in S["A_levels"]]
    n = len(keys)
    ncol = 5 if n > 5 else n
    nrow = int(np.ceil(n / ncol))
    fig, axes = pf.figure("double", height=1.9 * nrow + 0.6, nrows=nrow, ncols=ncol,
                          sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(nrow, ncol)
    sc = None
    for i, k in enumerate(keys):
        ax = axes[i // ncol, i % ncol]
        pts = pre[k]["points"]
        x = np.array([p["Ns_mae"] for p in pts])
        y = np.array([100 * p["omega_err"] for p in pts])
        c = np.array([p["W1_rel"] for p in pts])
        sc = ax.scatter(x, y, c=c, cmap="RdYlGn_r", vmin=0.7, vmax=1.5, s=6, lw=0, alpha=0.85)
        ax.add_patch(plt.Rectangle((0, -100 * e13.BOX_OM), e13.BOX_NS, 200 * e13.BOX_OM,
                                   fill=False, ec="tab:red", lw=1.0, ls="--"))
        cl = pre[k]["classical"]
        ax.plot(cl["Ns_mae"], 100 * cl["omega_err"], "s", color=pf.COL["classical"], ms=5,
                mec="k", mew=0.4, label=CL)
        w0, _ = _winner(ladder, k, 0.0)
        ax.plot(w0["parts_dt05"]["Ns_mae"], 100 * w0["parts_dt05"]["omega_err"], "o",
                color=pf.COL["model"], ms=5, mec="k", mew=0.4, label=LBL_E12)
        ax.plot(0, 0, "*", color="k", ms=7, label="SUMO")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_title(f"A = {k[1:]} ({pre[k]['n_in_box_field_ok']} in box & field ok)", fontsize=7.5)
        ax.axhline(0, color="0.6", lw=0.5)
        if i % ncol == 0:
            ax.set_ylabel(r"$\epsilon_\omega$ [%]")
        if i // ncol == nrow - 1:
            ax.set_xlabel(r"MAE$_N$ [veh]")
    for j in range(n, nrow * ncol):
        axes[j // ncol, j % ncol].axis("off")
    hs, ls = axes[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.45, -0.02))
    cb = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, pad=0.01)
    cb.set_label(r"W1 / W1$_{cl}$")
    fig.suptitle("reachable set of the pure model on the (ratio, magnitude, $\\gamma$, $w_s$) "
                 "grid; dashed box = mechanism target", fontsize=8, y=1.01)
    return f12._save(fig, name, S["out"])


def fig_tradeoff(S, ladder, sel, name="fig_e13_tradeoff"):
    lams = list(S["lambdas"])
    keys = [f"A{a:g}" for a in S["A_levels"]]
    rows = sel["sweep"]
    fig, axes = pf.figure("double", height=7.0, nrows=3, ncols=2)
    ax = axes[0, 0]
    x = [r["sum_penalty"] for r in rows]
    y = [r["sum_W1_rel"] for r in rows]
    ax.plot(x, y, "-o", color=pf.COL["model"], ms=3.5, label="pure model, per-A winners")
    for r, xi, yi in zip(rows, x, y):
        ax.annotate(f"{r['lam']:g}", (xi, yi), textcoords="offset points", xytext=(3, 3),
                    fontsize=5.5)
    r_last = rows[-1]                                  # J_cl = n_A + lam * sum P_cl
    cl_pen = (r_last["sum_J_classical"] - len(keys)) / r_last["lam"] if r_last["lam"] > 0 else np.nan
    ax.plot([cl_pen], [len(keys)], "s", color=pf.COL["classical"], ms=6, label="classical LWR+MB")
    ls = sel["lambda_star"]
    i = [r["lam"] for r in rows].index(ls)
    ax.plot(x[i], y[i], "o", ms=10, mfc="none", mec="tab:red", mew=1.3,
            label=rf"$\lambda^*$ = {ls:g} ({sel['rule']})")
    if sel.get("lambda_knee_restricted") is not None:
        j = [r["lam"] for r in rows].index(sel["lambda_knee_restricted"])
        ax.plot(x[j], y[j], "D", ms=8, mfc="none", mec="tab:purple", mew=1.0,
                label=rf"restricted knee $\lambda$ = {sel['lambda_knee_restricted']:g}")
    ax.set_xlabel(r"$\sum_A$ penalty [MAE$_N$/$N_0$ + |$\epsilon_\omega$|/$\epsilon_0$ + $\epsilon_R$]")
    ax.set_ylabel(r"$\sum_A$ W1 / W1$_{cl}$")
    ax.legend(fontsize=5.5, loc="upper right")
    ax = axes[0, 1]
    ax.plot(lams, [r["n_field_ok"] for r in rows], "-o", ms=3, color=pf.COL["model"],
            label="field ok (W1 $\\leq$ W1$_{cl}$)")
    ax.plot(lams, [r["n_collapsed"] for r in rows], "-s", ms=3, color="tab:red", label="collapsed")
    ax.plot(lams, [r["n_in_box"] for r in rows], "-^", ms=3, color="tab:green", label="in mechanism box")
    ax.plot(lams, [r["n_events_ok"] for r in rows], "-v", ms=3, color="tab:purple",
            label=r"$\epsilon_R \leq$ 0.5 dex")
    ax.plot(lams, [r["n_beats_classical"] for r in rows], "--x", ms=3, color="0.4",
            label="J < J classical")
    _lam_axis(ax, lams)
    ax.set_ylabel("number of A (of 10)")
    ax.set_ylim(-0.3, len(keys) + 0.3)
    ax.legend(fontsize=5.5, loc="best")
    if sel.get("lambda_max") is not None:
        for a in axes.ravel()[1:]:
            a.axvspan(sel["lambda_max"], max(lams) * 1.5, color="0.85", alpha=0.5, lw=0)
    cmap = plt.get_cmap("viridis")
    panels = [(axes[1, 0], "W1", "W1 [veh km]"), (axes[1, 1], "Ns_mae", r"MAE$_N$ [veh]"),
              (axes[2, 0], "omega_err", r"|$\epsilon_\omega$| [%]"), (axes[2, 1], "eps_R", r"$\epsilon_R$ [dex]")]
    for ax, key, lab in panels:
        for j, k in enumerate(keys):
            vals = []
            for lam in lams:
                w, _ = _winner(ladder, k, lam)
                v = w["parts_dt05"][key]
                vals.append(100 * abs(v) if key == "omega_err" else v)
            col = cmap(j / max(1, len(keys) - 1))
            ax.plot(lams, vals, "-o", ms=2.2, lw=0.8, color=col, label=f"A = {k[1:]}")
            cl = ladder[k]["classical"][KEY_FIT]
            clv = 100 * abs(cl[key]) if key == "omega_err" else cl[key]
            ax.axhline(clv, color=col, ls=":", lw=0.5, alpha=0.7)
        _lam_axis(ax, lams)
        ax.set_ylabel(lab)
        ax.axvline(ls, color="tab:red", lw=0.8, ls="--")
        if key in ("Ns_mae", "eps_R"):
            ax.set_yscale("symlog", linthresh=1.0)
    axes[1, 0].legend(fontsize=5, ncol=2, loc="upper left")
    axes[1, 0].text(0.98, 0.02, "dotted: classical; grey: $\\lambda \\geq \\lambda_{max}$ (collapsed)",
                    transform=axes[1, 0].transAxes, fontsize=5.5, ha="right", va="bottom")
    for ax, letter in zip(axes.ravel(), "abcdef"):
        ax.set_title(f"({letter})", loc="left", fontsize=8)
    fig.tight_layout()
    return f12._save(fig, name, S["out"])


def fig_gap_vs_lambda(S, hyp, sel, name="fig_e13_gap_vs_lambda"):
    lams = [q["lam"] for q in hyp["gap_vs_lambda"]]
    A = np.array(list(S["A_levels"]), float)
    keys = [f"A{a:g}" for a in S["A_levels"]]
    ls = sel["lambda_star"]
    fig, axes = pf.figure("double", height=7.0, nrows=3, ncols=2)
    ax = axes[0, 0]
    for n, c, lst, lab in (("M_none", "0.5", "-", "shared (2)"), ("M_c", pf.COL["model"], "-", r"$\kappa_c(A)$ (11)"),
                           ("M_r", pf.COL["classical"], "-", r"$\kappa_r(A)$ (11)"), ("M_both", "0.2", "--", "both (20)")):
        ax.plot(lams, [q["totals"][n] for q in hyp["gap_vs_lambda"]], lst, color=c, marker="o", ms=3, label=lab)
    _lam_axis(ax, lams)
    ax.set_ylabel(r"$\sum_A J_\lambda$ (dt = 1 s)")
    ax.legend(fontsize=6, loc="best")
    ax = axes[0, 1]
    ax.plot(lams, [100 * q["gap_rel"] for q in hyp["gap_vs_lambda"]], "-o", color="0.2", ms=3.5,
            label=r"($M_c - M_r$) / $M_{both}$")
    for lk, jk in hyp.get("jackknife", {}).items():
        tot_both = hyp["per_lambda"][lk]["totals_dt1"]["M_both"]
        ax.errorbar([jk["lam"]], [100 * jk["gap_mean"] / tot_both], yerr=[200 * jk["gap_se_jack"] / tot_both],
                    fmt="D", color="tab:red", ms=4, capsize=3, lw=1,
                    label=("jackknife mean $\\pm$ 2 SE" if lk == list(hyp["jackknife"])[0] else None))
    for tag, mk, lab in (("queue_only", "s", "queue-only"), ("overtaking_only", "^", "overtaking-only"),
                         ("events_only", "v", "events-only"), ("robust_ws", "D", r"$w_s$ = 0.6 w")):
        if tag in hyp["at_star"]:
            ax.plot([ls], [100 * hyp["at_star"][tag]["gap_rel_both"]], mk, ms=5, mfc="none", mew=1.1,
                    label=f"{lab} at $\\lambda^*$")
    ax.axhline(0, color="k", lw=0.6)
    _lam_axis(ax, lams)
    ax.set_ylabel(r"gap [% of $M_{both}$]")
    ax.legend(fontsize=5, loc="best")
    ax = axes[1, 0]
    ax.plot(lams, [q["eta_c"] for q in hyp["gap_vs_lambda"]], "-o", ms=3, color=pf.COL["model"],
            label=r"$\eta_c$ = ($J_{none}-J_c$)/($J_{none}-J_{both}$)")
    ax.plot(lams, [q["eta_r"] for q in hyp["gap_vs_lambda"]], "-s", ms=3, color=pf.COL["classical"],
            label=r"$\eta_r$")
    ax.axhline(0.8, color="0.5", lw=0.6, ls=":")
    ax.axhline(0.5, color="0.5", lw=0.6, ls=":")
    _lam_axis(ax, lams)
    ax.set_ylabel("explained fraction of the A-gain")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6, loc="best")
    ax = axes[1, 1]
    ax.plot(lams, [q["ridge_residual_dex"] for q in hyp["gap_vs_lambda"]], "-o", ms=3, color="0.2")
    ax.axhline(0.1, color="0.5", lw=0.6, ls=":")
    _lam_axis(ax, lams)
    ax.set_ylabel("ridge residual [dex]")
    ax.text(0.02, 0.95, "< 0.1 dex: $M_c$ and $M_r$ are the same\nratio ladder written two ways",
            transform=ax.transAxes, fontsize=6, va="top")
    q = hyp["per_lambda"][e13.lam_key(ls)]
    ev = hyp["events"].get(KEY_FIT, {})
    for ax, free, other, lab in ((axes[2, 0], "M_c", "M_r", r"$\kappa_c$"), (axes[2, 1], "M_r", "M_c", r"$\kappa_r$")):
        which = "kappa_c" if free == "M_c" else "kappa_r"
        ax.plot(A, [q[free]["ladder"][k] for k in keys], "-o", ms=3.5,
                color=(pf.COL["model"] if free == "M_c" else pf.COL["classical"]),
                label=f"${free[0]}_{{{free[2]}}}$: {lab}(A), other shared")
        ax.plot(A, [q["M_both"]["ladder_kc" if which == "kappa_c" else "ladder_kr"][k] for k in keys],
                "--s", ms=3, mfc="white", color="0.3", label=r"$M_{both}$")
        ax.axhline(q[other]["shared"]["kc" if which == "kappa_c" else "kr"], ls=":", lw=1,
                   color=(pf.COL["classical"] if free == "M_c" else pf.COL["model"]),
                   label=f"${other[0]}_{{{other[2]}}}$: shared {lab}")
        if ev:
            ax.plot([float(k[1:]) for k in keys if k in ev],
                    [max(ev[k][which][0], 1e-6) for k in keys if k in ev], "^", color="0.15", ms=3.5,
                    ls="none", label=LBL_EV)
        ax.set_yscale("log")
        ax.set_ylabel(lab + r" [veh$^{-1}$]")
        ax.set_xlabel("assertiveness A")
        ax.set_xticks(list(S["A_levels"]))
        ax.set_title(rf"$\lambda^*$ = {ls:g}", fontsize=7.5)
        ax.legend(fontsize=5, loc="best")
    for ax, letter in zip(axes.ravel(), "abcdef"):
        ax.set_title(f"({letter})", loc="left", fontsize=8)
    fig.tight_layout()
    return f12._save(fig, name, S["out"])


def fig_kappa_vs_A(S, hyp, sel, name="fig_e13_kappa_vs_A"):
    A = np.array(list(S["A_levels"]), float)
    keys = [f"A{a:g}" for a in S["A_levels"]]
    ls = sel["lambda_star"]
    q0 = hyp["per_lambda"][e13.lam_key(0.0)]["M_both"]
    qs = hyp["per_lambda"][e13.lam_key(ls)]["M_both"]
    ev = hyp["events"].get(KEY_FIT, {})
    fig, axes = pf.figure("double", height=2.6, nrows=1, ncols=3)
    for ax, which, lab in ((axes[0], "kappa_c", r"$\kappa_c$ [veh$^{-1}$]"), (axes[1], "kappa_r", r"$\kappa_r$ [veh$^{-1}$]")):
        lk = "ladder_kc" if which == "kappa_c" else "ladder_kr"
        ax.plot(A, [q0[lk][k] for k in keys], "--s", color="0.5", ms=3, mfc="white",
                label=r"$M_{both}$, $\lambda$ = 0 (E12)")
        ax.plot(A, [qs[lk][k] for k in keys], "-o", color=pf.COL["model"], ms=3.5,
                label=rf"$M_{{both}}$, $\lambda^*$ = {ls:g}")
        if ev:
            a = np.array([float(k[1:]) for k in keys if k in ev])
            hat = np.array([max(ev[k][which][0], 1e-6) for k in keys if k in ev])
            lo = np.array([max(ev[k][which][1], 1e-6) for k in keys if k in ev])
            hi = np.array([max(ev[k][which][2], 1e-6) for k in keys if k in ev])
            ax.errorbar(a, hat, yerr=[hat - lo, hi - hat], fmt="^", color="0.15", ms=3.5, capsize=1.5,
                        elinewidth=0.6, label=LBL_EV)
            ax.fill_between(a, hat / 10 ** e13.AGREE_DEX, hat * 10 ** e13.AGREE_DEX, color="0.15",
                            alpha=0.08, lw=0, label=rf"$\pm${e13.AGREE_DEX:g} dex band")
        ax.set_yscale("log")
        ax.set_ylabel(lab)
    ax = axes[2]
    ax.plot(A, [q0["ladder_kc"][k] / q0["ladder_kr"][k] for k in keys], "--s", color="0.5", ms=3, mfc="white",
            label=r"$M_{both}$, $\lambda$ = 0")
    ax.plot(A, [qs["ladder_kc"][k] / qs["ladder_kr"][k] for k in keys], "-o", color=pf.COL["model"], ms=3.5,
            label=rf"$M_{{both}}$, $\lambda^*$ = {ls:g}")
    if ev:
        ax.plot([float(k[1:]) for k in keys if k in ev],
                [max(ev[k]["kappa_c"][0], 1e-6) / max(ev[k]["kappa_r"][0], 1e-6) for k in keys if k in ev],
                "^", color="0.15", ms=3.5, ls="none", label=LBL_EV)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\kappa_c / \kappa_r$")
    for ax, letter in zip(axes, "abc"):
        ax.set_xlabel("assertiveness A")
        ax.set_xticks(list(S["A_levels"]))
        ax.set_title(f"({letter})", loc="left", fontsize=8)
        ax.legend(fontsize=5, loc="best")
    fig.tight_layout()
    return f12._save(fig, name, S["out"])


def fig_metrics_vs_A(S, ladder, sel, extras, name="fig_e13_metrics_vs_A"):
    A = np.array(list(S["A_levels"]), float)
    keys = [f"A{a:g}" for a in S["A_levels"]]
    ls = sel["lambda_star"]
    fig, axes = pf.figure("double", height=6.8, nrows=3, ncols=2)
    spec = [(axes[0, 0], "W1", "W1 [veh km]", 1.0), (axes[0, 1], "Ns_mae", r"MAE$_N$ [veh]", 1.0),
            (axes[1, 0], "omega_err", r"$\epsilon_\omega$ [%]", 100.0), (axes[1, 1], "eps_R", r"$\epsilon_R$ [dex]", 1.0),
            (axes[2, 0], "tau_model_s", r"turnover $\tau$ [s]", 1.0), (axes[2, 1], "wake", "wake density [veh/km]", 1.0)]
    for ax, key, lab, sc in spec:
        def val(d):
            v = d.get(key)
            return np.nan if v is None else sc * v
        cl = [val(ladder[k]["classical"][KEY_FIT]) for k in keys]
        e12v = [val(_winner(ladder, k, 0.0)[0]["eval"][KEY_FIT]) for k in keys]
        e13v = [val(_winner(ladder, k, ls)[0]["eval"][KEY_FIT]) for k in keys]
        if key != "tau_model_s":
            ax.plot(A, cl, "--s", color=pf.COL["classical"], ms=3.5, label=CL)
        ax.plot(A, e12v, "-.o", color="0.5", ms=3.5, mfc="white", label=LBL_E12)
        ax.plot(A, e13v, "-o", color=pf.COL["model"], ms=3.5, label=LBL_E13)
        if key == "wake":
            ax.plot(A, [ladder[k]["data"][KEY_FIT]["wake"] for k in keys], "-", color="k", lw=1.2, label="SUMO (5-run mean)")
        if key == "tau_model_s":
            td = [ladder[k]["ref"]["tau_data_s"] for k in keys]
            ax.plot(A, [np.nan if t is None else t for t in td], "-*", color="k", lw=1.2, ms=6,
                    label="SUMO (Little's law; undefined where no release)")
            ax.set_yscale("log")
        if key == "omega_err":
            ax.axhline(0, color="k", lw=0.6)
        if key == "Ns_mae":
            ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylabel(lab)
        ax.set_xticks(list(S["A_levels"]))
    axes[0, 0].legend(fontsize=5.5, loc="best")
    axes[2, 0].legend(fontsize=5.5, loc="best")
    axes[2, 1].legend(fontsize=5.5, loc="best")
    for ax in axes[2]:
        ax.set_xlabel("assertiveness A")
    for ax, letter in zip(axes.ravel(), "abcdef"):
        ax.set_title(f"({letter})", loc="left", fontsize=8)
    fig.tight_layout()
    return f12._save(fig, name, S["out"])


def make_figures(S):
    pf.setup()
    ladder, sel, hyp, extras, final, pre, fields = load(S)
    made = [fig_tradeoff(S, ladder, sel), fig_metrics_vs_A(S, ladder, sel, extras)]
    if pre is not None:
        made.append(fig_hull(S, pre, ladder))
    if hyp is not None:
        made.append(fig_gap_vs_lambda(S, hyp, sel))
        made.append(fig_kappa_vs_A(S, hyp, sel))
    if final is not None and fields:
        for qin in (e13.QIN_FIT, e13.QIN_TRANSFER):
            for chunk in f12._chunks(S["A_levels"]):
                made.append(f12.fig_heat_grid(S, final, fields, qin, chunk,
                                              f"fig_e13_heat_q{qin:g}_A{chunk[0]}-{chunk[-1]}"))
        made.append(f12.fig_profiles(S, final, fields, name="fig_e13_profiles"))
    if extras is not None:
        made.append(f12.fig_ridge(S, extras, name="fig_e13_ridge",
                                  ylabel=rf"$J_{{\lambda^*}}$ ($\lambda^*$ = {sel['lambda_star']:g})"))
    for p in made:
        print("wrote", p)
    return made


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _pct(v):
    return "n/a" if v is None else f"{100 * v:+.1f}"


def _f(v, fmt=".2f"):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)


def write_summary(S):
    ladder, sel, hyp, extras, final, pre, _ = load(S)
    keys = [f"A{a:g}" for a in S["A_levels"]]
    ls = sel["lambda_star"]
    lk = e13.lam_key(ls)
    L = [f"# E13 v2 summary — joint calibration (W1 + queue + overtaking + event counts), A = "
         f"{S['A_levels'][0]}..{S['A_levels'][-1]}\n",
         f"Objective: {ladder['_meta']['objective']}; N_0(A) = max({e13.N_MIN:g}, peak queue), "
         f"EPS_0 = {e13.EPS_0:g}; eps_R = {ladder['_meta']['eps_R']}. Sweep {ladder['_meta']['lambdas']}. "
         f"lambda* = {ls:g} (rule: {sel['rule']}; dominance {sel['lambda_dominance']}, restricted knee "
         f"{sel['lambda_knee_restricted']}, lambda_max {sel['lambda_max']}, per-A exchange-rate spread "
         f"{_f(sel['per_A_spread'], '.1f')}, two-lambda ladder: {sel['two_lambda_ladder']}). "
         "Model unchanged from E12 (pure E7).\n"]
    if pre is not None:
        L.append("## 0. Reachable-set pre-check (stage p): can the pure model reach the mechanism box "
                 f"(MAE_N <= {e13.BOX_NS:g} veh, |eps_omega| <= {100 * e13.BOX_OM:.0f} %) at all?\n")
        L.append("| A | grid points | in box | in box & field ok | + eps_R <= 0.5 | min MAE_N | min |eps_omega| | "
                 "min eps_R | classical (MAE_N, eps_omega, eps_R) | best-penalty point (slice, kc, kr, W1/W1_cl, MAE_N, eps, eps_R) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for k in keys:
            p = pre[k]
            b = p["best_penalty_point"]
            c = p["classical"]
            L.append(f"| {k[1:]} | {p['n_points']} | {p['n_in_box']} | {p['n_in_box_field_ok']} | "
                     f"{p['n_in_box_field_ok_events_ok']} | {p['min_Ns_mae']:.1f} | {100 * p['min_abs_omega']:.0f} % | "
                     f"{p['min_eps_R']:.2f} | ({c['Ns_mae']:.1f}, {_pct(c['omega_err'])} %, {c['eps_R']:.2f}) | "
                     f"({b['slice']}, {b['kappa_c']:.2e}, {b['kappa_r']:.2e}, {b['W1_rel']:.2f}, {b['Ns_mae']:.1f}, "
                     f"{_pct(b['omega_err'])} %, {b['eps_R']:.2f}) |")
    L.append("\n## 1. Lambda sweep (per-A winners, dt = 0.5); classical J = 1 + lambda P_cl\n")
    L.append("| lambda | sum W1/W1_cl | sum penalty | sum J | sum J classical | field ok | collapsed | in box | "
             "eps_R ok | J < J_cl | mean W1 | mean MAE_N | mean |eps| | mean eps_R | median tau [s] | knee dist |")
    L.append("|" + "---|" * 16)
    for r in sel["sweep"]:
        L.append(f"| {r['lam']:g} | {r['sum_W1_rel']:.3f} | {r['sum_penalty']:.3f} | {r['sum_J']:.3f} | "
                 f"{r['sum_J_classical']:.3f} | {r['n_field_ok']} | {r['n_collapsed']} | {r['n_in_box']} | "
                 f"{r['n_events_ok']} | {r['n_beats_classical']} | {r['mean_W1']:.1f} | {r['mean_Ns_mae']:.1f} | "
                 f"{100 * r['mean_abs_omega']:.0f} % | {r['mean_eps_R']:.2f} | {_f(r['median_tau_s'], '.1f')} | "
                 f"{_f(r['knee_distance'], '.3f')} |")
    L.append(f"\nPer-A exchange rates dW/dP (lambda 0 -> {sel['lambda_max'] if sel['lambda_max'] is not None else 'max'}): "
             + ", ".join(f"A{k[1:]} {_f(v, '.3f')}" for k, v in sel["per_A_exchange_rate"].items())
             + f". Closed-form knee: {sel['closed_form_knee']} (agree within x1.5: {sel['closed_form_agree_x1p5']}).\n")
    L.append(f"## 2. Per-A at lambda* = {ls:g}: classical / lambda = 0 (E12) / E13\n")
    L.append("| A | winner | kappa_c | kappa_r | gamma | w_s/w | start | W1 | MAE_N | eps_omega [%] | eps_R [dex] | "
             "tau [s] (data) | C_m / C_d | R_m / R_d | wake (data) | W1 q2000 | flags |")
    L.append("|" + "---|" * 17)
    for k in keys:
        b = ladder[k]
        w, blk = _winner(ladder, k, ls)
        w0, blk0 = _winner(ladder, k, 0.0)
        cl = b["classical"][KEY_FIT]
        p, p0 = w["parts_dt05"], w0["parts_dt05"]
        e = w["eval"][KEY_FIT]; e0 = w0["eval"][KEY_FIT]
        g_s = "-" if w["gamma"] is None else f"{w['gamma']:.3f}"
        ws_s = "-" if w["ws_frac"] is None else f"{w['ws_frac']:.3f}"
        fl = ",".join(n for n, v in blk["flags"].items() if v) or "-"
        L.append(f"| {k[1:]} | {blk['winner']} | {w['kappa_c']:.3e} | {w['kappa_r']:.3e} | {g_s} | {ws_s} | "
                 f"{w['start_used']} | {cl['W1']:.1f} / {p0['W1']:.1f} / {p['W1']:.1f} | "
                 f"{cl['Ns_mae']:.1f} / {p0['Ns_mae']:.1f} / {p['Ns_mae']:.1f} | "
                 f"{_pct(cl['omega_err'])} / {_pct(p0['omega_err'])} / {_pct(p['omega_err'])} | "
                 f"{cl['eps_R']:.2f} / {p0['eps_R']:.2f} / {p['eps_R']:.2f} | "
                 f"{_f(p0['tau_model_s'], '.1f')} / {_f(p['tau_model_s'], '.1f')} ({_f(b['ref']['tau_data_s'], '.0f')}) | "
                 f"{p['C_m']:.0f} / {b['ref']['C_d']:.0f} | {p['R_m']:.0f} / {b['ref']['R_d']:.0f} | "
                 f"{cl['wake']:.1f} / {e0['wake']:.1f} / {e['wake']:.1f} ({b['data'][KEY_FIT]['wake']:.1f}) | "
                 f"{b['classical'][KEY_TR]['W1']:.1f} / {blk0['transfer'][KEY_TR]['W1']:.1f} / "
                 f"{blk['transfer'][KEY_TR]['W1']:.1f} | {fl} |")
    L.append(f"\n## 3. Configuration comparison at lambda* (J at dt = 0.5; W1 / MAE_N / eps / eps_R)\n")
    names = [n for n, _ in e13.CONFIGS]
    L.append("| A | J classical | " + " | ".join(names) + " |")
    L.append("|---|---|" + "---|" * len(names))
    for k in keys:
        blk = ladder[k]["lambdas"][lk]
        cells = [k[1:], f"{blk['J_classical']:.3f}"]
        for n in names:
            f = blk["fits"][n]; p = f["parts_dt05"]
            star = "**" if n == blk["winner"] else ""
            cells.append(f"{star}{f['J_dt05']:.3f}{star} ({p['W1']:.0f} / {p['Ns_mae']:.0f} / "
                         f"{100 * p['omega_err']:+.0f}% / {p['eps_R']:.1f})")
        L.append("| " + " | ".join(cells) + " |")
    L.append("\nOptimizer path check (winner W1 non-decreasing / penalty non-increasing in lambda): "
             + ", ".join(f"A{k[1:]} {ladder[k]['path_check']['W1_nondecreasing_violations']}/"
                         f"{ladder[k]['path_check']['penalty_nonincreasing_violations']}" for k in keys) + "\n")
    p12 = e12.SETTINGS["out"] / "ladder.json"
    if p12.exists():
        L12 = json.loads(p12.read_text())
        L.append("## 4. lambda = 0 consistency vs E12 C1 (W1 at dt = 0.5; E13 C1 may be better: wider grid, dt = 0.5 polish)\n")
        L.append("| A | E12 C1 W1 | E13 C1 (lambda=0) W1 | diff | start |")
        L.append("|---|---|---|---|---|")
        for k in keys:
            if k in L12:
                a = L12[k]["fits"]["C1"]["eval"][KEY_FIT]["W1"]
                f = ladder[k]["lambdas"][e13.lam_key(0.0)]["fits"]["C1"]
                bv = f["parts_dt05"]["W1"]
                L.append(f"| {k[1:]} | {a:.2f} | {bv:.2f} | {bv - a:+.2f} | {f['start_used']} |")
    if hyp:
        L.append("\n## 5. Nested shared-parameter models vs lambda (sum_A J at dt = 1)\n")
        L.append("| lambda | M_none | M_c | M_r | M_both | gap / M_both | eta_c | eta_r | ridge resid [dex] | "
                 "M_c kc(A) Spearman | M_r kr(A) Spearman | M_both kc/kr Spearman | micro-macro kappa_r (median dex, rho) |")
        L.append("|" + "---|" * 13)
        for lkk, q in hyp["per_lambda"].items():
            t = q["totals_dt1"]; mm = q["micro_macro"]["kappa_r"]
            L.append(f"| {q['lam']:g} | {t['M_none']:.3f} | {t['M_c']:.3f} | {t['M_r']:.3f} | {t['M_both']:.3f} | "
                     f"{100 * q['gap_rel_both']:+.2f} % | {_f(q['eta_c'])} | {_f(q['eta_r'])} | {q['ridge_residual_dex']:.2f} | "
                     f"{q['M_c']['spearman']['rho']:+.2f} (p={q['M_c']['spearman']['p']:.3f}) | "
                     f"{q['M_r']['spearman']['rho']:+.2f} (p={q['M_r']['spearman']['p']:.3f}) | "
                     f"{q['M_both']['spearman_kc']['rho']:+.2f} / {q['M_both']['spearman_kr']['rho']:+.2f} | "
                     f"{mm['median_abs_dex']:.2f} dex, rho {_f(mm['spearman_macro_vs_micro']['rho'])} |")
        if hyp.get("jackknife"):
            L.append("\nJackknife (leave-one-run-out refits) of the gap:\n")
            L.append("| lambda | gap mean | jackknife SE | |gap| > 2 SE | same sign in all folds | eta_c folds | eta_r folds |")
            L.append("|---|---|---|---|---|---|---|")
            for lkk, jk in hyp["jackknife"].items():
                L.append(f"| {jk['lam']:g} | {jk['gap_mean']:+.4f} | {jk['gap_se_jack']:.4f} | "
                         f"{abs(jk['gap_mean']) > 2 * jk['gap_se_jack']} | {jk['same_sign']} | "
                         f"{', '.join(_f(v) for v in jk['eta_c'])} | {', '.join(_f(v) for v in jk['eta_r'])} |")
        L.append(f"\nAt lambda* = {ls:g}: robustness and ablations (which observable carries the gap?)\n")
        L.append("| variant | M_none | M_c | M_r | M_both | gap / M_both | eta_c | eta_r | ridge resid | nested ok |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for tag, q in [("all terms", hyp["per_lambda"][lk])] + list(hyp["at_star"].items()):
            t = q["totals_dt1"]
            L.append(f"| {tag} | {t['M_none']:.3f} | {t['M_c']:.3f} | {t['M_r']:.3f} | {t['M_both']:.3f} | "
                     f"{100 * q['gap_rel_both']:+.2f} % | {_f(q['eta_c'])} | {_f(q['eta_r'])} | "
                     f"{q['ridge_residual_dex']:.2f} | {all(q['nested'].values())} |")
        q = hyp["per_lambda"][lk]
        L.append(f"\nPer-A ladders at lambda* = {ls:g}:\n")
        L.append("| A | M_c kappa_c | M_r kappa_r | M_both kappa_c | M_both kappa_r | ratio | M_both tau [s] | "
                 "M_both eps_R | event kappa_c | event kappa_r |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        ev = hyp["events"].get(KEY_FIT, {})
        for k in keys:
            e = ev.get(k)
            pb = q["M_both"]["per_A_parts"][k]
            L.append(f"| {k[1:]} | {q['M_c']['ladder'][k]:.3e} | {q['M_r']['ladder'][k]:.3e} | "
                     f"{q['M_both']['ladder_kc'][k]:.3e} | {q['M_both']['ladder_kr'][k]:.3e} | "
                     f"{q['M_both']['ladder_kc'][k] / q['M_both']['ladder_kr'][k]:.1f} | {_f(pb['tau_model_s'], '.1f')} | "
                     f"{pb['eps_R']:.2f} | " + ("n/a | n/a |" if e is None else f"{e['kappa_c'][0]:.3e} | {e['kappa_r'][0]:.3e} |"))
        mm = q["micro_macro"]
        L.append(f"\nMicro-macro agreement at lambda* (criterion: median |log10 ratio| <= {e13.AGREE_DEX} dex and "
                 f"Spearman(macro, micro) >= {e13.AGREE_RHO}): kappa_c median {mm['kappa_c']['median_abs_dex']:.2f} dex, "
                 f"rho {_f(mm['kappa_c']['spearman_macro_vs_micro']['rho'])} -> magnitude {mm['kappa_c']['magnitude_agree']}, "
                 f"trend {mm['kappa_c']['trend_agree']}; kappa_r median {mm['kappa_r']['median_abs_dex']:.2f} dex, "
                 f"rho {_f(mm['kappa_r']['spearman_macro_vs_micro']['rho'])} -> magnitude {mm['kappa_r']['magnitude_agree']}, "
                 f"trend {mm['kappa_r']['trend_agree']}.\n")
    if extras:
        L.append(f"## 6. Identifiability under J_lambda* (ridge scan, dt = 1), turnover, event-MLE kappas\n")
        L.append("| A | J min | argmin (ratio, kappa_r) | basin fixed ratio [dex] | basin along valley 1 % / 2 % [dex] | "
                 "best kappa_c per magnitude | magnitude identified (valley rule)? | "
                 "winner kappa_r | tau data / lambda0 / lambda* [s] | C_d, R_d | J event-MLE (+w_s) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for k in keys:
            r = extras["ridge"][k]; t = extras["turnover"][k]; e = extras["events_field"].get(k)
            bv = r["basin_width_dex_along_valley"]
            kcs = ", ".join(f"{v:.2g}" for v in r["best_kappa_c_per_magnitude"])
            L.append(f"| {k[1:]} | {r['min']:.3f} | ({r['argmin']['ratio']:.1f}, {r['argmin']['magnitude']:g}) | "
                     f"{r['basin_width_dex_along_best_ratio']:.1f} | {bv['1pct']:.1f} / {bv['2pct']:.1f} | {kcs} | "
                     f"{r['magnitude_identified']} | {r['winner_kappa_r']:.2e} | "
                     f"{_f(t['tau_data_s'], '.0f')} / {_f(t['tau_lam0_s'], '.1f')} / {_f(t['tau_lamstar_s'], '.1f')} | "
                     f"{t['C_d']:.0f}, {t['R_d']:.0f} | " + ("n/a |" if e is None else f"{e['J_dt05']:.3f} ({e['J_ws_dt05']:.3f}) |"))
        L.append("\n## 7. Resolution sensitivity (CAV density = 1/dx, so dx belongs to the model definition; 50 m = KDE scale)\n")
        L.append("| A | model | dx/dt | W1 | MAE_N | eps_omega [%] | eps_R | wake | peak CAV-cell rho [veh/km] | N_s max |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for k, d in extras["dx_sensitivity"].items():
            for model, dd in d.items():
                for key in sorted(kk for kk in dd if kk.startswith("dx")):
                    v = dd[key]
                    L.append(f"| {k[1:]} | {model} | {key} | {v['W1']:.1f} | {v['Ns_mae']:.1f} | {_pct(v['omega_err'])} | "
                             f"{v['eps_R']:.2f} | {v['wake']:.1f} | {v['peak_cav_cell_rho_vehkm']:.0f} | {v['N_s_max_veh']:.1f} |")
    text = "\n".join(L) + "\n"
    (Path(S["out"]) / "summary13.md").write_text(text)
    return text
