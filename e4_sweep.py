"""E4: assertiveness sweep -- the micro-macro link figure (analog of
Krook, Cicic & Johansson ECC 2022 Fig. 1).

For every assertiveness A available at u_xi = 15 m/s and q_in in
{2000, 2500} veh/h (True files, 14 values of A, 5 reps each):

  1. E1 classification of every rep (e1_classify.process_rep, unchanged)
     -> out/e4/A{A:g}_u15_q{q:g}_r{r}.npz  (same keys as out/e1/*.npz)
  2. Poisson-exposure MLE per (A, q_in), pooled over the 5 reps
     (ev3_calibrate_kappa.rate_series / mle, unchanged):
       kappa_c   : ell f form, model Delta v(rho)   [1/veh]
       kappa_r   : (P - rho) s form, model Delta v  [1/veh]
     with 95% CIs (rule-of-three upper bound when 0 events), per-capita
     time rates sigma_hat, mu_hat [1/h], and the occupancies
       chi_free = t_free / (t_free + t_caught)   (E-V3 "chi_occ")
       chi_sync = 1 - chi_free                   (stuck fraction in the zone)
  3. Data overtaking flow omega_xi(A, q_in): mean of the measured overtaking
     flow over the slow window [260, 740] s per rep, then mean/min/max over
     reps; reference omega_0 = q_in - rho_in u_xi with rho_in = q_in / v_f
     (free-flow inverse, ECC22 dashed line).
  4. out/paper/fig_assertiveness.{png,pdf}: (a) omega_xi vs A,
     (b) kappa_r(A), kappa_c(A), (c) chi_sync(A).

Output: out/e4/*.npz, out/e4/kappa_vs_A.json, out/paper/fig_assertiveness.*
        (the paper caption lives in out/paper/captions.md)
Usage : python3 e4_sweep.py [--force]   (--force re-runs the classification)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter

import e1_classify as e1
import fd
import paperfig as pf
from ev3_calibrate_kappa import DT_H, P, V_F, mle, rate_series
from loader import list_scenarios, load_scenario

HERE = Path(__file__).parent
OUT = HERE / "out" / "e4"
SECOND = HERE.parent / "Second"

U_XI = 15.0                       # [m/s]
Q_SET = (2000.0, 2500.0)          # [veh/h]
T_LO, T_HI = e1.T_LO, e1.T_HI     # [s] slow window used everywhere


def tag_of(A, qin, r=None):
    base = f"A{A:g}_u{U_XI:g}_q{qin:g}"
    return base if r is None else f"{base}_r{r}"


# --------------------------------------------------------------------------
# 1. classification
# --------------------------------------------------------------------------
def classify(force: bool = False):
    OUT.mkdir(parents=True, exist_ok=True)
    scens = [(p, A, qin) for p, A, uc, qin, _ in list_scenarios(SECOND, flag=True)
             if uc == U_XI and qin in Q_SET]
    scens.sort(key=lambda s: (s[2], s[1]))
    n_done = 0
    t0 = time.time()
    for p, A, qin in scens:
        todo = [r for r in range(5)
                if force or not (OUT / f"{tag_of(A, qin, r)}.npz").exists()]
        if not todo:
            continue
        sc = load_scenario(p, fields=True, ctrl=True, trajs=True)
        for r in sorted(sc.reps):
            if r not in todo:
                continue
            rep = sc.reps[r]
            if rep.rho is None or rep.ctrl is None or not rep.trajs:
                print(f"  skip {tag_of(A, qin, r)}: incomplete rep", flush=True)
                continue
            res = e1.process_rep(sc, r)
            np.savez_compressed(OUT / f"{tag_of(A, qin, r)}.npz", **res)
            n_done += 1
            print(f"{tag_of(A, qin, r)}: cap={int(res['cap'].sum()):4d} "
                  f"rel={int(res['rel'].sum()):4d} "
                  f"maxN_s={int(res['n_caught'].max()):3d}  "
                  f"[{time.time() - t0:5.0f} s]", flush=True)
    print(f"classification: {n_done} new runs, {len(scens)} scenarios", flush=True)
    return [(A, qin) for _, A, qin in scens]


# --------------------------------------------------------------------------
# 2./3. MLE + overtaking flow per (A, q_in)
# --------------------------------------------------------------------------
def omega_0(qin: float) -> float:
    """Free-flow inverse reference [veh/h]: q_in - rho_in u_xi, rho_in = q_in/v_f."""
    rho_in = qin / V_F                       # [veh/km]
    return qin - rho_in * U_XI * fd.MS_TO_KMH


def analyse(pairs):
    results = {}
    for A, qin in pairs:
        files = sorted(OUT.glob(f"{tag_of(A, qin)}_r*.npz"))
        if not files:
            continue
        acc = defaultdict(float)
        n_cap = n_rel = 0
        t_caught = t_free = 0.0
        per_rep = dict(cap=[], rel=[], omega=[], max_caught=[], rel_unconf=[],
                       overtakes=[], passthru=[])
        for f in files:
            d = np.load(f)
            w = d["in_win"]
            rs = rate_series(d, U_XI)
            for k, v in rs.items():
                acc[k] += float(v[w].sum()) * DT_H
            c, rl = int(d["cap"][w].sum()), int(d["rel"][w].sum())
            n_cap += c
            n_rel += rl
            t_caught += float(d["n_caught"][w].sum()) * DT_H
            t_free += float(d["n_free_zone"][w].sum()) * DT_H
            # overtake_meas lives on t = (0..100)*10 s; drop t=0 -> tt grid
            ov = np.asarray(d["overtake_meas"], float)[1:]
            per_rep["omega"].append(float(ov[w].mean()))          # [veh/h]
            per_rep["cap"].append(c)
            per_rep["rel"].append(rl)
            per_rep["rel_unconf"].append(int(d["rel_unconf"][w].sum()))
            per_rep["overtakes"].append(int(d["overtakes"][w].sum()))
            per_rep["passthru"].append(int(d["passthru"][w].sum()))
            per_rep["max_caught"].append(int(d["n_caught"].max()))
        om = np.array(per_rep["omega"])
        tot = t_free + t_caught
        row = dict(A=A, uc=U_XI, qin=qin, n_reps=len(files),
                   n_cap=n_cap, n_rel=n_rel,
                   t_caught_vehh=t_caught, t_free_vehh=t_free,
                   sigma_time=n_cap / t_free if t_free > 0 else np.nan,
                   mu_time=n_rel / t_caught if t_caught > 0 else np.nan,
                   chi_free=t_free / tot if tot > 0 else np.nan,
                   chi_sync=t_caught / tot if tot > 0 else np.nan,
                   kappa_c=list(mle(n_cap, acc["Ecl_mod"])),
                   kappa_r=list(mle(n_rel, acc["Er_mod"])),
                   kappa_cA=list(mle(n_cap, acc["EcA_mod"])),
                   kappa_c_emp=list(mle(n_cap, acc["Ecl_emp"])),
                   kappa_r_emp=list(mle(n_rel, acc["Er_emp"])),
                   exp_c=acc["Ecl_mod"], exp_r=acc["Er_mod"],
                   exp_cA=acc["EcA_mod"],
                   omega_xi=dict(mean=float(om.mean()), min=float(om.min()),
                                 max=float(om.max()), per_rep=per_rep["omega"]),
                   omega_0=omega_0(qin),
                   per_rep={k: v for k, v in per_rep.items() if k != "omega"})
        results[tag_of(A, qin)] = row
        kc, kr = row["kappa_c"], row["kappa_r"]
        print(f"{tag_of(A, qin):>16s}: cap={n_cap:4d} rel={n_rel:4d} "
              f"chi_sync={row['chi_sync']:.2f} "
              f"kc={kc[0]:.2e} [{kc[1]:.1e},{kc[2]:.1e}] "
              f"kr={kr[0]:.2e} [{kr[1]:.1e},{kr[2]:.1e}] "
              f"mu={row['mu_time']:6.1f}/h  omega={om.mean():6.0f} "
              f"[{om.min():.0f},{om.max():.0f}] (omega_0={row['omega_0']:.0f})",
              flush=True)
    return results


# --------------------------------------------------------------------------
# 4. figure
# --------------------------------------------------------------------------
Q_STYLE = {2500.0: dict(color=pf.COL["data"], marker="o", label=r"$q_{in}=2500$ veh/h"),
           2000.0: dict(color="0.55", marker="s", label=r"$q_{in}=2000$ veh/h")}
KC_COL, KR_COL = pf.COL["stuck"], "tab:purple"   # panel (b) coefficient colours
YMIN_B = 5e-6                     # [1/veh] floor of the log axis in panel (b)


def _log_x(ax):
    ax.set_xscale("log")
    ax.set_xlim(0.9, 11.0)
    ax.set_xticks([1, 2, 3, 5, 10])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("assertiveness $A$ [-]")


def make_figure(results):
    pf.setup()
    fig, axes = pf.figure("double", height=2.35, nrows=1, ncols=3)
    ax_a, ax_b, ax_c = axes

    for qin in Q_SET:
        rows = sorted((r for r in results.values() if r["qin"] == qin),
                      key=lambda r: r["A"])
        if not rows:
            continue
        A = np.array([r["A"] for r in rows])
        st = Q_STYLE[qin]

        # (a) overtaking flow -----------------------------------------------
        om = np.array([[r["omega_xi"][k] for k in ("mean", "min", "max")]
                       for r in rows])
        ax_a.fill_between(A, om[:, 1], om[:, 2], color=st["color"], alpha=0.15,
                          lw=0)
        ax_a.plot(A, om[:, 0], "-", color=st["color"], marker=st["marker"],
                  ms=3.5, mfc=(st["color"] if qin == 2500.0 else "white"),
                  label=st["label"])
        ax_a.axhline(rows[0]["omega_0"], ls="--", color=st["color"], lw=1.0,
                     alpha=0.8)
        ax_a.text(1.3, rows[0]["omega_0"] + 6, rf"$\omega_0({qin:g})$",
                  color=st["color"], ha="left", va="bottom", fontsize=7)

        # (b) kappas ------------------------------------------------------------
        for key, col, name in (("kappa_c", KC_COL, r"$\kappa_c$"),
                               ("kappa_r", KR_COL, r"$\kappa_r$")):
            k = np.array([r[key] for r in rows])
            zero = k[:, 0] <= 0
            fill = col if qin == 2500.0 else "white"
            if (~zero).any():
                lo = np.maximum(k[~zero, 1], YMIN_B)   # lo95 = 0 -> axis floor
                ax_b.errorbar(A[~zero], k[~zero, 0],
                              yerr=[k[~zero, 0] - lo,
                                    k[~zero, 2] - k[~zero, 0]],
                              fmt=st["marker"], color=col, mfc=fill, mec=col,
                              ms=3.5, capsize=1.5, elinewidth=0.7, lw=0.8,
                              ls="-", label="_nolegend_")
            if zero.any():
                ax_b.scatter(A[zero], k[zero, 2], marker="v", s=22, color=col,
                             facecolors=fill, zorder=5)

        # (c) synchronized occupancy -------------------------------------------
        chi = np.array([r["chi_sync"] for r in rows])
        ax_c.plot(A, chi, "-", color=pf.COL["stuck"], marker=st["marker"],
                  mfc=pf.COL["stuck"] if qin == 2500.0 else "white",
                  mec=pf.COL["stuck"], ms=3.5, label=st["label"])

    # cosmetics ------------------------------------------------------------------
    for ax, lab in zip(axes, "abc"):
        _log_x(ax)
        ax.text(0.03, 0.97, f"({lab})", transform=ax.transAxes, ha="left",
                va="top", fontsize=9)
    ax_a.set_ylabel(r"overtaking flow $\omega_\xi$ [veh/h]")
    ax_a.set_ylim(650, 1280)
    ax_a.legend(loc="lower right")

    ax_b.set_yscale("log")
    ax_b.set_ylim(YMIN_B, 0.2)
    ax_b.set_ylabel(r"$\hat\kappa_c,\ \hat\kappa_r$ [1/veh]")
    # marker shape/fill (q_in) is defined in panels (a) and (c); here only the
    # colour (coefficient) and the censored-point symbol need a legend entry
    handles = [Line2D([], [], color=KC_COL, marker="o", ms=3.5, lw=0.8,
                      label=r"$\hat\kappa_c$"),
               Line2D([], [], color=KR_COL, marker="o", ms=3.5, lw=0.8,
                      label=r"$\hat\kappa_r$"),
               Line2D([], [], color=KR_COL, marker="v", ms=4.5, lw=0,
                      label="upper bound")]
    ax_b.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.0),
                fontsize=6.5, handlelength=1.6, borderaxespad=0.3,
                labelspacing=0.3)

    ax_c.set_ylabel(r"synchronized occupancy $\chi_s$ [-]")
    ax_c.set_ylim(0, 1)
    ax_c.legend(loc="upper right")
    return pf.save(fig, "fig_assertiveness")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-run the classification even if npz files exist")
    args = ap.parse_args()

    pairs = classify(force=args.force)
    results = analyse(pairs)

    meta = dict(
        u_xi_ms=U_XI, window_s=[T_LO, T_HI], v_f_kmh=V_F, P_vehkm=P,
        kappa_c="ell f form (kappa_c (a+s) f Dv), model Dv(rho), 1/veh; "
                "[hat, lo95, hi95]; 0 events -> [0, 0, 3/exposure]",
        kappa_r="kappa_r (P-rho) s Dv, model Dv(rho), 1/veh; same packing",
        kappa_c_emp="same with empirical left-lane Dv (E-V3 'emp')",
        chi_free="t_free/(t_free+t_caught) (E-V3 chi_occ)",
        chi_sync="t_caught/(t_free+t_caught): stuck fraction in the queue zone",
        omega_xi="mean of the measured overtaking flow over the slow window, "
                 "per rep; mean/min/max over reps [veh/h]",
        omega_0="q_in - (q_in/v_f) u_xi [veh/h]",
        source="e4_sweep.py; classification = e1_classify.process_rep",
    )
    payload = dict(_meta=meta, **results)
    (OUT / "kappa_vs_A.json").write_text(json.dumps(payload, indent=2,
                                                    default=float))
    png = make_figure(results)
    print("wrote", OUT / "kappa_vs_A.json", png)


if __name__ == "__main__":
    main()
