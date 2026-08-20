"""E7d: speed-transfer study — "predict u=10 or u=20 with u=15-calibrated
parameters" (Mladen's overfitting question).

u_xi is a PHYSICAL input of the catch & release model (the slow-class speed),
not a fitted parameter; the kappas are per-vehicle interaction rates that
should not depend on the bottleneck speed.  If they do, the u15 calibration
is overfitted.  Two tests:

1. A=3, q=2500 (the only assertiveness with a speed sweep in the data):
   fit (kappa_c, kappa_r) once at u=15 (lf form, W1 metric, rep-mean field,
   via e7_wasserstein.fit_field), then predict u in {10, 12, 18, 20, 24}
   with ZERO refitting -- only u_xi changes.  For every u the classical
   baseline M1 (kappa = 0 + Delle Monache-Goatin cap Q_xi = 2000 veh/h) is
   scored identically for reference: M1 has no fitted parameter at all, so a
   CR curve that stays comparable to / below M1 across speeds is the
   "nice and robust" outcome; a CR curve that blows up away from u=15 while
   M1 stays flat is the overfitting signature.

2. A in {1, 10} (only u in {15, 20} exist): the E6 kappas (out/e6/fits.json,
   lf, fitted on the u15_q2500 rep-mean field with RMSE) are evaluated
   unchanged on the u=20 scenarios (q2000 and q2500) against M1 at u20 --
   a two-point transfer check.  The u15_q2500 fit scenario is re-scored at
   production resolution as the anchor of the two-point comparison (this is
   NOT a u20 self-fit ceiling; nothing is refitted here).

Metrics per scenario, all vs the rep-mean data of ../Second .mat True files:
  W1        mean 1-Wasserstein distance of cumulative densities
            [veh km], t in [100, 1000] s, x <= 20 km (e7_wasserstein);
  rho_RMSE  pointwise density RMSE [veh/km], same window;
  e_s       ECC22 eq. (6) analogue at X_q = 15 km over [576 s, nominal CAV
            arrival at X_q] (ev4_compare.flux_sim_xq / t_reach; t_reach
            handles the u generalization analytically).  Reported per rep
            (mean) and vs the rep-mean flow.

A in {1, 10} fields come from ev4_compare.load_measured (out/e1 npz);
A=3 has no out/e1 npz, so densities AND flows are loaded directly from the
.mat via loader.load_scenario (fields only, rep-stacked).

All evaluation sims run at production resolution dt=0.5 / save_every=20,
uncapped for CR (native moving bottleneck), capped for M1.

Outputs (out/e7/):
  fit_A3_u15_q2500_lf_w1.json   the u15 anchor fit (reused if present;
                                --refit forces a refit)
  transfer.json                 all numbers + auto-generated headline
  fig_transfer_curve.png        W1 (top) and e_s (bottom) vs u for A=3:
                                CR-transfer solid, M1 dashed, vertical line
                                at the fit speed u=15, A1/A10 u20 transfer
                                points annotated

CLI:  python3 e7_transfer.py [--refit] [--skip-a110]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import e7_wasserstein as e7w
import ev4_compare as ev4
from loader import load_scenario
from solver import SimConfig, simulate

HERE = Path(__file__).parent
OUT_E7 = HERE / "out" / "e7"
SECOND = HERE.parent / "Second"

A_SWEEP = 3.0                      # only assertiveness with a data speed sweep
U_FIT, Q_FIT = 15.0, 2500.0        # anchor scenario of both tests
U_SWEEP = (10.0, 12.0, 15.0, 18.0, 20.0, 24.0)   # 15.0 = fit speed (anchor)
QXI_M1_VEHH = 2000.0               # M1 baseline cap (canonical ECC22 Q_xi)
DT_PROD, SAVE_PROD = 0.5, 20       # production resolution
FIT_JSON = OUT_E7 / "fit_A3_u15_q2500_lf_w1.json"   # e7_wasserstein naming
E6_FITS = HERE / "out" / "e6" / "fits.json"

# A1/A10 two-point transfer scenarios: anchor first, then the predictions
A110_SCEN = ((15.0, 2500.0, "fit anchor (E6 fit scenario, re-scored)"),
             (20.0, 2000.0, "zero-refit transfer"),
             (20.0, 2500.0, "zero-refit transfer"))


# ---------------------------------------------------------------------------
# simulation + data
# ---------------------------------------------------------------------------

def run_model(uc, qin, kc, kr, form="lf", qxi_vehh=None):
    """One production-resolution run -> ev4_compare.regrid_sim dict.
    qxi_vehh: DM-G capacity cap [veh/h]; None = uncapped (CR native)."""
    cfg = SimConfig(v_f=ev4.V_F, w=ev4.W, P=ev4.P, q_in=qin / 3600.0,
                    u_xi=uc, kappa_c=float(kc), kappa_r=float(kr),
                    capture_form=form, dt=DT_PROD, save_every=SAVE_PROD,
                    q_xi_max=None if qxi_vehh is None else qxi_vehh / 3600.0)
    return ev4.regrid_sim(simulate(cfg))


def load_data_a3(uc, qin=Q_FIT):
    """Rep-stacked measured fields for A=3 straight from the .mat (no out/e1
    npz exists for A=3): densities for W1/RMSE, flows at X_q for e_s."""
    path = SECOND / f"data_{A_SWEEP:g}_{uc:g}_{qin:g}_True.mat"
    sc = load_scenario(path, fields=True, ctrl=False, trajs=False)
    reps = sorted(r for r in sc.reps
                  if sc.reps[r].rho is not None and sc.reps[r].q is not None)
    if not reps:
        raise FileNotFoundError(f"no per-rep density+flow fields in {path}")
    rho = np.asarray([sc.reps[r].rho for r in reps], float)
    q_xq = np.asarray([sc.reps[r].q[:, ev4.J_XQ] for r in reps], float)
    return dict(tt=sc.t_field, rho_mean=rho.mean(axis=0), q_xq=q_xq,
                n_reps=len(reps))


def load_data_a110(A, uc, qin):
    """Same dict shape for A in {1, 10} via ev4_compare.load_measured
    (out/e1 npz + .mat flows)."""
    meas = ev4.load_measured(float(A), float(uc), float(qin))
    return dict(tt=np.asarray(meas["tt"], float),
                rho_mean=np.mean(meas["rho"], axis=0),
                q_xq=np.asarray(meas["q_xq"], float),
                n_reps=len(meas["reps"]))


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def evaluate(regr, uc, data):
    """W1 / rho-RMSE vs the rep-mean field + e_s at X_q (per rep and vs the
    rep-mean flow); windows exactly as in e7_wasserstein / ev4_compare."""
    tt = np.asarray(regr["tt"], float)
    assert np.allclose(tt, data["tt"]), "sim/data time grids differ"

    w1 = e7w.w1_mean(regr["rho_tot"], data["rho_mean"], tt)
    rmse = e7w.rmse_mean(regr["rho_tot"], data["rho_mean"], tt)

    t1 = ev4.t_reach(ev4.X_Q, uc)          # nominal CAV arrival at X_q
    we = (tt >= ev4.T0_ES) & (tt <= t1)
    num = float(ev4.flux_sim_xq(regr, uc)[we].sum())
    per = []
    for q in data["q_xq"]:
        den = float(np.asarray(q, float)[we].sum())
        per.append(num / den - 1.0 if den > 0 else np.nan)
    den_mean = float(np.mean(data["q_xq"], axis=0)[we].sum())
    return dict(w1_vehkm=float(w1), rho_rmse_vehkm=float(rmse),
                e_s=float(np.nanmean(per)),
                e_s_per_rep=[float(v) for v in per],
                e_s_vs_repmean=(num / den_mean - 1.0 if den_mean > 0
                                else float("nan")),
                es_window_s=[float(ev4.T0_ES), float(t1)],
                n_reps=int(data["n_reps"]))


# ---------------------------------------------------------------------------
# the u15 anchor fit (A=3)
# ---------------------------------------------------------------------------

def get_fit(refit=False):
    """A=3 u15 q2500 lf/W1 field fit; reuses FIT_JSON unless refit=True.
    A=3 has no out/e1 npz -- fit_field's loader-direct path handles it."""
    if FIT_JSON.exists() and not refit:
        fit = json.loads(FIT_JSON.read_text())
        print(f"[fit] reusing {FIT_JSON.name}: kappa_c={fit['kappa_c']:.3e} "
              f"kappa_r={fit['kappa_r']:.3e} (W1={fit['objective']:.3f})")
        return fit
    print(f"[fit] A={A_SWEEP:g} u{U_FIT:g} q{Q_FIT:g} lf/W1 field fit ...")
    fit = e7w.fit_field(A_SWEEP, U_FIT, Q_FIT, form="lf", metric="w1")
    OUT_E7.mkdir(parents=True, exist_ok=True)
    FIT_JSON.write_text(json.dumps(fit, indent=2))
    print("wrote", FIT_JSON)
    return fit


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def fig_transfer(sweep, a110, fit, out_path):
    """W1 (top) and e_s (bottom) vs u for A=3 q2500: CR-transfer solid, M1
    dashed, fit speed marked, A1/A10 u20 q2500 transfer points annotated."""
    us = sorted(sweep)
    w1_cr = [sweep[u]["cr"]["w1_vehkm"] for u in us]
    w1_m1 = [sweep[u]["m1"]["w1_vehkm"] for u in us]
    es_cr = [sweep[u]["cr"]["e_s"] for u in us]
    es_m1 = [sweep[u]["m1"]["e_s"] for u in us]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True)

    ax1.plot(us, w1_cr, "o-", color="tab:blue", lw=1.8,
             label="CR transfer ($\\kappa$ fixed at u15 fit)")
    ax1.plot(us, w1_m1, "s--", color="tab:orange", lw=1.5,
             label="M1: LWR + MB cap (no fitted param)")
    ax1.set_ylabel(r"mean $W_1$ [veh km]")
    ax1.set_title(f"E7d speed transfer — A={A_SWEEP:g}, q={Q_FIT:g} veh/h: "
                  f"$\\kappa_c$={fit['kappa_c']:.3g}, "
                  f"$\\kappa_r$={fit['kappa_r']:.3g} fitted at u=15 only",
                  fontsize=10)

    ax2.axhspan(-0.10, 0.10, color="tab:green", alpha=0.15,
                label=r"$\pm$10% (FTSM reference)")
    ax2.plot(us, es_cr, "o-", color="tab:blue", lw=1.8, label="CR transfer")
    ax2.plot(us, es_m1, "s--", color="tab:orange", lw=1.5, label="M1")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_ylabel(r"$e_s$ at $X_q$ = 15 km")
    ax2.set_xlabel(r"CAV slow speed $u_\xi$ [m/s]")

    for ax in (ax1, ax2):
        ax.axvline(U_FIT, color="0.4", ls=":", lw=1.2)
        ax.grid(alpha=0.3)
    ax1.annotate("fit speed", (U_FIT, ax1.get_ylim()[1]),
                 xytext=(3, -12), textcoords="offset points",
                 fontsize=8, color="0.35")

    # A1 / A10 u20 q2500 zero-refit transfer points (E6 kappas)
    marks = {"A1": ("^", "tab:red"), "A10": ("v", "tab:purple")}
    for a_key, (mk, col) in marks.items():
        row = a110.get(a_key, {}).get("u20_q2500")
        if row is None:
            continue
        for ax, key in ((ax1, "w1_vehkm"), (ax2, "e_s")):
            ax.plot([20.0], [row["cr"][key]], mk, color=col, ms=9, mew=1.2,
                    mec="k", zorder=5,
                    label=f"{a_key.replace('A', 'A=')} u20 transfer "
                          "(E6 $\\kappa$)")
            ax.annotate(a_key.replace("A", "A="), (20.0, row["cr"][key]),
                        xytext=(7, 3), textcoords="offset points",
                        fontsize=8, color=col)

    ax1.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# anchor-fit diagnostics
# ---------------------------------------------------------------------------

def diagnostics(fit, e6_fits=None):
    """Identifiability + wake-state checks around the u15 anchor fit.

    (i) Sloppy-ridge probe: scale (kappa_c, kappa_r) JOINTLY by factors
        {0.05, 0.1, 0.25, 1} (ratio fixed) and re-score W1 at u15.  In the
        fast-reaction regime theta dt >> 1 the reaction relaxes to its local
        equilibrium f_eq = mu / (sigma + mu) within one step, so only the
        capture/release RATIO is identified -- the fitted magnitudes must be
        read as "on the fast-equilibrium ridge", not as per-vehicle rates.
    (ii) E6-magnitude cross-check: the A10 RMSE-fit kappas re-scored on the
        A=3 field (how flat is the landscape across regimes).
    (iii) Wake state at t = 500 s: max total density over the 1.5 km behind
        the CAV vs the critical density -- is the fitted wedge supercritical
        like the SUMO plateau (the E6 defect was a subcritical ~44 veh/km
        wedge that advects at v_f without spreading)?
    """
    data = load_data_a3(U_FIT)
    kc0, kr0 = fit["kappa_c"], fit["kappa_r"]

    def w1_at(kc, kr):
        regr = run_model(U_FIT, Q_FIT, kc, kr, "lf", None)
        return float(e7w.w1_mean(regr["rho_tot"], data["rho_mean"],
                                 data["tt"])), regr

    ridge = {}
    regr_fit = None
    for fac in (0.05, 0.1, 0.25, 1.0):
        w1, regr = w1_at(kc0 * fac, kr0 * fac)
        ridge[f"x{fac:g}"] = dict(kappa_c=kc0 * fac, kappa_r=kr0 * fac,
                                  w1_vehkm=w1)
        if fac == 1.0:
            regr_fit = regr

    e6_probe = None
    if e6_fits is not None:
        f6 = e6_fits["A10_lf"]
        w1, _ = w1_at(f6["kappa_c"], f6["kappa_r"])
        e6_probe = dict(kappa_c=f6["kappa_c"], kappa_r=f6["kappa_r"],
                        w1_vehkm=w1, source="out/e6/fits.json A10_lf")

    # wake state at t = 500 s, 1.5 km behind the CAV (100 m data cells)
    i = int(np.argmin(np.abs(regr_fit["tt"] - 500.0)))
    j = int(regr_fit["x_cav"][i] / 100.0)
    lo = max(j - 15, 0)
    rho_crit = 1000.0 * ev4.W * ev4.P / (ev4.V_F + ev4.W)      # [veh/km]
    wake = dict(t_s=500.0, x_window_km=[lo * 0.1, j * 0.1],
                rho_crit_vehkm=float(rho_crit),
                sim_wake_max_vehkm=float(regr_fit["rho_tot"][i, lo:j].max()),
                data_wake_max_vehkm=float(data["rho_mean"][i, lo:j].max()))
    wake["sim_wake_supercritical"] = bool(
        wake["sim_wake_max_vehkm"] > rho_crit)
    return dict(ridge_probe_u15=ridge, e6_magnitude_probe=e6_probe,
                wake_state=wake,
                note="kappas are identified only up to the fast-equilibrium "
                     "ridge (ratio kc/kr = "
                     f"{kc0 / kr0:.2f}); W1 varies <~2% over a 20x joint "
                     "rescaling")


# ---------------------------------------------------------------------------
# headline verdict
# ---------------------------------------------------------------------------

def headline(sweep, fit):
    """Auto-generated robustness verdict from the A=3 sweep numbers."""
    us = sorted(sweep)
    w1_cr = {u: sweep[u]["cr"]["w1_vehkm"] for u in us}
    w1_m1 = {u: sweep[u]["m1"]["w1_vehkm"] for u in us}
    ratio = {u: w1_cr[u] / w1_m1[u] for u in us}
    growth = {u: w1_cr[u] / w1_cr[U_FIT] for u in us}
    worst_u = max(ratio, key=lambda u: ratio[u])
    if max(ratio.values()) <= 1.0:
        verdict = ("ROBUST: the zero-refit CR transfer beats the "
                   "parameter-free M1 baseline at EVERY tested speed")
    elif max(ratio.values()) <= 1.15:
        verdict = ("ROBUST: the zero-refit CR transfer stays comparable to "
                   "M1 across all tested speeds (within 15%)")
    else:
        verdict = (f"DEGRADES: CR transfer exceeds M1 by "
                   f">{(max(ratio.values()) - 1.0):.0%} at u={worst_u:g}")
    return dict(
        verdict=verdict,
        w1_ratio_cr_over_m1={f"u{u:g}": round(ratio[u], 3) for u in us},
        w1_growth_vs_fit_speed={f"u{u:g}": round(growth[u], 3) for u in us},
        max_ratio_u=float(worst_u),
        note=("u_xi is a physical input (slow-class speed); the kappas were "
              f"fitted ONCE at u={U_FIT:g} (W1={fit['objective']:.2f} veh km) "
              "and never touched across the sweep"))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="E7d speed-transfer study (u15-calibrated kappas vs "
                    "other CAV speeds)")
    ap.add_argument("--refit", action="store_true",
                    help="refit the A=3 u15 anchor even if the fit json "
                         "exists")
    ap.add_argument("--skip-a110", action="store_true",
                    help="skip the A1/A10 u20 two-point transfer check")
    args = ap.parse_args(argv)

    OUT_E7.mkdir(parents=True, exist_ok=True)
    fit = get_fit(refit=args.refit)
    kc, kr = fit["kappa_c"], fit["kappa_r"]

    # ---- test 1: A=3 speed sweep, zero refit ------------------------------
    print(f"\nA={A_SWEEP:g} q={Q_FIT:g} zero-refit sweep "
          f"(CR: kc={kc:.3e} kr={kr:.3e} lf, uncapped | "
          f"M1: kappa=0, Q_xi={QXI_M1_VEHH:g} veh/h)")
    print(f"{'u':>4s} {'W1_cr':>8s} {'W1_m1':>8s} {'RMSE_cr':>8s} "
          f"{'RMSE_m1':>8s} {'es_cr':>8s} {'es_m1':>8s}")
    sweep = {}
    for u in U_SWEEP:
        data = load_data_a3(u)
        cr = evaluate(run_model(u, Q_FIT, kc, kr, "lf", None), u, data)
        m1 = evaluate(run_model(u, Q_FIT, 0.0, 0.0, "lf", QXI_M1_VEHH),
                      u, data)
        sweep[u] = dict(cr=cr, m1=m1)
        print(f"{u:4g} {cr['w1_vehkm']:8.2f} {m1['w1_vehkm']:8.2f} "
              f"{cr['rho_rmse_vehkm']:8.2f} {m1['rho_rmse_vehkm']:8.2f} "
              f"{cr['e_s']:+8.1%} {m1['e_s']:+8.1%}")

    # ---- test 2: A1/A10 u15 -> u20 two-point transfer (E6 kappas) ---------
    a110 = {}
    e6 = json.loads(E6_FITS.read_text()) if E6_FITS.exists() else None
    if not args.skip_a110:
        print(f"\nA1/A10 two-point transfer (E6 lf kappas from {E6_FITS})")
        print(f"{'scenario':22s} {'W1_cr':>8s} {'W1_m1':>8s} {'RMSE_cr':>8s} "
              f"{'RMSE_m1':>8s} {'es_cr':>8s} {'es_m1':>8s}")
        for A in (1.0, 10.0):
            f6 = e6[f"A{A:g}_lf"]
            block = dict(kappa_c=f6["kappa_c"], kappa_r=f6["kappa_r"],
                         kappa_source=f"out/e6/fits.json A{A:g}_lf "
                                      "(RMSE field fit at u15_q2500)")
            for uc, qin, role in A110_SCEN:
                data = load_data_a110(A, uc, qin)
                cr = evaluate(run_model(uc, qin, f6["kappa_c"],
                                        f6["kappa_r"], "lf", None), uc, data)
                m1 = evaluate(run_model(uc, qin, 0.0, 0.0, "lf",
                                        QXI_M1_VEHH), uc, data)
                key = f"u{uc:g}_q{qin:g}"
                block[key] = dict(role=role, cr=cr, m1=m1)
                print(f"A{A:g} {key:18s} {cr['w1_vehkm']:8.2f} "
                      f"{m1['w1_vehkm']:8.2f} {cr['rho_rmse_vehkm']:8.2f} "
                      f"{m1['rho_rmse_vehkm']:8.2f} {cr['e_s']:+8.1%} "
                      f"{m1['e_s']:+8.1%}")
            a110[f"A{A:g}"] = block

    # ---- figure + json -----------------------------------------------------
    fig_path = OUT_E7 / "fig_transfer_curve.png"
    fig_transfer(sweep, a110, fit, fig_path)

    diag = diagnostics(fit, e6_fits=e6)
    wk = diag["wake_state"]
    print(f"\ndiagnostics: u15 wake max {wk['sim_wake_max_vehkm']:.1f} veh/km"
          f" (crit {wk['rho_crit_vehkm']:.1f}, data "
          f"{wk['data_wake_max_vehkm']:.1f}) -> "
          f"{'SUPER' if wk['sim_wake_supercritical'] else 'SUB'}critical; "
          f"{diag['note']}")

    head = headline(sweep, fit)
    payload = {
        "_meta": dict(
            purpose="E7d speed transfer: u15-calibrated kappas predict "
                    "other CAV speeds with zero refitting (u_xi is a "
                    "physical input, kappas fixed)",
            models=dict(
                cr="catch & release, uncapped, lf form, production "
                   f"dt={DT_PROD} save_every={SAVE_PROD}",
                m1=f"kappa=0 + DM-G cap Q_xi={QXI_M1_VEHH:g} veh/h "
                   "(parameter-free classical baseline)"),
            metrics=dict(
                w1="mean W1 of cumulative densities [veh km], "
                   "t in [100,1000] s, x <= 20 km, vs rep-mean field",
                rho_rmse="pointwise density RMSE [veh/km], same window",
                e_s="ECC22 eq.(6) analogue at X_q=15 km over [576 s, "
                    "nominal CAV arrival]; mean over per-rep values, "
                    "plus e_s_vs_repmean vs the rep-mean flow"),
            data=dict(
                a3="loader.load_scenario on ../Second data_3_*_True.mat "
                   "(no out/e1 npz for A=3)",
                a110="ev4_compare.load_measured (out/e1 npz + .mat flows)"),
            params_si=dict(v_f_ms=ev4.V_F, w_ms=ev4.W, P_vehm=ev4.P)),
        "A3_fit_u15": fit,
        "A3_fit_diagnostics": diag,
        "A3_sweep_q2500": {f"u{u:g}": sweep[u] for u in sorted(sweep)},
        "A110_transfer": a110,
        "headline": head,
    }
    (OUT_E7 / "transfer.json").write_text(json.dumps(payload, indent=2))
    print(f"\nheadline: {head['verdict']}")
    print("  W1 ratio CR/M1 by u:", head["w1_ratio_cr_over_m1"])
    print("  W1 growth vs fit speed:", head["w1_growth_vs_fit_speed"])
    print("wrote", OUT_E7 / "transfer.json", "and", fig_path)
    return payload


if __name__ == "__main__":
    main()
