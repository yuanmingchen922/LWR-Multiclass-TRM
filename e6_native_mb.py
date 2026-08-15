"""E6: can catch & release model the moving bottleneck *natively*?

Mladen's question (point 4): instead of bolting the Delle Monache-Goatin flux
constraint onto the model, calibrate (kappa_c, kappa_r) so that the UNCAPPED
catch-release dynamics reproduce the observed state evolution. If that works,
catch & release is a new way to model moving bottlenecks, and (point 3) it can
natively represent the split into a stuck stream and a free stream - a state
whose aggregate (rho, q) lies INSIDE the flux function, which scalar LWR cannot
express.

Three models, all given the same information (inflow, empty road, CAV schedule):

  M1  LWR+MB    : kappa = 0 + DM-G capacity cap (Q_xi = 2000 veh/h).  This is
                  the canonical scalar LWR + moving bottleneck baseline
                  (ECC22-equivalent prediction).
  M2  CR-native : catch & release ONLY, no cap.  (kappa_c, kappa_r) fitted per
                  assertiveness by matching the rep-mean density field of the
                  u15_q2500 scenario (coarse log-grid + Nelder-Mead polish),
                  then TRANSFERRED unchanged to u15_q2000 as validation.
  M3  CR-capped : E-V4b reference (event-MLE kappas + cap), for context.

Focus is u_xi = 15 m/s per Mladen's point 2 (u=20 is a known weak regime of
MB models); u20 numbers are reported in the JSON for completeness.

Outputs (out/e6/):
  fits.json                 fitted kappas + objective per (A, form)
  metrics_e6.json           rho-RMSE / Ns_MAE / omega_err / e_s for M1/M2/M3
  fig_heat3_{tag}.png       data vs M1 vs M2 heatmaps
  fig_profiles_{tag}.png    density profiles at 300/500/700 s with the M2
                            f/s sub-class split
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import ev4_compare as ev4
from loader import CELL_LEN
from solver import SimConfig, simulate

HERE = Path(__file__).parent
OUT = HERE / "out" / "e6"

QXI_VEHH = 2000.0            # canonical bottleneck capacity (ECC22 Q_xi)
FIT_SCEN = 2500.0            # fit at q_in = 2500, transfer to 2000
U_FOCUS = 15.0
KAPPA = json.loads((HERE / "out" / "ev3_kappa.json").read_text())

# objective window (matches ev4_compare rho_rmse convention)
T_WIN = (100.0, 1000.0)
X_CELLS = slice(0, 200)      # x <= 20 km on the data grid


def run_model(A, uc, qin, kc, kr, form, qxi_vehh):
    cfg = SimConfig(v_f=ev4.V_F, w=ev4.W, P=ev4.P, q_in=qin / 3600.0,
                    u_xi=uc, kappa_c=kc, kappa_r=kr, capture_form=form,
                    dt=0.5, save_every=20,
                    q_xi_max=None if qxi_vehh is None else qxi_vehh / 3600.0)
    return ev4.regrid_sim(simulate(cfg))


def objective(regr, rho_mean):
    """RMSE of the total density vs the rep-mean measured field [veh/km]."""
    tt = regr["tt"]
    sl = (tt >= T_WIN[0]) & (tt <= T_WIN[1])
    d = regr["rho_tot"][sl][:, X_CELLS] - rho_mean[sl][:, X_CELLS]
    return float(np.sqrt(np.mean(d ** 2)))


def fit_native(A, form, rho_mean, uc=U_FOCUS, qin=FIT_SCEN):
    """Coarse log-grid search + Nelder-Mead polish of (kappa_c, kappa_r)."""
    kc_grid = np.logspace(-2.5, 0.3, 8)
    kr_grid = np.concatenate([[0.0], np.logspace(-6.0, -1.0, 6)])
    best = (np.inf, None, None)
    for kc in kc_grid:
        for kr in kr_grid:
            j = objective(run_model(A, uc, qin, kc, kr, form, None), rho_mean)
            if j < best[0]:
                best = (j, kc, kr)
    j0, kc0, kr0 = best

    if kr0 == 0.0:            # polish kappa_c alone on the kr = 0 axis
        res = minimize(
            lambda z: objective(
                run_model(A, uc, qin, 10 ** z[0], 0.0, form, None), rho_mean),
            x0=[np.log10(kc0)], method="Nelder-Mead",
            options=dict(maxfev=40, xatol=1e-3, fatol=1e-3))
        kc, kr, j = 10 ** res.x[0], 0.0, float(res.fun)
    else:
        res = minimize(
            lambda z: objective(
                run_model(A, uc, qin, 10 ** z[0], 10 ** z[1], form, None),
                rho_mean),
            x0=[np.log10(kc0), np.log10(kr0)], method="Nelder-Mead",
            options=dict(maxfev=60, xatol=1e-3, fatol=1e-3))
        kc, kr, j = 10 ** res.x[0], 10 ** res.x[1], float(res.fun)
    if j0 < j:                # keep the grid optimum if polish regressed
        kc, kr, j = kc0, kr0, j0
    return dict(kappa_c=float(kc), kappa_r=float(kr), rmse=j,
                grid_rmse=j0, n_polish=int(res.nfev))


def rho_mean_of(meas):
    return np.mean(meas["rho"], axis=0)


def fig_heat3(tag, meas, m1, m2, out_dir):
    fields = [(np.mean(meas["rho"], axis=0), "SUMO (rep mean)"),
              (m1["rho_tot"], "M1: LWR + MB (DM-G cap)"),
              (m2["rho_tot"], "M2: catch & release, no cap")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    tt = m1["tt"]
    for ax, (rho, name) in zip(axes, fields):
        im = ax.imshow(rho, origin="lower", aspect="auto",
                       extent=[CELL_LEN / 1000, 30.0, tt[0] / 60, tt[-1] / 60],
                       cmap="turbo", vmin=0, vmax=90)
        ax.plot(m1["x_cav"] / 1000, tt / 60, "w-", lw=1.5)
        ax.set_title(f"{tag} — {name}", fontsize=9)
        ax.set_xlabel("x [km]")
        ax.set_xlim(0, 20)
    axes[0].set_ylabel("t [min]")
    fig.colorbar(im, ax=axes, label=r"$\rho$ [veh/km]", shrink=0.9)
    fig.savefig(out_dir / f"fig_heat3_{tag}.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def fig_profiles(tag, meas, m1, m2, out_dir):
    rho_d = np.mean(meas["rho"], axis=0)
    tt = m1["tt"]
    xs = (np.arange(rho_d.shape[1]) + 1) * CELL_LEN / 1000
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.5), sharex=True)
    for ax, t_snap in zip(axes, (300.0, 500.0, 700.0)):
        i = int(np.argmin(np.abs(tt - t_snap)))
        ax.plot(xs, rho_d[i], "k-", lw=1.8, label="SUMO (rep mean)")
        ax.plot(xs, m1["rho_tot"][i], color="tab:orange", lw=1.5,
                label="M1: LWR + MB")
        ax.plot(xs, m2["rho_tot"][i], color="tab:blue", lw=1.5,
                label=r"M2: CR total $\rho$")
        ax.fill_between(xs, 0, m2["s"][i], color="tab:red", alpha=0.30,
                        label="M2: caught $s$")
        ax.fill_between(xs, m2["s"][i], m2["s"][i] + m2["f"][i],
                        color="tab:green", alpha=0.20, label="M2: free $f$")
        xc = m1["x_cav"][i] / 1000
        if np.isfinite(xc):
            ax.axvline(xc, color="0.4", ls=":", lw=1)
        ax.set_ylabel(r"$\rho$ [veh/km]")
        ax.set_title(f"t = {t_snap:.0f} s", fontsize=9, loc="left")
        ax.set_xlim(0, 16)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("x [km]")
    fig.suptitle(f"{tag}: density profiles — data vs classical vs native "
                 "catch & release (with sub-class split)")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_profiles_{tag}.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- fit M2 at u15_q2500 per assertiveness, both capture forms --------
    fits = {}
    for A in (1.0, 10.0):
        meas_fit = ev4.load_measured(A, U_FOCUS, FIT_SCEN)
        rho_mean = rho_mean_of(meas_fit)
        for form in ("lf", "af"):
            key = f"A{A:g}_{form}"
            fits[key] = fit_native(A, form, rho_mean)
            print(f"[fit] {key}: kappa_c={fits[key]['kappa_c']:.3e} "
                  f"kappa_r={fits[key]['kappa_r']:.3e} "
                  f"rho-RMSE={fits[key]['rmse']:.2f} veh/km "
                  f"(grid {fits[key]['grid_rmse']:.2f})")
        best = min(("lf", "af"), key=lambda fm: fits[f"A{A:g}_{fm}"]["rmse"])
        fits[f"A{A:g}_best_form"] = best
    (OUT / "fits.json").write_text(json.dumps(fits, indent=2))

    # ---- evaluate M1 / M2 / M3 on the four u15 scenarios (+ u20 in JSON) --
    results = {}
    scen_list = [(A, uc, qin) for A in (1.0, 10.0)
                 for uc in (15.0, 20.0) for qin in (2000.0, 2500.0)]
    print(f"\n{'scenario':16s} {'model':10s} {'rho_RMSE':>9s} {'Ns_MAE':>8s} "
          f"{'omega_err':>10s} {'e_s':>8s}")
    for A, uc, qin in scen_list:
        tag = f"A{A:g}_u{uc:g}_q{qin:g}"
        meas = ev4.load_measured(A, uc, qin)
        form = fits[f"A{A:g}_best_form"]
        fit = fits[f"A{A:g}_{form}"]
        kmle = KAPPA[tag]
        kc_key = "kappa_cl_mod" if form == "lf" else "kappa_cA_mod"
        models = {
            "M1_lwr_mb": run_model(A, uc, qin, 0.0, 0.0, "lf", QXI_VEHH),
            "M2_cr_native": run_model(A, uc, qin, fit["kappa_c"],
                                      fit["kappa_r"], form, None),
            "M3_cr_capped": run_model(A, uc, qin, float(kmle[kc_key][0]),
                                      float(kmle["kappa_r_mod"][0]), form,
                                      QXI_VEHH),
        }
        results[tag] = {"fit_form": form}
        for name, regr in models.items():
            met = ev4.metrics(regr, meas)
            results[tag][name] = {k: met[k]["mean"] for k in
                                  ("rho_rmse", "Ns_mae", "omega_cum_rel_err",
                                   "e_s")}
            if uc == U_FOCUS:
                r = results[tag][name]
                print(f"{tag:16s} {name:10s} {r['rho_rmse']:9.2f} "
                      f"{r['Ns_mae']:8.2f} {r['omega_cum_rel_err']:+10.1%} "
                      f"{r['e_s']:+8.1%}")
        if uc == U_FOCUS:
            fig_heat3(tag, meas, models["M1_lwr_mb"], models["M2_cr_native"],
                      OUT)
            fig_profiles(tag, meas, models["M1_lwr_mb"],
                         models["M2_cr_native"], OUT)
    (OUT / "metrics_e6.json").write_text(json.dumps(results, indent=2))
    print("\nwrote", OUT / "metrics_e6.json", "and figures")


if __name__ == "__main__":
    main()
