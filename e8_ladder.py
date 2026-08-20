"""E8: one-at-a-time ablation ladder for the downstream-release fix.

Mladen's 3rd review: "try adding the changes one by one, to identify what
are the effects of each."  His confirmed diagnosis: the E7 winner leaks
stuck vehicles DOWNSTREAM of the CAV, causing (a) wrong downstream density
and e_s, (b) free-flow 'waviness' (R1 dispersion growth rate
lambda = Dv (kappa_c rho - kappa_r (P - rho)) > 0 downstream), and (c) an
intermediate-density wedge (downstream s travels at u_s < v_f).  The fix is
DEFINITIONAL, zero parameters: s strictly downstream of the CAV converts to
f immediately (solver.SimConfig.downstream_release).

Ladder (scenario u15 q2500, capture form lf, A in {1, 10}); every rung's
(kappa_c, kappa_r) is fitted with the SAME protocol as E7
(e7_wasserstein.fit_field: 6x6 log grid + Nelder-Mead maxfev 40 at
dt_fit=1, production re-evaluation at dt=0.5; fixed knobs enter via the E8
extra_cfg passthrough, NOT the optimization vector):

  L0  metric=rmse, no knobs                      E6 baseline reproduction
  L1  metric=w1,   no knobs                      effect of the metric alone
  L2  metric=w1 + w_s = 0.6 w fixed              effect of the stuck-class flux
  L3  metric=w1 + w_s = 0.6 w + downstream_release   the E8 fix, full
  L4  metric=w1 + downstream_release (NO w_s)    is w_s still needed with DR?

Per rung (production dt=0.5), recorded in out/e8/ladder.json:
  kappas, W1, RMSE, e_s, omega_err (ev4_compare.metrics);
  wake density   mean rho_tot over [x_cav-1 km, x_cav-0.2 km], t in
                 [600, 740] s (data: A1 58.6 / A10 55.4 veh/km);
  ds stuck frac  (int s dx)/(int rho dx) over data cells fully downstream
                 of x_cav at t = 700 s;
  ds waviness    std of linearly detrended rho_tot over
                 [x_cav+0.5 km, x_cav+5 km], mean over t in [400, 740] s
                 (DATA value computed for reference, nominal trajectory);
  wedge          fraction of cells in [x_cav+0.5, x_cav+5 km] at t=600 s
                 with rho strictly between (overtake-zone density + 4) and
                 (background inflow density - 4) veh/km, overtake zone =
                 mean rho over [x_cav+0.5, x_cav+1.5 km];
  rarefaction    width growth of {rho_tot > 35 veh/km} between t=780 and
                 960 s (e7_ablation.d2_rarefaction);
  s-layer        max over t in [600, 740] s of max_x {x_cav - x : s > 1
                 veh/km} (upstream extent; mean over snapshots as aux);
  ds max |s|     max |s| over data cells fully inside the DR-zeroed region
                 (must be exactly 0.0 for L3/L4: source = 0 downstream).

ANALYTICAL CHECK (rung L2, waviness present): the downstream s-lump grows
at the linearized R1 rate.  Downstream a = 0, so with the lf form ell = s
and (tex eq. exact-reaction-update, s << f ~ rho, frozen background):

    ds/dt = dv s (kappa_c f - kappa_r (P - rho))
          ~ [dv (kappa_c rho - kappa_r (P - rho))] s  =: lambda s .

We log-fit the amplitude max_x s over x > x_cav (t in [450, 650] s) and
compare the measured e-fold time to 1/lambda with the background taken at
the lump peak of the simulated field, in two instantiations: the naive
s << f ~ rho form above (Mladen's diagnosis), and the frozen-f form
dv (kappa_c f - kappa_r (P - rho)) which stays exact for the reaction ODE
once the lump saturates.  Further assumptions (documented in json):
ell = s (a = 0 downstream), amplitude preserved by transport (lump
advects at u_s = u_xi ~ CAV speed, no compression), reaction-layer
Delta v from the shared road FD.  Expect agreement within a factor ~2
WHEN the linear
regime is resolvable: |lambda| must be slow vs the 10 s save cadence and
fast vs the CAV-cell seed influx.  The L2 rungs are fitted, so their
kappas may land outside that regime (they do -- see json statuses); the
closed form is therefore ALSO verified on the E7 C3 winner point (the
exact configuration of Mladen's diagnosis, predicted e-fold ~10^2 s),
reported as analytic_check_reference_e7C3.  For L3/L4 the check is
instead exactness: max |s| = 0 in the DR zone, hence source = 0 there.

A=10 GUARD: if the W1-optimal A=10 rung has s-layer > 1.5 km (the
fast-churn broad band Mladen rejected), ALSO fit a structure-anchored
alternative of the same rung with kc_grid capped at logspace(-2.5, -0.5)
(the E6/E-V3 magnitude regime); the full-fix rung L3 is anchor-refit as
well (if distinct) so the feasibility verdict is not an artifact of
anchoring only a no-DR rung.  The FINAL A=10 pick must have s-layer <=
1.5 km AND W1 within 15% of the unconstrained optimum if possible; any
conflict between the two demands is reported honestly in the json
(guard.candidates / guard.resolution), a no-DR pick that re-leaks
downstream is flagged in pick_caveat, and when no candidate meets the
s-layer bound the pick minimizes its violation (Mladen's structural veto
outranks the W1 band).

Final picks (both A) get figures (fig_profiles_e8_{tag}.png with the E7
winner in gray dashed for contrast; fig_heat3_e8_{tag}.png adapted from
e6_native_mb) and a zero-refit transfer evaluation on q2000.

CLI:  python3 e8_ladder.py --run       full program -> out/e8/ladder.json
      python3 e8_ladder.py --smoke     tiny-grid mechanics check
                                       (-> out/e8/ladder_smoke.json)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import e7_ablation as e7a          # d1_wake / d2_rarefaction conventions
import e7_wasserstein as e7
import ev4_compare as ev4
from loader import CELL_LEN, N_CELL

HERE = Path(__file__).parent
OUT_E8 = HERE / "out" / "e8"
E7_ABLATION_JSON = HERE / "out" / "e7" / "ablation.json"

UC = 15.0                      # [m/s]  focus CAV slow speed
QIN_FIT = 2500.0               # [veh/h] fit inflow
QIN_TRANSFER = 2000.0          # [veh/h] zero-refit transfer inflow
FORM = "lf"
WS_FRAC_FIX = 0.6              # fixed stuck-class wave speed w_s = 0.6 w
RHO_CRIT = 48.5                # [veh/km]
SOLVER_DX = 50.0               # [m] solver cell (DR zeroes j > int(x_cav//dx))

# data-grid geometry (upper-edge convention XS = (1:300)*100 m)
X_UP = (np.arange(N_CELL) + 1) * CELL_LEN     # upper edges [m]
X_LO = X_UP - CELL_LEN                        # lower edges [m]
X_CTR = X_UP - CELL_LEN / 2.0                 # centers [m]

# diagnostic windows
WAVI_T = (400.0, 740.0)        # [s]
WAVI_XREL = (500.0, 5000.0)    # [m] window [x_cav+0.5 km, x_cav+5 km]
WEDGE_T = 600.0                # [s]
WEDGE_OT_XREL = (500.0, 1500.0)  # [m] overtake-zone window
WEDGE_MARGIN = 4.0             # [veh/km]
STUCK_T = 700.0                # [s]
SLAYER_T = (600.0, 740.0)      # [s]
SLAYER_S_MIN = 1.0             # [veh/km]
GROWTH_T = (450.0, 650.0)      # [s]  L2 amplitude-growth window
GROWTH_X_MARGIN = 200.0        # [m]  skip the CAV-straddling cells
GROWTH_AMP_FLOOR = 1e-6        # [veh/km]
SLAYER_GUARD_M = 1500.0        # A=10 guard threshold
W1_GUARD_TOL = 0.15            # A=10 guard: W1 within 15% of optimum
KC_GRID_ANCHOR = np.logspace(-2.5, -0.5, 6)   # E6/E-V3 magnitude regime

RUNGS = (
    # (name, metric, extra_cfg, description)
    ("L0", "rmse", {}, "rmse, no knobs (E6 baseline reproduction)"),
    ("L1", "w1", {}, "w1, no knobs (metric alone)"),
    ("L2", "w1", {"w_s": WS_FRAC_FIX * ev4.W},
     f"w1 + w_s={WS_FRAC_FIX:g}w fixed (stuck-class flux)"),
    ("L3", "w1", {"w_s": WS_FRAC_FIX * ev4.W, "downstream_release": True},
     f"w1 + w_s={WS_FRAC_FIX:g}w + downstream_release (E8 fix, full)"),
    ("L4", "w1", {"downstream_release": True},
     "w1 + downstream_release, NO w_s"),
)


# ---------------------------------------------------------------------------
# diagnostics (data grid, veh/km; wake and rarefaction reuse e7_ablation)
# ---------------------------------------------------------------------------

def _snap(tt, t):
    return int(np.argmin(np.abs(np.asarray(tt, float) - t)))


def ds_stuck_fraction(regr, t_snap=STUCK_T):
    """(int s dx)/(int rho dx) over data cells fully downstream of x_cav
    (lower edge >= x_cav) at the snapshot nearest t_snap."""
    i = _snap(regr["tt"], t_snap)
    xc = float(regr["x_cav"][i])
    if not np.isfinite(xc):
        return dict(frac=None, s_veh=0.0, t_s=float(regr["tt"][i]))
    sel = X_LO >= xc
    s_int = float(np.sum(regr["s"][i, sel]))          # [veh/km * cells]
    r_int = float(np.sum(regr["rho_tot"][i, sel]))
    return dict(frac=(s_int / r_int if r_int > 0.0 else 0.0),
                s_veh=s_int * CELL_LEN / 1000.0,
                t_s=float(regr["tt"][i]))


def waviness(rho, tt, x_cav_series, t_win=WAVI_T, x_rel=WAVI_XREL):
    """Mean over t in t_win of the std of linearly detrended rho_tot over
    cell centers in [x_cav + x_rel[0], x_cav + x_rel[1]]."""
    tt = np.asarray(tt, float)
    rho = np.asarray(rho, float)
    xc_s = np.asarray(x_cav_series, float)
    vals = []
    for i in np.where((tt >= t_win[0]) & (tt <= t_win[1]))[0]:
        xc = xc_s[i]
        if not np.isfinite(xc):
            continue
        sel = (X_CTR >= xc + x_rel[0]) & (X_CTR <= xc + x_rel[1])
        if np.sum(sel) < 3:
            continue
        x_km = X_CTR[sel] / 1000.0
        r = rho[i, sel]
        resid = r - np.polyval(np.polyfit(x_km, r, 1), x_km)
        vals.append(float(np.std(resid)))
    return dict(std_vehkm=(float(np.mean(vals)) if vals else None),
                n_snapshots=len(vals))


def wedge_indicator(rho, tt, x_cav_series, qin, t_snap=WEDGE_T):
    """Fraction of cells in [x_cav+0.5, x_cav+5 km] at t_snap with rho
    strictly between (overtake-zone density + 4) and (background inflow
    density - 4) veh/km; overtake zone = mean rho over
    [x_cav+0.5, x_cav+1.5 km]."""
    i = _snap(tt, t_snap)
    xc = float(np.asarray(x_cav_series, float)[i])
    rho_bg = qin / 3600.0 / ev4.V_F * 1000.0            # [veh/km]
    if not np.isfinite(xc):
        return dict(frac=None, background_vehkm=rho_bg)
    rho_i = np.asarray(rho, float)[i]
    sel_w = (X_CTR >= xc + WAVI_XREL[0]) & (X_CTR <= xc + WAVI_XREL[1])
    sel_ot = (X_CTR >= xc + WEDGE_OT_XREL[0]) & \
             (X_CTR <= xc + WEDGE_OT_XREL[1])
    ot = float(np.mean(rho_i[sel_ot])) if sel_ot.any() else np.nan
    lo, hi = ot + WEDGE_MARGIN, rho_bg - WEDGE_MARGIN
    band_ok = bool(np.isfinite(ot) and lo < hi)
    frac = (float(np.mean((rho_i[sel_w] > lo) & (rho_i[sel_w] < hi)))
            if band_ok and sel_w.any() else 0.0)
    return dict(frac=frac, overtake_zone_vehkm=ot, background_vehkm=rho_bg,
                band_lo_vehkm=lo, band_hi_vehkm=hi, band_nonempty=band_ok,
                n_cells=int(np.sum(sel_w)), t_s=float(np.asarray(tt)[i]))


def s_layer_extent(regr, t_win=SLAYER_T, s_min=SLAYER_S_MIN):
    """Per snapshot: max_x {x_cav - x_center : s > s_min} (0 if the layer is
    empty or entirely downstream).  Returns the max and mean over t_win."""
    tt = np.asarray(regr["tt"], float)
    ext = []
    for i in np.where((tt >= t_win[0]) & (tt <= t_win[1]))[0]:
        xc = float(regr["x_cav"][i])
        if not np.isfinite(xc):
            continue
        d = xc - X_CTR[regr["s"][i] > s_min]
        d = d[d > 0.0]
        ext.append(float(d.max()) if d.size else 0.0)
    if not ext:
        return dict(max_m=0.0, mean_m=0.0, n_snapshots=0)
    return dict(max_m=float(np.max(ext)), mean_m=float(np.mean(ext)),
                n_snapshots=len(ext))


def ds_max_s(regr):
    """Max |s| [veh/km] over data cells fully inside the DR-zeroed region
    (solver cells j > int(x_cav // 50 m)), over all snapshots with the CAV
    on road.  Exactly 0.0 when downstream_release is active (pair-averaging
    of exact zeros is exact)."""
    tt = np.asarray(regr["tt"], float)
    worst = 0.0
    for i in range(len(tt)):
        xc = float(regr["x_cav"][i])
        if not np.isfinite(xc):
            continue
        zero_from = (np.floor(xc / SOLVER_DX) + 1.0) * SOLVER_DX
        sel = X_LO >= zero_from
        if sel.any():
            worst = max(worst, float(np.max(np.abs(regr["s"][i, sel]))))
    return worst


# ---------------------------------------------------------------------------
# analytical check (rung L2): downstream s-lump growth vs the R1 rate
# ---------------------------------------------------------------------------

def analytic_check_growth(regr, kc, kr, uc, t_win=GROWTH_T):
    """Log-fit the downstream s-lump amplitude and compare its e-fold time
    to the closed-form lf reaction rate downstream (ell = s, a = 0):

        ds/dt = dv s (kappa_c f - kappa_r (P - rho))

    in TWO instantiations, both with the background evaluated from the
    simulated field at the lump peak and averaged over the window:
      naive     s << f ~ rho:  lambda = dv (kappa_c rho - kappa_r (P-rho))
                (the form quoted in Mladen's diagnosis);
      frozen_f  observed f at the peak: lambda = dv (kappa_c f - kappa_r
                (P-rho)) -- exact for the ODE given the local (f, rho), so
                it stays valid once the lump saturates (s no longer << f).
    """
    from solver import speed as fd_speed
    tt = np.asarray(regr["tt"], float)
    ts, amps, rho_pk, f_pk = [], [], [], []
    for i in np.where((tt >= t_win[0]) & (tt <= t_win[1]))[0]:
        xc = float(regr["x_cav"][i])
        if not np.isfinite(xc):
            continue
        sel = X_CTR > xc + GROWTH_X_MARGIN
        if not sel.any():
            continue
        s_ds = regr["s"][i, sel]
        j = int(np.argmax(s_ds))
        if s_ds[j] <= GROWTH_AMP_FLOOR:
            continue
        ts.append(tt[i])
        amps.append(float(s_ds[j]))
        rho_pk.append(float(regr["rho_tot"][i, sel][j]))
        f_pk.append(float(regr["f"][i, sel][j]))
    out = dict(t_win_s=list(t_win), n_points=len(ts),
               assumptions=("closed-form lf reaction downstream: ell = s "
                            "(a = 0), background (rho, f, dv) from the "
                            "simulated lump peak, per-snapshot rates "
                            "averaged over the window, amplitude preserved "
                            "by transport (lump advects at u_s ~ CAV "
                            "speed), Delta v from the shared road FD; the "
                            "naive form additionally assumes s << f ~ rho, "
                            "and resolvability needs the e-fold slow vs "
                            "the 10 s save cadence and fast vs the "
                            "CAV-cell seed influx"))
    if len(ts) < 5:
        out["status"] = "insufficient points (no downstream s-lump)"
        return out
    ts = np.asarray(ts)
    amps = np.asarray(amps)
    rho_si = np.asarray(rho_pk) / 1000.0                          # [veh/m]
    f_si = np.asarray(f_pk) / 1000.0
    save_dt = float(np.median(np.diff(tt)))
    lam_meas = float(np.polyfit(ts, np.log(amps), 1)[0])          # [1/s]
    dv = np.maximum(fd_speed(rho_si, ev4.V_F, ev4.W, ev4.P) - uc, 0.0)
    lam_naive = float(np.mean(dv * (kc * rho_si
                                    - kr * (ev4.P - rho_si))))    # [1/s]
    lam_frozf = float(np.mean(dv * (kc * f_si
                                    - kr * (ev4.P - rho_si))))    # [1/s]
    out.update(
        amp_first_vehkm=float(amps[0]), amp_last_vehkm=float(amps[-1]),
        s_over_rho_first=float(amps[0] / rho_pk[0]),
        s_over_rho_last=float(amps[-1] / rho_pk[-1]),
        save_cadence_s=save_dt,
        rho_ds_vehkm=float(np.mean(rho_pk)),
        f_ds_vehkm=float(np.mean(f_pk)),
        dv_ds_ms=float(np.mean(dv)),
        lambda_measured_per_s=lam_meas,
        lambda_naive_per_s=lam_naive,
        lambda_frozen_f_per_s=lam_frozf,
        efold_measured_s=(1.0 / lam_meas if lam_meas > 0 else None),
        efold_naive_s=(1.0 / lam_naive if lam_naive > 0 else None),
        efold_frozen_f_s=(1.0 / lam_frozf if lam_frozf > 0 else None))
    if lam_meas > 0 and lam_frozf > 0:
        ratio_f = (1.0 / lam_meas) / (1.0 / lam_frozf)
        out["efold_ratio_meas_over_frozen_f"] = ratio_f
        out["within_factor_2_frozen_f"] = bool(0.5 <= ratio_f <= 2.0)
        if lam_naive > 0:
            ratio_n = (1.0 / lam_meas) / (1.0 / lam_naive)
            out["efold_ratio_meas_over_naive"] = ratio_n
            out["within_factor_2_naive"] = bool(0.5 <= ratio_n <= 2.0)
        out["status"] = "ok"
    elif lam_naive > 0 and 1.0 / lam_naive < 2.0 * save_dt:
        out["status"] = (
            f"linear regime unresolvable: predicted e-fold "
            f"{1.0 / lam_naive:.1f} s is below the {save_dt:g} s save "
            "cadence, so s reaches its local equilibrium between saves and "
            "the measured amplitude tracks the quasi-steady wedge, not the "
            "linear mode")
    elif lam_naive <= 0.0 and lam_meas <= 0.0:
        out["status"] = (
            "closed form predicts DECAY and the measured amplitude is "
            "indeed non-growing (sign agreement); decay-rate magnitudes "
            "are not comparable because the lump is continuously re-seeded "
            "from the CAV cell (source-fed quasi-equilibrium)")
    else:
        out["status"] = ("sign disagreement between measured and predicted "
                         "rate; see lambda_* fields")
    return out


# ---------------------------------------------------------------------------
# per-rung production evaluation
# ---------------------------------------------------------------------------

def eval_rung(A, uc, qin, kc, kr, extra_cfg):
    """Run at production dt=0.5, score vs data, all E8 diagnostics."""
    regr = e7.run_sim(uc, qin, kc, kr, FORM, dt=e7.DT_PRODUCTION,
                      **extra_cfg)
    tt, rho_mean = e7.load_rho_mean(A, uc, qin)
    meas = ev4.load_measured(A, uc, qin)
    met = ev4.metrics(regr, meas)
    summary = dict(
        W1=e7.w1_mean(regr["rho_tot"], rho_mean, tt),
        RMSE=e7.rmse_mean(regr["rho_tot"], rho_mean, tt),
        e_s=met["e_s"]["mean"],
        omega_err=met["omega_cum_rel_err"]["mean"],
        wake=e7a.d1_wake(regr["rho_tot"], tt, regr["x_cav"]),
        ds_stuck=ds_stuck_fraction(regr),
        waviness_sim=waviness(regr["rho_tot"], tt, regr["x_cav"]),
        wedge=wedge_indicator(regr["rho_tot"], tt, regr["x_cav"], qin),
        rarefaction=e7a.d2_rarefaction(regr["rho_tot"], tt),
        s_layer=s_layer_extent(regr),
        ds_max_s_vehkm=ds_max_s(regr),
    )
    return summary, regr, meas


def data_refs(A, uc, qin):
    """Data-side reference values (nominal CAV trajectory)."""
    tt, rho_mean = e7.load_rho_mean(A, uc, qin)
    x_nom = ev4.x_cav_nominal(tt, uc)
    return dict(
        wake=e7a.d1_wake(rho_mean, tt, x_nom),
        waviness=waviness(rho_mean, tt, x_nom),
        wedge=wedge_indicator(rho_mean, tt, x_nom, qin),
        rarefaction=e7a.d2_rarefaction(rho_mean, tt))


# ---------------------------------------------------------------------------
# figures (final picks): profiles with the E7 winner as gray-dashed
# contrast, and a 3-panel heatmap adapted from e6_native_mb.fig_heat3
# ---------------------------------------------------------------------------

PROFILE_SNAPS = (300.0, 500.0, 700.0, 850.0)


def load_e7_winner(A):
    """(regr at dt=0.5, label) of the E7 ablation winner for this A."""
    d = json.loads(E7_ABLATION_JSON.read_text())
    blk = d[f"A{A:g}"]
    name = blk["winner"]
    cfgw = blk["configs"][name]
    extra = cfgw.get("extra") or {}
    regr = e7.run_sim(UC, QIN_FIT, cfgw["kappa_c"], cfgw["kappa_r"], FORM,
                      gamma=extra.get("gamma"), ws_frac=extra.get("ws_frac"),
                      dt=e7.DT_PRODUCTION)
    lbl = f"E7 winner ({name}: {cfgw['metric']}" + "".join(
        f", {k}={v:.3g}" for k, v in extra.items()) + ")"
    return regr, lbl


def fig_profiles_e8(tag, meas, e7_regr, e7_lbl, e8_regr, e8_lbl, out_dir):
    rho_d = np.mean(meas["rho"], axis=0)
    tt = e8_regr["tt"]
    xs = X_UP / 1000.0
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 11.0), sharex=True)
    for ax, t_snap in zip(axes, PROFILE_SNAPS):
        i = _snap(tt, t_snap)
        ax.plot(xs, rho_d[i], "k-", lw=1.8, label="SUMO (rep mean)")
        ax.plot(xs, e7_regr["rho_tot"][i], color="0.55", ls="--", lw=1.4,
                label=e7_lbl)
        ax.plot(xs, e8_regr["rho_tot"][i], color="tab:blue", lw=1.6,
                label=f"{e8_lbl}: total " + r"$\rho$")
        ax.fill_between(xs, 0, e8_regr["s"][i], color="tab:red", alpha=0.30,
                        label="E8: caught $s$")
        ax.fill_between(xs, e8_regr["s"][i],
                        e8_regr["s"][i] + e8_regr["f"][i],
                        color="tab:green", alpha=0.20, label="E8: free $f$")
        ax.axhline(RHO_CRIT, color="0.55", ls="--", lw=0.9)
        xc = e8_regr["x_cav"][i] / 1000.0
        if np.isfinite(xc):
            ax.axvline(xc, color="0.4", ls=":", lw=1)
        ax.set_ylabel(r"$\rho$ [veh/km]")
        ax.set_title(f"t = {t_snap:.0f} s"
                     + ("  (post-release)" if t_snap > 750.0 else ""),
                     fontsize=9, loc="left")
        ax.set_xlim(0, 16)
        ax.grid(alpha=0.3)
    axes[0].text(0.15, RHO_CRIT + 1.5, r"$\rho_{crit}$ = 48.5",
                 fontsize=7, color="0.45")
    axes[0].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("x [km]")
    fig.suptitle(f"{tag}: density profiles — data vs {e7_lbl} vs {e8_lbl}")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_profiles_e8_{tag}.png", dpi=160)
    plt.close(fig)


def fig_heat3_e8(tag, meas, e7_regr, e7_lbl, e8_regr, e8_lbl, out_dir):
    """3-panel heatmap (adapted from e6_native_mb.fig_heat3): SUMO rep mean
    vs E7 winner vs E8 pick, shared scale, CAV trajectory overlaid."""
    fields = [(np.mean(meas["rho"], axis=0), f"{tag} — SUMO (rep mean)"),
              (e7_regr["rho_tot"], e7_lbl),
              (e8_regr["rho_tot"], e8_lbl)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    tt = e8_regr["tt"]
    for ax, (rho, name) in zip(axes, fields):
        im = ax.imshow(rho, origin="lower", aspect="auto",
                       extent=[CELL_LEN / 1000, 30.0,
                               tt[0] / 60, tt[-1] / 60],
                       cmap="turbo", vmin=0, vmax=90)
        ax.plot(e8_regr["x_cav"] / 1000, tt / 60, "w-", lw=1.5)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("x [km]")
        ax.set_xlim(0, 20)
    axes[0].set_ylabel("t [min]")
    fig.colorbar(im, ax=axes, label=r"$\rho$ [veh/km]", shrink=0.9)
    fig.savefig(out_dir / f"fig_heat3_e8_{tag}.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# ladder table
# ---------------------------------------------------------------------------

def _fmt_row(name, r):
    ev = r["eval_q2500"]
    wav = ev["waviness_sim"]["std_vehkm"]
    return (f"{name:5s} {r['kappa_c']:9.3e} {r['kappa_r']:9.3e} "
            f"{ev['W1']:7.1f} {ev['RMSE']:6.2f} {ev['e_s']:+7.1%} "
            f"{ev['omega_err']:+7.1%} "
            f"{ev['wake']['wake_mean_vehkm']:6.1f} "
            f"{(ev['ds_stuck']['frac'] if ev['ds_stuck']['frac'] is not None else float('nan')):7.4f} "
            f"{(wav if wav is not None else float('nan')):6.2f} "
            f"{(ev['wedge']['frac'] if ev['wedge']['frac'] is not None else float('nan')):6.3f} "
            f"{ev['rarefaction']['width_growth_m']:7.0f} "
            f"{ev['s_layer']['max_m']:7.0f} "
            f"{ev['ds_max_s_vehkm']:9.2e}")


def print_table(akey, block):
    refs = block["data_refs_q2500"]
    print(f"\n===== ladder {akey} (u{UC:g} q{QIN_FIT:g} {FORM}) =====")
    print(f"data refs: wake={refs['wake']['wake_mean_vehkm']:.1f} veh/km, "
          f"waviness={refs['waviness']['std_vehkm']:.2f} veh/km, "
          f"wedge_frac={refs['wedge']['frac']:.3f}, "
          f"rarefaction_growth={refs['rarefaction']['width_growth_m']:.0f} m")
    print(f"{'rung':5s} {'kappa_c':>9s} {'kappa_r':>9s} {'W1':>7s} "
          f"{'RMSE':>6s} {'e_s':>7s} {'om_err':>7s} {'wake':>6s} "
          f"{'stuckf':>7s} {'wavi':>6s} {'wedge':>6s} {'rare_m':>7s} "
          f"{'slayer':>7s} {'ds|s|':>9s}")
    for name in sorted(block["rungs"]):
        print(_fmt_row(name, block["rungs"][name]))


# ---------------------------------------------------------------------------
# main program
# ---------------------------------------------------------------------------

def run(smoke=False):
    t_start = time.time()
    OUT_E8.mkdir(parents=True, exist_ok=True)
    fit_kwargs = ({} if not smoke else
                  dict(kc_grid=[0.1, 0.4], kr_grid=[0.005, 0.05], maxfev=2))
    anchor_grid = (KC_GRID_ANCHOR if not smoke else np.array([0.05, 0.2]))

    results = {"_meta": dict(
        scenario=f"u{UC:g} q{QIN_FIT:g} form={FORM}",
        protocol=("e7_wasserstein.fit_field: 6x6 log grid + Nelder-Mead "
                  "maxfev 40 at dt_fit=1, production re-eval dt=0.5; fixed "
                  "knobs via extra_cfg passthrough (NOT fitted)"),
        rungs={n: dict(metric=m,
                       extra_cfg={k: (v if not isinstance(v, bool) else
                                      bool(v)) for k, v in x.items()},
                       description=d)
               for n, m, x, d in RUNGS},
        ws_fixed=dict(frac=WS_FRAC_FIX, w_s_ms=WS_FRAC_FIX * ev4.W),
        dt_fit=1.0, dt_production=e7.DT_PRODUCTION,
        pick_rule=("min W1 at dt=0.5 (q2500); A=10 guard: pick must have "
                   f"s-layer <= {SLAYER_GUARD_M:g} m AND W1 within "
                   f"{W1_GUARD_TOL:.0%} of the unconstrained optimum if "
                   "possible (structure-anchored refit with kc_grid "
                   "logspace(-2.5,-0.5) when the optimum violates the "
                   "s-layer bound); conflicts reported honestly"),
        diagnostics=dict(
            wake=("mean rho_tot over [x_cav-1 km, x_cav-0.2 km], t in "
                  "[600,740] s (e7_ablation.d1_wake; data A1 58.6 / "
                  "A10 55.4)"),
            ds_stuck=("(int s dx)/(int rho dx) over data cells with lower "
                      "edge >= x_cav at t=700 s"),
            waviness=("std of linearly detrended rho_tot over cell centers "
                      "in [x_cav+0.5 km, x_cav+5 km], mean over t in "
                      "[400,740] s; data value uses the nominal CAV "
                      "trajectory"),
            wedge=("fraction of cells in [x_cav+0.5, x_cav+5 km] at t=600 "
                   "s with rho strictly between (mean rho over "
                   "[x_cav+0.5, x_cav+1.5 km]) + 4 and (q_in/v_f) - 4 "
                   "veh/km"),
            rarefaction=("width growth of {rho_tot > 35 veh/km} between "
                         "t=780 and 960 s (e7_ablation.d2_rarefaction)"),
            s_layer=("max over t in [600,740] s of max_x {x_cav - "
                     "x_center : s > 1 veh/km}; mean over snapshots as "
                     "aux"),
            ds_max_s=("max |s| over data cells fully inside the DR-zeroed "
                      "region (solver cells j > int(x_cav//50)), all "
                      "snapshots; must be exactly 0.0 for L3/L4")),
        smoke=smoke)}

    for A in (1.0, 10.0):
        akey = f"A{A:g}"
        tag = f"A{A:g}_u{UC:g}_q{QIN_FIT:g}"
        block = dict(tag=tag,
                     data_refs_q2500=data_refs(A, UC, QIN_FIT),
                     rungs={})
        kept = {}
        for name, metric, extra_cfg, desc in RUNGS:
            print(f"\n=== {akey} {name}: {desc} ===", flush=True)
            fit = e7.fit_field(A, UC, QIN_FIT, form=FORM, metric=metric,
                               extra=None, dt_fit=1.0,
                               extra_cfg=(extra_cfg or None), **fit_kwargs)
            ev, regr, meas = eval_rung(A, UC, QIN_FIT, fit["kappa_c"],
                                       fit["kappa_r"], extra_cfg)
            kept[name] = (fit, regr, meas, extra_cfg)
            block["rungs"][name] = dict(
                metric=metric, description=desc,
                extra_cfg={k: v for k, v in extra_cfg.items()},
                kappa_c=fit["kappa_c"], kappa_r=fit["kappa_r"],
                fit_objective_dtfit=fit["objective"],
                n_sim_fit=fit["n_sim"], eval_q2500=ev)
            print(f"[eval dt=0.5] W1={ev['W1']:.2f} RMSE={ev['RMSE']:.2f} "
                  f"e_s={ev['e_s']:+.1%} "
                  f"wake={ev['wake']['wake_mean_vehkm']:.1f} "
                  f"stuckf={ev['ds_stuck']['frac']} "
                  f"wavi={ev['waviness_sim']['std_vehkm']:.2f} "
                  f"wedge={ev['wedge']['frac']} "
                  f"slayer={ev['s_layer']['max_m']:.0f} m "
                  f"ds|s|={ev['ds_max_s_vehkm']:.3e}", flush=True)

        # ---- analytical checks -------------------------------------------
        fit2, regr2, _, _ = kept["L2"]
        block["analytic_check_L2"] = analytic_check_growth(
            regr2, fit2["kappa_c"], fit2["kappa_r"], UC)
        def _print_growth(label, g):
            if g.get("status") == "ok":
                naive = (f" | naive pred={g['efold_naive_s']:.0f} s "
                         f"ratio={g['efold_ratio_meas_over_naive']:.2f}"
                         if g.get("efold_naive_s") else
                         " | naive predicts decay")
                print(f"{label} efold meas={g['efold_measured_s']:.0f} s "
                      f"vs frozen-f pred={g['efold_frozen_f_s']:.0f} s -> "
                      f"ratio={g['efold_ratio_meas_over_frozen_f']:.2f} "
                      f"(within factor 2: {g['within_factor_2_frozen_f']})"
                      + naive, flush=True)
            else:
                print(f"{label} {g.get('status')}", flush=True)

        _print_growth(f"\n[{akey} L2 analytic]", block["analytic_check_L2"])

        # closed-form verification in its regime of validity: the E7 C3
        # winner point (the exact configuration of Mladen's diagnosis)
        if A == 1.0:
            d7 = json.loads(E7_ABLATION_JSON.read_text())
            c3 = d7["A1"]["configs"]["C3"]
            kc3, kr3 = c3["kappa_c"], c3["kappa_r"]
            wsf3 = c3["extra"]["ws_frac"]
            regr3 = e7.run_sim(UC, QIN_FIT, kc3, kr3, FORM, ws_frac=wsf3,
                               dt=e7.DT_PRODUCTION)
            ttd, rho_mean_d = e7.load_rho_mean(A, UC, QIN_FIT)
            ref = dict(
                config=("E7 A1 winner C3 (Mladen's diagnosed leak case): "
                        f"kappa_c={kc3:.4g}, kappa_r={kr3:.4g}, "
                        f"w_s={wsf3:.4g}*w, no downstream_release"),
                kappa_c=kc3, kappa_r=kr3, ws_frac=wsf3,
                W1=e7.w1_mean(regr3["rho_tot"], rho_mean_d, ttd),
                ds_stuck=ds_stuck_fraction(regr3),
                waviness_sim=waviness(regr3["rho_tot"], ttd,
                                      regr3["x_cav"]),
                growth=analytic_check_growth(regr3, kc3, kr3, UC))
            block["analytic_check_reference_e7C3"] = ref
            _print_growth(f"[{akey} e7C3 analytic ref]", ref["growth"])
        block["exactness_check_L3_L4"] = {
            n: dict(ds_max_s_vehkm=block["rungs"][n]["eval_q2500"]
                    ["ds_max_s_vehkm"],
                    exact_zero=bool(block["rungs"][n]["eval_q2500"]
                                    ["ds_max_s_vehkm"] == 0.0))
            for n in ("L3", "L4")}

        # ---- pick + A=10 guard -------------------------------------------
        w1_of = {n: block["rungs"][n]["eval_q2500"]["W1"]
                 for n in block["rungs"]}
        opt = min(w1_of, key=w1_of.get)
        w1_unc = w1_of[opt]
        pick, guard = opt, dict(applies=False)
        if A == 10.0:
            slayer_opt = block["rungs"][opt]["eval_q2500"]["s_layer"]["max_m"]
            guard = dict(applies=True, w1_optimal_rung=opt,
                         w1_unconstrained=w1_unc,
                         s_layer_optimal_m=slayer_opt,
                         guard_threshold_m=SLAYER_GUARD_M)
            if slayer_opt > SLAYER_GUARD_M:
                print(f"\n[{akey} guard] {opt} s-layer {slayer_opt:.0f} m > "
                      f"{SLAYER_GUARD_M:g} m -> structure-anchored refit "
                      f"(kc_grid capped)", flush=True)
                # anchor-refit the W1-optimal rung (the task's alternative)
                # and, if different, the full-fix rung L3 as well, so the
                # feasibility verdict below is not an artifact of anchoring
                # only a no-DR rung
                bases = [opt] + (["L3"] if opt != "L3" else [])
                guard["anchored_rungs"] = []
                for base in bases:
                    _, metric_o, cfg_o, desc_o = next(
                        r for r in RUNGS if r[0] == base)
                    fk = dict(fit_kwargs)
                    fk["kc_grid"] = anchor_grid
                    fit_a = e7.fit_field(A, UC, QIN_FIT, form=FORM,
                                         metric=metric_o, extra=None,
                                         dt_fit=1.0,
                                         extra_cfg=(cfg_o or None), **fk)
                    name_a = base + "anc"
                    ev_a, regr_a, meas_a = eval_rung(A, UC, QIN_FIT,
                                                     fit_a["kappa_c"],
                                                     fit_a["kappa_r"], cfg_o)
                    kept[name_a] = (fit_a, regr_a, meas_a, cfg_o)
                    block["rungs"][name_a] = dict(
                        metric=metric_o,
                        description=desc_o + " [structure-anchored: "
                        "kc_grid logspace(-2.5,-0.5)]",
                        extra_cfg={k: v for k, v in cfg_o.items()},
                        kappa_c=fit_a["kappa_c"], kappa_r=fit_a["kappa_r"],
                        fit_objective_dtfit=fit_a["objective"],
                        n_sim_fit=fit_a["n_sim"], eval_q2500=ev_a)
                    w1_of[name_a] = ev_a["W1"]
                    guard["anchored_rungs"].append(name_a)
                guard["anchored_rung"] = bases[0] + "anc"
            # final pick: feasible = s-layer <= 1.5 km AND W1 <= 1.15 * opt
            feas = [n for n in w1_of
                    if block["rungs"][n]["eval_q2500"]["s_layer"]["max_m"]
                    <= SLAYER_GUARD_M
                    and w1_of[n] <= (1.0 + W1_GUARD_TOL) * w1_unc]
            if feas:
                pick = min(feas, key=w1_of.get)
                guard["conflict"] = False
                guard["resolution"] = (
                    f"pick {pick}: s-layer "
                    f"{block['rungs'][pick]['eval_q2500']['s_layer']['max_m']:.0f}"
                    f" m <= {SLAYER_GUARD_M:g} and W1 {w1_of[pick]:.1f} "
                    f"within {W1_GUARD_TOL:.0%} of unconstrained "
                    f"{w1_unc:.1f}")
            else:
                def _sl(n, key="max_m"):
                    return block["rungs"][n]["eval_q2500"]["s_layer"][key]

                slay_ok = [n for n in w1_of if _sl(n) <= SLAYER_GUARD_M]
                if slay_ok:
                    pick = min(slay_ok, key=w1_of.get)
                    why = "thin s-layer satisfied; only the W1 demand fails"
                else:
                    # the structural demand is the veto Mladen already
                    # exercised on E7's broad band: minimize its violation
                    # rather than keep the fast-churn W1 optimum
                    pick = min(w1_of, key=_sl)
                    why = ("no rung meets the 1.5 km bound on the "
                           "max-reach measure; picked the thinnest layer")
                guard["conflict"] = True
                guard["candidates"] = {
                    n: dict(W1=w1_of[n], s_layer_max_m=_sl(n),
                            s_layer_mean_m=_sl(n, "mean_m"))
                    for n in sorted(w1_of)}
                guard["resolution"] = (
                    f"CONFLICT: no rung satisfies both demands; picked "
                    f"{pick} ({why}): s-layer max {_sl(pick):.0f} m / "
                    f"snapshot-mean {_sl(pick, 'mean_m'):.0f} m vs the "
                    f"{SLAYER_GUARD_M:g} m bound, W1 {w1_of[pick]:.1f} = "
                    f"{w1_of[pick] / w1_unc - 1.0:+.1%} vs unconstrained "
                    f"{w1_unc:.1f} ({opt}, s-layer max "
                    f"{_sl(opt):.0f} m)")
        if guard.get("applies") and "resolution" in guard:
            print(f"[{akey} guard] {guard['resolution']}", flush=True)
        block["guard"] = guard
        block["pick"] = pick
        ev_pick = block["rungs"][pick]["eval_q2500"]
        if (ev_pick["ds_max_s_vehkm"] > 0.0
                and (ev_pick["ds_stuck"]["frac"] or 0.0) > 0.005):
            block["pick_caveat"] = (
                f"final pick {pick} carries no downstream_release and "
                "re-leaks stuck vehicles downstream: stuck fraction "
                f"{ev_pick['ds_stuck']['frac']:.1%} at t=700 s, max "
                f"downstream s {ev_pick['ds_max_s_vehkm']:.1f} veh/km")
            print(f"[{akey}] CAVEAT: {block['pick_caveat']}", flush=True)
        block["pick_rule"] = (f"min W1 among rungs = {opt} "
                              f"(W1={w1_unc:.1f})"
                              + ("" if pick == opt else
                                 f"; A=10 guard moved the pick to {pick}"))
        print(f"\n[{akey}] final pick: {pick}  ({block['pick_rule']})",
              flush=True)

        # ---- zero-refit transfer of the pick to q2000 --------------------
        fit_p, regr_p, meas_p, cfg_p = kept[pick]
        ev_t, _, _ = eval_rung(A, UC, QIN_TRANSFER, fit_p["kappa_c"],
                               fit_p["kappa_r"], cfg_p)
        block["pick_transfer_q2000"] = ev_t
        block["data_refs_q2000"] = data_refs(A, UC, QIN_TRANSFER)
        print(f"[{akey}] transfer q2000: W1={ev_t['W1']:.2f} "
              f"RMSE={ev_t['RMSE']:.2f} e_s={ev_t['e_s']:+.1%} "
              f"omega_err={ev_t['omega_err']:+.1%} "
              f"wake={ev_t['wake']['wake_mean_vehkm']:.1f}", flush=True)

        # ---- figures for the final pick ----------------------------------
        e7_regr, e7_lbl = load_e7_winner(A)
        knobs = block["rungs"][pick]["extra_cfg"]
        knob_abbr = " + ".join(
            [block["rungs"][pick]["metric"]]
            + [{"w_s": f"w_s={WS_FRAC_FIX:g}w",
                "downstream_release": "DR"}.get(k, k) for k in knobs])
        e8_lbl = f"E8 {pick} ({knob_abbr})"
        fig_tag = ("smoke_" if smoke else "") + tag
        fig_profiles_e8(fig_tag, meas_p, e7_regr, e7_lbl, regr_p,
                        f"E8 {pick}", OUT_E8)
        fig_heat3_e8(fig_tag, meas_p, e7_regr, e7_lbl, regr_p, e8_lbl,
                     OUT_E8)
        results[akey] = block
        print_table(akey, block)

    results["_meta"]["runtime_s"] = round(time.time() - t_start, 1)
    out_path = OUT_E8 / ("ladder_smoke.json" if smoke else "ladder.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path} ({results['_meta']['runtime_s']} s)")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="E8 one-at-a-time ablation ladder")
    ap.add_argument("--run", action="store_true", help="full ladder")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny-grid mechanics check")
    args = ap.parse_args(argv)
    if args.smoke:
        run(smoke=True)
    elif args.run:
        run(smoke=False)
    else:
        ap.error("choose --run or --smoke")


if __name__ == "__main__":
    main()
