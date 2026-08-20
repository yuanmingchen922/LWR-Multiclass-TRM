"""E7 calibration ablation: RMSE vs W1 objective x structural knobs.

For each assertiveness A in {1, 10} (scenario u15 q2500, capture form 'lf')
four configurations are fitted with e7_wasserstein.fit_field:

  C0  metric='rmse', no extra            E6 reproduction / reference
  C1  metric='w1',   no extra            Mladen's main suggestion
  C2  metric='w1',   gamma in (0, 1)     capture localization, ell = a + g s
  C3  metric='w1',   ws_frac in (.5, 1)  stuck-class congested branch,
                                         w_s = ws_frac * w (P_s = legacy)

Every fitted config is re-run at the production resolution dt=0.5 and scored:
W1, RMSE (both vs the rep-mean field), e_s and omega_cum_rel_err (reused from
ev4_compare.metrics via regrid), plus three physical diagnostics:

  d1  wake wedge density: mean rho_tot over x in [x_cav-1 km, x_cav-0.2 km]
      at t in [600, 740] s.  SUMO plateau ~58 veh/km; supercritical means
      > rho_crit = 48.5 veh/km (a subcritical wedge advects at v_f under a
      triangular FD and never spreads -- the single defect behind both of
      Mladen's complaints).
  d2  post-release rarefaction: width growth of the {rho_tot > 35 veh/km}
      region between t=780 and t=960 s [m], and the peak-density decay rate
      over the same span [veh/km/s].
  d3  dynamic equilibrium (the A=10 story): extent of the s > 1 veh/km layer
      behind the CAV [m], turnover time (int s dx)/(int J_r dx) in the slow
      window [s] with J_r = kappa_r (P-rho)_+ dv s, and whether a 'slow but
      not stuck' band exists (mean speed in [x_cav-2 km, x_cav] strictly
      inside (u_xi+2, v_f-3) m/s AND band density > 1.2x inflow density).

Winner per A: primary criterion W1 at dt=0.5; ties (within 5% of the min)
broken by |d1 - 58| then by d2 width growth > 0.  Winners get E6-style
figures (fig_profiles_e7_{tag}.png with a post-release 850 s snapshot,
fig_heat3_e7_{tag}.png via e6_native_mb.fig_heat3) and a zero-refit transfer
evaluation on q2000.

Output: out/e7/ablation.json (+ figures).  Fits run at dt=1.0 (allowed for
coarse fitting), everything reported here is re-evaluated at dt=0.5.

CLI:  python3 e7_ablation.py --run            full program
      python3 e7_ablation.py --smoke          tiny-grid mechanics check
                                              (-> out/e7/ablation_smoke.json)
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

import e6_native_mb as e6
import e7_wasserstein as e7
import ev4_compare as ev4
from loader import CELL_LEN, N_CELL

HERE = Path(__file__).parent
OUT_E7 = HERE / "out" / "e7"

UC = 15.0                    # [m/s] focus CAV slow speed
QIN_FIT = 2500.0             # [veh/h] fit scenario inflow
QIN_TRANSFER = 2000.0        # [veh/h] zero-refit transfer inflow
FORM = "lf"
QXI_M1_VEHH = 2000.0         # classical M1 reference: kappa=0 + DM-G cap

RHO_CRIT = 48.5              # [veh/km] critical density of the road FD
D1_TARGET = 58.0             # [veh/km] SUMO wake plateau
D1_T = (600.0, 740.0)        # [s]
D1_BEHIND = (1000.0, 200.0)  # [m] window [x_cav-1000, x_cav-200]
D2_T = (780.0, 960.0)        # [s]
D2_THRESH = 35.0             # [veh/km]
D3_T = (ev4.SLOW_LO, ev4.SLOW_HI)   # [260, 740] s slow window
D3_S_MIN = 1.0               # [veh/km] s-layer threshold
D3_BAND_M = 2000.0           # [m] band [x_cav-2 km, x_cav]
W1_TIE_TOL = 0.05            # winner tie tolerance on W1

X_EDGES = (np.arange(N_CELL) + 1) * CELL_LEN   # [m] data-cell upper edges
X_MAX = e7.X_MAX_CELLS                          # 200 cells = 20 km

CONFIGS = (
    ("C0", "rmse", None),
    ("C1", "w1", None),
    ("C2", "w1", {"gamma": (0.0, 1.0)}),
    ("C3", "w1", {"ws_frac": (0.5, 1.0)}),
)


# ---------------------------------------------------------------------------
# diagnostics d1 / d2 / d3
# ---------------------------------------------------------------------------

def d1_wake(rho, tt, x_cav):
    """Mean rho_tot [veh/km] over [x_cav-1 km, x_cav-0.2 km] x t in D1_T."""
    tt = np.asarray(tt, float)
    idx = np.where((tt >= D1_T[0]) & (tt <= D1_T[1]))[0]
    vals = []
    for i in idx:
        xc = float(x_cav[i])
        if not np.isfinite(xc):
            continue
        sel = (X_EDGES >= xc - D1_BEHIND[0] - 1e-9) & \
              (X_EDGES <= xc - D1_BEHIND[1] + 1e-9)
        if sel.any():
            vals.append(float(np.mean(np.asarray(rho, float)[i, sel])))
    mean = float(np.mean(vals))
    return dict(wake_mean_vehkm=mean, supercritical=bool(mean > RHO_CRIT),
                n_snapshots=len(vals))


def d2_rarefaction(rho, tt):
    """Width growth of {rho > 35 veh/km} and peak decay over [780, 960] s."""
    rho = np.asarray(rho, float)[:, :X_MAX]
    tt = np.asarray(tt, float)

    def at(t):
        i = int(np.argmin(np.abs(tt - t)))
        width = float(np.sum(rho[i] > D2_THRESH) * CELL_LEN)
        return width, float(np.max(rho[i]))

    w0, p0 = at(D2_T[0])
    w1, p1 = at(D2_T[1])
    span = D2_T[1] - D2_T[0]
    return dict(width_780_m=w0, width_960_m=w1, width_growth_m=w1 - w0,
                peak_780_vehkm=p0, peak_960_vehkm=p1,
                peak_decay_vehkm_per_s=(p0 - p1) / span,
                rarefaction_present=bool(w1 - w0 > 0.0))


def _class_speeds_ms(rho_vehkm, uc, ws_frac):
    """(v_f-class, v_s-class) speeds [m/s] at total density rho [veh/km].

    f-class keeps the shared road FD; the s-class congested branch uses
    w_s = ws_frac * w when the config carries the C3 knob (P_s = legacy P),
    matching solver.transport_step.  Slow window assumed: c_s = u_xi.
    """
    rho_si = np.maximum(np.asarray(rho_vehkm, float) / 1000.0, 1e-12)
    v_f = np.minimum(ev4.V_F, np.maximum(ev4.W * (ev4.P / rho_si - 1.0), 0.0))
    w_s = (1.0 if ws_frac is None else float(ws_frac)) * ev4.W
    v_s = np.minimum(uc, np.maximum(w_s * (ev4.P / rho_si - 1.0), 0.0))
    return v_f, v_s


def d3_dyn_eq(regr, uc, qin, kappa_r, ws_frac):
    """Dynamic-equilibrium diagnostics in the slow window [260, 740] s."""
    tt = np.asarray(regr["tt"], float)
    idx = np.where((tt >= D3_T[0]) & (tt <= D3_T[1]))[0]
    a, f, s = regr["a"], regr["f"], regr["s"]
    rho = regr["rho_tot"]
    x_cav = np.asarray(regr["x_cav"], float)

    # (i) extent of the s > 1 veh/km layer behind the CAV
    ext = []
    for i in idx:
        if not np.isfinite(x_cav[i]):
            continue
        sel = (X_EDGES <= x_cav[i] + 1e-9) & (s[i] > D3_S_MIN)
        ext.append(float(np.sum(sel) * CELL_LEN))
    extent = float(np.mean(ext)) if ext else 0.0

    # (ii) turnover time: time-aggregated (int s dx) / (int J_r dx)
    from solver import speed as fd_speed
    rho_si = rho[idx] / 1000.0
    s_si = s[idx] / 1000.0
    dv = np.maximum(fd_speed(rho_si, ev4.V_F, ev4.W, ev4.P) - uc, 0.0)
    mu = float(kappa_r) * np.maximum(ev4.P - rho_si, 0.0) * dv       # [1/s]
    j_r = float(np.sum(mu * s_si) * CELL_LEN)                        # [veh/s]*nt
    n_s = float(np.sum(s_si) * CELL_LEN)                             # [veh]*nt
    turnover = n_s / j_r if j_r > 0.0 else np.inf

    # (iii) 'slow but not stuck' band in [x_cav-2 km, x_cav]
    v_f_cls, v_s_cls = _class_speeds_ms(rho, uc, ws_frac)
    q_ms = (a + s) * v_s_cls + f * v_f_cls        # [veh/km * m/s]
    v_num = v_den = 0.0
    rho_band = []
    for i in idx:
        if not np.isfinite(x_cav[i]):
            continue
        sel = (X_EDGES >= x_cav[i] - D3_BAND_M - 1e-9) & \
              (X_EDGES <= x_cav[i] + 1e-9)
        v_num += float(np.sum(q_ms[i, sel]))
        v_den += float(np.sum(rho[i, sel]))
        rho_band.append(float(np.mean(rho[i, sel])))
    v_band = v_num / v_den if v_den > 0.0 else float(ev4.V_F)
    rho_band_mean = float(np.mean(rho_band)) if rho_band else 0.0
    rho_in = qin / 3600.0 / ev4.V_F * 1000.0                 # [veh/km]
    band = bool((uc + 2.0 < v_band < ev4.V_F - 3.0)
                and rho_band_mean > 1.2 * rho_in)

    return dict(s_layer_extent_m=extent,
                turnover_time_s=(float(turnover) if np.isfinite(turnover)
                                 else None),
                band_speed_ms=float(v_band),
                band_rho_vehkm=rho_band_mean,
                rho_inflow_vehkm=float(rho_in),
                slow_not_stuck_band=band)


# ---------------------------------------------------------------------------
# per-config production evaluation
# ---------------------------------------------------------------------------

def eval_production(A, uc, qin, kappa_c, kappa_r, extra):
    """Run at dt=0.5, score vs data, compute diagnostics.  Returns
    (summary dict, regr, meas)."""
    gamma = extra.get("gamma")
    ws_frac = extra.get("ws_frac")
    regr = e7.run_sim(uc, qin, kappa_c, kappa_r, FORM, gamma=gamma,
                      ws_frac=ws_frac, dt=e7.DT_PRODUCTION)
    tt, rho_mean = e7.load_rho_mean(A, uc, qin)
    meas = ev4.load_measured(A, uc, qin)
    met = ev4.metrics(regr, meas)
    summary = dict(
        W1=e7.w1_mean(regr["rho_tot"], rho_mean, tt),
        RMSE=e7.rmse_mean(regr["rho_tot"], rho_mean, tt),
        e_s=met["e_s"]["mean"],
        omega_err=met["omega_cum_rel_err"]["mean"],
        rho_rmse_per_rep=met["rho_rmse"]["mean"],
        Ns_mae=met["Ns_mae"]["mean"],
        d1=d1_wake(regr["rho_tot"], tt, regr["x_cav"]),
        d2=d2_rarefaction(regr["rho_tot"], tt),
        d3=d3_dyn_eq(regr, uc, qin, kappa_r, ws_frac),
    )
    return summary, regr, meas


def data_diagnostics(A, uc, qin):
    """d1/d2 of the rep-mean measured field (nominal CAV trajectory)."""
    tt, rho_mean = e7.load_rho_mean(A, uc, qin)
    x_nom = ev4.x_cav_nominal(tt, uc)
    return dict(d1=d1_wake(rho_mean, tt, x_nom),
                d2=d2_rarefaction(rho_mean, tt))


# ---------------------------------------------------------------------------
# winner selection
# ---------------------------------------------------------------------------

def pick_winner(rows):
    """Primary: min W1 at dt=0.5.  Tie set: W1 within W1_TIE_TOL of the min;
    broken by |d1 - 58| then by d2 width growth > 0."""
    w1_min = min(r["W1"] for r in rows.values())
    cands = [k for k, r in rows.items() if r["W1"] <= (1.0 + W1_TIE_TOL) * w1_min]
    rule = (f"min W1 (dt=0.5) = {w1_min:.2f} veh km; tie set (within "
            f"{W1_TIE_TOL:.0%}): {sorted(cands)}")
    if len(cands) > 1:
        cands.sort(key=lambda k: (
            abs(rows[k]["d1"]["wake_mean_vehkm"] - D1_TARGET),
            0.0 if rows[k]["d2"]["width_growth_m"] > 0.0 else 1.0))
        rule += f"; tie-break |d1-{D1_TARGET:g}| then d2>0 -> {cands[0]}"
    return cands[0], rule


# ---------------------------------------------------------------------------
# figures (E6 style; heat3 reused from e6_native_mb, profiles adapted to add
# the post-release 850 s snapshot)
# ---------------------------------------------------------------------------

PROFILE_SNAPS = (300.0, 500.0, 700.0, 850.0)


def fig_profiles_e7(tag, meas, m1, m2, out_dir, winner_label):
    """Adapted from e6_native_mb.fig_profiles: 4 snapshots including 850 s
    (post-release rarefaction), rho_crit reference line."""
    rho_d = np.mean(meas["rho"], axis=0)
    tt = m1["tt"]
    xs = (np.arange(rho_d.shape[1]) + 1) * CELL_LEN / 1000.0
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 11.0), sharex=True)
    for ax, t_snap in zip(axes, PROFILE_SNAPS):
        i = int(np.argmin(np.abs(tt - t_snap)))
        ax.plot(xs, rho_d[i], "k-", lw=1.8, label="SUMO (rep mean)")
        ax.plot(xs, m1["rho_tot"][i], color="tab:orange", lw=1.5,
                label="M1: LWR + MB (DM-G cap)")
        ax.plot(xs, m2["rho_tot"][i], color="tab:blue", lw=1.5,
                label=f"E7 winner ({winner_label}): total " + r"$\rho$")
        ax.fill_between(xs, 0, m2["s"][i], color="tab:red", alpha=0.30,
                        label="winner: caught $s$")
        ax.fill_between(xs, m2["s"][i], m2["s"][i] + m2["f"][i],
                        color="tab:green", alpha=0.20, label="winner: free $f$")
        ax.axhline(RHO_CRIT, color="0.55", ls="--", lw=0.9)
        xc = m1["x_cav"][i] / 1000.0
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
    fig.suptitle(f"{tag}: density profiles — data vs classical vs E7 winner "
                 f"({winner_label}, with sub-class split)")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_profiles_e7_{tag}.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main program
# ---------------------------------------------------------------------------

def run(smoke=False):
    t_start = time.time()
    OUT_E7.mkdir(parents=True, exist_ok=True)
    fit_kwargs = ({} if not smoke else
                  dict(kc_grid=[0.2, 0.5], kr_grid=[0.01, 0.04], maxfev=2))

    results = {"_meta": dict(
        scenario=f"u{UC:g} q{QIN_FIT:g} form={FORM}",
        transfer=f"winner zero-refit on q{QIN_TRANSFER:g}",
        configs={n: dict(metric=m, extra=(None if x is None else
                                          {k: list(v) for k, v in x.items()}))
                 for n, m, x in CONFIGS},
        dt_fit=1.0, dt_production=e7.DT_PRODUCTION,
        winner_rule=("primary min W1 at dt=0.5; ties within "
                     f"{W1_TIE_TOL:.0%} broken by |d1-{D1_TARGET:g}| "
                     "then d2 width growth > 0"),
        diagnostics=dict(
            d1=(f"mean rho_tot over [x_cav-{D1_BEHIND[0]:g} m, "
                f"x_cav-{D1_BEHIND[1]:g} m], t in {list(D1_T)} s; data ~58 "
                f"veh/km; supercritical iff > {RHO_CRIT} veh/km"),
            d2=(f"width growth of {{rho_tot > {D2_THRESH:g} veh/km}} and "
                f"peak decay between t={D2_T[0]:g} and {D2_T[1]:g} s, "
                "x <= 20 km"),
            d3=(f"s > {D3_S_MIN:g} veh/km layer extent behind CAV, turnover "
                "(int s dx)/(int J_r dx) with J_r = kappa_r (P-rho)+ dv s, "
                f"and slow-not-stuck band in [x_cav-{D3_BAND_M:g} m, x_cav]: "
                f"speed in (u_xi+2, v_f-3) m/s AND rho > 1.2x inflow; "
                f"slow window {list(D3_T)} s")),
        smoke=smoke)}

    for A in (1.0, 10.0):
        akey = f"A{A:g}"
        tag = f"A{A:g}_u{UC:g}_q{QIN_FIT:g}"
        block = dict(tag=tag,
                     data_diagnostics_q2500=data_diagnostics(A, UC, QIN_FIT),
                     configs={})
        rows, kept = {}, {}
        for name, metric, extra in CONFIGS:
            print(f"\n=== {akey} {name}: metric={metric} extra={extra} ===",
                  flush=True)
            fit = e7.fit_field(A, UC, QIN_FIT, form=FORM, metric=metric,
                               extra=extra, dt_fit=1.0, **fit_kwargs)
            ev, regr, meas = eval_production(A, UC, QIN_FIT, fit["kappa_c"],
                                             fit["kappa_r"], fit["extra"])
            rows[name] = ev
            kept[name] = (fit, regr, meas)
            block["configs"][name] = dict(
                metric=metric, kappa_c=fit["kappa_c"], kappa_r=fit["kappa_r"],
                extra=fit["extra"], fit_objective_dtfit=fit["objective"],
                n_sim_fit=fit["n_sim"], eval_q2500=ev)
            print(f"[eval dt=0.5] W1={ev['W1']:.2f} RMSE={ev['RMSE']:.2f} "
                  f"e_s={ev['e_s']:+.1%} omega_err={ev['omega_err']:+.1%} | "
                  f"d1={ev['d1']['wake_mean_vehkm']:.1f} "
                  f"(super={ev['d1']['supercritical']}) "
                  f"d2_growth={ev['d2']['width_growth_m']:.0f} m "
                  f"d3_band={ev['d3']['slow_not_stuck_band']}", flush=True)

        winner, rule = pick_winner(rows)
        block["winner"] = winner
        block["winner_rule"] = rule
        print(f"\n[{akey}] winner: {winner}  ({rule})", flush=True)

        # zero-refit transfer of the winner to q2000
        fit_w, regr_w, meas_w = kept[winner]
        ev_t, _, _ = eval_production(A, UC, QIN_TRANSFER, fit_w["kappa_c"],
                                     fit_w["kappa_r"], fit_w["extra"])
        block["winner_transfer_q2000"] = ev_t
        block["data_diagnostics_q2000"] = data_diagnostics(A, UC,
                                                           QIN_TRANSFER)
        print(f"[{akey}] transfer q2000: W1={ev_t['W1']:.2f} "
              f"RMSE={ev_t['RMSE']:.2f} e_s={ev_t['e_s']:+.1%} "
              f"omega_err={ev_t['omega_err']:+.1%}", flush=True)

        # E6-style figures for the winner at the fit scenario
        m1 = e6.run_model(A, UC, QIN_FIT, 0.0, 0.0, "lf", QXI_M1_VEHH)
        extra_lbl = "".join(f", {k}={v:.3g}" for k, v in
                            fit_w["extra"].items())
        winner_label = f"{winner}: {block['configs'][winner]['metric']}" \
                       + extra_lbl
        fig_tag = ("smoke_" if smoke else "") + tag
        e6.fig_heat3(f"e7_{fig_tag}", meas_w, m1, regr_w, OUT_E7)
        fig_profiles_e7(fig_tag, meas_w, m1, regr_w, OUT_E7, winner_label)
        results[akey] = block

    results["_meta"]["runtime_s"] = round(time.time() - t_start, 1)
    out_path = OUT_E7 / ("ablation_smoke.json" if smoke else "ablation.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path} ({results['_meta']['runtime_s']} s)")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="E7 calibration ablation driver")
    ap.add_argument("--run", action="store_true", help="full ablation")
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
