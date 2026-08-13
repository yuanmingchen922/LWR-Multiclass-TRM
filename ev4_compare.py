"""E-V4: simulation-vs-data comparison harness for the two-class LWR
transition model.

Runs the finite-volume solver (solver.py, developed in parallel) for the 8
core scenarios A in {1, 10} x u_c in {15, 20} m/s x q_in in {2000, 2500}
veh/h with the E-V3 kappa point estimates, regrids the solution onto the SUMO
measurement grid (100 m x 10 s, upper-edge convention XS = (1:300)*100 m,
t = (1..100)*10 s) and scores it against the E1-classified data:

  rho_rmse           total-density RMSE (a+f+s vs measured rho), per rep,
                     over t in [100, 1000] s, x in [0, 20] km;
  Ns_mae             queue-size MAE |N_s^sim(t) - n_caught(t)| over the slow
                     window [260, 740] s;
  omega_cum_rel_err  relative error of cumulative passings over the slow
                     window.  Comparability caveat: the measured overtake
                     series counts ALL passings (captured-then-released AND
                     free pass-throughs); the model's omega = f (v - u_s) at
                     the CAV is the same total-passing concept, so the two
                     series are directly comparable;
  e_s                ECC22 eq. (6) analogue: relative outflow error at
                     X_q = 15 km, window [t0 = 576 s, CAV arrival at X_q]
                     with the arrival time taken from the nominal (analytic)
                     CAV schedule.  q_sim is rebuilt from the regridded class
                     densities with the FD class speeds
                     V_c(rho) = min{c, w (P/rho - 1)} (tex
                     eq. intrinsic-speed-explicit): classes a, s move at
                     V_{u_s(t)}, class f at V_{v_f}.

The solver is imported lazily inside run_all() only, so `import ev4_compare`
has no side effects and works while solver.py is still being built.  The
expected contract is

    from solver import SimConfig, simulate
    cfg = SimConfig(v_f, w, P, q_in, u_xi, kappa_c, kappa_r,
                    capture_form="lf")           # SI: veh/m, m/s, veh/s
    res = simulate(cfg)   # res.t (n_save,), res.x (nx=600, dx=50 m),
                          # res.a/f/s (n_save, nx) [veh/m], res.x_cav [m],
                          # res.omega [veh/s], res.N_s [veh]

Validation without solver.py:  python3 ev4_compare.py --selftest  fabricates
a SimResult-shaped mock and exercises regridding + metrics + one heatmap.

Input : out/e1/*.npz, ../Second/data_*_True.mat, out/params.json,
        out/ev3_kappa.json
Output: out/ev4/metrics.json + figures   (--form af -> out/ev4_af/)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from loader import CELL_LEN, N_CELL, T_SAMPLE, load_scenario

HERE = Path(__file__).parent
OUT = HERE / "out"
E1 = OUT / "e1"
SECOND = HERE.parent / "Second"

# Calibrated FD parameters (params.json stores km/h and veh/km) -> SI.
_PARAMS = json.loads((OUT / "params.json").read_text())
V_F_KMH, W_KMH, P_VEHKM = (_PARAMS["v_f_kmh"], _PARAMS["w_kmh"],
                           _PARAMS["P_vehkm"])
V_F, W, P = V_F_KMH / 3.6, W_KMH / 3.6, P_VEHKM / 1000.0  # m/s, m/s, veh/m

V_CAV_FREE = 27.78                   # [m/s] CAV free-driving speed
T_ENTER, T_SLOW, T_FAST, T_END = 100.0, 250.0, 750.0, 1000.0
N_T = 100                            # field samples, t = (1..N_T)*10 s

SLOW_LO, SLOW_HI = 260.0, 740.0      # [s] event window (matches E1 / E-V3)
RMSE_T = (100.0, 1000.0)             # [s] density-RMSE time window
RMSE_X_KM = 20.0                     # [km] density-RMSE space window
X_Q = 15_000.0                       # [m] outflow measurement station
J_XQ = 149                           # data cell with upper edge X_q (=(j+1)*100)
T0_ES = 576.0                        # [s] ECC22 eq. (6) window start at X_q

CORE = [(A, uc, qin) for A in (1.0, 10.0) for uc in (15.0, 20.0)
        for qin in (2000.0, 2500.0)]

OMEGA_NOTE = ("measured overtake counts ALL passings (captured-then-released "
              "AND free pass-throughs); sim omega = f (v - u_s) at the CAV "
              "is the same total-passing concept -> directly comparable")


# ---------------------------------------------------------------------------
# nominal CAV schedule (prescribed analytically, not transported numerically)
# ---------------------------------------------------------------------------

def x_cav_nominal(t, u_xi):
    """Nominal CAV trajectory [m] (NaN before entry at T_ENTER)."""
    t = np.asarray(t, float)
    x1 = V_CAV_FREE * (T_SLOW - T_ENTER)                 # 4167 m
    x2 = x1 + u_xi * (T_FAST - T_SLOW)
    x = np.where(t < T_SLOW, V_CAV_FREE * (t - T_ENTER),
                 np.where(t <= T_FAST, x1 + u_xi * (t - T_SLOW),
                          x2 + V_CAV_FREE * (t - T_FAST)))
    return np.where(t >= T_ENTER, x, np.nan)


def t_reach(x_target, u_xi):
    """First time [s] the nominal CAV trajectory reaches x_target [m]."""
    x1 = V_CAV_FREE * (T_SLOW - T_ENTER)
    x2 = x1 + u_xi * (T_FAST - T_SLOW)
    if x_target <= x1:
        return T_ENTER + x_target / V_CAV_FREE
    if x_target <= x2:
        return T_SLOW + (x_target - x1) / u_xi
    return T_FAST + (x_target - x2) / V_CAV_FREE


def u_s_of_t(t, u_xi):
    """CAV speed schedule [m/s]: u_xi in the slow phase, free otherwise."""
    t = np.asarray(t, float)
    return np.where((t >= T_SLOW) & (t <= T_FAST), u_xi, V_CAV_FREE)


# ---------------------------------------------------------------------------
# measured data
# ---------------------------------------------------------------------------

def load_measured(A, uc, qin):
    """Per-rep measured series for one scenario.

    Sources: out/e1/*.npz (classification counts, density field, measured
    overtake flow, CAV position) and the raw .mat (flows field, needed for
    e_s at X_q).  Returns arrays stacked over reps plus mean / min-max
    envelopes of n_caught(t) and overtake(t); rho of rep 0 serves the
    heatmaps.
    """
    tag = f"A{A:g}_u{uc:g}_q{qin:g}"
    files = sorted(E1.glob(f"{tag}_r*.npz"))
    if not files:
        raise FileNotFoundError(f"no E1 npz for {tag} under {E1}")
    tt = None
    n_caught, overtake, rho, x_cav, l_zone = [], [], [], [], []
    for f in files:
        d = np.load(f)
        tt = d["tt"]
        n_caught.append(d["n_caught"].astype(float))
        # overtake_meas lives on t = (0..100)*10 s; drop t=0 to align with tt
        overtake.append(d["overtake_meas"][1:].astype(float))
        rho.append(np.asarray(d["rho"], float))
        x_cav.append(d["x_cav"])
        l_zone.append(d["L_zone"])
    n_caught = np.array(n_caught)
    overtake = np.array(overtake)

    sc = load_scenario(SECOND / f"data_{A:g}_{uc:g}_{qin:g}_True.mat",
                       fields=True, ctrl=False, trajs=False)
    reps = [int(f.stem.rsplit("_r", 1)[1]) for f in files]
    q_xq = np.array([sc.reps[r].q[:, J_XQ] for r in reps])   # [veh/h]

    return dict(A=A, uc=uc, qin=qin, tag=tag, tt=tt, reps=reps,
                n_caught=n_caught, overtake=overtake, rho=np.array(rho),
                x_cav=np.array(x_cav), L_zone=np.array(l_zone), q_xq=q_xq,
                n_caught_mean=n_caught.mean(0), n_caught_min=n_caught.min(0),
                n_caught_max=n_caught.max(0),
                overtake_mean=overtake.mean(0), overtake_min=overtake.min(0),
                overtake_max=overtake.max(0))


# ---------------------------------------------------------------------------
# regridding: solver grid (dx = 50 m, save every 10 s) -> data grid
# ---------------------------------------------------------------------------

def regrid_sim(res):
    """Sim solution -> measurement grid.

    Space: pairs of 50 m solver cells averaged into the 300 x 100 m data
    cells (upper-edge convention, exact because 2 x 50 m tile each data
    cell).  Time: solver save instants matched to t = (1..100)*10 s.
    Units: veh/m -> veh/km (x1000), veh/s -> veh/h (x3600).
    """
    t_data = (np.arange(N_T) + 1) * T_SAMPLE
    res_t = np.asarray(res.t, float)
    res_x = np.asarray(res.x, float)
    if res_x.size != 2 * N_CELL:
        raise ValueError(f"expected {2 * N_CELL} solver cells (dx = 50 m), "
                         f"got {res_x.size}")
    dx = np.median(np.diff(res_x))
    if not np.isclose(dx, CELL_LEN / 2.0, atol=1e-6):
        raise ValueError(f"expected dx = {CELL_LEN / 2.0} m, got {dx}")
    it = np.argmin(np.abs(res_t[None, :] - t_data[:, None]), axis=1)
    err = np.max(np.abs(res_t[it] - t_data))
    if err > 0.5:
        raise ValueError(f"solver save times miss the 10 s data grid "
                         f"(worst offset {err:.2f} s)")

    def field(arr):
        arr = np.asarray(arr, float)[it]                    # (N_T, 600)
        return arr.reshape(N_T, N_CELL, 2).mean(axis=2) * 1000.0

    a, f, s = field(res.a), field(res.f), field(res.s)
    return dict(tt=t_data, a=a, f=f, s=s, rho_tot=a + f + s,
                x_cav=np.asarray(res.x_cav, float)[it],
                N_s=np.asarray(res.N_s, float)[it],
                omega=np.asarray(res.omega, float)[it] * 3600.0)   # veh/h


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def v_class(rho_vehkm, c_kmh):
    """FD class speed V_c [km/h] = min{c, w (P/rho - 1)} (shared congested
    branch); c may be a scalar or an array broadcast against rho."""
    with np.errstate(divide="ignore"):
        vw = W_KMH * (P_VEHKM / np.maximum(np.asarray(rho_vehkm, float),
                                           1e-9) - 1.0)
    return np.minimum(c_kmh, np.maximum(vw, 0.0))


def flux_sim_xq(regr, u_xi):
    """Model outflow q_sim(X_q, t) [veh/h] rebuilt from regridded states:
    q = (a+s) V_{c_s(t)}(rho) + f V_{v_f}(rho), rho the total density.
    c_s follows the SOLVER's class-speed schedule (v_f outside the slow
    phase, not the CAV cruise speed 27.78 m/s)."""
    rho = regr["rho_tot"][:, J_XQ]
    tt = regr["tt"]
    us_kmh = np.where((tt >= T_SLOW) & (tt <= T_FAST), u_xi * 3.6, V_F_KMH)
    return ((regr["a"][:, J_XQ] + regr["s"][:, J_XQ]) * v_class(rho, us_kmh)
            + regr["f"][:, J_XQ] * v_class(rho, V_F_KMH))


def metrics(regr, meas):
    """Score one scenario: per-rep values and their means (see module
    docstring for the definitions and windows)."""
    tt = regr["tt"]
    assert np.allclose(tt, meas["tt"]), "sim/data time grids differ"
    uc = meas["uc"]
    dt_h = T_SAMPLE / 3600.0

    # density RMSE, t in [100, 1000] s, x in [0, 20] km
    wt = (tt >= RMSE_T[0]) & (tt <= RMSE_T[1])
    jx = int(RMSE_X_KM * 1000.0 / CELL_LEN)       # upper edges <= 20 km
    sim_rho = regr["rho_tot"][wt][:, :jx]
    rho_rmse = [float(np.sqrt(np.mean((sim_rho - r[wt][:, :jx]) ** 2)))
                for r in meas["rho"]]

    # queue size MAE over the slow window
    ws = (tt >= SLOW_LO) & (tt <= SLOW_HI)
    ns_sim = regr["N_s"][ws]
    ns_mae = [float(np.mean(np.abs(ns_sim - nc[ws])))
              for nc in meas["n_caught"]]

    # cumulative passings [veh] over the slow window; see OMEGA_NOTE
    cum_sim = float(regr["omega"][ws].sum() * dt_h)
    cum_meas = [float(ov[ws].sum() * dt_h) for ov in meas["overtake"]]
    om_err = [(cum_sim - c) / c if c > 0 else np.nan for c in cum_meas]

    # e_s at X_q over [T0_ES, nominal CAV arrival]
    t1 = t_reach(X_Q, uc)
    we = (tt >= T0_ES) & (tt <= t1)
    num = float(flux_sim_xq(regr, uc)[we].sum())
    e_s = []
    for q in meas["q_xq"]:
        den = float(q[we].sum())
        e_s.append(num / den - 1.0 if den > 0 else np.nan)

    def pack(vals):
        return dict(per_rep=[float(v) for v in vals],
                    mean=float(np.nanmean(vals)))

    return dict(rho_rmse=pack(rho_rmse), Ns_mae=pack(ns_mae),
                omega_cum_rel_err=pack(om_err), e_s=pack(e_s),
                cum_overtake_sim_veh=cum_sim,
                cum_overtake_meas_veh=cum_meas,
                es_window_s=[T0_ES, t1])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_heat(regr, meas, out_dir, prefix=""):
    """1x2 heatmap: sim total density vs measured (rep 0), shared scale."""
    tag = meas["tag"]
    tt = regr["tt"]
    ext = (tt[0] / 60.0, tt[-1] / 60.0, 0.0, N_CELL * CELL_LEN / 1000.0)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    panels = ((axes[0], regr["rho_tot"], regr["x_cav"], "model"),
              (axes[1], meas["rho"][0], meas["x_cav"][0], "SUMO (rep 0)"))
    for ax, z, xcv, title in panels:
        im = ax.imshow(np.asarray(z, float).T, origin="lower", aspect="auto",
                       extent=ext, vmin=0.0, vmax=90.0, cmap="turbo")
        ax.plot(tt / 60.0, np.asarray(xcv, float) / 1000.0, "w--", lw=1.2)
        ax.set_title(f"{tag.replace('_', ' ')} — {title}", fontsize=10)
        ax.set_xlabel("t [min]")
    axes[0].set_ylabel("x [km]")
    fig.colorbar(im, ax=list(axes), label=r"$\rho$ [veh/km]", shrink=0.92)
    fig.savefig(out_dir / f"{prefix}fig_heat_{tag}.png", dpi=160)
    plt.close(fig)


def _grid_axes(rows, title):
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.8), sharex=True)
    fig.suptitle(title, fontsize=11)
    return fig, axes


def fig_ns_grid(rows, out_dir):
    """2x4 grid: model N_s(t) vs measured n_caught mean +- envelope."""
    fig, axes = _grid_axes(rows, "E-V4: queue size — model $N_s$ vs "
                                 "measured caught count (5 reps)")
    for ax, row in zip(axes.flat, rows):
        meas, regr = row["meas"], row["regr"]
        tm = meas["tt"] / 60.0
        ax.axvspan(SLOW_LO / 60.0, SLOW_HI / 60.0, color="0.9", zorder=0)
        ax.fill_between(tm, meas["n_caught_min"], meas["n_caught_max"],
                        color="tab:blue", alpha=0.25, label="SUMO min–max")
        ax.plot(tm, meas["n_caught_mean"], color="tab:blue", lw=1.4,
                label="SUMO mean")
        ax.plot(tm, regr["N_s"], color="tab:red", lw=1.6, label="model $N_s$")
        ax.set_title(meas["tag"].replace("_", " "), fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("t [min]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$N_s$ [veh]")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_Ns_grid.png", dpi=160)
    plt.close(fig)


def fig_omega_grid(rows, out_dir):
    """2x4 grid: cumulative passings over the slow window (see OMEGA_NOTE:
    both series count total passings, so they are comparable)."""
    dt_h = T_SAMPLE / 3600.0
    fig, axes = _grid_axes(rows, "E-V4: cumulative passings, slow window — "
                                 r"model $\int\omega\,dt$ vs measured "
                                 "overtake count (all passings)")
    for ax, row in zip(axes.flat, rows):
        meas, regr = row["meas"], row["regr"]
        tt = meas["tt"]
        w = (tt >= SLOW_LO) & (tt <= SLOW_HI)
        tm = tt[w] / 60.0
        cum = np.cumsum(meas["overtake"][:, w], axis=1) * dt_h
        ax.fill_between(tm, cum.min(0), cum.max(0), color="tab:blue",
                        alpha=0.25, label="SUMO min–max")
        ax.plot(tm, cum.mean(0), color="tab:blue", lw=1.4, label="SUMO mean")
        ax.plot(tm, np.cumsum(regr["omega"][w]) * dt_h, color="tab:red",
                lw=1.6, label=r"model $\int\omega$")
        ax.set_title(meas["tag"].replace("_", " "), fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("t [min]")
    for ax in axes[:, 0]:
        ax.set_ylabel("cumulative passings [veh]")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_omega_grid.png", dpi=160)
    plt.close(fig)


def fig_es_bar(rows, out_dir):
    """e_s per scenario with the +-10% FTSM reference band."""
    tags = [r["meas"]["tag"].replace("_", " ") for r in rows]
    means = [r["met"]["e_s"]["mean"] for r in rows]
    xpos = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.axhspan(-0.10, 0.10, color="tab:green", alpha=0.15,
               label=r"$\pm$10% (FTSM reference)")
    ax.bar(xpos, means, width=0.6, color="tab:red", alpha=0.85,
           label=r"mean $e_s$")
    for i, r in enumerate(rows):
        pr = r["met"]["e_s"]["per_rep"]
        ax.plot(np.full(len(pr), xpos[i]), pr, "k.", ms=5, alpha=0.6)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"$e_s$ at $X_q$ = 15 km")
    ax.set_title("E-V4: relative outflow error (ECC22 eq. (6) analogue); "
                 "dots = reps")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_es_bar.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main runs
# ---------------------------------------------------------------------------

def run_all(form="lf", qxi=None):
    """Simulate the 8 core scenarios with the E-V3 kappas and compare.
    qxi [veh/h]: E-V4b bottleneck capacity cap (None = uncapped E-V4)."""
    from solver import SimConfig, simulate   # lazy: built in parallel

    kap = json.loads((OUT / "ev3_kappa.json").read_text())
    if qxi is None:
        out_dir = OUT / ("ev4" if form == "lf" else "ev4_af")
    else:
        out_dir = OUT / (f"ev4b_q{qxi:g}" + ("" if form == "lf" else "_af"))
    out_dir.mkdir(parents=True, exist_ok=True)
    kc_key = "kappa_cl_mod" if form == "lf" else "kappa_cA_mod"

    print(f"{'scenario':16s} {'kappa_c':>10s} {'kappa_r':>10s} "
          f"{'rho_RMSE':>9s} {'Ns_MAE':>8s} {'omega_err':>10s} {'e_s':>8s}"
          f"   (form={form}, kappa_c from {kc_key})")
    rows = []
    for A, uc, qin in CORE:
        tag = f"A{A:g}_u{uc:g}_q{qin:g}"
        kc = float(kap[tag][kc_key][0])
        kr = float(kap[tag]["kappa_r_mod"][0])   # may be exactly 0.0 -> pass 0.0
        cfg = SimConfig(v_f=V_F, w=W, P=P, q_in=qin / 3600.0, u_xi=uc,
                        kappa_c=kc, kappa_r=kr, capture_form=form,
                        dt=0.5, save_every=20,
                        q_xi_max=None if qxi is None else qxi / 3600.0)
        res = simulate(cfg)
        regr = regrid_sim(res)
        meas = load_measured(A, uc, qin)
        met = metrics(regr, meas)
        fig_heat(regr, meas, out_dir)
        rows.append(dict(meas=meas, regr=regr, met=met,
                         kappa_c=kc, kappa_r=kr))
        print(f"{tag:16s} {kc:10.3e} {kr:10.3e} "
              f"{met['rho_rmse']['mean']:9.2f} {met['Ns_mae']['mean']:8.2f} "
              f"{met['omega_cum_rel_err']['mean']:+10.1%} "
              f"{met['e_s']['mean']:+8.1%}")

    fig_ns_grid(rows, out_dir)
    fig_omega_grid(rows, out_dir)
    fig_es_bar(rows, out_dir)

    payload = {"_meta": dict(
        capture_form=form, kappa_c_key=kc_key, kappa_r_key="kappa_r_mod",
        kappa_source="out/ev3_kappa.json (model-Delta-v MLE point estimates)",
        params_si=dict(v_f_ms=V_F, w_ms=W, P_vehm=P),
        windows=dict(rho_rmse_t_s=list(RMSE_T), rho_rmse_x_km=[0.0, RMSE_X_KM],
                     slow_s=[SLOW_LO, SLOW_HI], es_t0_s=T0_ES, es_Xq_m=X_Q),
        omega_note=OMEGA_NOTE)}
    if qxi is not None:
        payload["_meta"]["q_xi_max_vehh"] = qxi
        payload["_meta"]["q_xi_max"] = qxi / 3600.0
        payload["_meta"]["omega_cap_note"] = (
            "sim omega diagnostic clipped at omega_max under the DM-G cap")
    for row in rows:
        payload[row["meas"]["tag"]] = dict(kappa_c=row["kappa_c"],
                                           kappa_r=row["kappa_r"],
                                           **row["met"])
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print("wrote", out_dir / "metrics.json", "and figures")
    return rows


# ---------------------------------------------------------------------------
# selftest (no solver.py needed)
# ---------------------------------------------------------------------------

def _mock_sim(u_xi=15.0, q_in=2000.0 / 3600.0):
    """SimResult-shaped mock: free inflow plateau advected at v_f, a queue
    block growing behind the CAV in the slow phase, CAV point mass smeared
    over 2 cells, omega = f (v_f - u_s) at the CAV."""
    from types import SimpleNamespace
    dx = CELL_LEN / 2.0
    nx = 2 * N_CELL
    x = (np.arange(nx) + 0.5) * dx
    t = np.arange(0.0, T_END + 1e-9, T_SAMPLE)
    n = len(t)
    a = np.zeros((n, nx))
    f = np.zeros((n, nx))
    s = np.zeros((n, nx))
    rho_free = q_in / V_F                                     # [veh/m]
    xc = x_cav_nominal(t, u_xi)
    us = u_s_of_t(t, u_xi)
    for i, ti in enumerate(t):
        f[i, x <= V_F * ti] = rho_free
        if np.isfinite(xc[i]):
            jc = min(int(xc[i] // dx), nx - 1)
            a[i, jc] += 0.5 / dx
            a[i, max(jc - 1, 0)] += 0.5 / dx
            if T_SLOW <= ti <= T_FAST:
                lo = max(jc - int(20.0 * (ti - T_SLOW) / dx), 0)
                s[i, lo:jc] = 0.1                              # 100 veh/km
                f[i, lo:jc] = 0.0
    omega = np.where(np.isfinite(xc),
                     rho_free * np.maximum(V_F - us, 0.0), 0.0)  # [veh/s]
    return SimpleNamespace(t=t, x=x, a=a, f=f, s=s, x_cav=xc,
                           omega=omega, N_s=s.sum(axis=1) * dx)


def selftest():
    """Exercise regridding + metrics + one heatmap end-to-end with the mock
    solver result and the real measured data of A1 u15 q2000."""
    res = _mock_sim(u_xi=15.0, q_in=2000.0 / 3600.0)
    regr = regrid_sim(res)
    assert regr["rho_tot"].shape == (N_T, N_CELL)
    assert regr["tt"][0] == T_SAMPLE and regr["tt"][-1] == T_END
    # pair-averaging must conserve mass exactly
    m_fine = float((res.a + res.f + res.s)[-1].sum()) * CELL_LEN / 2.0
    m_coarse = float(regr["rho_tot"][-1].sum()) / 1000.0 * CELL_LEN
    assert np.isclose(m_fine, m_coarse), (m_fine, m_coarse)

    meas = load_measured(1.0, 15.0, 2000.0)
    met = metrics(regr, meas)
    for key in ("rho_rmse", "Ns_mae", "omega_cum_rel_err", "e_s"):
        assert len(met[key]["per_rep"]) == len(meas["reps"]), key
        assert np.isfinite(met[key]["mean"]), key

    out_dir = OUT / "ev4"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_heat(regr, meas, out_dir, prefix="selftest_")
    print("selftest OK (mock sim vs measured A1_u15_q2000):")
    for key in ("rho_rmse", "Ns_mae", "omega_cum_rel_err", "e_s"):
        print(f"  {key:18s} mean = {met[key]['mean']:+.4f}  "
              f"per-rep = {[f'{v:+.3f}' for v in met[key]['per_rep']]}")
    print(f"  e_s window = {met['es_window_s']} s "
          f"(nominal CAV reaches X_q at t = {t_reach(X_Q, 15.0):.0f} s)")
    print("  wrote", out_dir / f"selftest_fig_heat_{meas['tag']}.png")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="E-V4 simulation-vs-data comparison harness")
    ap.add_argument("--form", choices=("lf", "af"), default="lf",
                    help="capture source form: lf = kappa_c l f (default), "
                         "af = strict-tex kappa_c a f (-> out/ev4_af/)")
    ap.add_argument("--selftest", action="store_true",
                    help="validate regrid/metrics/figure with a mock solver "
                         "result (no solver.py needed)")
    ap.add_argument("--qxi", type=float, default=None,
                    help="E-V4b bottleneck capacity cap q_xi_max [veh/h] "
                         "(-> out/ev4b_q{QXI}[_af]/; omitted = uncapped E-V4)")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
    else:
        run_all(args.form, args.qxi)


if __name__ == "__main__":
    main()
