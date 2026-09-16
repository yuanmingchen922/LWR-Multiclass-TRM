"""E13 (v2): joint calibration of the pure E7 catch & release model on the
density field AND the per-run event observables, integer assertiveness
A = 1..10.  See E13_plan.md (v2, after the two methodology reviews).

Nothing in the model changes with respect to E12 (same equations, same
parameters kappa_c, kappa_r [+ gamma, w_s], same purity guard -- every
catch & release run goes through e12_assertiveness.run_cr).  Only the
scoring rule changes:

    J_lam(theta) = W1/W1_cl(A)
                   + lam * [ w_N MAE_N/N_0(A) + w_O |eps_omega|/EPS_0 + w_R eps_R ]

    W1        cumulative-density Wasserstein distance vs the 5-run mean field,
              normalised by the classical LWR+MB value of the same A;
    MAE_N     queue-size MAE vs the classified stuck count, slow window
              [260, 740] s, N_0(A) = max(5 veh, rep-mean peak queue of A);
    eps_omega relative error of the cumulative overtaking count, EPS_0 = 0.2;
    eps_R     event-count error in dex (the magnitude-sensitive term, v2):
              |log10((R_m+1)/(R_d+1))| + |log10((C_m+1)/(C_d+1))| with
              R_m, C_m the model's gross releases / captures over the slow
              window (exact solver counters, solver.reaction_exact
              return_gross) and R_d, C_d the 5-run mean event counts of the
              E1/E4 classification ('rel', 'cap');
    lam       a weight (not a model parameter); lam = 0 reproduces E12;
    w_N, w_O, w_R  1 by default; the ablations set two of them to 0.

Why v2: with MAE_N and eps_omega alone (v1, out/e13_v1) the M_c - M_r gap
stayed < 0.4 % at every lam -- in the fast-equilibrium regime the queue and
the overtaking flow are functions of the ratio kappa_c/kappa_r only, like
the field.  The gross event counts are not: the E12 optima produce
10^3-10^5 captures per window against 14-80 in the data.

Parametrisation: ridge coordinates (r, m) = (log10 kappa_c/kappa_r,
log10 kappa_r); grid 13 x 11, Nelder-Mead in (r, m, extras) with
continuation in lam, multi-start (grid, previous lam, E12 optimum,
event-MLE), polish at dt = 0.5.

Stages
  p  reachable-set pre-check: (W1, MAE_N, eps_omega, eps_R) on the (ratio,
     magnitude) grid, slices gamma in {None, 0, 0.3}, w_s = 0.6 w, every A;
     pre-registered falsification rule (plan §3).   -> prehull13.json
  a  per (A, lam): C1..C4 joint fits, flags (field_ok, collapsed, classical
     J), winners, transfers.                          -> ladder13.json
  f  lam selection (dominance rule primary, restricted knee secondary,
     per-A knees), fields at lam*.                    -> lambda_star.json, final13.json
  b  nested models M_none / M_c / M_r / M_both for every lam (0.1-dex
     pre-scan + bounded Brent, parent fallback), jackknife over the five
     SUMO runs at lam = 0 and lam*, ablations (queue-only, overtaking-only,
     events-only, w_s = 0.6 w) at lam*, eta fractions, ridge residual,
     micro-macro agreement.                           -> hypothesis13.json
  e  identifiability scan under J_lam*, event-MLE kappas under J_lam*,
     turnover diagnostics, dx/dt sensitivity.         -> extras13.json

CLI:  python3 e13_joint.py --run [--resume]   stages p, a, f, b, e
      python3 e13_joint.py --stage p|a|f|b|bl|bs|e
      python3 e13_joint.py --figures | --summary | --smoke
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

import e12_assertiveness as e12
import e7_wasserstein as e7
import ev4_compare as ev4

HERE = Path(__file__).parent
OUT_E12 = HERE / "out" / "e12"

UC, QIN_FIT, QIN_TRANSFER = e12.UC, e12.QIN_FIT, e12.QIN_TRANSFER
SCENARIOS = e12.SCENARIOS
SCEN_FULL = e12.SCEN_FULL
KEY_FIT = e12.scen_key(UC, QIN_FIT)
KEY_TR = e12.scen_key(UC, QIN_TRANSFER)
CONFIGS = e12.CONFIGS
SLOW = (ev4.SLOW_LO, ev4.SLOW_HI)           # [260, 740] s
N_MIN = 5.0                                 # [veh] floor of N_0(A)
EPS_0 = 0.2                                 # overtaking-error scale
LAMBDAS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)
KAPPA_FLOOR = 1e-6
# pre-registered boxes / rules (plan v2 §2-3)
BOX_NS, BOX_OM, BOX_R = 5.0, 0.10, 0.5      # mechanism box: MAE_N <= 5 veh,
#                                              |eps_omega| <= 10 %, eps_R <= 0.5 dex
COLLAPSE_NS_MARGIN, COLLAPSE_W1 = 2.0, 1.10 # collapsed: MAE_N <= cl + 2, W1 > 1.1 W1_cl
DOM_FRAC, COLLAPSE_FRAC = 0.8, 0.5
IDENT_KR_MAX, IDENT_BASIN_DEX = 0.1, 1.0    # magnitude identified: kr < 0.1, basin < 1 dex
AGREE_DEX, AGREE_RHO = 0.5, 0.6             # micro-macro agreement criterion

SETTINGS = dict(
    out=HERE / "out" / "e13",
    A_levels=e12.A_LEVELS,
    lambdas=LAMBDAS,
    r_grid=np.linspace(0.0, 3.0, 13),        # log10 kappa_c/kappa_r
    m_grid=np.linspace(-4.0, 1.0, 11),       # log10 kappa_r
    maxfev=100, polish_maxfev=20,
    r_bounds=(-1.0, 4.0), m_bounds=(-5.0, 2.0),
    shared_kr=np.logspace(-4.0, 1.0, 9),
    shared_kc=np.logspace(-2.5, 2.0, 9),
    log_kc_bounds=(-2.5, 2.0),
    log_kr_bounds=(-4.0, 1.0),
    prescan_dex=0.1, brent_maxiter=15,
    none_maxfev=40, both_maxfev=60,
    robust_ws=0.6,
    hull_ratios=np.logspace(0.0, 3.0, 13), hull_mags=(1e-3, 1e-2, 1e-1, 1.0, 10.0),
    hull_slices=(("base", None, None), ("gamma0", 0.0, None),
                 ("gamma0.3", 0.3, None), ("ws0.6", None, 0.6)),
    ridge_ratios=np.logspace(0.0, 3.0, 13), ridge_mags=np.logspace(-4.0, 1.0, 11),
    dx_grid=((50.0, 0.5), (25.0, 0.5), (50.0, 0.25), (25.0, 0.25), (12.5, 0.25)),
    jackknife=True,
    n_proc=max(1, min(10, os.cpu_count() - 2)),
)


def lam_key(lam):
    return f"lam{lam:g}"


# ---------------------------------------------------------------------------
# data (per A, optional leave-one-run-out fold) + objective
# ---------------------------------------------------------------------------

_DATA = {}


def _npz_files(A):
    d = e12.OUT_E1 if float(A) in (1.0, 10.0) else e12.OUT_E4
    return sorted(glob.glob(str(d / f"A{A:g}_u{UC:g}_q{QIN_FIT:g}_r*.npz")))


def _data(A, fold=None):
    """(tt, rho_mean, meas, W1_cl, ref) of the fit scenario; fold = index of
    the run left out (jackknife) or None = all five runs.  ref carries the
    data event counts and scales: C_d, R_d, N_0, tau_data, queue_noise."""
    key = (float(A), fold)
    if key in _DATA:
        return _DATA[key]
    A = float(A)
    meas_all = e12.load_measured_any(A, UC, QIN_FIT)
    files = _npz_files(A)
    assert len(files) == len(meas_all["reps"]), (files, meas_all["reps"])
    cap = np.array([np.load(f)["cap"] for f in files], float)
    rel = np.array([np.load(f)["rel"] for f in files], float)
    tt = np.asarray(meas_all["tt"], float)
    keep = [i for i in range(len(meas_all["reps"])) if i != fold]
    meas = dict(meas_all)
    for k in ("n_caught", "overtake", "rho", "x_cav", "L_zone", "q_xq"):
        meas[k] = np.asarray(meas_all[k])[keep]
    meas["reps"] = [meas_all["reps"][i] for i in keep]
    rho_mean = np.mean(meas["rho"], axis=0)
    cl = e12.run_classical(UC, QIN_FIT)
    w1_cl = float(e7.w1_mean(cl["rho_tot"], rho_mean, tt))
    w = (tt >= SLOW[0]) & (tt <= SLOW[1])
    nc = np.asarray(meas["n_caught"], float)
    C_d = float(np.mean(cap[keep][:, w].sum(axis=1)))
    R_d = float(np.mean(rel[keep][:, w].sum(axis=1)))
    n_max = float(np.mean(nc[:, w].max(axis=1)))
    tau_d = float(nc[:, w].sum() * ev4.T_SAMPLE / max(rel[keep][:, w].sum(), 1e-9))
    ref = dict(C_d=C_d, R_d=R_d, N_max=n_max, N_0=max(N_MIN, n_max),
               tau_data_s=(tau_d if R_d > 0 else None),
               queue_noise_veh=float(np.mean(np.mean(np.abs(nc[:, w] - nc[:, w].mean(0)), 0))),
               n_runs=len(keep), fold=fold)
    _DATA[key] = (tt, rho_mean, meas, w1_cl, ref)
    return _DATA[key]


def preload(S, folds=(None,)):
    for A in S["A_levels"]:
        for fold in folds:
            _data(A, fold)


def window_counts(regr):
    """Gross captures / releases of a run over the slow window [veh] and the
    Little's-law turnover time (int N_s dt) / R [s]."""
    tt = np.asarray(regr["tt"], float)
    i0 = int(np.argmin(np.abs(tt - SLOW[0])))
    i1 = int(np.argmin(np.abs(tt - SLOW[1])))
    C = float(regr["cum_cap"][i1] - regr["cum_cap"][i0])
    R = float(regr["cum_rel"][i1] - regr["cum_rel"][i0])
    w = (tt >= SLOW[0]) & (tt <= SLOW[1])
    n_int = float(np.sum(regr["N_s"][w]) * ev4.T_SAMPLE)       # veh s
    return C, R, (n_int / R if R > 0 else np.inf)


def parts_of(regr, A, fold=None):
    tt, rho_mean, meas, w1_cl, ref = _data(A, fold)
    met = ev4.metrics(regr, meas)
    w1 = float(e7.w1_mean(regr["rho_tot"], rho_mean, tt))
    om = met["omega_cum_rel_err"]["mean"]
    om = 0.0 if om is None or not np.isfinite(om) else float(om)
    C, R, tau = window_counts(regr)
    eps_R = abs(np.log10((R + 1.0) / (ref["R_d"] + 1.0))) \
        + abs(np.log10((C + 1.0) / (ref["C_d"] + 1.0)))
    return dict(W1=w1, W1_rel=w1 / w1_cl, Ns_mae=float(met["Ns_mae"]["mean"]),
                omega_err=om, C_m=C, R_m=R, eps_R=float(eps_R),
                tau_model_s=(float(tau) if np.isfinite(tau) else None),
                N_0=ref["N_0"])


def penalty(parts, wN=1.0, wO=1.0, wR=1.0):
    return float(wN * parts["Ns_mae"] / parts["N_0"] + wO * abs(parts["omega_err"]) / EPS_0
                 + wR * parts["eps_R"])


def J_of(parts, lam, wN=1.0, wO=1.0, wR=1.0):
    return float(parts["W1_rel"] + lam * penalty(parts, wN, wO, wR))


def parts_one(A, kc, kr, ws=None, gamma=None, dt=1.0, fold=None):
    return parts_of(e12.run_cr(UC, QIN_FIT, kc, kr, gamma, ws, dt), A, fold)


def j_one(A, kc, kr, lam, ws=None, gamma=None, wN=1.0, wO=1.0, wR=1.0, dt=1.0,
          fold=None):
    return J_of(parts_one(A, kc, kr, ws, gamma, dt, fold), lam, wN, wO, wR)


def classical_parts(A, fold=None, dt=e7.DT_PRODUCTION):
    return parts_of(e12.run_classical(UC, QIN_FIT, dt), A, fold)


def rm_to_kappa(r, m):
    return 10.0 ** (r + m), 10.0 ** m


def kappa_to_rm(kc, kr):
    return float(np.log10(kc / kr)), float(np.log10(kr))


# ---------------------------------------------------------------------------
# stage P: reachable-set pre-check
# ---------------------------------------------------------------------------

def _hull_task(args):
    A, kc, kr, gamma, ws, slice_name = args
    p = parts_one(A, kc, kr, ws, gamma)
    return dict(A=A, kappa_c=kc, kappa_r=kr, gamma=gamma, ws_frac=ws,
                slice=slice_name, **p)


_HULL_KEYS = ("kappa_c", "kappa_r", "gamma", "ws_frac", "slice", "W1", "W1_rel",
              "Ns_mae", "omega_err", "eps_R", "tau_model_s")


def stage_p(S):
    out = Path(S["out"])
    out.mkdir(parents=True, exist_ok=True)
    preload(S)
    tasks = [(float(A), float(r * m), float(m), g, ws, name)
             for A in S["A_levels"] for name, g, ws in S["hull_slices"]
             for m in S["hull_mags"] for r in S["hull_ratios"]]
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=S["n_proc"]) as pool:
        res = pool.map(_hull_task, tasks, chunksize=8)
    pre = {"_meta": dict(ratios=[float(r) for r in S["hull_ratios"]],
                         magnitudes=[float(m) for m in S["hull_mags"]],
                         slices=[s[0] for s in S["hull_slices"]], dt=1.0,
                         box=dict(Ns_mae_max=BOX_NS, abs_omega_max=BOX_OM, eps_R_max=BOX_R),
                         rule="pure model can reproduce the mechanism at A iff some grid "
                              "point satisfies the box; with field_ok (W1 <= W1_cl) and "
                              "with eps_R <= 0.5 reported separately",
                         runtime_s=None)}
    for A in S["A_levels"]:
        k = f"A{A:g}"
        rows = [r for r in res if r["A"] == float(A)]
        cl = classical_parts(A, dt=1.0)
        inbox = [r for r in rows if r["Ns_mae"] <= BOX_NS and abs(r["omega_err"]) <= BOX_OM]
        inbox_f = [r for r in inbox if r["W1_rel"] <= 1.0]
        inbox_fr = [r for r in inbox_f if r["eps_R"] <= BOX_R]
        best_pen = min(rows, key=lambda r: penalty(r))
        best_pen_f = min([r for r in rows if r["W1_rel"] <= 1.0] or rows, key=lambda r: penalty(r))
        pre[k] = dict(points=[{kk: r[kk] for kk in _HULL_KEYS} for r in rows],
                      classical={kk: cl[kk] for kk in ("W1", "W1_rel", "Ns_mae", "omega_err", "eps_R")},
                      n_points=len(rows), n_in_box=len(inbox), n_in_box_field_ok=len(inbox_f),
                      n_in_box_field_ok_events_ok=len(inbox_fr),
                      mechanism_reachable=bool(inbox), mechanism_reachable_field_ok=bool(inbox_f),
                      min_Ns_mae=float(min(r["Ns_mae"] for r in rows)),
                      min_abs_omega=float(min(abs(r["omega_err"]) for r in rows)),
                      min_eps_R=float(min(r["eps_R"] for r in rows)),
                      best_penalty_point={kk: best_pen[kk] for kk in _HULL_KEYS},
                      best_penalty_field_ok={kk: best_pen_f[kk] for kk in _HULL_KEYS})
        print(f"[stage p] A={A:g}: in box {len(inbox)} / field_ok {len(inbox_f)} / +events "
              f"{len(inbox_fr)} of {len(rows)}; min MAE_N {pre[k]['min_Ns_mae']:.1f} min|eps| "
              f"{pre[k]['min_abs_omega']:.2f} min eps_R {pre[k]['min_eps_R']:.2f}", flush=True)
    pre["_meta"]["runtime_s"] = round(time.time() - t0, 1)
    (out / "prehull13.json").write_text(json.dumps(pre, indent=1, default=e12._json))
    print(f"[stage p] wrote {out / 'prehull13.json'}", flush=True)
    return pre


# ---------------------------------------------------------------------------
# stage A: per-(A, lam) joint fits in ridge coordinates
# ---------------------------------------------------------------------------

_EXTRA_BOUNDS = {"gamma": (0.0, 1.0), "ws_frac": (0.5, 1.0)}


def _starts_for(A, S):
    starts = {}
    p = OUT_E12 / "ladder.json"
    if p.exists():
        L = json.loads(p.read_text())
        c1 = L.get(f"A{A:g}", {}).get("fits", {}).get("C1")
        if c1:
            starts["e12_C1"] = (c1["kappa_c"], c1["kappa_r"])
    ev = e12.event_ladders(S)[KEY_FIT].get(f"A{A:g}")
    if ev:
        starts["event_mle"] = (ev["kappa_c"][0], max(ev["kappa_r"][0], KAPPA_FLOOR))
    return starts


def _unpack(z, extra_names, S):
    r = float(np.clip(z[0], *S["r_bounds"]))
    m = float(np.clip(z[1], *S["m_bounds"]))
    kc, kr = rm_to_kappa(r, m)
    ex = {nm: float(np.clip(z[2 + i], *_EXTRA_BOUNDS[nm])) for i, nm in enumerate(extra_names)}
    return kc, kr, ex


def polish(A, lam, extra_names, x0, S, dt=1.0, maxfev=None, wN=1.0, wO=1.0, wR=1.0):
    """Nelder-Mead on z = [r, m, *extras] for J_lam.  Returns
    ((kc, kr, extras), J, n_eval, parts) of the best point seen."""
    from scipy.optimize import minimize
    n = [0]
    best = [np.inf, None, None]

    def f(z):
        n[0] += 1
        kc, kr, ex = _unpack(z, extra_names, S)
        parts = parts_one(A, kc, kr, ex.get("ws_frac"), ex.get("gamma"), dt)
        j = J_of(parts, lam, wN, wO, wR)
        if j < best[0]:
            best[:] = [j, (kc, kr, ex), parts]
        return j

    minimize(f, np.asarray(x0, float), method="Nelder-Mead",
             options=dict(maxfev=maxfev or S["maxfev"], xatol=1e-3, fatol=1e-6))
    return best[1], best[0], n[0], best[2]


def fit_joint(A, lam, cfg_name, extra, grid_parts, S, starts, prev=None):
    """Grid (precomputed parts) + NM polish per start (dt = 1) + dt = 0.5
    polish; the best point of every stage is kept."""
    extra_names = sorted(extra or {})
    mid = {nm: 0.5 * sum(_EXTRA_BOUNDS[nm]) for nm in extra_names}
    j_grid = [J_of(g["parts"], lam) for g in grid_parts]
    ig = int(np.argmin(j_grid))
    g0 = grid_parts[ig]
    cands = {"grid": (g0["kappa_c"], g0["kappa_r"], dict(mid))}
    if prev is not None:
        cands["continuation"] = (prev["kappa_c"], prev["kappa_r"],
                                 {nm: prev.get(nm) for nm in extra_names})
    if cfg_name == "C1":
        for name, (kc, kr) in starts.items():
            cands[name] = (kc, kr, dict(mid))
    best = (j_grid[ig], (g0["kappa_c"], g0["kappa_r"], dict(mid)), g0["parts"], "grid_no_polish")
    n_sim = 0
    for name, (kc0, kr0, ex0) in cands.items():
        ex0 = {nm: (mid[nm] if ex0.get(nm) is None else ex0[nm]) for nm in extra_names}
        x0 = list(kappa_to_rm(kc0, kr0)) + [ex0[nm] for nm in extra_names]
        th, j, n, parts = polish(A, lam, extra_names, x0, S)
        n_sim += n
        if j < best[0]:
            best = (j, th, parts, name)
    j1, (kc, kr, ex), parts1, used = best
    x1 = list(kappa_to_rm(kc, kr)) + [ex[nm] for nm in extra_names]
    th5, j5, n5, parts5 = polish(A, lam, extra_names, x1, S, dt=e7.DT_PRODUCTION,
                                 maxfev=S["polish_maxfev"])
    kc, kr, ex = th5
    return dict(kappa_c=float(kc), kappa_r=float(kr), gamma=ex.get("gamma"),
                ws_frac=ex.get("ws_frac"), J_dt1=float(j1), parts_dt1=parts1,
                J_dt05=float(j5), parts_dt05=parts5, grid_J_min=float(j_grid[ig]),
                start_used=used, n_sim=n_sim + n5)


def _grid_task(args):
    A, kc, kr, gamma, ws = args
    return dict(kappa_c=float(kc), kappa_r=float(kr), parts=parts_one(A, kc, kr, ws, gamma))


def stage_a_one(A, S):
    t0 = time.time()
    out = Path(S["out"])
    A = float(A)
    tt, rho_mean, meas, w1_cl, ref = _data(A)
    res = dict(A=A, W1_cl=w1_cl, ref=ref, classical={}, data={}, lambdas={},
               structure="pure E7 (no cap, no DR, no leader-loss, no impermeable "
                         "interface); joint objective v2")
    for uc, qin in SCENARIOS:
        key = e12.scen_key(uc, qin)
        summ, regr = e12.evaluate(A, uc, qin, "classical")
        if key == KEY_FIT:
            summ.update(parts_of(regr, A))
        res["classical"][key] = summ
        res["data"][key] = e12.data_diagnostics(A, uc, qin)
    cl = res["classical"][KEY_FIT]
    P_cl = penalty(cl)
    grids = {}
    for name, extra in CONFIGS:
        mid = {nm: 0.5 * sum(_EXTRA_BOUNDS[nm]) for nm in sorted(extra or {})}
        grids[name] = [_grid_task((A, *rm_to_kappa(r, m), mid.get("gamma"), mid.get("ws_frac")))
                       for r in S["r_grid"] for m in S["m_grid"]]
    starts = _starts_for(A, S)
    res["starts"] = {k: dict(kappa_c=v[0], kappa_r=v[1]) for k, v in starts.items()}
    prev = {name: None for name, _ in CONFIGS}
    for lam in S["lambdas"]:
        lk = lam_key(lam)
        block = dict(lam=lam, fits={}, transfer={}, J_classical=1.0 + lam * P_cl)
        for name, extra in CONFIGS:
            fit = fit_joint(A, lam, name, extra, grids[name], S, starts, prev[name])
            prev[name] = fit
            block["fits"][name] = fit
        winner = min(block["fits"], key=lambda n: block["fits"][n]["J_dt05"])
        block["winner"] = winner
        w = block["fits"][winner]
        summ, regr = e12.evaluate(A, UC, QIN_FIT, "cr", w["kappa_c"], w["kappa_r"],
                                  w["gamma"], w["ws_frac"])
        summ.update(parts_of(regr, A))
        w["eval"] = {KEY_FIT: summ}
        p = w["parts_dt05"]
        block["flags"] = dict(
            field_ok=bool(p["W1_rel"] <= 1.0),
            collapsed=bool(p["Ns_mae"] <= cl["Ns_mae"] + COLLAPSE_NS_MARGIN
                           and p["W1_rel"] > COLLAPSE_W1),
            in_box=bool(p["Ns_mae"] <= BOX_NS and abs(p["omega_err"]) <= BOX_OM),
            events_ok=bool(p["eps_R"] <= BOX_R),
            beats_classical_J=bool(w["J_dt05"] < block["J_classical"]))
        for uc, qin in SCENARIOS[1:]:
            key = e12.scen_key(uc, qin)
            summ, regr = e12.evaluate(A, uc, qin, "cr", w["kappa_c"], w["kappa_r"],
                                      w["gamma"], w["ws_frac"])
            if uc == UC:
                C, R, tau = window_counts(regr)
                summ["counts"] = dict(C_m=C, R_m=R,
                                      tau_model_s=(float(tau) if np.isfinite(tau) else None))
            block["transfer"][key] = summ
        res["lambdas"][lk] = block
        print(f"  A={A:g} {lk}: {winner} kc={w['kappa_c']:.3e} kr={w['kappa_r']:.3e} "
              f"g={w['gamma']} ws={w['ws_frac']} W1={p['W1']:.1f} Ns={p['Ns_mae']:.1f} "
              f"om={p['omega_err']:+.2f} epsR={p['eps_R']:.2f} tau={p['tau_model_s']} "
              f"J={w['J_dt05']:.3f} (cl {block['J_classical']:.3f}) [{w['start_used']}]",
              flush=True)
    lams = list(S["lambdas"])
    W = [_winner(res, None, l, direct=True)["parts_dt05"]["W1"] for l in lams]
    P = [penalty(_winner(res, None, l, direct=True)["parts_dt05"]) for l in lams]
    res["path_check"] = dict(W1_nondecreasing_violations=int(np.sum(np.diff(W) < -1e-6)),
                             penalty_nonincreasing_violations=int(np.sum(np.diff(P) > 1e-6)))
    res["runtime_s"] = round(time.time() - t0, 1)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"ladder13_A{A:g}.json").write_text(json.dumps(res, indent=1, default=e12._json))
    print(f"A={A:g} done ({res['runtime_s']} s) path check {res['path_check']}", flush=True)
    return res


def _winner(ladder, k, lam, direct=False):
    """(winner fit, block) at lam; direct=True when `ladder` is one A's block."""
    blk = (ladder if direct else ladder[k])["lambdas"][lam_key(lam)]
    w = blk["fits"][blk["winner"]]
    return w if direct else (w, blk)


def _stage_a_worker(args):
    return stage_a_one(*args)


def meta(S):
    S12 = dict(S, kc_grid=[float(10.0 ** (r + m)) for r in S["r_grid"] for m in S["m_grid"]],
               kr_grid=[float(10.0 ** m) for m in S["m_grid"]])
    m12 = e12.meta(S12)
    return dict(objective="J = W1/W1_cl + lam*(w_N*MAE_N/N_0(A) + w_O*|eps_omega|/EPS_0 + w_R*eps_R)",
                N_MIN=N_MIN, EPS_0=EPS_0,
                eps_R="|log10((R_m+1)/(R_d+1))| + |log10((C_m+1)/(C_d+1))| [dex], slow window",
                lambdas=list(S["lambdas"]), structure=m12["structure"],
                forbidden_knobs=m12["forbidden_knobs"], classical=m12["classical"],
                A_levels=list(S["A_levels"]),
                r_grid=np.asarray(S["r_grid"]).tolist(), m_grid=np.asarray(S["m_grid"]).tolist(),
                maxfev=S["maxfev"], polish_maxfev=S["polish_maxfev"], dt_fit=1.0,
                dt_production=e7.DT_PRODUCTION,
                windows=dict(W1=list(e7.T_WIN), events=list(SLOW)),
                winner_rule="min J at dt = 0.5 over C1..C4 (after a dt = 0.5 polish)",
                flags=dict(field_ok="W1 <= W1_cl",
                           collapsed=f"MAE_N <= MAE_N_cl + {COLLAPSE_NS_MARGIN} veh and W1 > {COLLAPSE_W1} W1_cl",
                           in_box=f"MAE_N <= {BOX_NS} veh and |eps_omega| <= {BOX_OM}",
                           events_ok=f"eps_R <= {BOX_R} dex"),
                configs=m12["configs"])


def stage_a(S, resume=False):
    out = Path(S["out"])
    out.mkdir(parents=True, exist_ok=True)
    preload(S)
    todo = [A for A in S["A_levels"] if not (resume and (out / f"ladder13_A{A:g}.json").exists())]
    print(f"[stage a] A = {todo} on {S['n_proc']} processes", flush=True)
    if todo:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(S["n_proc"], len(todo))) as pool:
            pool.map(_stage_a_worker, [(A, S) for A in todo], chunksize=1)
    merged = {"_meta": meta(S)}
    for A in S["A_levels"]:
        merged[f"A{A:g}"] = json.loads((out / f"ladder13_A{A:g}.json").read_text())
    (out / "ladder13.json").write_text(json.dumps(merged, indent=1))
    print(f"[stage a] wrote {out / 'ladder13.json'}", flush=True)
    return merged


# ---------------------------------------------------------------------------
# stage F: lam selection + fields
# ---------------------------------------------------------------------------

def knee(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return 0, np.zeros(len(x))
    xn = (x - x.min()) / (x.max() - x.min() if x.max() > x.min() else 1.0)
    yn = (y - y.min()) / (y.max() - y.min() if y.max() > y.min() else 1.0)
    p0, p1 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    d = p1 - p0
    nrm = np.linalg.norm(d)
    dist = np.array([abs(d[0] * (yn[i] - p0[1]) - d[1] * (xn[i] - p0[0])) / nrm if nrm > 0 else 0.0
                     for i in range(len(x))])
    inner = np.arange(1, len(x) - 1)
    return int(inner[np.argmax(dist[inner])]), dist


def sweep_table(S, ladder):
    keys = [f"A{A:g}" for A in S["A_levels"]]
    rows = []
    for lam in S["lambdas"]:
        r = dict(lam=lam, sum_W1_rel=0.0, sum_penalty=0.0, sum_J=0.0, sum_J_classical=0.0,
                 n_field_ok=0, n_collapsed=0, n_in_box=0, n_events_ok=0, n_beats_classical=0)
        W1, Ns, om, eR, taus = [], [], [], [], []
        for k in keys:
            w, blk = _winner(ladder, k, lam)
            p = w["parts_dt05"]
            r["sum_W1_rel"] += p["W1_rel"]
            r["sum_penalty"] += penalty(p)
            r["sum_J"] += w["J_dt05"]
            r["sum_J_classical"] += blk["J_classical"]
            for f in ("field_ok", "collapsed", "in_box", "events_ok"):
                r["n_" + f] += int(blk["flags"][f])
            r["n_beats_classical"] += int(blk["flags"]["beats_classical_J"])
            W1.append(p["W1"]); Ns.append(p["Ns_mae"]); om.append(abs(p["omega_err"]))
            eR.append(p["eps_R"]); taus.append(p["tau_model_s"])
        r.update(mean_W1=float(np.mean(W1)), mean_Ns_mae=float(np.mean(Ns)),
                 mean_abs_omega=float(np.mean(om)), mean_eps_R=float(np.mean(eR)))
        taus = [t for t in taus if t is not None]
        r["median_tau_s"] = float(np.median(taus)) if taus else None
        rows.append(r)
    return rows


def stage_f(S):
    out = Path(S["out"])
    ladder = json.loads((out / "ladder13.json").read_text())
    keys = [f"A{A:g}" for A in S["A_levels"]]
    nA = len(keys)
    rows = sweep_table(S, ladder)
    lams = [r["lam"] for r in rows]
    lam_max = next((r["lam"] for r in rows if r["n_collapsed"] >= COLLAPSE_FRAC * nA), None)
    sub = [r for r in rows if lam_max is None or r["lam"] < lam_max]
    i_knee, dist = knee([r["sum_penalty"] for r in sub], [r["sum_W1_rel"] for r in sub])
    lam_knee = sub[i_knee]["lam"] if len(sub) >= 3 else None

    def closed(lm):
        r0 = rows[0]
        r1 = next(r for r in rows if r["lam"] == lm)
        dP = r0["sum_penalty"] - r1["sum_penalty"]
        return (r1["sum_W1_rel"] - r0["sum_W1_rel"]) / dP if dP > 0 else None

    closed_form = {f"lam_max={lm:g}": closed(lm) for lm in (0.2, 0.5, 1.0) if lm in lams}
    cf_vals = [v for v in closed_form.values() if v]
    closed_agree = bool(cf_vals and max(cf_vals) / min(cf_vals) <= 1.5)
    dom = [r["lam"] for r in rows if r["n_field_ok"] >= DOM_FRAC * nA]
    lam_dom = max(dom) if dom else None
    lm_ref = lam_max if lam_max is not None else lams[-1]
    per_A = {}
    for k in keys:
        w0, _ = _winner(ladder, k, 0.0)
        w1, _ = _winner(ladder, k, lm_ref)
        dP = penalty(w0["parts_dt05"]) - penalty(w1["parts_dt05"])
        dW = w1["parts_dt05"]["W1_rel"] - w0["parts_dt05"]["W1_rel"]
        per_A[k] = (dW / dP if dP > 0 else None)
    pa = [v for v in per_A.values() if v and v > 0]
    spread = (max(pa) / min(pa)) if pa else None
    lam_star = lam_dom if lam_dom is not None else (lam_knee if lam_knee is not None else 0.0)
    lambdas_for_B = sorted({0.0, lam_star} | ({lam_knee} if lam_knee is not None else set()))
    sel = dict(lambda_star=lam_star, rule=("dominance" if lam_dom is not None else "knee"),
               lambda_dominance=lam_dom, lambda_knee_restricted=lam_knee, lambda_max=lam_max,
               closed_form_knee=closed_form, closed_form_agree_x1p5=closed_agree,
               per_A_exchange_rate=per_A, per_A_spread=spread,
               two_lambda_ladder=bool(spread is not None and spread > 3.0),
               lambdas_for_B=lambdas_for_B,
               rules=dict(dominance=f"largest lam with >= {DOM_FRAC:.0%} of A field_ok",
                          knee="chord-distance knee restricted to lam < lambda_max",
                          lambda_max=f"smallest lam with >= {COLLAPSE_FRAC:.0%} of A collapsed"),
               sweep=[dict(r, knee_distance=(float(dist[sub.index(r)]) if r in sub else None))
                      for r in rows])
    (out / "lambda_star.json").write_text(json.dumps(sel, indent=1, default=e12._json))
    print(f"[stage f] lambda* = {lam_star} ({sel['rule']}); dominance {lam_dom}, restricted knee "
          f"{lam_knee}, lambda_max {lam_max}, per-A spread {spread}", flush=True)
    final = {"_meta": dict(meta(S), lambda_star=lam_star, source="stage f")}
    for A in S["A_levels"]:
        k = f"A{A:g}"
        w, blk = _winner(ladder, k, lam_star)
        fields = {}
        evals = dict(w["eval"])
        evals.update(blk["transfer"])
        for uc, qin in SCEN_FULL:
            key = e12.scen_key(uc, qin)
            tt, rho_mean = e7.load_rho_mean(A, uc, qin)
            meas = e12.load_measured_any(A, uc, qin)
            fields["tt"] = tt
            fields[f"rho_mean_{key}"] = rho_mean
            fields[f"xcav_data_{key}"] = e12.cav_data_mean(meas)
            cl = e12.run_classical(uc, qin)
            fields[f"classical_rho_{key}"] = cl["rho_tot"]
            fields[f"classical_xcav_{key}"] = cl["x_cav"]
            regr = e12.run_cr(uc, qin, w["kappa_c"], w["kappa_r"], w["gamma"], w["ws_frac"],
                              e7.DT_PRODUCTION)
            fields[f"winner_rho_{key}"] = regr["rho_tot"]
            fields[f"winner_xcav_{key}"] = regr["x_cav"]
            fields[f"winner_s_{key}"] = regr["s"]
            fields[f"winner_f_{key}"] = regr["f"]
        np.savez_compressed(out / f"fields13_A{A:g}.npz", **fields)
        final[k] = dict(A=float(A), winner=blk["winner"],
                        fits={blk["winner"]: dict(kappa_c=w["kappa_c"], kappa_r=w["kappa_r"],
                                                  gamma=w["gamma"], ws_frac=w["ws_frac"],
                                                  J_dt05=w["J_dt05"], eval=evals)},
                        classical=ladder[k]["classical"], data=ladder[k]["data"],
                        W1_cl=ladder[k]["W1_cl"], flags=blk["flags"])
    (out / "final13.json").write_text(json.dumps(final, indent=1, default=e12._json))
    print(f"[stage f] wrote {out / 'final13.json'} + fields13_A*.npz", flush=True)
    return sel


def selection(S):
    p = Path(S["out"]) / "lambda_star.json"
    if not p.exists():
        raise FileNotFoundError("run --stage f first (lambda_star.json)")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# stage B: nested shared-parameter models under J_lam
# ---------------------------------------------------------------------------

def _j_task(args):
    A, kc, kr, ws, lam, wN, wO, wR, dt, fold = args
    return j_one(A, kc, kr, lam, ws, None, wN, wO, wR, dt, fold)


def _scalar_task(args):
    """Per-A 1-D minimisation: pre-scan over the bounds (prescan_dex),
    bounded Brent in the best bracket, parent-optimum fallback."""
    from scipy.optimize import minimize_scalar
    A, free, fixed, bounds, ws, S, ref, lam, wN, wO, wR, fold = args
    n = [0]

    def f(z):
        n[0] += 1
        kc, kr = (10.0 ** z, fixed) if free == "kc" else (fixed, 10.0 ** z)
        return j_one(A, kc, kr, lam, ws, None, wN, wO, wR, 1.0, fold)

    zs = np.arange(bounds[0], bounds[1] + 1e-9, S["prescan_dex"])
    js = np.array([f(z) for z in zs])
    i = int(np.argmin(js))
    lo, hi = zs[max(i - 1, 0)], zs[min(i + 1, len(zs) - 1)]
    z_best, j_best = float(zs[i]), float(js[i])
    if hi > lo:
        r = minimize_scalar(f, bounds=(lo, hi), method="bounded",
                            options=dict(xatol=1e-3, maxiter=S["brent_maxiter"]))
        if r.fun < j_best:
            z_best, j_best = float(r.x), float(r.fun)
    if ref is not None:
        j_ref = f(float(ref))
        if j_ref < j_best:
            z_best, j_best = float(ref), j_ref
    return dict(A=A, value=10.0 ** z_best, J=j_best, n_eval=n[0])


def _shared_scan(pool, free, shared_vals, ws, S, ref, lam, wN, wO, wR, fold):
    fixed_name = "kr" if free == "kc" else "kc"
    bounds = S["log_kc_bounds"] if free == "kc" else S["log_kr_bounds"]
    tasks = [(A, free, float(v), bounds, ws, S, ref, lam, wN, wO, wR, fold)
             for v in shared_vals for A in S["A_levels"]]
    res = pool.map(_scalar_task, tasks, chunksize=1)
    out = []
    nA = len(S["A_levels"])
    for k, v in enumerate(shared_vals):
        rows = res[k * nA:(k + 1) * nA]
        out.append(dict(shared={fixed_name: float(v)},
                        per_A={f"A{r['A']:g}": dict(value=r["value"], J=r["J"]) for r in rows},
                        total=float(sum(r["J"] for r in rows))))
    return sorted(out, key=lambda d: d["total"])


def fit_shared(pool, free, S, lam, ws=None, parent=None, wN=1.0, wO=1.0, wR=1.0, fold=None):
    fixed_name = "kr" if free == "kc" else "kc"
    vals = list(S["shared_kr"] if free == "kc" else S["shared_kc"])
    ref = None
    if parent is not None:
        vals.append(parent[fixed_name])
        ref = float(np.log10(parent["kc" if free == "kc" else "kr"]))
    scan = _shared_scan(pool, free, vals, ws, S, ref, lam, wN, wO, wR, fold)
    v0 = scan[0]["shared"][fixed_name]
    refine = _shared_scan(pool, free, [v0 * 10 ** -0.2, v0 * 10 ** 0.2], ws, S, ref,
                          lam, wN, wO, wR, fold)
    win = sorted([scan[0]] + refine, key=lambda d: d["total"])[0]
    ladder = {a: d["value"] for a, d in win["per_A"].items()}
    per_A = {a: d["J"] for a, d in win["per_A"].items()}
    vals_l = [ladder[f"A{A:g}"] for A in S["A_levels"]]
    mono = (all(np.diff(vals_l) <= 1e-12) if free == "kc" else all(np.diff(vals_l) >= -1e-12))
    return dict(free=("kappa_c" if free == "kc" else "kappa_r"), shared=win["shared"],
                ws_frac=ws, lam=lam, wN=wN, wO=wO, wR=wR, fold=fold,
                ladder=ladder, per_A_J_dt1=per_A, total_dt1=win["total"],
                monotone=("non-increasing" if free == "kc" else "non-decreasing"),
                monotone_ok=bool(mono), spearman=e12.spearman(list(S["A_levels"]), vals_l),
                scan=[dict(shared=d["shared"], total=d["total"])
                      for d in sorted(scan + refine, key=lambda d: d["total"])],
                n_params=len(S["A_levels"]) + 1)


def fit_none(pool, S, lam, ws=None, wN=1.0, wO=1.0, wR=1.0, fold=None):
    from scipy.optimize import minimize
    A_levels = list(S["A_levels"])

    def total(kc, kr):
        return float(sum(pool.map(_j_task, [(A, kc, kr, ws, lam, wN, wO, wR, 1.0, fold)
                                            for A in A_levels])))

    grid = {}
    for r in S["r_grid"]:
        for m in S["m_grid"]:
            kc, kr = rm_to_kappa(r, m)
            grid[(float(kc), float(kr))] = total(kc, kr)
    (kc0, kr0), j0 = min(grid.items(), key=lambda kv: kv[1])
    r = minimize(lambda z: total(*rm_to_kappa(z[0], z[1])), list(kappa_to_rm(kc0, kr0)),
                 method="Nelder-Mead", options=dict(maxfev=S["none_maxfev"], xatol=1e-3, fatol=1e-6))
    kc, kr = rm_to_kappa(r.x[0], r.x[1])
    j = float(r.fun)
    if j0 < j:
        kc, kr, j = kc0, kr0, j0
    per_A = {f"A{A:g}": j_one(A, kc, kr, lam, ws, None, wN, wO, wR, 1.0, fold) for A in A_levels}
    return dict(shared=dict(kc=float(kc), kr=float(kr)), ws_frac=ws, lam=lam, wN=wN, wO=wO,
                wR=wR, fold=fold, per_A_J_dt1=per_A, total_dt1=float(sum(per_A.values())),
                grid_total_min=j0, n_params=2)


def _both_task(args):
    from scipy.optimize import minimize
    A, starts, S, lam, ws, wN, wO, wR, fold = args
    cands = [(j_one(A, kc, kr, lam, ws, None, wN, wO, wR, 1.0, fold), float(kc), float(kr))
             for kc, kr in starts]
    j0, kc0, kr0 = min(cands)
    r = minimize(lambda z: j_one(A, *rm_to_kappa(z[0], z[1]), lam, ws, None, wN, wO, wR, 1.0, fold),
                 list(kappa_to_rm(kc0, kr0)), method="Nelder-Mead",
                 options=dict(maxfev=S["both_maxfev"], xatol=1e-3, fatol=1e-6))
    kc, kr = rm_to_kappa(r.x[0], r.x[1])
    j = float(r.fun)
    if j0 <= j:
        kc, kr, j = kc0, kr0, j0
    p = parts_one(A, kc, kr, ws, None, 1.0, fold)
    return dict(A=A, kappa_c=float(kc), kappa_r=float(kr), J=j, parts=p,
                start=dict(kappa_c=kc0, kappa_r=kr0, J=j0), n_eval=int(r.nfev))


def fit_both(pool, S, lam, c1, m_none, m_c, m_r, ws=None, wN=1.0, wO=1.0, wR=1.0, fold=None):
    keys = [f"A{A:g}" for A in S["A_levels"]]
    tasks = []
    for A, k in zip(S["A_levels"], keys):
        starts = [(c1[k]["kappa_c"], c1[k]["kappa_r"]),
                  (m_c["ladder"][k], m_c["shared"]["kr"]),
                  (m_r["shared"]["kc"], m_r["ladder"][k]),
                  (m_none["shared"]["kc"], m_none["shared"]["kr"])]
        tasks.append((float(A), starts, S, lam, ws, wN, wO, wR, fold))
    res = pool.map(_both_task, tasks, chunksize=1)
    kcs = [r["kappa_c"] for r in res]
    krs = [r["kappa_r"] for r in res]
    return dict(lam=lam, ws_frac=ws, wN=wN, wO=wO, wR=wR, fold=fold,
                ladder_kc=dict(zip(keys, kcs)), ladder_kr=dict(zip(keys, krs)),
                per_A_J_dt1={k: r["J"] for k, r in zip(keys, res)},
                per_A_parts={k: r["parts"] for k, r in zip(keys, res)},
                total_dt1=float(sum(r["J"] for r in res)),
                spearman_kc=e12.spearman(list(S["A_levels"]), kcs),
                spearman_kr=e12.spearman(list(S["A_levels"]), krs),
                spearman_ratio=e12.spearman(list(S["A_levels"]), np.array(kcs) / np.array(krs)),
                n_params=2 * len(keys))


def nested_quartet(pool, S, lam, c1, ws=None, wN=1.0, wO=1.0, wR=1.0, fold=None, tag=""):
    tag = tag or lam_key(lam)
    print(f"[stage b] {tag}: M_none ...", flush=True)
    m_none = fit_none(pool, S, lam, ws, wN, wO, wR, fold)
    print(f"[stage b] {tag}: M_c / M_r ...", flush=True)
    m_c = fit_shared(pool, "kc", S, lam, ws, m_none["shared"], wN, wO, wR, fold)
    m_r = fit_shared(pool, "kr", S, lam, ws, m_none["shared"], wN, wO, wR, fold)
    print(f"[stage b] {tag}: M_both ...", flush=True)
    m_both = fit_both(pool, S, lam, c1, m_none, m_c, m_r, ws, wN, wO, wR, fold)
    tot = {n: m["total_dt1"] for n, m in (("M_none", m_none), ("M_c", m_c),
                                          ("M_r", m_r), ("M_both", m_both))}
    gap = tot["M_c"] - tot["M_r"]
    den = tot["M_none"] - tot["M_both"]
    keys = [f"A{A:g}" for A in S["A_levels"]]
    resid = max(abs(np.log10(m_c["ladder"][k] / m_c["shared"]["kr"])
                    - np.log10(m_r["shared"]["kc"] / m_r["ladder"][k])) for k in keys)
    q = dict(lam=lam, ws_frac=ws, wN=wN, wO=wO, wR=wR, fold=fold, M_none=m_none, M_c=m_c,
             M_r=m_r, M_both=m_both, totals_dt1=tot, gap_c_minus_r=gap,
             gap_rel_both=gap / tot["M_both"],
             eta_c=((tot["M_none"] - tot["M_c"]) / den if den > 0 else None),
             eta_r=((tot["M_none"] - tot["M_r"]) / den if den > 0 else None),
             gain_none_to_both=den, ridge_residual_dex=float(resid),
             nested=dict(none_ge_c=bool(tot["M_none"] >= tot["M_c"] - 1e-9),
                         none_ge_r=bool(tot["M_none"] >= tot["M_r"] - 1e-9),
                         c_ge_both=bool(tot["M_c"] >= tot["M_both"] - 1e-9),
                         r_ge_both=bool(tot["M_r"] >= tot["M_both"] - 1e-9)))
    print(f"[stage b] {tag}: none {tot['M_none']:.3f} c {tot['M_c']:.3f} r {tot['M_r']:.3f} "
          f"both {tot['M_both']:.3f} gap {gap:+.4f} ({gap / tot['M_both']:+.2%}) "
          f"eta_c {q['eta_c']} eta_r {q['eta_r']} ridge_res {resid:.2f} dex", flush=True)
    return q


def jackknife_gap(pool, S, lam, ladder):
    """Leave-one-run-out refits of the quartet; jackknife SE of the gap."""
    keys = [f"A{A:g}" for A in S["A_levels"]]
    c1 = {k: ladder[k]["lambdas"][lam_key(lam)]["fits"]["C1"] for k in keys}
    n_runs = _data(S["A_levels"][0])[4]["n_runs"]
    folds = []
    for fold in range(n_runs):
        preload(S, folds=(fold,))
        q = nested_quartet(pool, S, lam, c1, fold=fold, tag=f"{lam_key(lam)} fold {fold}")
        folds.append(dict(fold=fold, gap=q["gap_c_minus_r"], gap_rel=q["gap_rel_both"],
                          eta_c=q["eta_c"], eta_r=q["eta_r"], totals=q["totals_dt1"],
                          ladder_kc_Mc={k: q["M_c"]["ladder"][k] for k in keys},
                          ladder_kr_Mr={k: q["M_r"]["ladder"][k] for k in keys}))
    g = np.array([f["gap"] for f in folds])
    n = len(g)
    se = float(np.sqrt((n - 1) / n * np.sum((g - g.mean()) ** 2)))
    return dict(lam=lam, folds=folds, gap_mean=float(g.mean()), gap_se_jack=se,
                same_sign=bool(np.all(np.sign(g) == np.sign(g[0])) and np.all(g != 0)),
                eta_c=[f["eta_c"] for f in folds], eta_r=[f["eta_r"] for f in folds])


def micro_macro(S, m_both):
    """Agreement between the macroscopic M_both ladder and the event-MLE
    ladder: median |log10 ratio| <= AGREE_DEX and Spearman >= AGREE_RHO."""
    ev = e12.event_ladders(S)[KEY_FIT]
    keys = [f"A{A:g}" for A in S["A_levels"] if f"A{A:g}" in ev]
    out = {}
    for which, lad in (("kappa_c", m_both["ladder_kc"]), ("kappa_r", m_both["ladder_kr"])):
        macro = np.array([lad[k] for k in keys])
        micro = np.array([max(ev[k][which][0], KAPPA_FLOOR) for k in keys])
        dex = np.abs(np.log10(macro / micro))
        sp = e12.spearman(macro, micro) if len(keys) >= 3 else dict(rho=None, p=None)
        out[which] = dict(median_abs_dex=float(np.median(dex)), max_abs_dex=float(np.max(dex)),
                          per_A_dex={k: float(d) for k, d in zip(keys, dex)},
                          spearman_macro_vs_micro=sp,
                          magnitude_agree=bool(np.median(dex) <= AGREE_DEX),
                          trend_agree=bool(sp["rho"] is not None and sp["rho"] >= AGREE_RHO
                                           and sp["p"] is not None and sp["p"] < 0.05))
    return out


def stage_b(S, which="all"):
    out = Path(S["out"])
    ladder = json.loads((out / "ladder13.json").read_text())
    sel = selection(S)
    lam_s = float(sel["lambda_star"])
    keys = [f"A{A:g}" for A in S["A_levels"]]
    preload(S)
    t0 = time.time()
    ctx = mp.get_context("fork")
    hp = out / "hypothesis13.json"
    hyp = (json.loads(hp.read_text()) if which == "star" and hp.exists() else
           dict(_meta=dict(objective=meta(S)["objective"], lambdas=list(S["lambdas"]),
                           structure="C1 (kappa_c, kappa_r), pure E7",
                           shared_kr=np.asarray(S["shared_kr"]).tolist(),
                           shared_kc=np.asarray(S["shared_kc"]).tolist(),
                           prescan_dex=S["prescan_dex"],
                           criteria=dict(agree_dex=AGREE_DEX, agree_rho=AGREE_RHO,
                                         ident=dict(kr_max=IDENT_KR_MAX, basin_dex=IDENT_BASIN_DEX))),
                per_lambda={}, at_star={}, jackknife={}, events=e12.event_ladders(S)))
    hyp["_meta"]["lambda_star"] = lam_s
    hyp["_meta"]["lambdas_for_B"] = sel["lambdas_for_B"]
    with ctx.Pool(processes=S["n_proc"]) as pool:
        if which in ("all", "lambda"):
            for lam in S["lambdas"]:
                c1 = {k: ladder[k]["lambdas"][lam_key(lam)]["fits"]["C1"] for k in keys}
                q = nested_quartet(pool, S, lam, c1)
                q["micro_macro"] = micro_macro(S, q["M_both"])
                hyp["per_lambda"][lam_key(lam)] = q
        if which in ("all", "star"):
            c1s = {k: ladder[k]["lambdas"][lam_key(lam_s)]["fits"]["C1"] for k in keys}
            hyp["at_star"] = {}
            for tag, kw in (("robust_ws", dict(ws=S["robust_ws"])),
                            ("queue_only", dict(wO=0.0, wR=0.0)),
                            ("overtaking_only", dict(wN=0.0, wR=0.0)),
                            ("events_only", dict(wN=0.0, wO=0.0))):
                hyp["at_star"][tag] = nested_quartet(pool, S, lam_s, c1s,
                                                     tag=f"lam*={lam_s:g} {tag}", **kw)
            if S.get("jackknife", True):
                hyp["jackknife"] = {}
                for lam in sorted(set([0.0, lam_s] + [float(l) for l in sel["lambdas_for_B"]])):
                    hyp["jackknife"][lam_key(lam)] = jackknife_gap(pool, S, lam, ladder)
    hyp["_meta"]["runtime_s"] = round(time.time() - t0, 1)
    hyp["gap_vs_lambda"] = [dict(lam=q["lam"], gap=q["gap_c_minus_r"], gap_rel=q["gap_rel_both"],
                                 eta_c=q["eta_c"], eta_r=q["eta_r"],
                                 ridge_residual_dex=q["ridge_residual_dex"],
                                 gain_none_to_both=q["gain_none_to_both"], totals=q["totals_dt1"])
                            for q in hyp["per_lambda"].values()]
    (out / "hypothesis13.json").write_text(json.dumps(hyp, indent=1, default=e12._json))
    print(f"[stage b] wrote {out / 'hypothesis13.json'} ({hyp['_meta']['runtime_s']} s)", flush=True)
    return hyp


# ---------------------------------------------------------------------------
# stage E: identifiability under J_lam*, event kappas, turnover, resolution
# ---------------------------------------------------------------------------

def regrid_any(res):
    """ev4_compare.regrid_sim generalised to any solver dx dividing 100 m."""
    from loader import CELL_LEN, N_CELL
    t_data = (np.arange(ev4.N_T) + 1) * ev4.T_SAMPLE
    res_t = np.asarray(res.t, float)
    res_x = np.asarray(res.x, float)
    k = res_x.size // N_CELL
    assert k * N_CELL == res_x.size, res_x.size
    assert np.isclose(np.median(np.diff(res_x)), CELL_LEN / k, atol=1e-6)
    it = np.argmin(np.abs(res_t[None, :] - t_data[:, None]), axis=1)
    assert np.max(np.abs(res_t[it] - t_data)) <= 0.5

    def field(arr):
        return np.asarray(arr, float)[it].reshape(ev4.N_T, N_CELL, k).mean(axis=2) * 1000.0

    a, f, s = field(res.a), field(res.f), field(res.s)
    return dict(tt=t_data, a=a, f=f, s=s, rho_tot=a + f + s,
                x_cav=np.asarray(res.x_cav, float)[it], N_s=np.asarray(res.N_s, float)[it],
                omega=np.asarray(res.omega, float)[it] * 3600.0,
                cum_cap=np.asarray(res.cum_cap, float)[it],
                cum_rel=np.asarray(res.cum_rel, float)[it])


def run_at(uc, qin, kc, kr, gamma, ws, dx, dt, classical=False):
    from solver import SimConfig, simulate
    kw = dict(v_f=ev4.V_F, w=ev4.W, P=ev4.P, q_in=qin / 3600.0, u_xi=uc, kappa_c=float(kc),
              kappa_r=float(kr), capture_form=e12.FORM, dx=dx, dt=dt,
              save_every=int(round(10.0 / dt)))
    if classical:
        kw.update(kappa_c=0.0, kappa_r=0.0, q_xi_max=e12.QXI_CLASSICAL_VEHH / 3600.0)
    else:
        if gamma is not None:
            kw["gamma"] = float(gamma)
        if ws is not None:
            kw["w_s"] = float(ws) * ev4.W
            kw["P_s"] = None
    cfg = SimConfig(**kw)
    if not classical:
        e12.assert_pure(cfg)
    raw = simulate(cfg)
    return regrid_any(raw), raw


def peak_cav_cell_density(raw, dx):
    rho = np.asarray(raw.a, float) + np.asarray(raw.f, float) + np.asarray(raw.s, float)
    xc = np.asarray(raw.x_cav, float)
    nx = rho.shape[1]
    vals = [rho[i, min(int(xc[i] // dx), nx - 1)] for i in range(len(xc)) if np.isfinite(xc[i])]
    return float(np.max(vals) * 1000.0) if vals else 0.0


DX_KEYS = ("W1", "W1_rel", "Ns_mae", "omega_err", "eps_R", "wake", "s_layer_mean_m", "J",
           "peak_cav_cell_rho_vehkm", "N_s_max_veh", "tau_model_s")


def _dx_task(args):
    A, kc, kr, gamma, ws, dx, dt, classical, lam, tag = args
    regr, raw = run_at(UC, QIN_FIT, kc, kr, gamma, ws, dx, dt, classical)
    p = parts_of(regr, A)
    import e7_ablation as e7a
    import e8_ladder as e8
    p["wake"] = e7a.d1_wake(regr["rho_tot"], regr["tt"], regr["x_cav"])["wake_mean_vehkm"]
    p["J"] = J_of(p, lam)
    p["s_layer_mean_m"] = e8.s_layer_extent(regr)["mean_m"]
    p["peak_cav_cell_rho_vehkm"] = peak_cav_cell_density(raw, dx)
    p["N_s_max_veh"] = float(np.max(regr["N_s"]))
    return dict(A=A, dx=dx, dt=dt, model=tag, **p)


def dx_sensitivity(pool, S, ladder, lam_s, A_list=(1, 5, 10)):
    """Winners at lam = 0 and lam*, and the classical model, at finer dx / dt.
    NOTE: the CAV is represented as a density 1/dx in its cell, so dx is part
    of the model definition (dx = 50 m = the KDE scale of the data since
    E-V4); this is a sensitivity report, not a convergence study."""
    tasks = []
    for A in A_list:
        if A not in S["A_levels"]:
            continue
        for lam, tag in ((0.0, "winner_lam0"), (lam_s, "winner_lamstar")):
            w, _ = _winner(ladder, f"A{A:g}", lam)
            for dx, dt in S["dx_grid"]:
                tasks.append((float(A), w["kappa_c"], w["kappa_r"], w["gamma"], w["ws_frac"],
                              dx, dt, False, lam_s, tag))
        for dx, dt in S["dx_grid"]:
            tasks.append((float(A), 0.0, 0.0, None, None, dx, dt, True, lam_s, "classical"))
    res = pool.map(_dx_task, tasks, chunksize=1)
    out = {}
    for r in res:
        out.setdefault(f"A{r['A']:g}", {}).setdefault(r["model"], {})[
            f"dx{r['dx']:g}_dt{r['dt']:g}"] = {k: r[k] for k in DX_KEYS}
    for key, d in out.items():
        for model in d:
            base = d[model]["dx50_dt0.5"]
            for lbl, fk in (("change_50_to_25", "dx25_dt0.5"), ("change_50_to_12p5", "dx12.5_dt0.25"),
                            ("change_dt_0.5_to_0.25", "dx50_dt0.25")):
                if fk in d[model]:
                    d[model][lbl] = {k: d[model][fk][k] - base[k]
                                     for k in ("W1", "Ns_mae", "omega_err", "eps_R", "wake",
                                               "peak_cav_cell_rho_vehkm")}
    return out


def _ridge_task(args):
    A, ratio, mag, lam = args
    return j_one(A, ratio * mag, mag, lam)


def _ev_task(args):
    A, kc, kr, ws = args
    return parts_one(A, kc, kr, ws, None, e7.DT_PRODUCTION)


def stage_e(S):
    out = Path(S["out"])
    sel = selection(S)
    lam_s = float(sel["lambda_star"])
    ladder = json.loads((out / "ladder13.json").read_text())
    ev = e12.event_ladders(S)[KEY_FIT]
    keys = [f"A{A:g}" for A in S["A_levels"]]
    ratios = [float(r) for r in S["ridge_ratios"]]
    mags = [float(m) for m in S["ridge_mags"]]
    preload(S)
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=S["n_proc"]) as pool:
        tasks, index = [], []
        for A, k in zip(S["A_levels"], keys):
            if k in ev:
                kc, kr = ev[k]["kappa_c"][0], max(ev[k]["kappa_r"][0], KAPPA_FLOOR)
                for ws in (None, S["robust_ws"]):
                    tasks.append((float(A), kc, kr, ws))
                    index.append((k, ws))
        parts = pool.map(_ev_task, tasks, chunksize=1)
        events_field = {}
        for (k, ws), p in zip(index, parts):
            d = events_field.setdefault(k, dict(kappa_c=ev[k]["kappa_c"][0], kappa_r=ev[k]["kappa_r"][0]))
            d["parts" if ws is None else "parts_ws"] = p
            d["J_dt05" if ws is None else "J_ws_dt05"] = J_of(p, lam_s)
        tasks = [(float(A), r, m, lam_s) for A in S["A_levels"] for m in mags for r in ratios]
        vals = pool.map(_ridge_task, tasks, chunksize=4)
        dxs = dx_sensitivity(pool, S, ladder, lam_s)
    W = np.asarray(vals).reshape(len(keys), len(mags), len(ratios))
    ridge = {}
    for i, k in enumerate(keys):
        Wk = W[i]
        im, ir = np.unravel_index(np.argmin(Wk), Wk.shape)
        best = [dict(magnitude=mags[m], ratio=ratios[int(np.argmin(Wk[m]))], J=float(Wk[m].min()))
                for m in range(len(mags))]
        lm = np.log10(mags)
        line = Wk[:, ir]                       # fixed-ratio column through the argmin
        ok = line <= 1.01 * line.min()
        basin = float(lm[ok].max() - lm[ok].min())
        # basin along the valley (best ratio at every magnitude): the audit
        # showed the J_lam* valley runs along constant kappa_c, so the
        # fixed-ratio column cuts across it and its basin is 0 by construction
        valley = Wk.min(axis=1)
        basins_valley = {}
        for tol in (0.01, 0.02):
            okv = valley <= (1.0 + tol) * valley.min()
            basins_valley[f"{100 * tol:.0f}pct"] = float(lm[okv].max() - lm[okv].min())
        best_kc = [b_["ratio"] * b_["magnitude"] for b_ in best]
        w0, _ = _winner(ladder, k, lam_s)
        ridge[k] = dict(W1=Wk.tolist(), min=float(Wk.min()),
                        argmin=dict(magnitude=mags[im], ratio=ratios[ir]),
                        best_ratio_per_magnitude=best, best_kappa_c_per_magnitude=best_kc,
                        basin_width_dex_along_best_ratio=basin,
                        basin_width_dex_along_valley=basins_valley,
                        magnitude_identified_fixed_ratio=bool(
                            mags[im] < IDENT_KR_MAX and basin < IDENT_BASIN_DEX
                            and w0["parts_dt05"]["W1_rel"] <= 1.0),
                        magnitude_identified=bool(
                            mags[im] < IDENT_KR_MAX and basins_valley["1pct"] < IDENT_BASIN_DEX
                            and w0["parts_dt05"]["W1_rel"] <= 1.0),
                        winner_kappa_r=w0["kappa_r"], winner_W1_rel=w0["parts_dt05"]["W1_rel"])
    tau = {}
    for k in keys:
        w0, _ = _winner(ladder, k, 0.0)
        ws_, _ = _winner(ladder, k, lam_s)
        tau[k] = dict(tau_data_s=ladder[k]["ref"]["tau_data_s"],
                      tau_lam0_s=w0["parts_dt05"]["tau_model_s"],
                      tau_lamstar_s=ws_["parts_dt05"]["tau_model_s"],
                      C_d=ladder[k]["ref"]["C_d"], R_d=ladder[k]["ref"]["R_d"],
                      C_m_lam0=w0["parts_dt05"]["C_m"], R_m_lam0=w0["parts_dt05"]["R_m"],
                      C_m_lamstar=ws_["parts_dt05"]["C_m"], R_m_lamstar=ws_["parts_dt05"]["R_m"])
    extras = dict(_meta=dict(ratios=ratios, magnitudes=mags, lambda_star=lam_s,
                             objective="J_lam* (dt=1 ridge, dt=0.5 events)",
                             ident_rule=f"argmin kappa_r < {IDENT_KR_MAX}, basin < {IDENT_BASIN_DEX} dex, W1 <= W1_cl",
                             dx_note="CAV density = 1/dx: dx is part of the model definition "
                                     "(50 m = KDE scale); sensitivity, not convergence",
                             runtime_s=round(time.time() - t0, 1)),
                  events_field=events_field, ridge=ridge, turnover=tau, dx_sensitivity=dxs)
    (out / "extras13.json").write_text(json.dumps(extras, indent=1, default=e12._json))
    print(f"[stage e] wrote {out / 'extras13.json'} ({extras['_meta']['runtime_s']} s)", flush=True)
    return extras


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def smoke_settings():
    S = dict(SETTINGS)
    S.update(out=SETTINGS["out"] / "smoke", A_levels=(1, 5), lambdas=(0.0, 0.1, 0.3),
             r_grid=np.array([0.5, 1.5]), m_grid=np.array([-2.0, 0.0]), maxfev=3, polish_maxfev=2,
             shared_kr=np.array([1e-3, 0.3]), shared_kc=np.array([0.05, 3.0]), prescan_dex=1.0,
             brent_maxiter=3, none_maxfev=2, both_maxfev=2,
             hull_ratios=np.array([3.0, 30.0]), hull_mags=(1e-2, 1.0),
             ridge_ratios=np.array([3.0, 30.0]), ridge_mags=np.array([1e-2, 1.0]),
             dx_grid=((50.0, 0.5), (25.0, 0.5)), jackknife=True)
    return S


def main(argv=None):
    ap = argparse.ArgumentParser(description="E13 v2 joint calibration")
    ap.add_argument("--run", action="store_true", help="stages p, a, f, b, e")
    ap.add_argument("--stage", choices=["p", "a", "f", "b", "bl", "bs", "e"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    S = smoke_settings() if args.smoke else SETTINGS
    if args.smoke:
        stage_p(S); stage_a(S); stage_f(S); stage_b(S); stage_e(S)
        import e13_figures as fig
        fig.make_figures(S)
        print(fig.write_summary(S))
        return
    if args.run or args.stage == "p":
        stage_p(S)
    if args.run or args.stage == "a":
        stage_a(S, resume=args.resume)
    if args.run or args.stage == "f":
        stage_f(S)
    if args.run or args.stage == "b":
        stage_b(S)
    elif args.stage == "bl":
        stage_b(S, which="lambda")
    elif args.stage == "bs":
        stage_b(S, which="star")
    if args.run or args.stage == "e":
        stage_e(S)
    if args.figures:
        import e13_figures as fig
        fig.make_figures(S)
    if args.summary:
        import e13_figures as fig
        print(fig.write_summary(S))
    if not any((args.run, args.stage, args.figures, args.summary)):
        ap.error("choose --run, --stage, --figures, --summary or --smoke")


if __name__ == "__main__":
    main()
