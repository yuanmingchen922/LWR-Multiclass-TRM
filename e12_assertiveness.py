"""E12: the pure E7 catch & release model across the integer assertiveness
ladder A = 1..10 (Mladen's fifth review, see E12_plan.md).

Model = E7 structure ONLY: two-class catch & release, uncapped, source terms
functions of the state only (kappa_c, kappa_r, optional gamma and w_s).
Everything added after E7 is forbidden here and the driver refuses to run
with any of it switched on (see PURE_OFF / assert_pure):

    q_xi_max (DM-G capacity cap, E-V4b), downstream_release (E8 kludge),
    eta_la / leader-loss (E9/E10), s_impermeable (E10 interface constraint).

The classical baseline (scalar LWR + Delle Monache-Goatin moving bottleneck,
Q_xi = 2000 veh/h, kappa = 0) is the only capped run; it is the reference
model, not part of catch & release.

Stage A (--stage a): per A on u_xi = 15 m/s, q_in = 2500 veh/h, four E7
configurations fitted with e7_wasserstein.fit_field (W1 objective, 6x6 log
grid + Nelder-Mead, dt = 1 s), each re-evaluated at dt = 0.5 s on the fit
scenario, zero-refit on q_in = 2000 and (W1 only) on u_xi = 20 m/s:

    C1  kappa_c, kappa_r                    pure E7
    C2  + gamma in [0, 1]                   capture agent ell = a + gamma s
    C3  + w_s / w in [0.5, 1]               stuck-class congested branch
    C4  + gamma, + w_s / w                  joint (new)

Winner per A by the E7 rule (e7_ablation.pick_winner).  The C1 W1 landscape
on the 6x6 grid is stored for the identifiability analysis.  Per-run event
metrics (e_s, omega, N_s) come from out/e1 (A = 1, 10) or out/e4 (other A,
same file naming and keys, u_xi = 15 only).

Stage B (--stage b): nested shared-parameter models over all ten A (C1
structure, objective sum_A W1 at dt = 1 on the q2500 fields):
    M_none  one (kappa_c, kappa_r) for all A
    M_c     kappa_c free per A, kappa_r shared      (Mladen's hypothesis)
    M_r     kappa_r free per A, kappa_c shared      (E-V3 event picture)
    M_both  the per-A C1 fits
plus M_c / M_r with w_s = 0.6 w fixed, Spearman rank correlations of the
parameter ladders with A (field fits and event-based MLE), monotonicity
flags.

Outputs (out/e12/): ladder_A{A}.json, ladder.json, fields_A{A}.npz,
hypothesis.json; figures and summary.md via e12_figures (--figures,
--summary).  --smoke runs a reduced program into out/e12/smoke/.

Stage E (--stage e): extras -- the event-based (E-V3/E4) kappas evaluated
in the field without any fitting, and a (ratio, magnitude) identifiability
scan of the C1 W1 landscape per A -> out/e12/extras.json.

CLI:  python3 e12_assertiveness.py --run [--resume]      stages a, b, e
      python3 e12_assertiveness.py --stage a|b|e [--resume]
      python3 e12_assertiveness.py --figures
      python3 e12_assertiveness.py --summary
      python3 e12_assertiveness.py --smoke
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

import e7_ablation as e7a
import e7_wasserstein as e7
import e8_ladder as e8
import ev4_compare as ev4
from loader import CELL_LEN, N_CELL
from solver import SimConfig, simulate

HERE = Path(__file__).parent
OUT_E1 = HERE / "out" / "e1"
OUT_E4 = HERE / "out" / "e4"
KAPPA_EVENTS = OUT_E4 / "kappa_vs_A.json"

A_LEVELS = tuple(range(1, 11))          # the ONLY assertiveness set in E12
UC = 15.0                               # [m/s] focus CAV slow speed
QIN_FIT = 2500.0                        # [veh/h] fit inflow
QIN_TRANSFER = 2000.0                   # [veh/h] zero-refit inflow
UC_TRANSFER = 20.0                      # [m/s] zero-refit speed (W1 only)
SCEN_FULL = ((UC, QIN_FIT), (UC, QIN_TRANSFER))
SCEN_FIELD = ((UC_TRANSFER, QIN_TRANSFER), (UC_TRANSFER, QIN_FIT))
SCENARIOS = SCEN_FULL + SCEN_FIELD
FORM = "lf"
QXI_CLASSICAL_VEHH = 2000.0             # classical baseline cap [veh/h]
T_SNAP_DS = 700.0                       # [s] downstream-stuck snapshot
NEAR_OPT_TOL = 0.03                     # near-optimal set: <= 3 % above min
RIDGE_TOL = 0.01                        # |M_c - M_r| below this fraction of
#                                         M_both = "field cannot tell" branch
X_UP = (np.arange(N_CELL) + 1) * CELL_LEN

PURE_OFF = dict(q_xi_max=None, downstream_release=False, eta_la=None,
                s_impermeable=False)
FORBIDDEN = tuple(PURE_OFF)

CONFIGS = (("C1", None),
           ("C2", {"gamma": (0.0, 1.0)}),
           ("C3", {"ws_frac": (0.5, 1.0)}),
           ("C4", {"gamma": (0.0, 1.0), "ws_frac": (0.5, 1.0)}))

# production settings (overridden by --smoke)
SETTINGS = dict(
    out=HERE / "out" / "e12",
    A_levels=A_LEVELS,
    kc_grid=np.logspace(-2.5, 0.5, 6),
    kr_grid=np.logspace(-4.0, -0.5, 6),
    maxfev=40,
    shared_kr=np.logspace(-4.0, 1.0, 11),      # M_c shared kappa_r values
    shared_kc=np.logspace(-2.5, 2.0, 10),      # M_r shared kappa_c values
    log_kc_bounds=(-2.5, 2.0),                 # per-A 1-D search box (log10);
    log_kr_bounds=(-4.0, 1.0),                 # wide: C1 optima sit on a ridge
    scalar_maxiter=40,
    scalar_xatol=0.02,
    none_maxfev=40,
    both_maxfev=60,                            # per-A 2-D polish of M_both
    robust_ws=0.6,
    ridge_ratios=np.logspace(0.0, 3.0, 13),    # stage e: kappa_c/kappa_r scan
    ridge_mags=(1e-3, 1e-2, 1e-1, 1.0, 10.0),  # stage e: kappa_r magnitudes
    n_proc=max(1, min(10, os.cpu_count() - 2)),
)


def scen_key(uc, qin):
    return f"u{uc:g}_q{qin:g}"


# ---------------------------------------------------------------------------
# purity guard
# ---------------------------------------------------------------------------

def assert_pure(obj):
    """Raise ValueError if `obj` (SimConfig or kwargs dict) switches on any
    knob that is forbidden in E12 (capacity cap, downstream release,
    leader-loss, impermeable interface)."""
    get = (obj.get if isinstance(obj, dict)
           else lambda k, d=None: getattr(obj, k, d))
    for name, off in PURE_OFF.items():
        val = get(name, off)
        if val is not off and val != off:
            raise ValueError(f"E12 purity violation: {name}={val!r} "
                             f"(must be {off!r}; pure E7 structure only)")
    return True


def _solver_defaults_are_pure():
    d = {f.name: f.default for f in dataclasses.fields(SimConfig)}
    bad = {k: d.get(k, "<missing>") for k, v in PURE_OFF.items()
           if k not in d or d[k] != v}
    if bad:
        raise RuntimeError(f"solver.SimConfig defaults are not the pure E7 "
                           f"path: {bad}")
    return True


_solver_defaults_are_pure()


def cr_config(uc, qin, kc, kr, gamma=None, ws_frac=None, dt=1.0,
              save_every=None):
    """SimConfig of a pure catch & release run (mirrors e7_wasserstein.run_sim
    without any extra_cfg passthrough) and assert its purity."""
    if save_every is None:
        save_every = int(round(10.0 / dt))
    kw = dict(v_f=ev4.V_F, w=ev4.W, P=ev4.P, q_in=qin / 3600.0, u_xi=uc,
              kappa_c=float(kc), kappa_r=float(kr), capture_form=FORM,
              dt=dt, save_every=save_every)
    if gamma is not None:
        kw["gamma"] = float(gamma)
    if ws_frac is not None:
        kw["w_s"] = float(ws_frac) * ev4.W
        kw["P_s"] = None
    cfg = SimConfig(**kw)
    assert_pure(cfg)
    return cfg


def run_cr(uc, qin, kc, kr, gamma=None, ws_frac=None, dt=1.0):
    """Pure catch & release run regridded to the data grid."""
    return ev4.regrid_sim(simulate(cr_config(uc, qin, kc, kr, gamma,
                                             ws_frac, dt)))


def run_classical(uc, qin, dt=e7.DT_PRODUCTION):
    """Classical baseline: kappa = 0 + DM-G cap Q_xi = 2000 veh/h (the only
    capped run in E12)."""
    cfg = SimConfig(v_f=ev4.V_F, w=ev4.W, P=ev4.P, q_in=qin / 3600.0,
                    u_xi=uc, kappa_c=0.0, kappa_r=0.0, capture_form=FORM,
                    dt=dt, save_every=int(round(10.0 / dt)),
                    q_xi_max=QXI_CLASSICAL_VEHH / 3600.0)
    return ev4.regrid_sim(simulate(cfg))


# ---------------------------------------------------------------------------
# measured data
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def e1_dir(path):
    """Temporarily point ev4_compare.load_measured at another npz folder."""
    old = ev4.E1
    ev4.E1 = Path(path)
    try:
        yield
    finally:
        ev4.E1 = old


def load_measured_any(A, uc, qin):
    """ev4_compare.load_measured for every integer A at u_xi = 15: out/e1
    for A in {1, 10}, out/e4 (E4 sweep, identical naming/keys) otherwise."""
    A = float(A)
    if A in (1.0, 10.0):
        return ev4.load_measured(A, float(uc), float(qin))
    if float(uc) != UC:
        raise ValueError(f"per-run classification npz exist only at "
                         f"u_xi = {UC:g} m/s for A = {A:g} (asked {uc})")
    with e1_dir(OUT_E4):
        return ev4.load_measured(A, float(uc), float(qin))


def cav_data_mean(meas):
    """Rep-mean measured CAV trajectory [m], NaN where no run has it."""
    xc = np.asarray(meas["x_cav"], float)
    ok = np.isfinite(xc)
    n = ok.sum(axis=0)
    out = np.full(xc.shape[1], np.nan)
    out[n > 0] = np.where(ok, xc, 0.0).sum(axis=0)[n > 0] / n[n > 0]
    return out


def data_diagnostics(A, uc, qin):
    tt, rho_mean = e7.load_rho_mean(A, uc, qin)
    x_nom = ev4.x_cav_nominal(tt, uc)
    return dict(wake=e7a.d1_wake(rho_mean, tt, x_nom)["wake_mean_vehkm"],
                d2=e7a.d2_rarefaction(rho_mean, tt))


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate(A, uc, qin, model, kc=0.0, kr=0.0, gamma=None, ws_frac=None,
             dt=e7.DT_PRODUCTION):
    """Production run of `model` ('cr' | 'classical') on one scenario.
    Returns (summary, regr).  Field metrics always; per-run event metrics
    only where classification npz exist (u_xi = 15)."""
    if model == "cr":
        regr = run_cr(uc, qin, kc, kr, gamma, ws_frac, dt)
    elif model == "classical":
        regr = run_classical(uc, qin, dt)
    else:
        raise ValueError(model)
    tt, rho_mean = e7.load_rho_mean(A, uc, qin)
    assert np.allclose(regr["tt"], tt)
    summ = dict(
        model=model,
        W1=e7.w1_mean(regr["rho_tot"], rho_mean, tt),
        RMSE=e7.rmse_mean(regr["rho_tot"], rho_mean, tt),
        wake=e7a.d1_wake(regr["rho_tot"], tt, regr["x_cav"])["wake_mean_vehkm"],
        d2=e7a.d2_rarefaction(regr["rho_tot"], tt),
        N_s700=float(regr["N_s"][int(np.argmin(np.abs(tt - T_SNAP_DS)))]),
        peak_omega_vehh=float(np.max(regr["omega"])),
        ds_stuck_700=e8.ds_stuck_fraction(regr, T_SNAP_DS)["frac"],
        s_layer_mean_m=e8.s_layer_extent(regr)["mean_m"],
        ds_max_s_vehkm=e8.ds_max_s(regr),
        full=False)
    if float(uc) == UC:
        meas = load_measured_any(A, uc, qin)
        met = ev4.metrics(regr, meas)
        summ.update(
            full=True,
            e_s=met["e_s"]["mean"], e_s_per_rep=met["e_s"]["per_rep"],
            omega_err=met["omega_cum_rel_err"]["mean"],
            omega_err_per_rep=met["omega_cum_rel_err"]["per_rep"],
            rho_rmse=met["rho_rmse"]["mean"],
            rho_rmse_per_rep=met["rho_rmse"]["per_rep"],
            Ns_mae=met["Ns_mae"]["mean"],
            cum_overtake_sim_veh=met["cum_overtake_sim_veh"])
        if model == "cr":
            summ["d3"] = e7a.d3_dyn_eq(regr, uc, qin, kr, ws_frac)
    return summ, regr


# ---------------------------------------------------------------------------
# stage A: per-A calibration ladder
# ---------------------------------------------------------------------------

def grid_landscape(A, S):
    """C1 W1 landscape on the fit grid (dt = 1) with the near-optimal set."""
    tt, rho_mean = e7.load_rho_mean(A, UC, QIN_FIT)
    kc_g, kr_g = np.asarray(S["kc_grid"]), np.asarray(S["kr_grid"])
    W = np.empty((len(kc_g), len(kr_g)))
    for i, kc in enumerate(kc_g):
        for j, kr in enumerate(kr_g):
            regr = e7.run_sim(UC, QIN_FIT, kc, kr, FORM, dt=1.0)
            W[i, j] = e7.w1_mean(regr["rho_tot"], rho_mean, tt)
    i0, j0 = np.unravel_index(np.argmin(W), W.shape)
    mask = W <= (1.0 + NEAR_OPT_TOL) * W.min()
    lkc = np.log10(kc_g)[:, None] * np.ones_like(W)
    lkr = np.log10(kr_g)[None, :] * np.ones_like(W)
    ratio = lkc - lkr
    ext = lambda arr: float(arr[mask].max() - arr[mask].min())
    return dict(kc_grid=kc_g.tolist(), kr_grid=kr_g.tolist(),
                W1=W.tolist(), min=float(W.min()),
                argmin=dict(kappa_c=float(kc_g[i0]), kappa_r=float(kr_g[j0])),
                near_opt=dict(tol=NEAR_OPT_TOL, n=int(mask.sum()),
                              extent_log_kc=ext(lkc), extent_log_kr=ext(lkr),
                              extent_log_ratio=ext(ratio),
                              points=[dict(kappa_c=float(kc_g[i]),
                                           kappa_r=float(kr_g[j]),
                                           W1=float(W[i, j]))
                                      for i, j in zip(*np.where(mask))]))


def stage_a_one(A, S):
    """Fit C1..C4 for one A, evaluate everything, write ladder/fields."""
    t0 = time.time()
    out = Path(S["out"])
    A = float(A)
    res = dict(A=A, tag=f"A{A:g}_u{UC:g}_q{QIN_FIT:g}", fits={},
               classical={}, data={}, structure="pure E7 (no cap, no DR, "
               "no leader-loss, no impermeable interface)")
    regrs = {"classical": {}, "winner": {}, "C1": {}}
    for uc, qin in SCENARIOS:
        key = scen_key(uc, qin)
        summ, regr = evaluate(A, uc, qin, "classical")
        res["classical"][key] = summ
        res["data"][key] = data_diagnostics(A, uc, qin)
        if (uc, qin) in SCEN_FULL:
            regrs["classical"][key] = regr
    rows = {}
    kept = {}
    for name, extra in CONFIGS:
        fit = e7.fit_field(A, UC, QIN_FIT, form=FORM, metric="w1",
                           extra=extra, dt_fit=1.0, kc_grid=S["kc_grid"],
                           kr_grid=S["kr_grid"], maxfev=S["maxfev"],
                           verbose=False)
        assert not fit["extra_cfg"], "fit_field must not receive extra_cfg"
        gamma = fit["extra"].get("gamma")
        ws = fit["extra"].get("ws_frac")
        block = dict(kappa_c=fit["kappa_c"], kappa_r=fit["kappa_r"],
                     gamma=gamma, ws_frac=ws,
                     fit=dict(objective_dt1=fit["objective"],
                              grid_objective_dt1=fit["grid_objective"],
                              n_sim=fit["n_sim"],
                              at_dt_fit=fit["at_dt_fit"],
                              at_production=fit["at_production"]),
                     eval={})
        for uc, qin in SCENARIOS:
            key = scen_key(uc, qin)
            summ, regr = evaluate(A, uc, qin, "cr", fit["kappa_c"],
                                  fit["kappa_r"], gamma, ws)
            block["eval"][key] = summ
            if (uc, qin) in SCEN_FULL:
                kept.setdefault(name, {})[key] = regr
        res["fits"][name] = block
        ev = block["eval"][scen_key(UC, QIN_FIT)]
        rows[name] = dict(W1=ev["W1"], d1=dict(wake_mean_vehkm=ev["wake"]),
                          d2=ev["d2"])
        print(f"  A={A:g} {name}: kc={fit['kappa_c']:.3e} "
              f"kr={fit['kappa_r']:.3e} gamma={gamma} ws={ws} "
              f"W1={ev['W1']:.1f} (classical "
              f"{res['classical'][scen_key(UC, QIN_FIT)]['W1']:.1f})",
              flush=True)
    winner, rule = e7a.pick_winner(rows)
    res["winner"], res["winner_rule"] = winner, rule
    res["grid_C1"] = grid_landscape(A, S)
    res["runtime_s"] = round(time.time() - t0, 1)

    # fields for the figures
    fields = {}
    for uc, qin in SCEN_FULL:
        key = scen_key(uc, qin)
        tt, rho_mean = e7.load_rho_mean(A, uc, qin)
        meas = load_measured_any(A, uc, qin)
        fields["tt"] = tt
        fields[f"rho_mean_{key}"] = rho_mean
        fields[f"xcav_data_{key}"] = cav_data_mean(meas)
        for lbl, regr in (("classical", regrs["classical"][key]),
                          ("winner", kept[winner][key]),
                          ("C1", kept["C1"][key])):
            fields[f"{lbl}_rho_{key}"] = regr["rho_tot"]
            fields[f"{lbl}_xcav_{key}"] = regr["x_cav"]
            if lbl != "classical":
                fields[f"{lbl}_s_{key}"] = regr["s"]
                fields[f"{lbl}_f_{key}"] = regr["f"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"fields_A{A:g}.npz", **fields)
    (out / f"ladder_A{A:g}.json").write_text(
        json.dumps(res, indent=1, default=_json))
    print(f"A={A:g} done: winner {winner} ({res['runtime_s']} s)", flush=True)
    return res


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _stage_a_worker(args):
    A, S = args
    return stage_a_one(A, S)


def meta(S):
    return dict(structure="pure E7 catch & release: uncapped, lf capture "
                "form, optional gamma / w_s (state-only)",
                forbidden_knobs={k: ("off" if v in (None, False) else v)
                                 for k, v in PURE_OFF.items()},
                classical=f"kappa=0 + DM-G cap Q_xi={QXI_CLASSICAL_VEHH:g} veh/h",
                A_levels=list(S["A_levels"]), fit_scenario=scen_key(UC, QIN_FIT),
                transfer_scenarios=[scen_key(*s) for s in SCENARIOS[1:]],
                configs={n: (None if x is None else
                             {k: list(v) for k, v in x.items()})
                         for n, x in CONFIGS},
                kc_grid=np.asarray(S["kc_grid"]).tolist(),
                kr_grid=np.asarray(S["kr_grid"]).tolist(),
                maxfev=S["maxfev"], dt_fit=1.0, dt_production=e7.DT_PRODUCTION,
                objective="mean W1 of cumulative densities, t in [100,1000] s,"
                          " x <= 20 km, vs 5-run-mean field",
                winner_rule="e7_ablation.pick_winner on the q2500 rows",
                near_opt_tol=NEAR_OPT_TOL,
                event_metrics_source="out/e1 (A=1,10) / out/e4 (other A), "
                                     "u_xi=15 only")


def stage_a(S, resume=False):
    out = Path(S["out"])
    out.mkdir(parents=True, exist_ok=True)
    todo = [A for A in S["A_levels"]
            if not (resume and (out / f"ladder_A{A:g}.json").exists())]
    print(f"[stage a] A = {todo} on {S['n_proc']} processes", flush=True)
    if todo:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(S["n_proc"], len(todo))) as pool:
            pool.map(_stage_a_worker, [(A, S) for A in todo], chunksize=1)
    merged = {"_meta": meta(S)}
    for A in S["A_levels"]:
        merged[f"A{A:g}"] = json.loads((out / f"ladder_A{A:g}.json").read_text())
    (out / "ladder.json").write_text(json.dumps(merged, indent=1))
    print(f"[stage a] wrote {out / 'ladder.json'}", flush=True)
    return merged


# ---------------------------------------------------------------------------
# stage B: nested shared-parameter hypothesis test
# ---------------------------------------------------------------------------

_FIELDS = {}


def _field(A):
    if A not in _FIELDS:
        _FIELDS[A] = e7.load_rho_mean(A, UC, QIN_FIT)
    return _FIELDS[A]


def w1_one(A, kc, kr, ws_frac=None, dt=1.0):
    """W1 objective of one A at (15, 2500), pure C1 structure (+ optional
    fixed w_s)."""
    tt, rho_mean = _field(A)
    regr = run_cr(UC, QIN_FIT, kc, kr, None, ws_frac, dt)
    return float(e7.w1_mean(regr["rho_tot"], rho_mean, tt))


def _w1_task(args):
    return w1_one(*args)


def _scalar_task(args):
    """Per-A 1-D minimisation of W1 over the free log-parameter with the
    other one shared.  args = (A, free, fixed_val, bounds, ws, S, ref):
    `ref` (log10 of a reference value of the free parameter, e.g. the
    M_none optimum) is evaluated as well and kept if it is better, so the
    nested model can never lose against its parent by optimizer failure."""
    from scipy.optimize import minimize_scalar
    A, free, fixed, bounds, ws, S, ref = args
    n = [0]

    def f(z):
        n[0] += 1
        kc, kr = (10.0 ** z, fixed) if free == "kc" else (fixed, 10.0 ** z)
        return w1_one(A, kc, kr, ws)

    r = minimize_scalar(f, bounds=bounds, method="bounded",
                        options=dict(xatol=S["scalar_xatol"],
                                     maxiter=S["scalar_maxiter"]))
    z_best, j_best = float(r.x), float(r.fun)
    if ref is not None:
        j_ref = f(float(ref))
        if j_ref < j_best:
            z_best, j_best = float(ref), j_ref
    return dict(A=A, free=free, value=10.0 ** z_best, W1=j_best, n_eval=n[0])


def _shared_scan(pool, free, shared_vals, ws, S, ref=None):
    """For every shared value: per-A optimum of the free parameter (parallel
    over (A, shared)).  Returns list of dicts sorted by total W1."""
    fixed_name = "kr" if free == "kc" else "kc"
    bounds = S["log_kc_bounds"] if free == "kc" else S["log_kr_bounds"]
    tasks = [(A, free, float(v), bounds, ws, S, ref)
             for v in shared_vals for A in S["A_levels"]]
    res = pool.map(_scalar_task, tasks, chunksize=1)
    out = []
    for k, v in enumerate(shared_vals):
        rows = res[k * len(S["A_levels"]):(k + 1) * len(S["A_levels"])]
        out.append(dict(shared={fixed_name: float(v)},
                        per_A={f"A{r['A']:g}": dict(value=r["value"],
                                                    W1=r["W1"])
                               for r in rows},
                        total=float(sum(r["W1"] for r in rows))))
    return sorted(out, key=lambda d: d["total"])


def fit_shared(pool, free, S, ws=None, parent=None):
    """M_c (free='kc', shared kappa_r) or M_r (free='kr', shared kappa_c):
    coarse scan of the shared value, then one refinement at x10^(+-0.2).
    `parent` = the M_none optimum dict(kc, kr) of the same w_s: its shared
    value joins the scan and its free value is a per-A fallback, which
    makes the nested inequality sum W1(M_x) <= sum W1(M_none) exact."""
    fixed_name = "kr" if free == "kc" else "kc"
    vals = list(S["shared_kr"] if free == "kc" else S["shared_kc"])
    ref = None
    if parent is not None:
        vals.append(parent[fixed_name])
        ref = float(np.log10(parent["kc" if free == "kc" else "kr"]))
    scan = _shared_scan(pool, free, vals, ws, S, ref)
    best = scan[0]
    v0 = best["shared"][fixed_name]
    refine = _shared_scan(pool, free, [v0 * 10 ** -0.2, v0 * 10 ** 0.2],
                          ws, S, ref)
    cands = sorted([best] + refine, key=lambda d: d["total"])
    win = cands[0]
    ladder = {a: d["value"] for a, d in win["per_A"].items()}
    per_A_W1_dt1 = {a: d["W1"] for a, d in win["per_A"].items()}
    # re-evaluate at production resolution
    prod = {}
    for A in S["A_levels"]:
        a = f"A{A:g}"
        kc, kr = ((ladder[a], win["shared"]["kr"]) if free == "kc"
                  else (win["shared"]["kc"], ladder[a]))
        prod[a] = w1_one(A, kc, kr, ws, dt=e7.DT_PRODUCTION)
    vals_ladder = [ladder[f"A{A:g}"] for A in S["A_levels"]]
    mono = (all(np.diff(vals_ladder) <= 1e-12) if free == "kc"
            else all(np.diff(vals_ladder) >= -1e-12))
    return dict(free=("kappa_c" if free == "kc" else "kappa_r"),
                shared=win["shared"], ws_frac=ws,
                ladder=ladder, per_A_W1_dt1=per_A_W1_dt1,
                total_dt1=win["total"],
                per_A_W1_dt05=prod, total_dt05=float(sum(prod.values())),
                monotone=("non-increasing" if free == "kc"
                          else "non-decreasing"), monotone_ok=bool(mono),
                scan=[dict(shared=d["shared"], total=d["total"])
                      for d in sorted(scan + refine,
                                      key=lambda d: d["total"])],
                n_params=len(S["A_levels"]) + 1)


def _both_task(args):
    """Per-A 2-D (log kc, log kr) Nelder-Mead polish of M_both starting from
    the best of several candidates (the C1 fit and the per-A points of the
    single-carrier models), so that M_both <= M_c, M_r holds exactly."""
    from scipy.optimize import minimize
    A, starts, S = args
    cands = [(w1_one(A, kc, kr), float(kc), float(kr)) for kc, kr in starts]
    j0, kc0, kr0 = min(cands)
    r = minimize(lambda z: w1_one(A, 10.0 ** z[0], 10.0 ** z[1]),
                 [np.log10(kc0), np.log10(kr0)], method="Nelder-Mead",
                 options=dict(maxfev=S["both_maxfev"], xatol=1e-3,
                              fatol=1e-3))
    kc, kr, j = 10.0 ** r.x[0], 10.0 ** r.x[1], float(r.fun)
    if j0 <= j:
        kc, kr, j = kc0, kr0, j0
    return dict(A=A, kappa_c=float(kc), kappa_r=float(kr), W1=j,
                start=dict(kappa_c=kc0, kappa_r=kr0, W1=j0),
                n_eval=int(r.nfev))


def fit_both(pool, S, c1, m_none, m_c, m_r):
    keys = [f"A{A:g}" for A in S["A_levels"]]
    tasks = []
    for A, k in zip(S["A_levels"], keys):
        starts = [(c1[k]["kappa_c"], c1[k]["kappa_r"]),
                  (m_c["ladder"][k], m_c["shared"]["kr"]),
                  (m_r["shared"]["kc"], m_r["ladder"][k]),
                  (m_none["shared"]["kc"], m_none["shared"]["kr"])]
        tasks.append((float(A), starts, S))
    res = pool.map(_both_task, tasks, chunksize=1)
    prod = {k: w1_one(float(k[1:]), r["kappa_c"], r["kappa_r"],
                      dt=e7.DT_PRODUCTION) for k, r in zip(keys, res)}
    return dict(ladder_kc={k: r["kappa_c"] for k, r in zip(keys, res)},
                ladder_kr={k: r["kappa_r"] for k, r in zip(keys, res)},
                per_A_W1_dt1={k: r["W1"] for k, r in zip(keys, res)},
                total_dt1=float(sum(r["W1"] for r in res)),
                per_A_W1_dt05=prod, total_dt05=float(sum(prod.values())),
                start={k: r["start"] for k, r in zip(keys, res)},
                n_params=2 * len(keys))


def fit_none(pool, S, ws=None):
    """M_none: one (kappa_c, kappa_r) for all A: grid on the sum + polish."""
    from scipy.optimize import minimize
    A_levels = list(S["A_levels"])

    def total(kc, kr, dt=1.0):
        return float(sum(pool.map(_w1_task, [(A, kc, kr, ws, dt)
                                             for A in A_levels])))

    kc_g, kr_g = np.asarray(S["kc_grid"]), np.asarray(S["kr_grid"])
    grid = {(float(kc), float(kr)): total(kc, kr) for kc in kc_g for kr in kr_g}
    (kc0, kr0), j0 = min(grid.items(), key=lambda kv: kv[1])
    r = minimize(lambda z: total(10 ** z[0], 10 ** z[1]),
                 [np.log10(kc0), np.log10(kr0)], method="Nelder-Mead",
                 options=dict(maxfev=S["none_maxfev"], xatol=1e-3,
                              fatol=1e-3))
    kc, kr, j = 10 ** r.x[0], 10 ** r.x[1], float(r.fun)
    if j0 < j:
        kc, kr, j = kc0, kr0, j0
    per_A = {f"A{A:g}": w1_one(A, kc, kr, ws) for A in A_levels}
    prod = {f"A{A:g}": w1_one(A, kc, kr, ws, dt=e7.DT_PRODUCTION)
            for A in A_levels}
    return dict(shared=dict(kc=float(kc), kr=float(kr)), ws_frac=ws,
                per_A_W1_dt1=per_A,
                total_dt1=float(sum(per_A.values())), per_A_W1_dt05=prod,
                total_dt05=float(sum(prod.values())), grid_total_min=j0,
                n_params=2)


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(np.asarray(x, float), np.asarray(y, float))
    return dict(rho=float(r.correlation), p=float(r.pvalue), n=len(x))


def event_ladders(S):
    """Event-based Poisson-MLE kappas (E4 sweep), integer A only."""
    d = json.loads(KAPPA_EVENTS.read_text())
    out = {}
    for qin in (QIN_FIT, QIN_TRANSFER):
        rows = {}
        for A in S["A_levels"]:
            k = f"A{A:g}_u15_q{qin:g}"
            if k in d:
                rows[f"A{A:g}"] = dict(kappa_c=d[k]["kappa_c"],
                                       kappa_r=d[k]["kappa_r"])
        out[scen_key(UC, qin)] = rows
    return out


def stage_b(S):
    out = Path(S["out"])
    ladder = json.loads((out / "ladder.json").read_text())
    A_levels = list(S["A_levels"])
    keys = [f"A{A:g}" for A in A_levels]
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=S["n_proc"]) as pool:
        c1 = {k: ladder[k]["fits"]["C1"] for k in keys}
        c1_dt1 = {k: w1_one(float(k[1:]), c1[k]["kappa_c"], c1[k]["kappa_r"])
                  for k in keys}
        print("[stage b] M_none ...", flush=True)
        m_none = fit_none(pool, S)
        print("[stage b] M_c ...", flush=True)
        m_c = fit_shared(pool, "kc", S, parent=m_none["shared"])
        print("[stage b] M_r ...", flush=True)
        m_r = fit_shared(pool, "kr", S, parent=m_none["shared"])
        print("[stage b] M_both (per-A 2-D polish) ...", flush=True)
        m_both = fit_both(pool, S, c1, m_none, m_c, m_r)
        m_both["C1_fit"] = dict(
            per_A_W1_dt1=c1_dt1, total_dt1=float(sum(c1_dt1.values())),
            per_A_W1_dt1_from_fit={k: c1[k]["fit"]["at_dt_fit"]["w1"]
                                   for k in keys},
            per_A_W1_dt05={k: c1[k]["eval"][scen_key(UC, QIN_FIT)]["W1"]
                           for k in keys})
        print(f"[stage b] robustness w_s = {S['robust_ws']} w ...", flush=True)
        m_none_ws = fit_none(pool, S, ws=S["robust_ws"])
        m_c_ws = fit_shared(pool, "kc", S, ws=S["robust_ws"],
                            parent=m_none_ws["shared"])
        m_r_ws = fit_shared(pool, "kr", S, ws=S["robust_ws"],
                            parent=m_none_ws["shared"])

    # trends
    A_arr = np.array(A_levels, float)
    c1_kc = [c1[k]["kappa_c"] for k in keys]
    c1_kr = [c1[k]["kappa_r"] for k in keys]
    win = {k: ladder[k]["fits"][ladder[k]["winner"]] for k in keys}
    ev = event_ladders(S)
    trends = dict(
        C1=dict(kappa_c=spearman(A_arr, c1_kc), kappa_r=spearman(A_arr, c1_kr),
                ratio=spearman(A_arr, np.array(c1_kc) / np.array(c1_kr))),
        winner=dict(kappa_c=spearman(A_arr, [win[k]["kappa_c"] for k in keys]),
                    kappa_r=spearman(A_arr, [win[k]["kappa_r"] for k in keys])),
        M_c=dict(kappa_c=spearman(A_arr, [m_c["ladder"][k] for k in keys])),
        M_r=dict(kappa_r=spearman(A_arr, [m_r["ladder"][k] for k in keys])))
    for sk, rows in ev.items():
        ks = [k for k in keys if k in rows]
        if len(ks) >= 3:
            a = np.array([float(k[1:]) for k in ks])
            trends[f"events_{sk}"] = dict(
                kappa_c=spearman(a, [rows[k]["kappa_c"][0] for k in ks]),
                kappa_r=spearman(a, [rows[k]["kappa_r"][0] for k in ks]))
    tot = {n: m["total_dt1"] for n, m in (("M_none", m_none), ("M_c", m_c),
                                          ("M_r", m_r), ("M_both", m_both))}
    nest_tol = 0.005
    nested = dict(
        none_ge_c=bool(tot["M_none"] >= tot["M_c"] * (1 - nest_tol)),
        none_ge_r=bool(tot["M_none"] >= tot["M_r"] * (1 - nest_tol)),
        c_ge_both=bool(tot["M_c"] >= tot["M_both"] * (1 - nest_tol)),
        r_ge_both=bool(tot["M_r"] >= tot["M_both"] * (1 - nest_tol)),
        none_ws_ge_c_ws=bool(m_none_ws["total_dt1"]
                             >= m_c_ws["total_dt1"] * (1 - nest_tol)),
        none_ws_ge_r_ws=bool(m_none_ws["total_dt1"]
                             >= m_r_ws["total_dt1"] * (1 - nest_tol)))
    # ridge diagnostic: C1 optima whose magnitude left the fit grid
    lo_kc, hi_kc = np.log10(S["kc_grid"][0]), np.log10(S["kc_grid"][-1])
    ridge = {k: dict(log10_kc=float(np.log10(c1[k]["kappa_c"])),
                     above_grid=bool(np.log10(c1[k]["kappa_c"]) > hi_kc + 1e-9),
                     ratio=c1[k]["kappa_c"] / c1[k]["kappa_r"])
             for k in keys}
    gap = tot["M_c"] - tot["M_r"]
    totals_txt = (f"sum W1 dt=1: M_none {tot['M_none']:.1f}, M_c {tot['M_c']:.1f}, "
                  f"M_r {tot['M_r']:.1f}, M_both {tot['M_both']:.1f}; "
                  f"gap M_c-M_r = {gap:+.2f} veh km "
                  f"({gap / tot['M_both']:+.2%} of M_both)")
    if abs(gap) < RIDGE_TOL * tot["M_both"]:
        verdict = ("RIDGE: the field cannot tell whether A acts through "
                   "kappa_c or kappa_r (|M_c - M_r| < "
                   f"{RIDGE_TOL:.0%} of M_both); only the ratio kappa_c/kappa_r "
                   f"is identified ({totals_txt})")
    else:
        best_single = "M_c" if gap < 0 else "M_r"
        verdict = (f"{best_single} is the better single-carrier model "
                   f"({totals_txt})")
    hyp = dict(_meta=dict(objective="sum over A of mean W1 at (u15, q2500), "
                          "dt=1 for the search, dt=0.5 re-evaluation",
                          structure="C1 (kappa_c, kappa_r), pure E7",
                          shared_kr=np.asarray(S["shared_kr"]).tolist(),
                          shared_kc=np.asarray(S["shared_kc"]).tolist(),
                          scalar=dict(xatol=S["scalar_xatol"],
                                      maxiter=S["scalar_maxiter"]),
                          nest_tol=nest_tol, runtime_s=round(time.time() - t0, 1)),
               M_none=m_none, M_c=m_c, M_r=m_r, M_both=m_both,
               M_none_ws=m_none_ws, M_c_ws=m_c_ws, M_r_ws=m_r_ws,
               totals_dt1=dict(tot, M_none_ws=m_none_ws["total_dt1"],
                               M_c_ws=m_c_ws["total_dt1"],
                               M_r_ws=m_r_ws["total_dt1"]),
               totals_dt05={n: m["total_dt05"] for n, m in
                            (("M_none", m_none), ("M_c", m_c), ("M_r", m_r),
                             ("M_both", m_both), ("M_none_ws", m_none_ws),
                             ("M_c_ws", m_c_ws), ("M_r_ws", m_r_ws))},
               nested=nested, ridge_C1=ridge, verdict=verdict,
               trends=trends, events=ev)
    (out / "hypothesis.json").write_text(json.dumps(hyp, indent=1,
                                                    default=_json))
    print(f"[stage b] {verdict}")
    print(f"[stage b] wrote {out / 'hypothesis.json'} "
          f"({hyp['_meta']['runtime_s']} s)", flush=True)
    return hyp


# ---------------------------------------------------------------------------
# stage E: extras -- event-MLE kappas in the field, ratio/magnitude ridge scan
# ---------------------------------------------------------------------------

def _ridge_task(args):
    A, ratio, mag, ws = args
    return w1_one(A, ratio * mag, mag, ws)


def stage_e(S):
    """(1) Zero-field-fit evaluation of the event-based (E-V3/E4 Poisson MLE)
    kappas of each A in the pure C1 structure (and with w_s = 0.6 w).
    (2) Identifiability scan: W1(ratio, magnitude) with kappa_c = ratio *
    kappa_r, kappa_r = magnitude, per A at (15, 2500), dt = 1.  Writes
    out/e12/extras.json (independent of ladder/hypothesis)."""
    out = Path(S["out"])
    ev = event_ladders(S)[scen_key(UC, QIN_FIT)]
    keys = [f"A{A:g}" for A in S["A_levels"]]
    ratios = [float(r) for r in S["ridge_ratios"]]
    mags = [float(m) for m in S["ridge_mags"]]
    ctx = mp.get_context("fork")
    t0 = time.time()
    with ctx.Pool(processes=S["n_proc"]) as pool:
        # (1) event kappas in the field (production dt), with/without w_s
        tasks, index = [], []
        for A, k in zip(S["A_levels"], keys):
            if k not in ev:
                continue
            kc, kr = ev[k]["kappa_c"][0], max(ev[k]["kappa_r"][0], 1e-12)
            for ws in (None, S["robust_ws"]):
                tasks.append((float(A), kc, kr, ws, e7.DT_PRODUCTION))
                index.append((k, ws))
        vals = pool.map(_w1_task, tasks, chunksize=1)
        events_field = {}
        for (k, ws), v in zip(index, vals):
            d = events_field.setdefault(k, dict(kappa_c=ev[k]["kappa_c"][0],
                                                kappa_r=ev[k]["kappa_r"][0]))
            d["W1_dt05" if ws is None else "W1_ws_dt05"] = float(v)
        # (2) ridge scan at dt = 1 (fit resolution)
        tasks = [(float(A), r, m, None) for A in S["A_levels"]
                 for m in mags for r in ratios]
        vals = pool.map(_ridge_task, tasks, chunksize=4)
    W = np.asarray(vals).reshape(len(keys), len(mags), len(ratios))
    ridge = {}
    for i, k in enumerate(keys):
        Wk = W[i]
        j_min = float(Wk.min())
        im, ir = np.unravel_index(np.argmin(Wk), Wk.shape)
        # per magnitude: best ratio and its W1; per ratio: spread over magnitudes
        best_ratio = [dict(magnitude=mags[m], ratio=ratios[int(np.argmin(Wk[m]))],
                           W1=float(Wk[m].min())) for m in range(len(mags))]
        fast = Wk[-1]                       # largest magnitude = fast equilibrium
        hi = [b["W1"] for b in best_ratio if b["magnitude"] >= 0.1]
        ridge[k] = dict(W1=Wk.tolist(), min=j_min,
                        argmin=dict(magnitude=mags[im], ratio=ratios[ir]),
                        best_ratio_per_magnitude=best_ratio,
                        fast_limit=dict(best_ratio=ratios[int(np.argmin(fast))],
                                        W1=float(fast.min())),
                        # slow kinetics (kappa_r <= 0.01) beat the fast limit?
                        slow_kinetics_better=bool(
                            min(b["W1"] for b in best_ratio
                                if b["magnitude"] <= 0.01) < 0.99 * float(fast.min())),
                        # fast-equilibrium ridge: W1 flat (<1 %) for kappa_r >= 0.1
                        ridge_flat_above_0p1=bool(
                            (max(hi) - min(hi)) < 0.01 * min(hi)) if hi else None)
    extras = dict(_meta=dict(ratios=ratios, magnitudes=mags, dt_ridge=1.0,
                             dt_events=e7.DT_PRODUCTION,
                             events_source=str(KAPPA_EVENTS),
                             runtime_s=round(time.time() - t0, 1)),
                  events_field=events_field, ridge=ridge)
    (out / "extras.json").write_text(json.dumps(extras, indent=1, default=_json))
    print(f"[stage e] wrote {out / 'extras.json'} "
          f"({extras['_meta']['runtime_s']} s)", flush=True)
    return extras


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def smoke_settings():
    S = dict(SETTINGS)
    S.update(out=SETTINGS["out"] / "smoke", A_levels=(1, 5),
             kc_grid=np.array([0.05, 0.3]), kr_grid=np.array([1e-3, 1e-2]),
             maxfev=2, shared_kr=np.array([1e-3, 1e-2]),
             shared_kc=np.array([0.05, 0.3]), scalar_maxiter=4,
             none_maxfev=2, both_maxfev=2,
             ridge_ratios=np.array([3.0, 30.0]), ridge_mags=(1e-2, 1.0))
    return S


def main(argv=None):
    ap = argparse.ArgumentParser(description="E12 pure-E7 assertiveness ladder")
    ap.add_argument("--run", action="store_true", help="stage a then b")
    ap.add_argument("--stage", choices=["a", "b", "e"])
    ap.add_argument("--resume", action="store_true",
                    help="stage a: skip A with an existing ladder json")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="reduced program into out/e12/smoke/")
    args = ap.parse_args(argv)
    S = smoke_settings() if args.smoke else SETTINGS
    if args.smoke:
        stage_a(S)
        stage_b(S)
        stage_e(S)
        import e12_figures as fig
        fig.make_figures(S)
        print(fig.write_summary(S))
        return
    if args.run or args.stage == "a":
        stage_a(S, resume=args.resume)
    if args.run or args.stage == "b":
        stage_b(S)
    if args.run or args.stage == "e":
        stage_e(S)
    if args.figures:
        import e12_figures as fig
        fig.make_figures(S)
    if args.summary:
        import e12_figures as fig
        print(fig.write_summary(S))
    if not any((args.run, args.stage, args.figures, args.summary)):
        ap.error("choose --run, --stage a|b, --figures, --summary or --smoke")


if __name__ == "__main__":
    main()
