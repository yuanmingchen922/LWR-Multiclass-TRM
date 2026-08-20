"""E7a: 1-Wasserstein calibration infrastructure for the catch & release
two-class LWR model vs SUMO moving-bottleneck data.

Mladen's suggestion: compare CUMULATIVE densities, not densities.  For each
time t build the unnormalized cumulative count

    N(x, t) = integral_0^x rho(x', t) dx'          [veh]

on the 100 m data grid and score

    W1(t) = integral_0^{x_max} |N_sim - N_data| dx  [veh km].

For equal masses this is exactly the 1-Wasserstein (earth-mover) distance,
which penalizes a shock in the wrong PLACE proportionally to (mass moved x
displacement) instead of rewarding smeared-out compromise fields the way
pointwise RMSE does.  The cumulative is deliberately NOT normalized: a mass
mismatch dM leaves |N_sim - N_data| = |dM| over the rest of the domain, so
missing/excess vehicles are charged |dM| x (distance to the domain edge).

Public API
----------
  w1_series(rho_sim, rho_data, dx_km=0.1, x_max_cells=200) -> (n_t,) [veh km]
  w1_mean  (rho_sim, rho_data, tt, t_win=(100, 1000), ...)  -> float
  rmse_mean(rho_sim, rho_data, tt, t_win=(100, 1000), ...)  -> float
      (identical signature, for side-by-side reporting; dx_km is ignored)
  load_rho_mean(A, uc, qin) -> (tt, rho_mean)  rep-mean measured field
      A in {1, 10}: via ev4_compare.load_measured (needs out/e1 npz);
      other A     : ../Second .mat directly via loader (fields only).
  run_sim(uc, qin, kc, kr, form, gamma=None, ws_frac=None, dt=1.0)
      -> ev4_compare.regrid_sim dict.  The E7 structural knobs are passed to
      solver.SimConfig ONLY when requested AND the (parallel-built) solver
      already exposes them; with all knobs None the call is bit-identical to
      the legacy solver, so this module never hard-requires the new fields.
  fit_field(A, uc, qin, form='lf', metric='w1', extra=None, dt_fit=1.0)
      -> dict.  Fits (kappa_c, kappa_r) by log-space 6x6 coarse grid +
      Nelder-Mead (maxfev 40) against the REP-MEAN measured field; optional
      extra structural parameters join the optimization vector:
          extra={'gamma': (lo, hi)}    capture localization ell = a + gamma s
          extra={'ws_frac': (lo, hi)}  stuck-class wave speed w_s = frac * w
                                       (P_s stays None = legacy jam density;
                                       the solver asserts w_s <= w, so the
                                       caller must keep hi <= 1)
      Fitting sims run at dt=1.0 / save_every=10 (CFL 0.56, allowed for
      coarse fitting); the returned dict re-evaluates the optimum at the
      production resolution dt=0.5 / save_every=20 and reports BOTH metrics
      at BOTH resolutions.

CLI
---
  python3 e7_wasserstein.py --selftest
      MANDATORY synthetic gate: (1) two rectangular blobs of equal mass
      M*width offset by d km  ->  w1_series == M*width*d within 1% (exact
      for cell-aligned blobs); (2) a mass-mismatch case must grow linearly
      in the remaining domain length with slope |dM|.  Everything else in
      this module is gated on this test (fit_field runs it once per
      process before touching the solver).
  python3 e7_wasserstein.py --smoke
      2-eval smoke of the fit_field mechanics (grid 2x2, maxfev 2) on
      A=1 u15 q2500, plus the extra-parameter guard check.
  python3 e7_wasserstein.py --fit --A 1 --uc 15 --qin 2500 [--metric w1]
      [--form lf] [--extra gamma:0,1 | --extra ws_frac:1,2] [--dt-fit 1.0]
      full calibration -> out/e7/fit_{tag}_{form}_{metric}[_{extra}].json

Units: measured and regridded sim densities in veh/km on the 300 x 100 m
data grid (upper-edge convention), t = (1..100)*10 s; x_max_cells=200 caps
the score at x <= 20 km as in E6/EV4.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT_E7 = HERE / "out" / "e7"
SECOND = HERE.parent / "Second"

T_WIN = (100.0, 1000.0)      # [s]   scoring window (matches E6 / EV4)
DX_KM = 0.1                  # [km]  data cell length
X_MAX_CELLS = 200            # cells: x <= 20 km on the data grid
DT_PRODUCTION = 0.5          # [s]   production resolution (save_every 20)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def w1_series(rho_sim, rho_data, dx_km: float = DX_KM,
              x_max_cells: int = X_MAX_CELLS):
    """Per-time 1-Wasserstein distance between unnormalized cumulatives.

    W1(t) = sum_j |N_sim(j,t) - N_data(j,t)| dx_km  [veh km], with
    N(j,t) = cumsum_j(rho * dx_km) [veh] over the first x_max_cells cells.
    rho_* in veh/km, shape (..., n_x); returns shape (...,) (i.e. (n_t,)
    for field input).  Mass differences persist in N and are charged over
    the remaining domain length (unnormalized on purpose).
    """
    rs = np.asarray(rho_sim, float)[..., :x_max_cells]
    rd = np.asarray(rho_data, float)[..., :x_max_cells]
    if rs.shape != rd.shape:
        raise ValueError(f"shape mismatch: sim {rs.shape} vs data {rd.shape}")
    d_n = np.cumsum(rs - rd, axis=-1) * dx_km          # [veh]
    return np.sum(np.abs(d_n), axis=-1) * dx_km        # [veh km]


def _t_mask(tt, t_win):
    tt = np.asarray(tt, float)
    m = (tt >= t_win[0]) & (tt <= t_win[1])
    if not m.any():
        raise ValueError(f"empty window {t_win} on t grid "
                         f"[{tt[0]:g}, {tt[-1]:g}]")
    return m


def w1_mean(rho_sim, rho_data, tt, t_win=T_WIN, dx_km: float = DX_KM,
            x_max_cells: int = X_MAX_CELLS) -> float:
    """Mean of w1_series over the time window t_win [veh km]."""
    m = _t_mask(tt, t_win)
    return float(np.mean(w1_series(np.asarray(rho_sim, float)[m],
                                   np.asarray(rho_data, float)[m],
                                   dx_km, x_max_cells)))


def rmse_mean(rho_sim, rho_data, tt, t_win=T_WIN, dx_km: float = DX_KM,
              x_max_cells: int = X_MAX_CELLS) -> float:
    """Pointwise density RMSE [veh/km] over the same window/domain (side-by-
    side baseline; dx_km is accepted only for signature parity)."""
    m = _t_mask(tt, t_win)
    d = (np.asarray(rho_sim, float)[m][:, :x_max_cells]
         - np.asarray(rho_data, float)[m][:, :x_max_cells])
    return float(np.sqrt(np.mean(d ** 2)))


METRICS = {"w1": w1_mean, "rmse": rmse_mean}


# ---------------------------------------------------------------------------
# measured data (rep-mean field)
# ---------------------------------------------------------------------------

def load_rho_mean(A, uc, qin):
    """(tt, rho_mean): rep-mean measured density field [veh/km], (n_t, 300).

    A in {1, 10} goes through ev4_compare.load_measured (same arrays the E6
    RMSE fit used; requires out/e1 npz); any other A loads the ../Second
    .mat directly (fields only) and rep-averages the densities.
    """
    if float(A) in (1.0, 10.0):
        import ev4_compare as ev4                     # lazy: needs out/e1
        meas = ev4.load_measured(float(A), float(uc), float(qin))
        return np.asarray(meas["tt"], float), np.mean(meas["rho"], axis=0)

    from loader import load_scenario
    a_tag = (f"{A:g}" if float(A).is_integer()
             else f"{int(round(float(A) * 100))}")   # 1.5 -> data_150_...
    path = SECOND / f"data_{a_tag}_{uc:g}_{qin:g}_True.mat"
    sc = load_scenario(path, fields=True, ctrl=False, trajs=False)
    rhos = [sc.reps[r].rho for r in sorted(sc.reps)
            if sc.reps[r].rho is not None]
    if not rhos:
        raise FileNotFoundError(f"no per-rep density fields in {path}")
    return sc.t_field, np.mean(np.asarray(rhos, float), axis=0)


# ---------------------------------------------------------------------------
# solver bridge (import INSIDE; new SimConfig fields optional)
# ---------------------------------------------------------------------------

def run_sim(uc, qin, kc, kr, form: str = "lf", gamma=None, ws_frac=None,
            dt: float = 1.0, save_every: int | None = None, **extra_cfg):
    """One solver run regridded to the data grid (ev4_compare.regrid_sim).

    gamma / ws_frac are the E7c structural knobs (solver.py extension built
    in parallel): they are forwarded to SimConfig ONLY when not None, and a
    clear RuntimeError is raised if the installed solver does not expose the
    field yet.  ws_frac scales the stuck-class wave speed w_s = ws_frac * w
    and passes P_s=None (legacy jam density); note the solver asserts
    w_s <= w (invariant domain), i.e. ws_frac <= 1.  With both None the
    SimConfig call is exactly the legacy one (bit-compatibility discipline).

    **extra_cfg (E8): FIXED SimConfig fields forwarded verbatim (e.g.
    w_s=0.6*W, downstream_release=True).  Each key must name an existing
    SimConfig field (RuntimeError otherwise) and must not collide with a
    field this function already sets.  Empty extra_cfg is bit-identical to
    the pre-E8 behavior.
    """
    import ev4_compare as ev4
    from solver import SimConfig, simulate

    if save_every is None:
        save_every = int(round(10.0 / dt))    # land on the 10 s data grid
    kw = dict(v_f=ev4.V_F, w=ev4.W, P=ev4.P, q_in=qin / 3600.0, u_xi=uc,
              kappa_c=float(kc), kappa_r=float(kr), capture_form=form,
              dt=dt, save_every=save_every)
    have = {f.name for f in dataclasses.fields(SimConfig)}
    if gamma is not None:
        if "gamma" not in have:
            raise RuntimeError(
                "solver.SimConfig has no 'gamma' field yet (E7 solver "
                "extension not built); run without extra={'gamma': ...}")
        kw["gamma"] = float(gamma)
    if ws_frac is not None:
        if "w_s" not in have:
            raise RuntimeError(
                "solver.SimConfig has no 'w_s' field yet (E7 solver "
                "extension not built); run without extra={'ws_frac': ...}")
        kw["w_s"] = float(ws_frac) * ev4.W
        if "P_s" in have:
            kw["P_s"] = None                  # explicit legacy jam density
    for name, val in extra_cfg.items():       # E8 fixed-knob passthrough
        if name not in have:
            raise RuntimeError(
                f"solver.SimConfig has no '{name}' field (extra_cfg "
                "passthrough targets existing SimConfig fields only)")
        if name in kw:
            raise RuntimeError(
                f"extra_cfg key '{name}' collides with a field run_sim "
                "already sets (use the dedicated argument instead)")
        kw[name] = val
    return ev4.regrid_sim(simulate(SimConfig(**kw)))


# ---------------------------------------------------------------------------
# field calibration
# ---------------------------------------------------------------------------

_EXTRA_NAMES = ("gamma", "ws_frac")
_selftest_passed = False


def _ensure_selftest():
    """Gate: the synthetic W1 check must pass once per process before any
    calibration touches the solver."""
    global _selftest_passed
    if not _selftest_passed:
        selftest(verbose=False)
        _selftest_passed = True


def fit_field(A, uc, qin, form: str = "lf", metric: str = "w1",
              extra: dict | None = None, dt_fit: float = 1.0,
              t_win=T_WIN, x_max_cells: int = X_MAX_CELLS,
              kc_grid=None, kr_grid=None, maxfev: int = 40,
              verbose: bool = True,
              extra_cfg: dict | None = None) -> dict:
    """Fit (kappa_c, kappa_r) [+ extras] to the rep-mean measured field.

    extra_cfg (E8): FIXED SimConfig fields (e.g. w_s=0.6*W,
    downstream_release=True) forwarded verbatim through run_sim to every
    fitting/re-eval sim; they are NOT part of the optimization vector.
    None/empty is bit-identical to the pre-E8 behavior.

    Stage 1: log-space coarse grid over kappa_c x kappa_r (default 6x6),
    extras frozen at their interval midpoints.  Stage 2: Nelder-Mead
    (maxfev 40) on z = [log10 kc, log10 kr, *extras] from the grid optimum;
    extras live in linear space and are clipped to their (lo, hi) bounds
    inside the objective.  The grid optimum is kept if the polish regresses
    (same convention as e6_native_mb.fit_native).

    Fitting sims run at dt_fit (default 1.0 s, save_every 10); the returned
    dict re-evaluates the optimum at dt=0.5 / save_every=20 and reports both
    w1 and rmse at both resolutions.
    """
    from scipy.optimize import minimize

    _ensure_selftest()
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}")
    metric_fn = METRICS[metric]

    extra = dict(extra or {})
    for name in extra:
        if name not in _EXTRA_NAMES:
            raise ValueError(f"unknown extra parameter '{name}' "
                             f"(supported: {_EXTRA_NAMES})")
    extra_names = sorted(extra)
    mid = {n: 0.5 * (extra[n][0] + extra[n][1]) for n in extra_names}

    kc_grid = (np.logspace(-2.5, 0.5, 6) if kc_grid is None
               else np.asarray(kc_grid, float))
    kr_grid = (np.logspace(-4.0, -0.5, 6) if kr_grid is None
               else np.asarray(kr_grid, float))

    tt_data, rho_mean = load_rho_mean(A, uc, qin)
    n_sim = 0

    def sim(kc, kr, extras, dt):
        nonlocal n_sim
        regr = run_sim(uc, qin, kc, kr, form,
                       gamma=extras.get("gamma"),
                       ws_frac=extras.get("ws_frac"), dt=dt,
                       **(extra_cfg or {}))
        n_sim += 1
        assert np.allclose(regr["tt"], tt_data), "sim/data time grids differ"
        return regr

    def score(regr, fn):
        return fn(regr["rho_tot"], rho_mean, tt_data, t_win=t_win,
                  x_max_cells=x_max_cells)

    def obj(kc, kr, extras):
        return score(sim(kc, kr, extras, dt_fit), metric_fn)

    # ---- stage 1: coarse log grid ----------------------------------------
    best = (np.inf, None, None)
    for kc in kc_grid:
        for kr in kr_grid:
            j = obj(kc, kr, mid)
            if j < best[0]:
                best = (j, float(kc), float(kr))
    j_grid, kc0, kr0 = best
    if verbose:
        print(f"[grid {len(kc_grid)}x{len(kr_grid)}] {metric}={j_grid:.3f} "
              f"at kappa_c={kc0:.3e} kappa_r={kr0:.3e} extras={mid}")

    # ---- stage 2: Nelder-Mead polish --------------------------------------
    def unpack(z):
        kc, kr = 10.0 ** z[0], 10.0 ** z[1]
        extras = {n: float(np.clip(z[2 + i], *extra[n]))
                  for i, n in enumerate(extra_names)}
        return kc, kr, extras

    x0 = [np.log10(kc0), np.log10(kr0)] + [mid[n] for n in extra_names]
    res = minimize(lambda z: obj(*unpack(z)), x0, method="Nelder-Mead",
                   options=dict(maxfev=maxfev, xatol=1e-3, fatol=1e-3))
    kc, kr, extras = unpack(res.x)
    j_fit = float(res.fun)
    if j_grid < j_fit:                        # polish regressed: keep grid
        kc, kr, extras, j_fit = kc0, kr0, dict(mid), j_grid
    if verbose:
        print(f"[polish nfev={res.nfev}] {metric}={j_fit:.3f} at "
              f"kappa_c={kc:.3e} kappa_r={kr:.3e} extras={extras}")

    # ---- re-evaluate the optimum at both resolutions ----------------------
    def both(regr):
        return {m: score(regr, METRICS[m]) for m in ("w1", "rmse")}

    at_fit = dict(dt=dt_fit, save_every=int(round(10.0 / dt_fit)),
                  **both(sim(kc, kr, extras, dt_fit)))
    at_prod = dict(dt=DT_PRODUCTION,
                   save_every=int(round(10.0 / DT_PRODUCTION)),
                   **both(sim(kc, kr, extras, DT_PRODUCTION)))
    if verbose:
        print(f"[re-eval] dt={dt_fit:g}: w1={at_fit['w1']:.3f} "
              f"rmse={at_fit['rmse']:.3f} | dt={DT_PRODUCTION:g}: "
              f"w1={at_prod['w1']:.3f} rmse={at_prod['rmse']:.3f} "
              f"({n_sim} sims total)")

    return dict(A=float(A), uc=float(uc), qin=float(qin), form=form,
                metric=metric, kappa_c=float(kc), kappa_r=float(kr),
                extra={n: float(extras[n]) for n in extra_names},
                extra_bounds={n: list(map(float, extra[n]))
                              for n in extra_names},
                extra_cfg=dict(extra_cfg or {}),
                objective=j_fit, grid_objective=float(j_grid),
                at_dt_fit=at_fit, at_production=at_prod,
                n_sim=n_sim, n_polish=int(res.nfev),
                t_win=[float(t_win[0]), float(t_win[1])],
                x_max_cells=int(x_max_cells),
                data_source=("ev4_compare.load_measured (out/e1)"
                             if float(A) in (1.0, 10.0)
                             else "loader.load_scenario (../Second, rep-mean)"))


# ---------------------------------------------------------------------------
# MANDATORY synthetic selftest
# ---------------------------------------------------------------------------

def selftest(verbose: bool = True):
    """Gate test for the W1 machinery (pure numpy, no solver/data needed).

    (1) Equal-mass shifted rectangles: rho = M veh/km over `wd` cells vs the
        same blob shifted by `sh` cells.  W1 must equal M * width * shift
        [veh km] within 1% (exact for cell-aligned blobs).
    (2) Mass mismatch dM over the same support: W1(x_max) must grow linearly
        in the remaining domain length with slope |dM| [veh], and match the
        closed-form triangle+plateau value.
    (3) w1_mean / rmse_mean window masking on a synthetic time grid.
    """
    dx, nx = 0.1, 200                 # data-grid geometry (0.1 km cells)
    M, wd, a0 = 60.0, 20, 30          # blob: 60 veh/km over 2 km from 3 km

    # (1) equal mass, shifted -----------------------------------------------
    shifts = np.array([3, 7, 12])     # cells -> 0.3 / 0.7 / 1.2 km
    ref = np.zeros((len(shifts), nx))
    shf = np.zeros_like(ref)
    for i, sh in enumerate(shifts):
        ref[i, a0:a0 + wd] = M
        shf[i, a0 + sh:a0 + sh + wd] = M
    w1 = w1_series(shf, ref, dx_km=dx, x_max_cells=nx)
    assert w1.shape == (len(shifts),), w1.shape
    expect = M * (wd * dx) * (shifts * dx)          # mass [veh] x shift [km]
    rel = np.abs(w1 / expect - 1.0)
    assert np.all(rel < 0.01), (w1, expect)
    # symmetry and self-distance
    assert np.allclose(w1_series(ref, shf, dx_km=dx, x_max_cells=nx), w1)
    assert np.allclose(w1_series(ref, ref, dx_km=dx, x_max_cells=nx), 0.0)

    # (2) mass mismatch: linear growth in remaining domain length -----------
    d_rho = 10.0                                   # [veh/km] density deficit
    dm = d_rho * wd * dx                           # [veh] mass deficit = 20
    r1 = np.zeros(nx); r1[a0:a0 + wd] = M
    r2 = np.zeros(nx); r2[a0:a0 + wd] = M - d_rho
    xms = np.array([100, 140, 180])
    w = np.array([float(w1_series(r1, r2, dx_km=dx, x_max_cells=k))
                  for k in xms])
    slopes = np.diff(w) / (np.diff(xms) * dx)      # [veh]
    assert np.allclose(slopes, dm, rtol=0.01), (slopes, dm)
    # closed form: ramp inside the blob + dM plateau to the domain edge
    w_exact = dx * (d_rho * dx * wd * (wd + 1) / 2.0
                    + (xms - a0 - wd) * dm)
    assert np.allclose(w, w_exact, rtol=1e-9), (w, w_exact)

    # (3) window masking ------------------------------------------------------
    tt = np.array([50.0, 100.0, 500.0, 1000.0, 1100.0])
    sim5 = np.zeros((5, nx)); dat5 = np.zeros((5, nx))
    sim5[0] = 1e6; sim5[4] = 1e6                  # poisoned out-of-window rows
    sim5[1:4] = shf                                # rows with known W1
    dat5[1:4] = ref
    wm = w1_mean(sim5, dat5, tt, t_win=(100.0, 1000.0), dx_km=dx,
                 x_max_cells=nx)
    assert np.isclose(wm, np.mean(expect), rtol=0.01), (wm, np.mean(expect))
    rm = rmse_mean(sim5, dat5, tt, t_win=(100.0, 1000.0), x_max_cells=nx)
    assert rm < 1e5, "rmse_mean leaked out-of-window rows"

    if verbose:
        print("selftest OK:")
        print(f"  shift  : W1 = {np.round(w1, 6)} veh km, expected "
              f"{np.round(expect, 6)} (M*width*d; max rel err "
              f"{rel.max():.2e})")
        print(f"  mass   : W1(x_max={xms}) = {np.round(w, 6)} veh km, "
              f"slope = {np.round(slopes, 6)} veh vs dM = {dm:g} veh "
              f"(closed form matched)")
        print(f"  window : w1_mean = {wm:.6f} veh km == mean shift W1 "
              f"{np.mean(expect):.6f}; rmse_mean windowed OK")
    return True


# ---------------------------------------------------------------------------
# smoke run: fit_field mechanics only (grid 2x2, maxfev 2)
# ---------------------------------------------------------------------------

def smoke():
    """2-eval smoke of fit_field on A=1 u15 q2500 (mechanics, not science)."""
    _ensure_selftest()
    import solver
    have = {f.name for f in dataclasses.fields(solver.SimConfig)}
    new = sorted({"gamma", "w_s", "P_s"} & have)
    print(f"solver.SimConfig E7 fields present: {new or 'none (legacy)'}")

    fit = fit_field(1.0, 15.0, 2500.0, form="lf", metric="w1", dt_fit=1.0,
                    kc_grid=[0.3, 0.6], kr_grid=[0.01, 0.04], maxfev=2)
    print(json.dumps(fit, indent=2))

    if "gamma" not in have:       # extra-param guard must fail loudly
        try:
            run_sim(15.0, 2500.0, 0.5, 0.03, gamma=0.5)
        except RuntimeError as e:
            print(f"extra-param guard OK (legacy solver): {e}")
        else:
            raise AssertionError("gamma accepted by a solver without the "
                                 "field -- guard broken")
    return fit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_extra(spec):
    """'gamma:0,1' -> {'gamma': (0.0, 1.0)}; None -> None."""
    if spec is None:
        return None
    name, _, rng = spec.partition(":")
    lo, hi = (float(v) for v in rng.split(","))
    return {name.strip(): (lo, hi)}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="E7a 1-Wasserstein calibration (cumulative densities)")
    ap.add_argument("--selftest", action="store_true",
                    help="mandatory synthetic W1 gate (pure numpy)")
    ap.add_argument("--smoke", action="store_true",
                    help="fit_field mechanics smoke: grid 2x2, maxfev 2, "
                         "A=1 u15 q2500")
    ap.add_argument("--fit", action="store_true",
                    help="full calibration -> out/e7/fit_*.json")
    ap.add_argument("--A", type=float, default=1.0)
    ap.add_argument("--uc", type=float, default=15.0)
    ap.add_argument("--qin", type=float, default=2500.0)
    ap.add_argument("--form", choices=("lf", "af"), default="lf")
    ap.add_argument("--metric", choices=tuple(METRICS), default="w1")
    ap.add_argument("--extra", default=None, metavar="NAME:LO,HI",
                    help="extra structural parameter, e.g. gamma:0,1 or "
                         "ws_frac:0.5,1 (solver asserts w_s <= w)")
    ap.add_argument("--dt-fit", type=float, default=1.0)
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return
    if args.smoke:
        smoke()
        return
    if args.fit:
        extra = _parse_extra(args.extra)
        fit = fit_field(args.A, args.uc, args.qin, form=args.form,
                        metric=args.metric, extra=extra, dt_fit=args.dt_fit)
        OUT_E7.mkdir(parents=True, exist_ok=True)
        tag = f"A{args.A:g}_u{args.uc:g}_q{args.qin:g}"
        suffix = "" if not extra else "_" + "_".join(sorted(extra))
        path = OUT_E7 / f"fit_{tag}_{args.form}_{args.metric}{suffix}.json"
        path.write_text(json.dumps(fit, indent=2))
        print("wrote", path)
        return
    ap.error("choose one of --selftest / --smoke / --fit")


if __name__ == "__main__":
    main()
