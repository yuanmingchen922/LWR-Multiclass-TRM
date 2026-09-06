"""E11: best instantiation of the general model per assertiveness + the
main paper figures (Figs 3-5 and the metrics table).

Model = the E10 general structure (NO downstream_release, NO eta_la):
    stuck-class congested branch w_s = ws_frac * w
    + s-impermeable bottleneck interface (only free vehicles overtake)
    + catch & release, lf form with the gamma override ell = a + gamma s,
    coefficients (kappa_c, kappa_r)
    [+ optionally the DM-G capacity cap Q_xi; the paper's FINAL set drops it]

Step A -- capped tuning (per assertiveness A in {1, 10}), u15 q2500 rep-mean:
    theta = (log10 kappa_c, log10 kappa_r, gamma, ws_frac, Q_xi [veh/h])
    box     [-3, 0] x [-6, -1] x [0, 1] x [0.4, 1.0] x [1400, 2600]
    J(theta) = W1 [veh km]  (e7_wasserstein.w1_mean, t in [100, 1000] s,
                             x <= 20 km)
             + 50 * max(0, ds_stuck - 0.01)
    ds_stuck = (int s dx)/(int rho dx) over data cells strictly downstream
    of the CAV at t = 700 s (e8_ladder.ds_stuck_fraction).
    Degenerate guard: N_s(700 s) < 1 veh -> J += 1e3 (never accepted).
    Search at dt = 1 s (save_every 10): Nelder-Mead from the E10 point
    (E10 kappas, gamma = 1, ws_frac = 0.6, Q_xi = 2000; maxfev 150), then
    two random restarts inside the box (seeded; maxfev 80 each); the best
    non-degenerate point is kept and re-evaluated at dt = 0.5 s.
    -> tuned_config.json[A]['capped_reference']

Step B -- no-cap ablation (same protocol, 4 parameters, q_xi_max = None):
    theta = (log10 kappa_c, log10 kappa_r, gamma, ws_frac), NM from the
    capped tuned point (maxfev 150) + 2 restarts (maxfev 80), dt = 1 s
    search, dt = 0.5 s final.  Both sets are scored on the fit scenario and
    the three zero-refit transfers; the no-cap set is ADOPTED as the paper's
    final model when its fit-scenario W1 is within NOCAP_TOL (1.5 %) of the
    capped one.  -> tuned_config.json[A]['nocap'], ['ablation'], ['final']

Evaluation (zero refit) on u15 q2500 (fit), u15 q2000, u20 q2000, u20 q2500
for the classical M1 (kappa = 0, cap 2000 veh/h, no w_s, permeable
interface) and the adopted final set: W1, RMSE, e_s, omega_err, wake,
ds_stuck, plus per-rep e_s and rho_rmse.

Outputs (out/paper/): tuned_config.json, table_metrics.md,
fig_heatmaps_q2500, fig_heatmaps_q2000, fig_es, fig_profiles (.png + .pdf).
The paper captions live in out/paper/captions.md (hand-written from the
JSON numbers).

CLI:  python3 e11_tune.py --run       full program (steps A + B + outputs)
      python3 e11_tune.py --ablate    step B from an existing
                                      tuned_config.json (capped results
                                      kept), then outputs
      python3 e11_tune.py --figures   figures + table from the 'final' set
                                      of an existing tuned_config.json
      python3 e11_tune.py --summary   compact console table (8 scenarios +
                                      the A = 3 transfer sweep)
      python3 e11_tune.py --smoke     tiny budgets (mechanics only; writes
                                      nothing under out/paper)
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import numpy as np
from matplotlib.ticker import PercentFormatter
from scipy.optimize import minimize

import e7_ablation as e7a
import e7_wasserstein as e7
import e8_ladder as L
import ev4_compare as ev4
import paperfig as pf
from loader import CELL_LEN, N_CELL

HERE = Path(__file__).parent
OUT = HERE / "out" / "paper"
E10_CONFIG = HERE / "out" / "e10" / "final_config.json"
TRANSFER_JSON = OUT / "transfer.json"

FORM = "lf"
UC_FIT, QIN_FIT = 15.0, 2500.0
QXI0 = 2000.0                    # [veh/h] classical / E10 capacity
WS0, GAMMA0 = 0.6, 1.0           # E10 point
RHO_CRIT = 48.5                  # [veh/km]
T_GUARD = 700.0                  # [s] degenerate guard / ds_stuck snapshot
DS_TOL, DS_WEIGHT = 0.01, 50.0   # penalty 50 * max(0, ds_stuck - 0.01)
DEGEN_PENALTY = 1e3
NS_MIN = 1.0                     # [veh]

BOX = dict(log_kc=(-3.0, 0.0), log_kr=(-6.0, -1.0), gamma=(0.0, 1.0),
           ws_frac=(0.4, 1.0), Q_xi=(1400.0, 2600.0))
NAMES = list(BOX)                                  # 5-parameter capped set
NAMES_NOCAP = [n for n in NAMES if n != "Q_xi"]    # 4-parameter set, no cap
NOCAP_TOL = 0.015                # adopt the no-cap set if fit W1 within 1.5 %
SCENARIOS = [(15.0, 2500.0), (15.0, 2000.0), (20.0, 2000.0), (20.0, 2500.0)]
ES_ORDER = [(15.0, 2000.0), (15.0, 2500.0), (20.0, 2000.0), (20.0, 2500.0)]
MODELS = ("classical", "final")
MODEL_LABEL = dict(classical="classical LWR+MB", final="catch & release")
DATA_LABEL = "SUMO (5-run mean)"
PROFILE_T = (500.0, 700.0, 850.0)
X_UP_KM = (np.arange(N_CELL) + 1) * CELL_LEN / 1000.0


# ---------------------------------------------------------------------------
# parameter box <-> unit cube (the parameter set is defined by theta's keys)
# ---------------------------------------------------------------------------

def names_of(theta):
    return [n for n in NAMES if n in theta]


def pack(theta):
    return np.array([(theta[n] - BOX[n][0]) / (BOX[n][1] - BOX[n][0])
                     for n in names_of(theta)])


def unpack(z, names=NAMES):
    z = np.clip(np.asarray(z, float), 0.0, 1.0)
    return {n: float(BOX[n][0] + z[i] * (BOX[n][1] - BOX[n][0]))
            for i, n in enumerate(names)}


def q_xi_max_of(theta):
    """SimConfig q_xi_max [veh/s] of a parameter set; None = no cap."""
    return theta["Q_xi"] / 3600.0 if theta.get("Q_xi") is not None else None


def theta_public(theta):
    """Human units: kappas linear, Q_xi in veh/h (None = no cap), w_s in m/s."""
    capped = theta.get("Q_xi") is not None
    return dict(kappa_c=10.0 ** theta["log_kc"],
                kappa_r=10.0 ** theta["log_kr"],
                gamma=theta["gamma"], ws_frac=theta["ws_frac"],
                w_s_ms=theta["ws_frac"] * ev4.W,
                Q_xi_vehh=(float(theta["Q_xi"]) if capped else None),
                capacity_cap=capped, n_params=len(names_of(theta)),
                s_impermeable=True, capture_form=FORM)


def fmt_theta(th):
    q = "none" if th["Q_xi_vehh"] is None else f"{th['Q_xi_vehh']:.0f}"
    return (f"kc={th['kappa_c']:.3e} kr={th['kappa_r']:.3e} "
            f"gamma={th['gamma']:.3f} ws={th['ws_frac']:.3f} Qxi={q}")


def e10_theta(A):
    e10 = json.loads(E10_CONFIG.read_text())[f"A{A:g}"]
    return dict(log_kc=float(np.log10(e10["kappa_c"])),
                log_kr=float(np.log10(e10["kappa_r"])),
                gamma=GAMMA0, ws_frac=WS0, Q_xi=QXI0)


def initial_simplex(z0, step=0.15):
    n = len(z0)
    S = np.tile(np.asarray(z0, float), (n + 1, 1))
    for i in range(n):
        S[i + 1, i] = z0[i] + step if z0[i] + step <= 1.0 else z0[i] - step
    return S


# ---------------------------------------------------------------------------
# simulation + scoring
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def data(A, uc, qin):
    tt, rho_mean = e7.load_rho_mean(A, uc, qin)
    meas = ev4.load_measured(A, uc, qin)
    return tt, rho_mean, meas


def _snap(tt, t):
    return int(np.argmin(np.abs(np.asarray(tt, float) - t)))


def sim_theta(uc, qin, theta, dt):
    return e7.run_sim(uc, qin, 10.0 ** theta["log_kc"], 10.0 ** theta["log_kr"],
                      FORM, gamma=theta["gamma"], ws_frac=theta["ws_frac"],
                      dt=dt, q_xi_max=q_xi_max_of(theta), s_impermeable=True)


def objective_parts(regr, rho_mean, tt):
    w1 = e7.w1_mean(regr["rho_tot"], rho_mean, tt)
    ds = L.ds_stuck_fraction(regr, T_GUARD)["frac"]
    ds = 0.0 if ds is None else float(ds)
    ns = float(regr["N_s"][_snap(tt, T_GUARD)])
    pen = DS_WEIGHT * max(0.0, ds - DS_TOL)
    degenerate = bool(ns < NS_MIN)
    return dict(W1=w1, ds_stuck=ds, N_s700=ns, penalty=pen,
                J=w1 + pen, degenerate=degenerate,
                J_search=w1 + pen + (DEGEN_PENALTY if degenerate else 0.0))


def spec_of(theta):
    """run_sim arguments of a catch & release parameter set."""
    return dict(kc=10.0 ** theta["log_kc"], kr=10.0 ** theta["log_kr"],
                gamma=theta["gamma"], ws_frac=theta["ws_frac"],
                cfg=dict(q_xi_max=q_xi_max_of(theta), s_impermeable=True))


def model_specs(theta):
    """run_sim arguments of the two compared models."""
    return dict(
        classical=dict(kc=0.0, kr=0.0, gamma=None, ws_frac=None,
                       cfg=dict(q_xi_max=QXI0 / 3600.0)),
        final=spec_of(theta))


def evaluate(A, uc, qin, spec, dt=e7.DT_PRODUCTION):
    """Production-resolution run of one model on one scenario -> (summary,
    regr).  Summary: W1, RMSE (rep-mean field), e_s / rho_rmse (mean and
    per rep), omega_err, wake, ds_stuck(700 s), N_s(700 s)."""
    regr = e7.run_sim(uc, qin, spec["kc"], spec["kr"], FORM,
                      gamma=spec["gamma"], ws_frac=spec["ws_frac"], dt=dt,
                      **spec["cfg"])
    tt, rho_mean, meas = data(A, uc, qin)
    met = ev4.metrics(regr, meas)
    parts = objective_parts(regr, rho_mean, tt)
    summ = dict(
        W1=parts["W1"],
        RMSE=e7.rmse_mean(regr["rho_tot"], rho_mean, tt),
        e_s=met["e_s"]["mean"], e_s_per_rep=met["e_s"]["per_rep"],
        rho_rmse=met["rho_rmse"]["mean"],
        rho_rmse_per_rep=met["rho_rmse"]["per_rep"],
        omega_err=met["omega_cum_rel_err"]["mean"],
        omega_err_per_rep=met["omega_cum_rel_err"]["per_rep"],
        Ns_mae=met["Ns_mae"]["mean"],
        wake=e7a.d1_wake(regr["rho_tot"], tt, regr["x_cav"])["wake_mean_vehkm"],
        ds_stuck=parts["ds_stuck"], N_s700=parts["N_s700"],
        penalty=parts["penalty"], J=parts["J"],
        degenerate=parts["degenerate"],
        peak_omega_vehh=float(np.max(regr["omega"])),
        cum_overtake_sim_veh=met["cum_overtake_sim_veh"],
        es_window_s=met["es_window_s"])
    return summ, regr


# ---------------------------------------------------------------------------
# tuning (parameter set = `names`; start point = `theta0`)
# ---------------------------------------------------------------------------

def tune(A, maxfev_main=150, maxfev_restart=80, n_restart=2, seed=0,
         dt_search=1.0, verbose=True, names=None, theta0=None,
         start_label="E10 point"):
    names = list(names or NAMES)
    tt, rho_mean, _ = data(A, UC_FIT, QIN_FIT)
    trace = []

    def f(z):
        theta = unpack(z, names)
        parts = objective_parts(sim_theta(UC_FIT, QIN_FIT, theta, dt_search),
                                rho_mean, tt)
        trace.append(dict(theta=theta, **parts))
        return parts["J_search"]

    runs = []

    def run_nm(z0, maxfev, label):
        i0 = len(trace)
        res = minimize(f, z0, method="Nelder-Mead",
                       options=dict(maxfev=maxfev, xatol=1e-4, fatol=1e-3,
                                    initial_simplex=initial_simplex(z0)))
        seg = trace[i0:]
        ok = [e for e in seg if not e["degenerate"]]
        best = min(ok, key=lambda e: e["J"]) if ok else None
        n_deg = sum(e["degenerate"] for e in seg)
        info = dict(label=label, start=unpack(z0, names), nfev=int(res.nfev),
                    n_degenerate=int(n_deg),
                    best=(None if best is None else
                          dict(theta=best["theta"], J=best["J"],
                               W1=best["W1"], ds_stuck=best["ds_stuck"],
                               N_s700=best["N_s700"])))
        runs.append(info)
        if verbose:
            if best is None:
                print(f"  [{label}] nfev={res.nfev}: ALL {n_deg} evaluations "
                      "degenerate (N_s(700) < 1 veh)", flush=True)
            else:
                print(f"  [{label}] nfev={res.nfev} degenerate={n_deg}: "
                      f"J={best['J']:.2f} W1={best['W1']:.2f} "
                      f"ds={best['ds_stuck']:.4f} N_s={best['N_s700']:.1f} | "
                      + fmt_theta(theta_public(best["theta"])), flush=True)
        return best

    t0 = time.time()
    theta0 = e10_theta(A) if theta0 is None else dict(theta0)
    theta0 = {n: float(theta0[n]) for n in names}
    if verbose:
        print(f"[A={A:g}] {len(names)}-parameter set {names}; start "
              f"({start_label}): " + fmt_theta(theta_public(theta0)), flush=True)
    j0 = f(pack(theta0))
    if verbose:
        print(f"  J(start, dt={dt_search:g}) = {j0:.2f}", flush=True)
    cands = [run_nm(pack(theta0), maxfev_main, f"NM from {start_label}")]
    rng = np.random.default_rng(seed)
    for k in range(n_restart):
        cands.append(run_nm(rng.uniform(0.0, 1.0, len(names)),
                            maxfev_restart, f"restart {k + 1}"))
    cands = [c for c in cands if c is not None]
    if not cands:
        raise RuntimeError(f"A={A:g}: every search run was degenerate")
    best = min(cands, key=lambda c: c["J"])

    # dt = 0.5 re-evaluation of every run optimum (consistency) + the pick
    prod = []
    for c in cands:
        p = objective_parts(sim_theta(UC_FIT, QIN_FIT, c["theta"],
                                      e7.DT_PRODUCTION), rho_mean, tt)
        prod.append(dict(theta=c["theta"], J_dt1=c["J"], J_dt05=p["J"],
                         W1_dt05=p["W1"], ds_stuck_dt05=p["ds_stuck"],
                         N_s700_dt05=p["N_s700"], degenerate_dt05=p["degenerate"]))
    pick = next(p for p in prod if p["theta"] is best["theta"])
    if pick["degenerate_dt05"]:
        raise RuntimeError(f"A={A:g}: picked optimum is degenerate at dt=0.5")
    deg = [e for e in trace if e["degenerate"]]
    best_deg = min(deg, key=lambda e: e["W1"]) if deg else None
    out = dict(
        names=names, capped=("Q_xi" in names), n_params=len(names),
        theta=best["theta"], theta_public=theta_public(best["theta"]),
        objective_dt1=best["J"], objective_dt05=pick["J_dt05"],
        W1_dt1=best["W1"], W1_dt05=pick["W1_dt05"],
        start=dict(label=start_label, theta=theta0,
                   theta_public=theta_public(theta0), J_dt1=j0),
        runs=runs, run_optima_at_dt05=prod,
        n_eval=len(trace), n_degenerate=len(deg),
        best_degenerate=(None if best_deg is None else
                         dict(theta=best_deg["theta"], W1=best_deg["W1"],
                              N_s700=best_deg["N_s700"])),
        seed=seed, dt_search=dt_search,
        budgets=dict(maxfev_main=maxfev_main, maxfev_restart=maxfev_restart,
                     n_restart=n_restart),
        runtime_s=round(time.time() - t0, 1),
        trace=[dict(theta={k: round(v, 5) for k, v in e["theta"].items()},
                    J=round(e["J"], 3), W1=round(e["W1"], 3),
                    ds=round(e["ds_stuck"], 5), Ns=round(e["N_s700"], 2))
               for e in trace])
    if verbose:
        print(f"[A={A:g}] TUNED ({len(names)} params): "
              + fmt_theta(out["theta_public"])
              + f" | J dt=1 {best['J']:.2f} -> dt=0.5 {pick['J_dt05']:.2f} "
              f"(W1 {pick['W1_dt05']:.2f}); {len(trace)} evals, "
              f"{len(deg)} degenerate, {out['runtime_s']} s", flush=True)
    return out


# ---------------------------------------------------------------------------
# post-tuning diagnostics (fit scenario): is the DM-G cap still active?
# how well is each parameter identified?
# ---------------------------------------------------------------------------

def cap_activity(A, theta):
    """DM-G cap activity at a capped tuned point (u15 q2500, dt = 0.5): the
    moving-frame bound omega_max = (Q_xi - u_xi sigma_xi)_+ [veh/h] with
    sigma_xi = beta w P / (v_f + w), the peak overtaking flow of the SAME
    theta with the cap removed (q_xi_max = None), and the W1 / e_s /
    omega_err change when the cap is removed.  max_abs_drho = 0 means the
    cap never engaged."""
    tt, rho_mean, meas = data(A, UC_FIT, QIN_FIT)
    kc, kr = 10.0 ** theta["log_kc"], 10.0 ** theta["log_kr"]
    kw = dict(gamma=theta["gamma"], ws_frac=theta["ws_frac"],
              dt=e7.DT_PRODUCTION, s_impermeable=True)
    r_cap = e7.run_sim(UC_FIT, QIN_FIT, kc, kr, FORM,
                       q_xi_max=theta["Q_xi"] / 3600.0, **kw)
    r_no = e7.run_sim(UC_FIT, QIN_FIT, kc, kr, FORM, q_xi_max=None, **kw)
    sigma = 0.5 * ev4.W * ev4.P / (ev4.V_F + ev4.W)            # beta = 0.5
    om_max = max(theta["Q_xi"] / 3600.0 - UC_FIT * sigma, 0.0) * 3600.0
    met_no = ev4.metrics(r_no, meas)
    d = float(np.max(np.abs(r_cap["rho_tot"] - r_no["rho_tot"])))
    return dict(omega_max_vehh=om_max,
                peak_omega_uncapped_vehh=float(r_no["omega"].max()),
                peak_omega_capped_vehh=float(r_cap["omega"].max()),
                W1_nocap=e7.w1_mean(r_no["rho_tot"], rho_mean, tt),
                e_s_nocap=met_no["e_s"]["mean"],
                omega_err_nocap=met_no["omega_cum_rel_err"]["mean"],
                max_abs_drho_vehkm=d, cap_active=bool(d > 0.0))


def sensitivity(A, theta, frac=0.1, dt=1.0):
    """dJ for a +-frac step (of the box width) in each parameter, others
    fixed, at the search resolution."""
    tt, rho_mean, _ = data(A, UC_FIT, QIN_FIT)
    names = names_of(theta)
    z0 = pack(theta)
    j0 = objective_parts(sim_theta(UC_FIT, QIN_FIT, unpack(z0, names), dt),
                         rho_mean, tt)["J"]
    out = dict(step_frac_of_box=frac, J0=j0)
    for i, n in enumerate(names):
        dj = []
        for sgn in (-1.0, 1.0):
            z = z0.copy()
            z[i] = float(np.clip(z[i] + sgn * frac, 0.0, 1.0))
            dj.append(objective_parts(sim_theta(UC_FIT, QIN_FIT,
                                                unpack(z, names), dt),
                                      rho_mean, tt)["J"] - j0)
        out[n] = dict(minus=dj[0], plus=dj[1])
    return out


# ---------------------------------------------------------------------------
# no-cap ablation: capped (5) vs no-cap (4) parameter set on the four
# scenarios, adoption rule
# ---------------------------------------------------------------------------

ABL_KEYS = ("W1", "RMSE", "e_s", "omega_err", "wake", "ds_stuck",
            "peak_omega_vehh")


def compare_sets(A, capped_theta, nocap_theta, verbose=True):
    rows = {}
    for uc, qin in SCENARIOS:
        tag = f"A{A:g}_u{uc:g}_q{qin:g}"
        row = dict(tag=tag, uc=uc, qin=qin,
                   fit=(uc == UC_FIT and qin == QIN_FIT))
        for name, th in (("capped", capped_theta), ("nocap", nocap_theta)):
            summ, _ = evaluate(A, uc, qin, spec_of(th))
            row[name] = {k: summ[k] for k in ABL_KEYS}
        rows[tag] = row
        if verbose:
            c, n = row["capped"], row["nocap"]
            print(f"  {tag:15s} capped W1={c['W1']:6.1f} RMSE={c['RMSE']:5.2f} "
                  f"e_s={c['e_s']:+6.1%} om={c['omega_err']:+6.1%} | "
                  f"no cap W1={n['W1']:6.1f} RMSE={n['RMSE']:5.2f} "
                  f"e_s={n['e_s']:+6.1%} om={n['omega_err']:+6.1%}",
                  flush=True)
    fit_tag = f"A{A:g}_u{UC_FIT:g}_q{QIN_FIT:g}"
    w_c = rows[fit_tag]["capped"]["W1"]
    w_n = rows[fit_tag]["nocap"]["W1"]
    rel = (w_n - w_c) / w_c
    return dict(scenarios=rows, fit_W1_capped=w_c, fit_W1_nocap=w_n,
                fit_W1_rel_diff=rel, tol=NOCAP_TOL,
                rule=(f"adopt the {len(NAMES_NOCAP)}-parameter no-cap set if "
                      f"(W1_nocap - W1_capped)/W1_capped <= {NOCAP_TOL:g} at "
                      "the fit scenario (dt = 0.5 s)"),
                adopt_nocap=bool(rel <= NOCAP_TOL))


def final_block(capped, nocap, abl):
    adopted = "nocap" if abl["adopt_nocap"] else "capped"
    src = nocap if adopted == "nocap" else capped
    th = theta_public(src["theta"])
    cap_txt = ("no capacity cap (q_xi_max = None)" if not th["capacity_cap"]
               else f"DM-G capacity cap Q_xi = {th['Q_xi_vehh']:.0f} veh/h")
    return dict(
        adopted=adopted, n_params=src["n_params"], parameters=list(src["names"]),
        theta=src["theta"], theta_public=th,
        W1_dt05=src["W1_dt05"], objective_dt05=src["objective_dt05"],
        definition=(f"catch & release ({FORM} form, ell = a + gamma s) with "
                    f"gamma = {th['gamma']:.3f}, stuck-class branch w_s = "
                    f"{th['ws_frac']:.3f} w, s-impermeable bottleneck "
                    f"interface, {cap_txt}; kappa_c = {th['kappa_c']:.3e}, "
                    f"kappa_r = {th['kappa_r']:.3e} veh^-1; "
                    f"{src['n_params']} fitted parameters"))


def ablate(results, budgets, verbose=True):
    """Step B: no-cap refit per A from the capped tuned point, comparison
    on the four scenarios, adoption, evaluation of the adopted set."""
    stores = {}
    for A in (1.0, 10.0):
        blk = results[f"A{A:g}"]
        capped = blk.get("capped_reference") or blk.pop("tuned")
        blk.pop("eval", None)
        print(f"\n===== no-cap ablation A = {A:g} =====", flush=True)
        theta0 = {n: capped["theta"][n] for n in NAMES_NOCAP}
        nocap = tune(A, names=NAMES_NOCAP, theta0=theta0,
                     start_label="capped tuned point", verbose=verbose,
                     **budgets)
        nocap["sensitivity"] = sensitivity(A, nocap["theta"])
        if verbose:
            print("[A=%g] dJ (+-10%% box): " % A + ", ".join(
                f"{n} {nocap['sensitivity'][n]['minus']:+.1f}/"
                f"{nocap['sensitivity'][n]['plus']:+.1f}"
                for n in NAMES_NOCAP), flush=True)
            print(f"----- capped (5 params) vs no cap (4 params), dt = 0.5 "
                  "-----", flush=True)
        abl = compare_sets(A, capped["theta"], nocap["theta"], verbose)
        final = final_block(capped, nocap, abl)
        print(f"[A={A:g}] fit W1 capped {abl['fit_W1_capped']:.2f} vs no cap "
              f"{abl['fit_W1_nocap']:.2f} ({abl['fit_W1_rel_diff']:+.2%}, "
              f"tol {NOCAP_TOL:.1%}) -> ADOPT "
              f"{'no-cap (4 params)' if abl['adopt_nocap'] else 'capped (5 params)'}",
              flush=True)
        blk.clear()
        blk.update(capped_reference=capped, nocap=nocap, ablation=abl,
                   final=final)
        if verbose:
            print(f"----- evaluating A = {A:g}, adopted set (dt = 0.5) -----",
                  flush=True)
        table, store = evaluate_all(A, final["theta"], verbose=verbose)
        blk["eval"] = table
        stores[A] = store
    results["_meta"]["final_model"] = {
        f"A{A:g}": results[f"A{A:g}"]["final"]["definition"] for A in (1.0, 10.0)}
    return stores


# ---------------------------------------------------------------------------
# evaluation of the two models on the four scenarios
# ---------------------------------------------------------------------------

def evaluate_all(A, theta, verbose=True):
    specs = model_specs(theta)
    store, table = {}, {}
    for uc, qin in SCENARIOS:
        tag = f"A{A:g}_u{uc:g}_q{qin:g}"
        tt, rho_mean, meas = data(A, uc, qin)
        row = dict(tag=tag, uc=uc, qin=qin, fit=(uc == UC_FIT and qin == QIN_FIT),
                   data_wake=e7a.d1_wake(rho_mean, tt, ev4.x_cav_nominal(tt, uc))
                   ["wake_mean_vehkm"])
        regrs = {}
        for m in MODELS:
            summ, regr = evaluate(A, uc, qin, specs[m])
            row[m] = summ
            regrs[m] = regr
            if verbose:
                print(f"  {tag:15s} {m:9s} W1={summ['W1']:6.1f} "
                      f"RMSE={summ['RMSE']:5.2f} e_s={summ['e_s']:+6.1%} "
                      f"om={summ['omega_err']:+6.1%} wake={summ['wake']:5.1f} "
                      f"ds={summ['ds_stuck']:.4f} Ns700={summ['N_s700']:5.1f}",
                      flush=True)
        table[tag] = row
        store[(uc, qin)] = dict(meas=meas, tt=tt, rho_mean=rho_mean,
                                regrs=regrs, row=row)
    return table, store


# ---------------------------------------------------------------------------
# figures (paperfig style)
# ---------------------------------------------------------------------------

def _cav_data(meas):
    """Rep-mean measured CAV trajectory [m]; NaN where no rep has the CAV
    on road (before entry / after exit), without the all-NaN warning."""
    xc = np.asarray(meas["x_cav"], float)
    ok = np.isfinite(xc)
    n = ok.sum(axis=0)
    out = np.full(xc.shape[1], np.nan)
    out[n > 0] = np.where(ok, xc, 0.0).sum(axis=0)[n > 0] / n[n > 0]
    return out


def fig_heatmaps(stores, qin, name):
    """2 rows (A=1, A=10) x 3 cols (SUMO 5-run mean, classical, catch &
    release); t in s, panel letters (a)-(f)."""
    fig, axes = pf.figure("double", height=4.3, nrows=2, ncols=3,
                          sharex=True, sharey=True, constrained_layout=True)
    im = None
    letters = "abcdef"
    for i, A in enumerate((1.0, 10.0)):
        st = stores[A][(UC_FIT, qin)]
        tt = st["tt"]
        panels = [(st["rho_mean"], _cav_data(st["meas"]), DATA_LABEL),
                  (st["regrs"]["classical"]["rho_tot"],
                   st["regrs"]["classical"]["x_cav"], MODEL_LABEL["classical"]),
                  (st["regrs"]["final"]["rho_tot"],
                   st["regrs"]["final"]["x_cav"], MODEL_LABEL["final"])]
        for j, (rho, xc, lbl) in enumerate(panels):
            ax = axes[i, j]
            im = ax.imshow(np.asarray(rho, float), origin="lower",
                           aspect="auto", cmap="turbo", vmin=0.0, vmax=90.0,
                           extent=[0.0, N_CELL * CELL_LEN / 1000.0,
                                   0.0, tt[-1]],
                           interpolation="nearest")
            ax.plot(np.asarray(xc, float) / 1000.0, tt, color="white",
                    lw=0.9, ls="-")
            ax.set_xlim(0.0, 20.0)
            ax.set_ylim(0.0, tt[-1])
            ax.set_yticks([0, 250, 500, 750, 1000])
            ax.grid(False)
            ax.set_title(f"({letters[3 * i + j]}) A = {A:g}: {lbl}",
                         fontsize=8.5)
            if i == 1:
                ax.set_xlabel("x [km]")
            if j == 0:
                ax.set_ylabel("t [s]")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, pad=0.01)
    cb.set_label(r"$\rho$ [veh/km]")
    return pf.save(fig, name, tight=False)


def fig_es(tables, name="fig_es"):
    """ECC22 Fig. 4 style: e_s per scenario, grouped by A (within a group
    u_xi/q_in = 54/2000, 54/2500, 72/2000, 72/2500); line = mean, shading =
    min-max over the 5 runs; grey +-10 % band; classical dashed with open
    squares, catch & release solid with filled circles."""
    fig, ax = pf.figure("single", height=2.5)
    xpos = {1.0: np.arange(4), 10.0: np.arange(4) + 4.7}
    ax.axhspan(-0.10, 0.10, color="0.5", alpha=0.15, lw=0,
               label=r"$\pm10\,\%$", zorder=0)
    ax.axhline(0.0, color="k", lw=0.6, zorder=1)
    style = dict(
        classical=dict(color=pf.COL["classical"], ls=pf.LS["classical"],
                       marker="s", ms=3.0, mfc="white", zorder=2),
        final=dict(color=pf.COL["model"], ls=pf.LS["model"], marker="o",
                   ms=3.2, zorder=3))
    for m in MODELS:
        for k, A in enumerate((1.0, 10.0)):
            rows = [tables[A][f"A{A:g}_u{uc:g}_q{qin:g}"][m]
                    for uc, qin in ES_ORDER]
            mean = np.array([r["e_s"] for r in rows])
            lo = np.array([min(r["e_s_per_rep"]) for r in rows])
            hi = np.array([max(r["e_s_per_rep"]) for r in rows])
            ax.fill_between(xpos[A], lo, hi, color=style[m]["color"],
                            alpha=0.2, lw=0, zorder=style[m]["zorder"] - 0.5)
            ax.plot(xpos[A], mean, label=(MODEL_LABEL[m] if k == 0 else None),
                    **style[m])
    ax.axvline(3.85, color="0.5", lw=0.6, ls=":")
    ticks = np.concatenate([xpos[1.0], xpos[10.0]])
    lab = [f"{uc * 3.6:.0f}\n{qin:.0f}" for uc, qin in ES_ORDER] * 2
    ax.set_xticks(ticks)
    ax.set_xticklabels(lab, fontsize=7)
    ax.set_xlabel(r"$u_\xi$ [km/h] / $q_{in}$ [veh/h]")
    ax.set_ylabel(r"$e_s$ at $X_q$ = 15 km")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    lo_all = min(min(tables[A][f"A{A:g}_u{uc:g}_q{qin:g}"][m]["e_s_per_rep"])
                 for A in (1.0, 10.0) for uc, qin in SCENARIOS for m in MODELS)
    hi_all = max(max(tables[A][f"A{A:g}_u{uc:g}_q{qin:g}"][m]["e_s_per_rep"])
                 for A in (1.0, 10.0) for uc, qin in SCENARIOS for m in MODELS)
    ymax = max(0.25, hi_all + 0.09)           # room for the group headers
    ymin = min(-0.25, lo_all - 0.05)
    ax.set_ylim(ymin, ymax)
    for A in (1.0, 10.0):
        ax.text(np.mean(xpos[A]), ymax - 0.02 * (ymax - ymin), f"A = {A:g}",
                ha="center", va="top", fontsize=8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              fontsize=7, handlelength=1.8, columnspacing=1.0,
              handletextpad=0.5, borderaxespad=0.0)
    return pf.save(fig, name)


def fig_profiles(stores, name="fig_profiles"):
    """2 cols (A=1, A=10) x 3 rows (t = 500, 700, 850 s), u15 q2500; one
    figure-level legend above the panels."""
    fig, axes = pf.figure("double", height=5.4, nrows=3, ncols=2,
                          sharex=True, sharey="row")
    row_max = np.zeros(len(PROFILE_T))
    for j, A in enumerate((1.0, 10.0)):
        st = stores[A][(UC_FIT, QIN_FIT)]
        tt = st["tt"]
        cl, tu = st["regrs"]["classical"], st["regrs"]["final"]
        for i, t_snap in enumerate(PROFILE_T):
            ax = axes[i, j]
            k = _snap(tt, t_snap)
            ax.fill_between(X_UP_KM, 0.0, tu["s"][k], color=pf.COL["stuck"],
                            alpha=0.3, lw=0, label="stuck class $s$")
            ax.fill_between(X_UP_KM, tu["s"][k], tu["s"][k] + tu["f"][k],
                            color=pf.COL["free"], alpha=0.2, lw=0,
                            label="free class $f$")
            ax.plot(X_UP_KM, st["rho_mean"][k], color=pf.COL["data"],
                    ls=pf.LS["data"], lw=1.3, label=DATA_LABEL)
            ax.plot(X_UP_KM, cl["rho_tot"][k], color=pf.COL["classical"],
                    ls=pf.LS["classical"], lw=1.1,
                    label=MODEL_LABEL["classical"])
            ax.plot(X_UP_KM, tu["rho_tot"][k], color=pf.COL["model"],
                    ls=pf.LS["model"], lw=1.2,
                    label=MODEL_LABEL["final"] + r" $\rho$")
            ax.axhline(RHO_CRIT, color="0.5", ls="--", lw=0.7)
            xc = tu["x_cav"][k] / 1000.0
            if np.isfinite(xc):
                ax.axvline(xc, color=pf.COL["cav"], ls=":", lw=0.8)
            sel = X_UP_KM <= 16.0
            row_max[i] = max(row_max[i], float(st["rho_mean"][k][sel].max()),
                             float(cl["rho_tot"][k][sel].max()),
                             float(tu["rho_tot"][k][sel].max()))
            ax.set_xlim(0.0, 16.0)
            ax.text(0.02, 0.95, f"A = {A:g}, t = {t_snap:.0f} s"
                    + ("  (post-release)" if t_snap > 750.0 else ""),
                    transform=ax.transAxes, ha="left", va="top", fontsize=8)
            if j == 0:
                ax.set_ylabel(r"$\rho$ [veh/km]")
            if i == len(PROFILE_T) - 1:
                ax.set_xlabel("x [km]")
    for i in range(len(PROFILE_T)):        # headroom above the tallest curve
        axes[i, 0].set_ylim(0.0, 5.0 * np.ceil(1.08 * row_max[i] / 5.0))
    axes[0, 0].text(15.9, RHO_CRIT + 2.0, r"$\rho_c$", ha="right",
                    va="bottom", fontsize=7, color="0.4")
    hs, ls = axes[0, 0].get_legend_handles_labels()
    order = [ls.index(l) for l in (DATA_LABEL, MODEL_LABEL["classical"],
                                   MODEL_LABEL["final"] + r" $\rho$",
                                   "stuck class $s$", "free class $f$")]
    fig.legend([hs[i] for i in order], [ls[i] for i in order],
               loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=7.5, handlelength=1.8,
               columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return pf.save(fig, name, tight=False)


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------

def _set_line(th):
    q = ("no capacity cap" if th["Q_xi_vehh"] is None
         else f"Q_xi = {th['Q_xi_vehh']:.0f} veh/h")
    return (f"kappa_c = {th['kappa_c']:.3e}, kappa_r = {th['kappa_r']:.3e} "
            f"veh^-1, gamma = {th['gamma']:.3f}, w_s = {th['ws_frac']:.3f} w, "
            f"{q}")


def write_table(results, path):
    lines = [
        "# Paper table: classical LWR+MB vs catch & release (adopted final set)",
        "",
        "Zero-refit evaluation at dt = 0.5 s. The catch & release parameters "
        "are fitted per A on u_xi = 54 km/h (15 m/s), q_in = 2500 veh/h only; "
        "every other scenario is a zero-refit transfer. Classical LWR+MB: "
        "scalar LWR + Delle Monache-Goatin moving bottleneck, Q_xi = 2000 "
        "veh/h, no fitted parameter.",
        "",
        "Columns: W1 = mean 1-Wasserstein distance between simulated and "
        "measured cumulative density profiles [veh km] and RMSE = density "
        "RMSE [veh/km], both against the SUMO 5-run-mean field (t in [100, "
        "1000] s, x <= 20 km); e_s [%] = relative cumulative-flow error at "
        "X_q = 15 km over the ECC22 eq. (6) window, mean over the 5 runs of "
        "the per-run values; omega error [%] = relative cumulative "
        "overtaking-flow error over the slow window, mean over runs; wake = "
        "mean density 0.2-1 km behind the controlled vehicle, t in [600, 740] "
        "s (SUMO value in brackets). Sign convention: e_s > 0 and omega error "
        "> 0 mean that the model passes MORE flow than SUMO. RMSE convention: "
        "the RMSE column is against the 5-run-mean field, whereas the "
        "rho_rmse min/max columns of the spread tables are per-run values "
        "(RMSE against each single run), which is why the two differ.", ""]
    for A in (1.0, 10.0):
        blk = results[f"A{A:g}"]
        fin = blk["final"]
        th = fin["theta_public"]
        lines += [f"## A = {A:g} -- catch & release, {fin['n_params']} fitted "
                  f"parameters ({', '.join(fin['parameters'])}): {_set_line(th)}",
                  "",
                  "| scenario | model | W1 [veh km] | RMSE [veh/km] | e_s [%] | "
                  "omega error [%] | wake [veh/km] |",
                  "|---|---|---|---|---|---|---|"]
        for uc, qin in SCENARIOS:
            tag = f"A{A:g}_u{uc:g}_q{qin:g}"
            row = blk["eval"][tag]
            sc = (f"{uc * 3.6:.0f} km/h, {qin:.0f} veh/h"
                  + (" (fit)" if row["fit"] else ""))
            for m in MODELS:
                r = row[m]
                lines.append(
                    f"| {sc} | {MODEL_LABEL[m]} | {r['W1']:.1f} | "
                    f"{r['RMSE']:.2f} | {100 * r['e_s']:+.1f} | "
                    f"{100 * r['omega_err']:+.1f} | {r['wake']:.1f} "
                    f"({row['data_wake']:.1f}) |")
                sc = ""
        lines += ["", "Per-run spread (min .. max over the 5 SUMO runs):", "",
                  "| scenario | model | e_s min [%] | e_s max [%] | rho_rmse "
                  "min [veh/km] | rho_rmse max [veh/km] |",
                  "|---|---|---|---|---|---|"]
        for uc, qin in SCENARIOS:
            tag = f"A{A:g}_u{uc:g}_q{qin:g}"
            row = blk["eval"][tag]
            sc = f"{uc * 3.6:.0f} km/h, {qin:.0f} veh/h"
            for m in MODELS:
                r = row[m]
                lines.append(
                    f"| {sc} | {MODEL_LABEL[m]} | "
                    f"{100 * min(r['e_s_per_rep']):+.1f} | "
                    f"{100 * max(r['e_s_per_rep']):+.1f} | "
                    f"{min(r['rho_rmse_per_rep']):.2f} | "
                    f"{max(r['rho_rmse_per_rep']):.2f} |")
                sc = ""
        lines.append("")
    lines += ["## Capacity-cap ablation (same protocol, 5 vs 4 parameters)", "",
              "Capped reference = the 5-parameter set with the DM-G cap Q_xi "
              "as a fitted parameter; no cap = the 4-parameter set with "
              "q_xi_max = None, refitted from the capped point. Both scored "
              "at dt = 0.5 s. Rule: " + results["A1"]["ablation"]["rule"]
              + ".", ""]
    for A in (1.0, 10.0):
        blk = results[f"A{A:g}"]
        abl = blk["ablation"]
        thc = theta_public(blk["capped_reference"]["theta"])
        thn = theta_public(blk["nocap"]["theta"])
        lines += [f"### A = {A:g}", "",
                  f"- capped reference (5): {_set_line(thc)}",
                  f"- no cap (4): {_set_line(thn)}",
                  f"- fit-scenario W1: capped {abl['fit_W1_capped']:.2f} vs no "
                  f"cap {abl['fit_W1_nocap']:.2f} veh km "
                  f"({100 * abl['fit_W1_rel_diff']:+.2f} %, tolerance "
                  f"{100 * abl['tol']:.1f} %) -> adopted: "
                  f"{'no cap (4 parameters)' if abl['adopt_nocap'] else 'capped (5 parameters)'}",
                  "",
                  "| scenario | set | W1 [veh km] | RMSE [veh/km] | e_s [%] | "
                  "omega error [%] | peak omega [veh/h] |",
                  "|---|---|---|---|---|---|---|"]
        for uc, qin in SCENARIOS:
            row = abl["scenarios"][f"A{A:g}_u{uc:g}_q{qin:g}"]
            sc = (f"{uc * 3.6:.0f} km/h, {qin:.0f} veh/h"
                  + (" (fit)" if row["fit"] else ""))
            for name, lbl in (("capped", "capped (5)"), ("nocap", "no cap (4)")):
                r = row[name]
                lines.append(f"| {sc} | {lbl} | {r['W1']:.1f} | {r['RMSE']:.2f} "
                             f"| {100 * r['e_s']:+.1f} | "
                             f"{100 * r['omega_err']:+.1f} | "
                             f"{r['peak_omega_vehh']:.0f} |")
                sc = ""
        lines.append("")
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# compact console summary (8 scenarios + the A = 3 transfer sweep)
# ---------------------------------------------------------------------------

def print_summary(results=None, transfer=None):
    results = results or json.loads((OUT / "tuned_config.json").read_text())
    transfer = transfer or (json.loads(TRANSFER_JSON.read_text())
                            if TRANSFER_JSON.exists() else None)
    print("\nFINAL METRICS -- classical LWR+MB | catch & release (adopted)")
    print(f"{'scenario':24s} {'W1 cl':>7s} {'W1 c&r':>7s} {'RMSE cl':>8s} "
          f"{'RMSE c&r':>9s} {'e_s cl':>8s} {'e_s c&r':>8s} "
          f"{'omega cl':>9s} {'omega c&r':>10s}")
    for A in (1.0, 10.0):
        blk = results[f"A{A:g}"]
        fin = blk["final"]
        print(f"A = {A:g}: {fin['n_params']} params "
              f"({'no cap' if fin['theta_public']['Q_xi_vehh'] is None else 'capped'}), "
              + fmt_theta(fin["theta_public"]))
        for uc, qin in SCENARIOS:
            row = blk["eval"][f"A{A:g}_u{uc:g}_q{qin:g}"]
            c, m = row["classical"], row["final"]
            sc = f"  {uc * 3.6:.0f} km/h {qin:.0f} veh/h" + (" *" if row["fit"] else "")
            print(f"{sc:24s} {c['W1']:7.1f} {m['W1']:7.1f} {c['RMSE']:8.2f} "
                  f"{m['RMSE']:9.2f} {c['e_s']:+8.1%} {m['e_s']:+8.1%} "
                  f"{c['omega_err']:+9.1%} {m['omega_err']:+10.1%}")
    print("  (* = calibration scenario)")
    if transfer and "A3_sweep_q2500" in transfer:
        fit = transfer["fit"]
        print(f"\nA = 3 transfer sweep, q_in = 2500 veh/h (anchor fit at 54 km/h: "
              f"kappa_c={fit['kappa_c']:.3e} kappa_r={fit['kappa_r']:.3e}, "
              f"extra_cfg={fit['extra_cfg']})")
        print(f"{'u_xi':>12s} {'W1 cl':>7s} {'W1 c&r':>7s} {'RMSE cl':>8s} "
              f"{'RMSE c&r':>9s} {'e_s cl':>8s} {'e_s c&r':>8s} "
              f"{'Xq flow SUMO':>13s} {'cl':>6s} {'c&r':>6s}")
        sw = transfer["A3_sweep_q2500"]
        for key in sorted(sw, key=lambda k: float(k[1:])):
            u = float(key[1:])
            c, m = sw[key]["classical"], sw[key]["model"]
            tag = f"{u * 3.6:5.1f} km/h" + (" *" if sw[key]["role"] == "fit anchor" else "  ")
            print(f"{tag:>12s} {c['w1_vehkm']:7.1f} {m['w1_vehkm']:7.1f} "
                  f"{c['rho_rmse_vehkm']:8.2f} {m['rho_rmse_vehkm']:9.2f} "
                  f"{c['e_s']:+8.1%} {m['e_s']:+8.1%} "
                  f"{m['wake_flow_Xq_data_vehh']:13.0f} "
                  f"{c['wake_flow_Xq_sim_vehh']:6.0f} "
                  f"{m['wake_flow_Xq_sim_vehh']:6.0f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _json_ready(o):
    if isinstance(o, dict):
        return {k: _json_ready(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_ready(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def make_outputs(results, stores):
    pf.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {A: results[f"A{A:g}"]["eval"] for A in (1.0, 10.0)}
    fig_heatmaps(stores, 2500.0, "fig_heatmaps_q2500")
    fig_heatmaps(stores, 2000.0, "fig_heatmaps_q2000")
    fig_es(tables)
    fig_profiles(stores)
    write_table(results, OUT / "table_metrics.md")


def _meta(smoke):
    return dict(
        structure=("stuck-class branch w_s + s_impermeable + catch & release "
                   "(lf, gamma override); DM-G cap Q_xi only in the capped "
                   "reference set; no downstream_release, no eta_la"),
        theta_capped=NAMES, theta_nocap=NAMES_NOCAP, box=BOX,
        objective=("W1 (e7_wasserstein.w1_mean, u15 q2500 rep-mean, t in "
                   "[100,1000] s, x<=20 km) + 50*max(0, ds_stuck(700 s) - "
                   "0.01); degenerate guard N_s(700) < 1 veh -> +1e3"),
        search=("Nelder-Mead in the unit cube (clipped), dt=1 s save_every "
                "10: capped set from the E10 point, no-cap set from the "
                "capped tuned point (maxfev 150) + 2 seeded random restarts "
                "(maxfev 80); best kept; evaluation at dt=0.5 s"),
        adoption=(f"the no-cap (4-parameter) set is adopted as 'final' when "
                  f"its fit-scenario W1 is within {NOCAP_TOL:.1%} of the "
                  "capped (5-parameter) set; both are recorded "
                  "('capped_reference', 'nocap')"),
        classical="kappa=0, q_xi_max=2000 veh/h, w_s None, permeable",
        e10_source=str(E10_CONFIG.relative_to(HERE)),
        scenarios=[f"u{uc:g}_q{qin:g}" for uc, qin in SCENARIOS],
        fit_scenario=f"u{UC_FIT:g}_q{QIN_FIT:g}", smoke=smoke)


def _budgets(smoke):
    return (dict(maxfev_main=6, maxfev_restart=4) if smoke
            else dict(maxfev_main=150, maxfev_restart=80))


def _write(results):
    (OUT / "tuned_config.json").write_text(
        json.dumps(_json_ready(results), indent=1))


def run(smoke=False):
    """Steps A + B + outputs."""
    t0 = time.time()
    budgets = _budgets(smoke)
    results = {"_meta": _meta(smoke)}
    for A in (1.0, 10.0):
        print(f"\n===== capped tuning A = {A:g} =====", flush=True)
        tuned = tune(A, **budgets)
        tuned["cap_activity"] = cap_activity(A, tuned["theta"])
        tuned["sensitivity"] = sensitivity(A, tuned["theta"])
        ca = tuned["cap_activity"]
        print(f"[A={A:g}] cap: omega_max={ca['omega_max_vehh']:.0f} veh/h vs "
              f"uncapped peak omega {ca['peak_omega_uncapped_vehh']:.0f}; "
              f"W1 without cap {ca['W1_nocap']:.2f}; max|drho|="
              f"{ca['max_abs_drho_vehkm']:.3f} -> cap "
              f"{'ACTIVE' if ca['cap_active'] else 'INACTIVE'}", flush=True)
        print("[A=%g] dJ (+-10%% box): " % A + ", ".join(
            f"{n} {tuned['sensitivity'][n]['minus']:+.1f}/"
            f"{tuned['sensitivity'][n]['plus']:+.1f}" for n in NAMES),
              flush=True)
        results[f"A{A:g}"] = dict(capped_reference=tuned)
    stores = ablate(results, budgets)
    results["_meta"]["runtime_s"] = round(time.time() - t0, 1)
    if smoke:
        print(json.dumps(_json_ready({k: v for k, v in results.items()
                                      if k == "_meta"}), indent=1))
        print("smoke OK (nothing written)")
        return results
    make_outputs(results, stores)
    _write(results)
    print(f"\nwrote {OUT / 'tuned_config.json'}, table_metrics.md and figures "
          f"({results['_meta']['runtime_s']} s)")
    print_summary(results)
    return results


def ablate_only():
    """Step B from an existing tuned_config.json (capped results kept)."""
    t0 = time.time()
    results = json.loads((OUT / "tuned_config.json").read_text())
    old_meta = results.get("_meta", {})
    results["_meta"] = _meta(False)
    results["_meta"]["capped_runtime_s"] = old_meta.get("runtime_s")
    stores = ablate(results, _budgets(False))
    results["_meta"]["ablation_runtime_s"] = round(time.time() - t0, 1)
    make_outputs(results, stores)
    _write(results)
    print(f"\nwrote {OUT / 'tuned_config.json'}, table_metrics.md and figures "
          f"({results['_meta']['ablation_runtime_s']} s)")
    print_summary(results)
    return results


def figures_only():
    results = json.loads((OUT / "tuned_config.json").read_text())
    stores = {}
    for A in (1.0, 10.0):
        theta = results[f"A{A:g}"]["final"]["theta"]
        table, store = evaluate_all(A, theta, verbose=False)
        results[f"A{A:g}"]["eval"] = table
        stores[A] = store
    make_outputs(results, stores)
    _write(results)
    print("figures + table regenerated")


def main(argv=None):
    ap = argparse.ArgumentParser(description="E11 tuning + paper figures")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        run(smoke=True)
    elif args.run:
        run()
    elif args.ablate:
        ablate_only()
    elif args.figures:
        figures_only()
    elif args.summary:
        print_summary()
    else:
        ap.error("choose --run, --ablate, --figures, --summary or --smoke")


if __name__ == "__main__":
    main()
