"""E9: general leader-loss (LL) release ladder -- the state-only replacement
for the downstream_release kludge.

Mladen's verdict on E8: SimConfig.downstream_release references the CAV
position x_cav, i.e. SCENARIO information, not traffic state; a general
model's source terms must be functions of the state only.  The flag stays
in the solver for the record but no new result may use it -- here it
appears ONLY in the RDR reference row (E8 final kappas re-evaluated, not
refit) so the general term can be judged against what the kludge did.

The general replacement (solver.py: eta_la / leader_loss_rate): being
synchronized is a relation to a slow LEADER AHEAD.  A stuck vehicle whose
look-ahead window (horizon eta_la, the only new quantity, ~200 m) contains
no slow vehicle -- no A vehicle (lane-blocking, full coverage) and less s
than its own cell -- has lost its leader and is released at the per-capita
rate mu_ll = (c_wave / ell_eff)(1 - c_j), c_wave = w_s if set else w.  The
rate scale is parameter-free: the release front recedes through a sharp-
edged platoon at exactly c_wave (test t18), which is also the START-UP
WAVE after t_fast -- something the kludge cannot produce (Delta v = 0 after
t_fast, nothing releases s).

Ladder (u15 q2500, lf, W1 metric, A in {1, 10}); every R-rung's
(kappa_c, kappa_r) is refit with the E7/E8 protocol (e7_wasserstein.
fit_field: 6x6 log grid + Nelder-Mead maxfev 40 at dt_fit=1, production
re-eval dt=0.5; A=10 uses kc_grid=logspace(-2.5,-0.5,6), the E6/E-V3
magnitude regime; fixed knobs via the extra_cfg passthrough, NOT the
optimization vector):

  R0   cap 2000 veh/h + w_s=0.6w                no DR, no LL: leaky reference
  R1   cap + w_s + eta_la=200 m                 the general fix
  R2   cap + w_s + eta_la=100 m                 horizon sensitivity
  R3   cap + w_s + eta_la=400 m                 horizon sensitivity
  R4   cap + eta_la=200 m, NO w_s               is w_s still needed?
  RDR  cap + w_s + downstream_release, E8 final kappas
       (out/e8/final_config.json) re-evaluated -- reference only, never
       a pick candidate; its W1 must reproduce final_config.json exactly
  RLL  cap + w_s + eta_la=200 m, E8 final kappas re-evaluated -- the
       like-for-like DR -> LL swap at identical kappas; reference only

FALLBACK STAGE (constrained refit, rungs "Rk_c"): the unconstrained W1
optimum of a rung may sit in the fast-churn regime (large kappas) where
the leader-loss rate c_wave / ell_eff (~0.03 1/s at eta_la = 200 m, dx =
50 m) cannot beat the downstream R1 growth rate lambda = Delta v (kappa_c
rho - kappa_r (P - rho)) nor the numerical forward smearing of s co-moving
with the CAV, so the rung fails Mladen's physical requirements although an
admissible basin exists at smaller kappas (test t19; row RLL).  Since the
pick rule REQUIRES admissibility, every R-rung whose unconstrained optimum
is inadmissible is refit with the SAME grid + Nelder-Mead protocol at the
production resolution dt = 0.5 on the penalized objective

    J = W1 + PEN_DS * (max ds s - 1)_+ / 1 + PEN_STUCK * (ds stuck - 0.01)_+ / 0.01

(PEN = 50 veh km per unit violation); the constrained rung is only a pick
candidate if its optimum is admissible at production resolution.  The
primary rungs stay exactly as prescribed; the constrained rows are extra,
clearly labelled, and the JSON records both.

Per rung (production dt=0.5; diagnostics via e8_ladder.eval_rung, imported
not duplicated): kappas, W1, RMSE, e_s, omega_err, wake density (data A1
58.6 / A10 55.4), downstream stuck fraction (t=700), max downstream s,
waviness (sim and sim/data), wedge indicator, post-release rarefaction,
s-layer extent, queue size N_s, and three NEW diagnostics:

  start-up wave   over t in [760, 900] s (after t_fast) the head of the
                  {s > 1 veh/km} region (downstream-most data cell) is
                  tracked; v_recede_cav = d/dt (x_cav - x_head) is the speed
                  at which the head recedes relative to the CAV.  After
                  t_fast the CAV drives at v_cav_free ~ v_f, so this is the
                  recession seen by a free-flow observer: exactly c_wave for
                  a subcritical sharp-edged platoon (t18), plus the head's
                  transport lag (v_f - v_s(rho_head)) while the head is
                  still on the s-class congested branch (rho > w_s P /
                  (v_f + w_s) = 31.4 veh/km).  Expect ~c_wave (w_s = 0.6 w
                  for R1-R3, w for R4) for LL rungs; ~0 for R0 (its head is
                  the leaked downstream s blob advecting at v_f); for RDR
                  ~0 as well but for the opposite reason: the kludge never
                  releases after t_fast, its s-platoon rarefies to the
                  s-critical density and rides along pinned behind the CAV.
                  v_recede_platoon = v_s(rho_head) - v_head subtracts the
                  transport lag (release front only).  Both are also
                  reported for the half-maximum head (cells with s > 0.5
                  max_x s, the t18 convention).  Because a residual plume
                  sitting at the 1 veh/km threshold contaminates the head
                  (it co-moves with the CAV), the MASS-based measure
                  v = -(dN_s/dt) / max_x s (the equivalent sharp-edged
                  front speed, t18 identity) and the fraction of N_s
                  released over the window are the robust numbers: exactly
                  0 for R0; for RDR small but nonzero because its s-platoon
                  rides at v_f > v_cav_free after t_fast, overtakes the CAV
                  and is converted by the DR projection (the kludge acting,
                  not a release wave); ~c_wave for the LL rungs.
  LL budget       where the leader-loss release acts during the slow window
                  [t_slow, t_fast]: mu_ll s dx integrated (10 s save
                  cadence) over cells strictly downstream of the CAV cell,
                  the CAV cell, the 4 cells upstream of it and the far
                  upstream queue body, as fractions of the total, plus the
                  in-queue LL release vs the kappa_r release.  This is the
                  direct test of whether the state-only term does the
                  kludge's job (downstream only) or something more: with
                  c_j = min{1, s_ahead/s_j} it also releases inside the
                  queue wherever the s-profile decreases toward the head.
  downstream rate closed form: lambda_ds at the background state vs the
                  LL rate c_wave / ell_eff; net < 0 means leaked s decays.

Pick rule per A: physical admissibility first (Mladen's requirements):
ds_stuck <= 0.01 AND max downstream s <= 1 veh/km (all snapshots, data
cells fully inside the DR zone); candidates = primary R-rungs + their
constrained refits; among admissible candidates pick min W1; reported if
none is admissible (figures/transfer then use the min-W1 rung, flagged).
Zero-refit q2000 transfer of the pick (and of RDR for contrast).  Figures:
fig_profiles_e9_{tag}.png (t=300/500/700/850; data black, RDR gray dashed,
pick blue with s/f fills; title states the labels) and
fig_heat3_e9_{tag}.png.  Verdict block: pick, R1 (and R1_c), RLL vs RDR,
component by component, with plain-language statements.

CLI:  python3 e9_ladder.py --run     -> out/e9/ladder.json + figures
      python3 e9_ladder.py --smoke   -> out/e9/ladder_smoke.json (tiny grid)
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

import e7_wasserstein as e7
import e8_ladder as L              # eval_rung / data_refs / grid constants
import ev4_compare as ev4
from loader import CELL_LEN
from solver import SimConfig, simulate, leader_loss_rate
from solver import speed as fd_speed

HERE = Path(__file__).parent
OUT_E9 = HERE / "out" / "e9"
E8_FINAL_CONFIG = HERE / "out" / "e8" / "final_config.json"

UC = 15.0                      # [m/s]   focus CAV slow speed
QIN_FIT = 2500.0               # [veh/h] fit inflow
QIN_TRANSFER = 2000.0          # [veh/h] zero-refit transfer inflow
FORM = "lf"
METRIC = "w1"
QXI = 2000.0 / 3600.0          # [veh/s] one-lane-blocked capacity cap
WS_FRAC_FIX = 0.6
W_S = WS_FRAC_FIX * ev4.W      # [m/s]   stuck-class wave speed
ETA_MAIN = 200.0               # [m]     the general fix's horizon
SOLVER_DX = L.SOLVER_DX        # [m]
KC_GRID_A1 = np.logspace(-2.5, 0.5, 6)     # fit_field default
KR_GRID = np.logspace(-4.0, -0.5, 6)       # fit_field default
KC_GRID_A10 = np.logspace(-2.5, -0.5, 6)   # E6/E-V3 magnitude regime
NM_MAXFEV = 40

# physical admissibility (Mladen's requirements)
DS_STUCK_MAX = 0.01            # downstream stuck fraction at t = 700 s
DS_MAX_S_VEHKM = 1.0           # max downstream s, all snapshots
DEGENERATE_NS_VEH = 1.0        # N_s(700) below this: no stuck population
CONSTRAINT_ACTIVE_FRAC = 0.95  # constrained optimum on the bound if >= this

# constrained-refit penalties [veh km per unit violation]
PEN_DS = 50.0
PEN_STUCK = 50.0

# start-up wave check
SUW_T = (760.0, 900.0)         # [s]
SUW_S_MIN = 1.0                # [veh/km]
SUW_HALF = 0.5                 # half-maximum head (t18 convention)
SUW_MIN_POINTS = 5

# LL release budget
BUDGET_T = (ev4.T_SLOW, ev4.T_FAST)
BUDGET_NEAR_CELLS = 4          # solver cells (200 m) upstream of the CAV cell

# verdict tolerances (rung vs RDR)
VERDICT_W1_TOL = 0.05
VERDICT_WAKE_TOL = 3.0         # [veh/km]
VERDICT_OMEGA_TOL = 0.05
VERDICT_ES_TOL = 0.05

RHO_CRIT = L.RHO_CRIT
X_CTR = L.X_CTR
X_UP = L.X_UP

RUNGS = (
    # (name, extra_cfg, description)
    ("R0", {"q_xi_max": QXI, "w_s": W_S},
     "cap 2000 + w_s=0.6w (no DR, no LL: the leaky reference)"),
    ("R1", {"q_xi_max": QXI, "w_s": W_S, "eta_la": 200.0},
     "cap + w_s + eta_la=200 m (the general fix)"),
    ("R2", {"q_xi_max": QXI, "w_s": W_S, "eta_la": 100.0},
     "cap + w_s + eta_la=100 m (horizon sensitivity)"),
    ("R3", {"q_xi_max": QXI, "w_s": W_S, "eta_la": 400.0},
     "cap + w_s + eta_la=400 m (horizon sensitivity)"),
    ("R4", {"q_xi_max": QXI, "eta_la": 200.0},
     "cap + eta_la=200 m, NO w_s (is w_s still needed?)"),
)
RDR = "RDR"
RDR_CFG = {"q_xi_max": QXI, "w_s": W_S, "downstream_release": True}
RDR_DESC = ("cap + w_s=0.6w + downstream_release, E8 final kappas "
            "re-evaluated (NOT refit; reference only, never a pick)")
RLL = "RLL"
RLL_CFG = {"q_xi_max": QXI, "w_s": W_S, "eta_la": ETA_MAIN}
RLL_DESC = (f"cap + w_s=0.6w + eta_la={ETA_MAIN:g} m, E8 final kappas "
            "re-evaluated (NOT refit; the like-for-like DR->LL swap; "
            "reference only, never a pick)")
CONSTRAINED_SUFFIX = "_c"

PROFILE_SNAPS = (300.0, 500.0, 700.0, 850.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _fmt(v, spec):
    return "n/a" if v is None else format(v, spec)


def _c_wave(extra_cfg):
    """Rate scale of the leader-loss release: w_s if set, else w (solver)."""
    ws = extra_cfg.get("w_s")
    return float(ws) if ws is not None else float(ev4.W)


def _ll_rate(extra_cfg, dx=SOLVER_DX):
    """c_wave / ell_eff [1/s], ell_eff = dx (n + 1) / 2 (solver)."""
    eta = extra_cfg.get("eta_la")
    if eta is None:
        return 0.0
    n = max(1, int(round(float(eta) / dx)))
    return _c_wave(extra_cfg) / (dx * (n + 1) / 2.0)


def rung_label(name, cfg):
    parts = ["cap"]
    if cfg.get("w_s") is not None:
        parts.append(f"w_s={WS_FRAC_FIX:g}w")
    if cfg.get("downstream_release"):
        parts.append("downstream_release")
    if cfg.get("eta_la") is not None:
        parts.append(f"eta_la={cfg['eta_la']:g} m")
    return f"{name}: " + " + ".join(parts)


def _raw_sim(uc, qin, kc, kr, extra_cfg, dt=e7.DT_PRODUCTION):
    """Production-resolution solver run with the SAME SimConfig that
    e7_wasserstein.run_sim builds (raw SimResult, for the LL budget)."""
    cfg = SimConfig(v_f=ev4.V_F, w=ev4.W, P=ev4.P, q_in=qin / 3600.0,
                    u_xi=uc, kappa_c=float(kc), kappa_r=float(kr),
                    capture_form=FORM, dt=dt,
                    save_every=int(round(10.0 / dt)), **extra_cfg)
    return simulate(cfg)


# ---------------------------------------------------------------------------
# NEW diagnostic 1: start-up wave check
# ---------------------------------------------------------------------------

def _track_head(regr, c_wave, t_win, head_of):
    """Linear-fit kinematics of a head position over t_win.

    head_of(s_row [veh/km]) -> index of the head cell or None.  Returns
    the head's absolute speed, the CAV speed, v_recede_cav = v_cav -
    v_head, the s-class FD speed at the head's total density (c_s = v_f
    after t_fast: min{v_f, c_wave (P/rho - 1)}) and v_recede_platoon =
    v_s_fd - v_head.
    """
    tt = np.asarray(regr["tt"], float)
    ts, xh, xc, rh, sh = [], [], [], [], []
    for i in np.where((tt >= t_win[0]) & (tt <= t_win[1]))[0]:
        x_cav = float(regr["x_cav"][i])
        if not np.isfinite(x_cav):
            continue
        j = head_of(np.asarray(regr["s"][i], float))
        if j is None:
            continue
        ts.append(float(tt[i]))
        xh.append(float(X_CTR[j]))
        xc.append(x_cav)
        rh.append(float(regr["rho_tot"][i, j]))
        sh.append(float(regr["s"][i, j]))
    if len(ts) < SUW_MIN_POINTS:
        return dict(status="insufficient points (no head with the CAV on "
                           "road in the window)", n_points=len(ts))
    ts, xh, xc = np.asarray(ts), np.asarray(xh), np.asarray(xc)
    v_head = float(np.polyfit(ts, xh, 1)[0])
    v_cav = float(np.polyfit(ts, xc, 1)[0])
    rho_si = np.maximum(np.asarray(rh) / 1000.0, 1e-12)
    v_s_fd = float(np.mean(np.minimum(
        ev4.V_F, np.maximum(c_wave * (ev4.P / rho_si - 1.0), 0.0))))
    return dict(
        status="ok", n_points=len(ts),
        head_ahead_of_cav_frac=float(np.mean(xh > xc)),
        v_head_abs_ms=v_head, v_cav_ms=v_cav,
        v_recede_cav_ms=v_cav - v_head,
        v_recede_cav_over_w=(v_cav - v_head) / float(ev4.W),
        v_recede_cav_over_cwave=(v_cav - v_head) / c_wave,
        v_s_head_fd_ms=v_s_fd,
        v_recede_platoon_ms=v_s_fd - v_head,
        v_recede_platoon_over_cwave=(v_s_fd - v_head) / c_wave,
        rho_head_mean_vehkm=float(np.mean(rh)),
        s_head_mean_vehkm=float(np.mean(sh)),
        head_lag_first_m=float(xc[0] - xh[0]),
        head_lag_last_m=float(xc[-1] - xh[-1]),
        head_first_m=float(xh[0]), head_last_m=float(xh[-1]))


def startup_wave(regr, extra_cfg, t_win=SUW_T, s_min=SUW_S_MIN):
    """Head recession of the {s > s_min} region relative to the CAV after
    t_fast (primary, flattened at the top level), plus the half-maximum
    head (cells with s > SUW_HALF max_x s, t18 convention)."""
    c_wave = _c_wave(extra_cfg)

    def head_gt(s):
        idx = np.where(s > s_min)[0]
        return int(idx.max()) if idx.size else None

    def head_half(s):
        smax = float(s.max())
        if smax <= s_min:
            return None
        idx = np.where(s > SUW_HALF * smax)[0]
        return int(idx.max()) if idx.size else None

    prim = _track_head(regr, c_wave, t_win, head_gt)
    half = _track_head(regr, c_wave, t_win, head_half)
    out = dict(
        t_win_s=[float(t_win[0]), float(t_win[1])], s_min_vehkm=float(s_min),
        c_wave_ms=c_wave, w_ms=float(ev4.W),
        definition=("x_head = center of the downstream-most data cell with "
                    "s > s_min; v_recede_cav = slope of (x_cav - x_head) "
                    "over t_win; v_s_head_fd = min{v_f, c_wave (P/rho_head "
                    "- 1)} (s-class FD after t_fast); v_recede_platoon = "
                    "v_s_head_fd - v_head_abs; half_max repeats it for the "
                    f"head of {{s > {SUW_HALF:g} max_x s}}"))
    out.update(prim)
    out["half_max"] = half
    # mass-based measure: immune to a residual plume at the s_min threshold
    tt = np.asarray(regr["tt"], float)
    m = (tt >= t_win[0]) & (tt <= t_win[1])
    ns = np.asarray(regr["N_s"], float)[m]                        # [veh]
    smax = np.max(np.asarray(regr["s"], float)[m], axis=1) / 1000.0  # veh/m
    if m.sum() >= SUW_MIN_POINTS and ns[0] > 0.0 and np.mean(smax) > 0.0:
        dns = float(np.polyfit(tt[m], ns, 1)[0])                  # [veh/s]
        v_eq = -dns / float(np.mean(smax))
        out["mass"] = dict(
            status="ok", N_s_first_veh=float(ns[0]),
            N_s_last_veh=float(ns[-1]),
            released_frac=float(1.0 - ns[-1] / ns[0]),
            dNs_dt_veh_per_s=dns,
            s_max_mean_vehkm=float(np.mean(smax) * 1000.0),
            v_startup_mass_ms=float(v_eq),
            v_startup_mass_over_cwave=float(v_eq / c_wave),
            definition=("v = -(dN_s/dt) / mean_t max_x s over t_win: front "
                        "speed of the equivalent sharp-edged platoon of "
                        "density max_x s (telescoping identity, t18); "
                        "exactly 0 when nothing releases after t_fast (R0: "
                        "no mechanism, Delta v = 0).  RDR is small but NOT "
                        "0: after t_fast its s-platoon rides at v_f > "
                        "v_cav_free, overtakes the CAV and is converted by "
                        "the DR projection -- the kludge acting, not a "
                        "release wave.  released_frac = 1 - "
                        "N_s(end)/N_s(start) over t_win"))
    else:
        out["mass"] = dict(status="no stuck mass in the window")
    return out


# ---------------------------------------------------------------------------
# NEW diagnostic 2: where the leader-loss release acts (slow window)
# ---------------------------------------------------------------------------

def ll_release_budget(uc, qin, kc, kr, extra_cfg, t_win=BUDGET_T):
    """Integrate mu_ll s dx over the saved states in t_win (10 s cadence,
    an estimate) split by position relative to the CAV cell
    j_cav = int(x_cav // dx): strictly downstream (j > j_cav), the CAV
    cell, the BUDGET_NEAR_CELLS cells upstream of it, and the far upstream
    queue body.  Also the kappa_r release integrated over j <= j_cav for
    the same states, so in-queue LL release can be compared to the
    modelled overtaking release."""
    eta = extra_cfg.get("eta_la")
    if eta is None:
        return dict(applies=False, status="no eta_la in this rung")
    res = _raw_sim(uc, qin, kc, kr, extra_cfg)
    dx = float(res.x[1] - res.x[0])
    save_dt = float(np.median(np.diff(res.t)))
    c_wave = _c_wave(extra_cfg)
    parts = dict(downstream=0.0, cav_cell=0.0, upstream_near=0.0,
                 upstream_far=0.0)
    kr_queue = 0.0
    s_queue_time = 0.0            # int s dx dt over j <= j_cav [veh s]
    n = 0
    for i in range(res.t.size):
        t = float(res.t[i])
        xc = float(res.x_cav[i])
        if not (t_win[0] <= t <= t_win[1]) or not np.isfinite(xc):
            continue
        a_i, f_i, s_i = res.a[i], res.f[i], res.s[i]
        mu = leader_loss_rate(a_i, s_i, dx, float(eta), c_wave)
        rel = mu * s_i * dx * save_dt                  # [veh] per interval
        jc = int(xc // dx)
        j0 = max(jc - BUDGET_NEAR_CELLS, 0)
        parts["downstream"] += float(rel[jc + 1:].sum())
        parts["cav_cell"] += float(rel[jc])
        parts["upstream_near"] += float(rel[j0:jc].sum())
        parts["upstream_far"] += float(rel[:j0].sum())
        rho = a_i + f_i + s_i
        u_s = uc if ev4.T_SLOW <= t <= ev4.T_FAST else ev4.V_F
        dv = np.maximum(fd_speed(rho, ev4.V_F, ev4.W, ev4.P) - u_s, 0.0)
        mu_kr = kr * np.maximum(ev4.P - rho, 0.0) * dv
        kr_queue += float((mu_kr * s_i * dx * save_dt)[:jc + 1].sum())
        s_queue_time += float(s_i[:jc + 1].sum() * dx * save_dt)
        n += 1
    total = sum(parts.values())
    in_queue = (parts["cav_cell"] + parts["upstream_near"]
                + parts["upstream_far"])
    frac = {k: (v / total if total > 0.0 else None) for k, v in parts.items()}
    return dict(
        applies=True, t_win_s=[float(t_win[0]), float(t_win[1])],
        n_snapshots=n, save_cadence_s=save_dt, eta_la_m=float(eta),
        c_wave_ms=c_wave, rate_scale_per_s=_ll_rate(extra_cfg, dx),
        total_ll_release_veh=total, parts_veh=parts, parts_frac=frac,
        in_queue_veh=in_queue,
        in_queue_frac=(in_queue / total if total > 0.0 else None),
        in_queue_mean_rate_per_s=(in_queue / s_queue_time
                                  if s_queue_time > 0.0 else None),
        kappa_r_release_in_queue_veh=kr_queue,
        kappa_r_mean_rate_per_s=(kr_queue / s_queue_time
                                 if s_queue_time > 0.0 else None),
        ll_over_kappa_r_in_queue=(in_queue / kr_queue if kr_queue > 0.0
                                  else None),
        definition=("mu_ll s dx per saved state in t_win, summed; "
                    "'in_queue' = cav_cell + upstream_near + upstream_far "
                    "(j <= j_cav); mean rates are s-weighted per-capita "
                    "[1/s] over the same cells"))


# ---------------------------------------------------------------------------
# NEW diagnostic 3: closed-form downstream rate balance
# ---------------------------------------------------------------------------

def downstream_rates(kc, kr, qin, extra_cfg):
    """Linearized downstream growth rate of leaked s at the background
    state (lf form, ell = s, a = 0; E8 analytic check):
    lambda = Delta v (kappa_c rho_bg - kappa_r (P - rho_bg)), rho_bg =
    q_in / v_f, Delta v = v_f - u_xi, vs the leader-loss rate
    c_wave / ell_eff.  net = lambda - ll_rate < 0: leaked s decays."""
    rho_bg = qin / 3600.0 / ev4.V_F
    dv = ev4.V_F - UC
    lam = dv * (kc * rho_bg - kr * (ev4.P - rho_bg))
    llr = _ll_rate(extra_cfg)
    return dict(rho_bg_vehkm=rho_bg * 1000.0, dv_ms=dv,
                lambda_ds_per_s=float(lam), ll_rate_per_s=float(llr),
                net_per_s=float(lam - llr), leaked_s_decays=bool(lam < llr))


# ---------------------------------------------------------------------------
# per-rung evaluation (E8 diagnostics + the new ones + admissibility)
# ---------------------------------------------------------------------------

def admissibility(ev):
    frac = ev["ds_stuck"]["frac"]
    stuck_ok = frac is not None and frac <= DS_STUCK_MAX
    dsmax_ok = ev["ds_max_s_vehkm"] <= DS_MAX_S_VEHKM
    return dict(ok=bool(stuck_ok and dsmax_ok), ds_stuck_ok=bool(stuck_ok),
                ds_max_s_ok=bool(dsmax_ok),
                thresholds=dict(ds_stuck_max=DS_STUCK_MAX,
                                ds_max_s_vehkm=DS_MAX_S_VEHKM))


def evaluate(A, qin, kc, kr, extra_cfg, refs):
    ev, regr, meas = L.eval_rung(A, UC, qin, kc, kr, extra_cfg)
    ev["startup_wave"] = startup_wave(regr, extra_cfg)
    ev["ll_budget"] = ll_release_budget(UC, qin, kc, kr, extra_cfg)
    ev["downstream_rates"] = downstream_rates(kc, kr, qin, extra_cfg)
    i700 = L._snap(regr["tt"], 700.0)
    ns700 = float(regr["N_s"][i700])
    ev["queue"] = dict(
        N_s_700_veh=ns700, N_s_end_veh=float(regr["N_s"][-1]),
        n_caught_data_700_veh=float(meas["n_caught_mean"][i700]),
        degenerate=bool(ns700 < DEGENERATE_NS_VEH),
        degenerate_rule=f"N_s(700) < {DEGENERATE_NS_VEH:g} veh: no stuck "
                        "population at all (not a catch & release model)")
    wd = refs["waviness"]["std_vehkm"]
    ws = ev["waviness_sim"]["std_vehkm"]
    ev["waviness_rel_data"] = (ws / wd if (ws is not None and wd) else None)
    ev["rarefaction_data_growth_m"] = refs["rarefaction"]["width_growth_m"]
    ev["wake_data_vehkm"] = refs["wake"]["wake_mean_vehkm"]
    ev["admissible"] = admissibility(ev)
    return ev, regr, meas


def _grids(A, smoke):
    if smoke:
        return np.array([0.1, 0.4]), np.array([0.005, 0.05]), 2
    return ((KC_GRID_A10 if A == 10.0 else KC_GRID_A1), KR_GRID, NM_MAXFEV)


def fit_rung(A, extra_cfg, smoke):
    """Primary stage: e7_wasserstein.fit_field exactly as prescribed."""
    kc_grid, kr_grid, maxfev = _grids(A, smoke)
    fk = dict(kr_grid=kr_grid, maxfev=maxfev)
    if smoke or A == 10.0:
        fk["kc_grid"] = kc_grid          # A=1 keeps fit_field's default grid
    return e7.fit_field(A, UC, QIN_FIT, form=FORM, metric=METRIC,
                        extra=None, dt_fit=1.0,
                        extra_cfg=(extra_cfg or None), **fk)


def fit_constrained(A, extra_cfg, smoke, verbose=True):
    """Fallback stage: same 6x6 log grid + Nelder-Mead (maxfev 40)
    protocol, at production dt = 0.5, on the penalized objective
    J = W1 + PEN_DS (max ds s - 1)_+ + PEN_STUCK (ds stuck - 0.01)_+ / 0.01.
    The grid optimum is kept if the polish regresses (fit_field
    convention)."""
    from scipy.optimize import minimize

    kc_grid, kr_grid, maxfev = _grids(A, smoke)
    tt_data, rho_mean = e7.load_rho_mean(A, UC, QIN_FIT)
    n_sim = 0

    def evalp(kc, kr):
        nonlocal n_sim
        regr = e7.run_sim(UC, QIN_FIT, float(kc), float(kr), FORM,
                          dt=e7.DT_PRODUCTION, **extra_cfg)
        n_sim += 1
        w1 = e7.w1_mean(regr["rho_tot"], rho_mean, tt_data)
        stuck = L.ds_stuck_fraction(regr)["frac"] or 0.0
        dsmax = L.ds_max_s(regr)
        pen = (PEN_DS * max(0.0, dsmax - DS_MAX_S_VEHKM) / DS_MAX_S_VEHKM
               + PEN_STUCK * max(0.0, stuck - DS_STUCK_MAX) / DS_STUCK_MAX)
        return w1 + pen, w1, stuck, dsmax

    best = (np.inf, None, None, None)
    for kc in kc_grid:
        for kr in kr_grid:
            j, w1, stuck, dsmax = evalp(kc, kr)
            if j < best[0]:
                best = (j, float(kc), float(kr), (w1, stuck, dsmax))
    j_grid, kc0, kr0, det0 = best
    if verbose:
        print(f"[constrained grid {len(kc_grid)}x{len(kr_grid)} dt=0.5] "
              f"J={j_grid:.3f} (W1={det0[0]:.3f}, stuck={det0[1]:.4f}, "
              f"ds|s|={det0[2]:.3f}) at kappa_c={kc0:.3e} kappa_r={kr0:.3e}")

    res = minimize(lambda z: evalp(10.0 ** z[0], 10.0 ** z[1])[0],
                   [np.log10(kc0), np.log10(kr0)], method="Nelder-Mead",
                   options=dict(maxfev=maxfev, xatol=1e-3, fatol=1e-3))
    kc, kr = 10.0 ** res.x[0], 10.0 ** res.x[1]
    j_fit, w1_fit, stuck_fit, dsmax_fit = evalp(kc, kr)
    if j_grid < j_fit:                        # polish regressed: keep grid
        kc, kr, j_fit = kc0, kr0, j_grid
        w1_fit, stuck_fit, dsmax_fit = det0
    if verbose:
        print(f"[constrained polish nfev={res.nfev}] J={j_fit:.3f} "
              f"(W1={w1_fit:.3f}, stuck={stuck_fit:.4f}, "
              f"ds|s|={dsmax_fit:.3f}) at kappa_c={kc:.3e} kappa_r={kr:.3e} "
              f"({n_sim} sims)")
    return dict(kappa_c=float(kc), kappa_r=float(kr), objective=float(j_fit),
                grid_objective=float(j_grid), W1_at_fit=float(w1_fit),
                ds_stuck_at_fit=float(stuck_fit),
                ds_max_s_at_fit=float(dsmax_fit), n_sim=n_sim,
                n_polish=int(res.nfev), dt_fit=e7.DT_PRODUCTION,
                penalties=dict(PEN_DS=PEN_DS, PEN_STUCK=PEN_STUCK))


# ---------------------------------------------------------------------------
# verdict: rung vs the kludge reference
# ---------------------------------------------------------------------------

def compare_to_rdr(block, name):
    a = block["rungs"][name]["eval_q2500"]
    d = block["rungs"][RDR]["eval_q2500"]

    def pair(x, y):
        return dict(rung=x, RDR=y)

    w1_rel = a["W1"] / d["W1"] - 1.0
    wake_d = a["wake"]["wake_mean_vehkm"] - d["wake"]["wake_mean_vehkm"]
    om_d = a["omega_err"] - d["omega_err"]
    es_d = a["e_s"] - d["e_s"]
    suw_a = a["startup_wave"].get("v_recede_cav_ms")
    suw_d = d["startup_wave"].get("v_recede_cav_ms")
    suwp_a = a["startup_wave"].get("v_recede_platoon_ms")
    ma, md = a["startup_wave"]["mass"], d["startup_wave"]["mass"]
    out = dict(
        rung=name, admissible=a["admissible"]["ok"],
        W1=dict(**pair(a["W1"], d["W1"]), rel=w1_rel,
                within=bool(abs(w1_rel) <= VERDICT_W1_TOL)),
        wake_vehkm=dict(**pair(a["wake"]["wake_mean_vehkm"],
                               d["wake"]["wake_mean_vehkm"]),
                        data=a["wake_data_vehkm"], diff=wake_d,
                        within=bool(abs(wake_d) <= VERDICT_WAKE_TOL)),
        ds_stuck_frac=pair(a["ds_stuck"]["frac"], d["ds_stuck"]["frac"]),
        ds_max_s_vehkm=pair(a["ds_max_s_vehkm"], d["ds_max_s_vehkm"]),
        omega_err=dict(**pair(a["omega_err"], d["omega_err"]), diff=om_d,
                       within=bool(abs(om_d) <= VERDICT_OMEGA_TOL)),
        e_s=dict(**pair(a["e_s"], d["e_s"]), diff=es_d,
                 within=bool(abs(es_d) <= VERDICT_ES_TOL)),
        rarefaction_growth_m=dict(**pair(a["rarefaction"]["width_growth_m"],
                                         d["rarefaction"]["width_growth_m"]),
                                  data=a["rarefaction_data_growth_m"]),
        waviness_rel_data=pair(a["waviness_rel_data"],
                               d["waviness_rel_data"]),
        wedge_frac=pair(a["wedge"]["frac"], d["wedge"]["frac"]),
        s_layer_max_m=pair(a["s_layer"]["max_m"], d["s_layer"]["max_m"]),
        N_s_700_veh=pair(a["queue"]["N_s_700_veh"],
                         d["queue"]["N_s_700_veh"]),
        N_s_end_veh=pair(a["queue"]["N_s_end_veh"], d["queue"]["N_s_end_veh"]),
        startup_wave_cav_ms=pair(suw_a, suw_d),
        startup_wave_platoon_ms=pair(suwp_a,
                                     d["startup_wave"].get("v_recede_platoon_ms")),
        startup_wave_mass_ms=pair(ma.get("v_startup_mass_ms"),
                                  md.get("v_startup_mass_ms")),
        released_frac_760_900=pair(ma.get("released_frac"),
                                   md.get("released_frac")),
        degenerate=a["queue"]["degenerate"],
        ll_in_queue_frac=(a["ll_budget"].get("in_queue_frac")
                          if a["ll_budget"].get("applies") else None),
        ll_over_kappa_r_in_queue=(a["ll_budget"].get("ll_over_kappa_r_in_queue")
                                  if a["ll_budget"].get("applies") else None),
        tolerances=dict(W1_rel=VERDICT_W1_TOL, wake_vehkm=VERDICT_WAKE_TOL,
                        omega_err=VERDICT_OMEGA_TOL, e_s=VERDICT_ES_TOL))
    out["reproduces_kludge"] = bool(out["admissible"] and out["W1"]["within"]
                                    and out["wake_vehkm"]["within"])
    llq = out["ll_in_queue_frac"]
    parts = [
        f"{name} vs RDR: W1 {a['W1']:.1f} vs {d['W1']:.1f} ({w1_rel:+.1%}), "
        f"wake {a['wake']['wake_mean_vehkm']:.1f} vs "
        f"{d['wake']['wake_mean_vehkm']:.1f} veh/km (data "
        f"{a['wake_data_vehkm']:.1f}), omega err {a['omega_err']:+.1%} vs "
        f"{d['omega_err']:+.1%}, e_s {a['e_s']:+.1%} vs {d['e_s']:+.1%}",
        f"downstream: stuck {a['ds_stuck']['frac']:.4f} (RDR "
        f"{d['ds_stuck']['frac']:.4f}), max s {a['ds_max_s_vehkm']:.2f} "
        f"veh/km (RDR {d['ds_max_s_vehkm']:.2f}) -> "
        + ("ADMISSIBLE" if out["admissible"] else "NOT admissible"),
        f"start-up wave: mass-equivalent front speed "
        f"{_fmt(ma.get('v_startup_mass_ms'), '.2f')} m/s, "
        f"{_fmt(ma.get('released_frac'), '.0%')} of the queue released over "
        f"[760,900] s (RDR {_fmt(md.get('v_startup_mass_ms'), '.2f')} m/s, "
        f"{_fmt(md.get('released_frac'), '.0%')}: the DR projection eating "
        f"s that overtakes the CAV after t_fast, not a release wave); "
        f"{{s>1}} head recedes at "
        f"{_fmt(suw_a, '.2f')} m/s rel. to the CAV vs RDR "
        f"{_fmt(suw_d, '.2f')}; c_wave "
        f"{a['startup_wave']['c_wave_ms']:.2f}, w {ev4.W:.2f}; post-release "
        f"rarefaction growth {a['rarefaction']['width_growth_m']:+.0f} m vs "
        f"RDR {d['rarefaction']['width_growth_m']:+.0f} m (data "
        f"{a['rarefaction_data_growth_m']:+.0f} m)"]
    if llq is not None:
        r = out["ll_over_kappa_r_in_queue"]
        parts.append(f"LL release budget (slow window): {llq:.0%} inside the "
                     f"queue body, {1 - llq:.0%} downstream of the CAV cell"
                     + (f"; in-queue LL release = {r:.1f}x the kappa_r "
                        "release" if r is not None else ""))
    if out["reproduces_kludge"]:
        verdict = ("the general LL term REPRODUCES the kludge (clean "
                   f"downstream, W1 within {VERDICT_W1_TOL:.0%}, wake within "
                   f"{VERDICT_WAKE_TOL:g} veh/km)")
    else:
        fails = [k for k, ok in (("downstream cleanliness", out["admissible"]),
                                 ("W1", out["W1"]["within"]),
                                 ("wake density", out["wake_vehkm"]["within"]))
                 if not ok]
        verdict = ("the general LL term does NOT reproduce the kludge on "
                   + ", ".join(fails))
    out["statement"] = "; ".join(parts) + " => " + verdict
    return out


# ---------------------------------------------------------------------------
# figures (winners): data black, RDR gray dashed, winner blue + s/f fills
# ---------------------------------------------------------------------------

def fig_profiles_e9(tag, meas, rdr_regr, rdr_lbl, win_regr, win_lbl,
                    out_dir, fname_tag=None):
    rho_d = np.mean(meas["rho"], axis=0)
    tt = win_regr["tt"]
    xs = X_UP / 1000.0
    short = win_lbl.split(":")[0]
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 11.0), sharex=True)
    for ax, t_snap in zip(axes, PROFILE_SNAPS):
        i = L._snap(tt, t_snap)
        ax.plot(xs, rho_d[i], "k-", lw=1.8, label="SUMO (rep mean)")
        ax.plot(xs, rdr_regr["rho_tot"][i], color="0.55", ls="--", lw=1.4,
                label=rdr_lbl)
        ax.plot(xs, win_regr["rho_tot"][i], color="tab:blue", lw=1.6,
                label=f"{win_lbl}: total " + r"$\rho$")
        ax.fill_between(xs, 0, win_regr["s"][i], color="tab:red", alpha=0.30,
                        label=f"{short}: caught $s$")
        ax.fill_between(xs, win_regr["s"][i],
                        win_regr["s"][i] + win_regr["f"][i],
                        color="tab:green", alpha=0.20,
                        label=f"{short}: free $f$")
        ax.axhline(RHO_CRIT, color="0.55", ls="--", lw=0.9)
        xc = win_regr["x_cav"][i] / 1000.0
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
    axes[0].legend(fontsize=7.5, ncol=2)
    axes[-1].set_xlabel("x [km]")
    fig.suptitle(f"{tag}: density profiles — SUMO rep mean (black) vs "
                 f"{rdr_lbl} (gray dashed)\nvs {win_lbl} (blue, s/f fills)",
                 fontsize=9.5)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(out_dir / f"fig_profiles_e9_{fname_tag or tag}.png", dpi=160)
    plt.close(fig)


def fig_heat3_e9(tag, meas, rdr_regr, rdr_lbl, win_regr, win_lbl, out_dir,
                 fname_tag=None):
    fields = [(np.mean(meas["rho"], axis=0), f"{tag} — SUMO (rep mean)"),
              (rdr_regr["rho_tot"], rdr_lbl),
              (win_regr["rho_tot"], win_lbl)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    tt = win_regr["tt"]
    for ax, (rho, name) in zip(axes, fields):
        im = ax.imshow(rho, origin="lower", aspect="auto",
                       extent=[CELL_LEN / 1000, 30.0,
                               tt[0] / 60, tt[-1] / 60],
                       cmap="turbo", vmin=0, vmax=90)
        ax.plot(win_regr["x_cav"] / 1000, tt / 60, "w-", lw=1.5)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("x [km]")
        ax.set_xlim(0, 20)
    axes[0].set_ylabel("t [min]")
    fig.colorbar(im, ax=axes, label=r"$\rho$ [veh/km]", shrink=0.9)
    fig.savefig(out_dir / f"fig_heat3_e9_{fname_tag or tag}.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# ladder table
# ---------------------------------------------------------------------------

def _nan(v):
    return float("nan") if v is None else v


def _row(name, r):
    ev = r["eval_q2500"]
    suw = ev["startup_wave"]
    ok = suw.get("status") == "ok"
    bud = ev["ll_budget"]
    llq = (100.0 * bud["in_queue_frac"]
           if bud.get("applies") and bud.get("in_queue_frac") is not None
           else float("nan"))
    return (f"{name:5s} {r['kappa_c']:9.3e} {r['kappa_r']:9.3e} "
            f"{ev['W1']:6.1f} {ev['RMSE']:5.2f} {ev['e_s']:+7.1%} "
            f"{ev['omega_err']:+7.1%} "
            f"{ev['wake']['wake_mean_vehkm']:5.1f} "
            f"{_nan(ev['ds_stuck']['frac']):7.4f} "
            f"{ev['ds_max_s_vehkm']:6.2f} "
            f"{_nan(ev['waviness_sim']['std_vehkm']):5.2f} "
            f"{_nan(ev['waviness_rel_data']):5.2f} "
            f"{_nan(ev['wedge']['frac']):5.3f} "
            f"{ev['rarefaction']['width_growth_m']:+6.0f} "
            f"{ev['s_layer']['max_m']:6.0f} "
            f"{ev['queue']['N_s_700_veh']:5.1f} "
            f"{(suw['v_recede_cav_ms'] if ok else float('nan')):6.2f} "
            f"{_nan(suw['mass'].get('v_startup_mass_ms')):6.2f} "
            f"{100.0 * _nan(suw['mass'].get('released_frac')):5.0f} "
            f"{ev['downstream_rates']['net_per_s']:+7.4f} "
            f"{llq:5.0f} "
            f"{('deg' if ev['queue']['degenerate'] else 'yes' if ev['admissible']['ok'] else 'NO'):>3s}")


def _rung_order(block):
    prim = [n for n, *_ in RUNGS]
    cons = [n + CONSTRAINED_SUFFIX for n in prim
            if n + CONSTRAINED_SUFFIX in block["rungs"]]
    return prim + cons + [RLL, RDR]


def print_table(akey, block):
    refs = block["data_refs_q2500"]
    print(f"\n===== E9 ladder {akey} (u{UC:g} q{QIN_FIT:g} {FORM}, W1 fit) "
          "=====")
    print(f"data refs: wake={refs['wake']['wake_mean_vehkm']:.1f} veh/km, "
          f"waviness={refs['waviness']['std_vehkm']:.2f} veh/km, "
          f"wedge_frac={refs['wedge']['frac']:.3f}, "
          f"rarefaction_growth={refs['rarefaction']['width_growth_m']:+.0f} m;"
          f"  c_wave: w_s={W_S:.2f} m/s (w_s rungs), w={ev4.W:.2f} m/s (R4)")
    print(f"{'rung':5s} {'kappa_c':>9s} {'kappa_r':>9s} {'W1':>6s} "
          f"{'RMSE':>5s} {'e_s':>7s} {'om_err':>7s} {'wake':>5s} "
          f"{'stuckf':>7s} {'ds|s|':>6s} {'wavi':>5s} {'w/dat':>5s} "
          f"{'wedge':>5s} {'rare_m':>6s} {'slayer':>6s} {'Ns700':>5s} "
          f"{'suw':>6s} {'suw_m':>6s} {'rel%':>5s} {'net_ds':>7s} "
          f"{'llq%':>5s} {'adm':>3s}")
    for name in _rung_order(block):
        print(_row(name, block["rungs"][name]))
    print("  suw = start-up wave: {s>1 veh/km} head recedes rel. to the CAV "
          "over t in [760,900] s [m/s]; suw_m = mass-equivalent front speed "
          "-(dN_s/dt)/max_x s [m/s] (0 when nothing releases after t_fast);\n"
          "  rel% = share of N_s released over [760,900] s; net_ds = "
          "lambda_ds - LL rate [1/s] (closed form; <0: leaked s decays); "
          "llq% = share of the LL release acting inside the queue body "
          "(j <= CAV cell) in [250,750] s;\n"
          "  adm = ds_stuck<=0.01 AND max downstream s<=1 veh/km (deg = "
          "degenerate, N_s(700) < 1 veh, excluded); '_c' rows = constrained "
          "refit (W1 + admissibility penalty, dt=0.5); RLL/RDR = E8 final "
          "kappas, not refit")


# ---------------------------------------------------------------------------
# main program
# ---------------------------------------------------------------------------

def _print_eval(ev, extra=""):
    bud = ev["ll_budget"]
    suw = ev["startup_wave"]
    print(f"[eval dt=0.5] W1={ev['W1']:.2f} RMSE={ev['RMSE']:.2f} "
          f"e_s={ev['e_s']:+.1%} om={ev['omega_err']:+.1%} "
          f"wake={ev['wake']['wake_mean_vehkm']:.1f} "
          f"stuckf={_nan(ev['ds_stuck']['frac']):.4f} "
          f"ds|s|={ev['ds_max_s_vehkm']:.3f} "
          f"slayer={ev['s_layer']['max_m']:.0f} m "
          f"Ns700={ev['queue']['N_s_700_veh']:.1f} "
          f"suw={_fmt(suw.get('v_recede_cav_ms'), '.2f')} m/s "
          f"suw_m={_fmt(suw['mass'].get('v_startup_mass_ms'), '.2f')} m/s "
          f"rel={_fmt(suw['mass'].get('released_frac'), '.0%')} "
          f"net_ds={ev['downstream_rates']['net_per_s']:+.4f} "
          + (f"llq={bud['in_queue_frac']:.0%} " if bud.get("applies")
             and bud.get("in_queue_frac") is not None else "")
          + f"adm={ev['admissible']['ok']}{extra}", flush=True)


def run(smoke=False):
    t_start = time.time()
    OUT_E9.mkdir(parents=True, exist_ok=True)
    e8_final = json.loads(E8_FINAL_CONFIG.read_text())

    results = {"_meta": dict(
        scenario=f"u{UC:g} q{QIN_FIT:g} form={FORM} metric={METRIC}",
        protocol=("primary rungs: e7_wasserstein.fit_field, 6x6 log grid + "
                  "Nelder-Mead maxfev 40 at dt_fit=1, production re-eval "
                  "dt=0.5; fixed knobs via extra_cfg passthrough (NOT "
                  "fitted); A=10 kc_grid=logspace(-2.5,-0.5,6); "
                  "constrained fallback ('_c' rungs, only for rungs whose "
                  "unconstrained optimum is inadmissible): same grid + NM "
                  "protocol at dt=0.5 on W1 + admissibility penalty "
                  f"(PEN_DS={PEN_DS:g} veh km per veh/km of max downstream "
                  f"s above {DS_MAX_S_VEHKM:g}, PEN_STUCK={PEN_STUCK:g} per "
                  f"{DS_STUCK_MAX:g} of stuck fraction above "
                  f"{DS_STUCK_MAX:g}); RDR/RLL = out/e8/final_config.json "
                  "kappas re-evaluated, NOT refit"),
        rungs=({n: dict(extra_cfg=dict(x), description=d)
                for n, x, d in RUNGS}
               | {RLL: dict(extra_cfg=dict(RLL_CFG), description=RLL_DESC),
                  RDR: dict(extra_cfg=dict(RDR_CFG), description=RDR_DESC)}),
        downstream_release_policy=("used ONLY in the RDR reference row; "
                                   "every R-rung is state-only (eta_la)"),
        knobs=dict(q_xi_max_vehs=QXI, q_xi_max_vehh=QXI * 3600.0,
                   ws_frac=WS_FRAC_FIX, w_s_ms=W_S, w_ms=float(ev4.W),
                   eta_main_m=ETA_MAIN,
                   ll_rate_per_s={f"eta{e:g}_ws": _ll_rate({"eta_la": e, "w_s": W_S})
                                  for e in (100.0, 200.0, 400.0)}
                   | {"eta200_w": _ll_rate({"eta_la": 200.0})}),
        dt_fit=1.0, dt_production=e7.DT_PRODUCTION,
        pick_rule=(f"admissible = ds_stuck(t=700) <= {DS_STUCK_MAX:g} AND "
                   f"max downstream s <= {DS_MAX_S_VEHKM:g} veh/km (all "
                   "snapshots, data cells fully inside the DR zone); "
                   "candidates = primary R-rungs + constrained refits, "
                   f"excluding degenerate ones (N_s(700) < {DEGENERATE_NS_VEH:g} "
                   "veh: no stuck population, i.e. not a catch & release "
                   "model); among admissible candidates pick min W1 at "
                   "dt=0.5 (q2500); RDR/RLL are reference only; if none is "
                   "admissible pick=None and figures/transfer use the "
                   "min-W1 rung, flagged"),
        verdict_rule=(f"reproduces_kludge = admissible AND |W1 rel diff| <= "
                      f"{VERDICT_W1_TOL:.0%} AND |wake diff| <= "
                      f"{VERDICT_WAKE_TOL:g} veh/km vs RDR"),
        diagnostics=dict(
            e8=("wake, ds_stuck, waviness, wedge, rarefaction, s_layer, "
                "ds_max_s: e8_ladder.eval_rung (imported); definitions in "
                "out/e8/ladder.json _meta"),
            waviness_rel_data="waviness_sim / waviness_data (nominal CAV)",
            startup_wave=startup_wave.__doc__ + " " + _track_head.__doc__,
            ll_budget=ll_release_budget.__doc__,
            downstream_rates=downstream_rates.__doc__,
            queue="N_s at t=700 s and at t_end [veh] (regrid_sim N_s)"),
        smoke=smoke)}

    for A in (1.0, 10.0):
        akey = f"A{A:g}"
        tag = f"A{A:g}_u{UC:g}_q{QIN_FIT:g}"
        refs = L.data_refs(A, UC, QIN_FIT)
        block = dict(tag=tag, data_refs_q2500=refs, rungs={})
        kept = {}

        # ---- references: E8 final kappas, re-evaluated, NOT refit --------
        e8k = e8_final[tag]
        kc8, kr8 = float(e8k["kappa_c"]), float(e8k["kappa_r"])
        w1_e8 = float(e8k["summary"]["W1"])
        for name, cfg, desc in ((RDR, RDR_CFG, RDR_DESC),
                                (RLL, RLL_CFG, RLL_DESC)):
            print(f"\n=== {akey} {name}: {desc} ===", flush=True)
            ev, regr, meas = evaluate(A, QIN_FIT, kc8, kr8, cfg, refs)
            row = dict(metric=METRIC, description=desc, extra_cfg=dict(cfg),
                       kappa_c=kc8, kappa_r=kr8, refit=False,
                       source=str(E8_FINAL_CONFIG.relative_to(HERE)),
                       eval_q2500=ev)
            if name == RDR:
                repro = bool(abs(ev["W1"] - w1_e8) <= 1e-6 * max(1.0, w1_e8))
                row.update(e8_final_W1=w1_e8, reproduces_e8_final=repro)
                _print_eval(ev, f" | E8 final_config W1 {w1_e8:.2f}, "
                                f"reproduced={repro}")
                if not repro:
                    print(f"[{akey}] WARNING: RDR re-evaluation does not "
                          "reproduce out/e8/final_config.json W1 "
                          f"({ev['W1']:.4f} vs {w1_e8:.4f}); the solver's "
                          "default path may have changed", flush=True)
            else:
                _print_eval(ev)
            block["rungs"][name] = row
            kept[name] = (regr, meas, cfg, kc8, kr8)

        # ---- primary R-rungs: fit_field as prescribed --------------------
        for name, cfg, desc in RUNGS:
            print(f"\n=== {akey} {name}: {desc} ===", flush=True)
            fit = fit_rung(A, cfg, smoke)
            ev, regr, meas = evaluate(A, QIN_FIT, fit["kappa_c"],
                                      fit["kappa_r"], cfg, refs)
            block["rungs"][name] = dict(
                metric=METRIC, description=desc, extra_cfg=dict(cfg),
                kappa_c=fit["kappa_c"], kappa_r=fit["kappa_r"], refit=True,
                stage="primary (fit_field, dt_fit=1)",
                fit_objective_dtfit=fit["objective"],
                fit_grid_objective=fit["grid_objective"],
                n_sim_fit=fit["n_sim"], fit_at_production=fit["at_production"],
                eval_q2500=ev)
            kept[name] = (regr, meas, cfg, fit["kappa_c"], fit["kappa_r"])
            _print_eval(ev)

        # ---- fallback: constrained refit of inadmissible rungs -----------
        block["constrained_stage"] = {}
        for name, cfg, desc in RUNGS:
            if block["rungs"][name]["eval_q2500"]["admissible"]["ok"]:
                continue
            name_c = name + CONSTRAINED_SUFFIX
            print(f"\n=== {akey} {name_c}: {desc} [constrained refit: W1 + "
                  "admissibility penalty, dt=0.5] ===", flush=True)
            fit = fit_constrained(A, cfg, smoke)
            ev, regr, meas = evaluate(A, QIN_FIT, fit["kappa_c"],
                                      fit["kappa_r"], cfg, refs)
            block["rungs"][name_c] = dict(
                metric=METRIC + " + admissibility penalty",
                description=desc + " [constrained refit]",
                extra_cfg=dict(cfg), kappa_c=fit["kappa_c"],
                kappa_r=fit["kappa_r"], refit=True,
                stage="constrained fallback (dt=0.5, penalized W1)",
                fit_objective=fit["objective"],
                fit_grid_objective=fit["grid_objective"],
                fit_W1=fit["W1_at_fit"], fit_ds_stuck=fit["ds_stuck_at_fit"],
                fit_ds_max_s=fit["ds_max_s_at_fit"], n_sim_fit=fit["n_sim"],
                penalties=fit["penalties"], eval_q2500=ev)
            kept[name_c] = (regr, meas, cfg, fit["kappa_c"], fit["kappa_r"])
            active = bool(
                ev["ds_max_s_vehkm"] >= CONSTRAINT_ACTIVE_FRAC * DS_MAX_S_VEHKM
                or (ev["ds_stuck"]["frac"] or 0.0)
                >= CONSTRAINT_ACTIVE_FRAC * DS_STUCK_MAX)
            block["rungs"][name_c]["constraint_active"] = active
            block["constrained_stage"][name] = dict(
                constrained_rung=name_c,
                admissible=ev["admissible"]["ok"],
                degenerate=ev["queue"]["degenerate"],
                constraint_active=active,
                W1_unconstrained=block["rungs"][name]["eval_q2500"]["W1"],
                W1_constrained=ev["W1"],
                W1_cost_of_admissibility=(
                    ev["W1"] - block["rungs"][name]["eval_q2500"]["W1"]))
            _print_eval(ev, f" | W1 cost vs unconstrained {name}: "
                            f"{ev['W1'] - block['rungs'][name]['eval_q2500']['W1']:+.1f}")

        # ---- pick: admissibility first, then min W1 ---------------------
        prim = [n for n, *_ in RUNGS]
        cands = prim + [n for n in block["rungs"]
                        if n.endswith(CONSTRAINED_SUFFIX)]
        w1_of = {n: block["rungs"][n]["eval_q2500"]["W1"] for n in cands}
        degen = [n for n in cands
                 if block["rungs"][n]["eval_q2500"]["queue"]["degenerate"]]
        adm = [n for n in cands
               if block["rungs"][n]["eval_q2500"]["admissible"]["ok"]
               and n not in degen]
        opt_all = min(prim, key=lambda n: w1_of[n])
        pick = min(adm, key=w1_of.get) if adm else None
        fig_rung = pick if pick is not None else opt_all
        block["candidates"] = cands
        block["degenerate_rungs"] = degen
        block["admissible_rungs"] = adm
        if degen:
            print(f"\n[{akey}] degenerate (no stuck population, excluded "
                  f"from the pick): {degen}", flush=True)
        block["w1_optimal_unconstrained"] = dict(rung=opt_all,
                                                  W1=w1_of[opt_all])
        block["pick"] = pick
        block["figure_rung"] = fig_rung
        if pick is None:
            block["pick_rule"] = (
                "NO ADMISSIBLE RUNG (ds_stuck <= 0.01 AND max downstream s "
                "<= 1 veh/km fails for every primary and constrained "
                f"R-rung); figures/transfer use the unconstrained min-W1 "
                f"rung {opt_all}, flagged")
            print(f"\n[{akey}] NO ADMISSIBLE RUNG; figures/transfer use "
                  f"{opt_all} (flagged)", flush=True)
        else:
            block["pick_rule"] = (
                f"admissible candidates {adm}; min W1 among them = {pick} "
                f"(W1={w1_of[pick]:.1f}); unconstrained min-W1 primary rung "
                f"= {opt_all} (W1={w1_of[opt_all]:.1f}, admissible="
                f"{block['rungs'][opt_all]['eval_q2500']['admissible']['ok']})")
            print(f"\n[{akey}] admissible: {adm}; pick = {pick} "
                  f"(W1={w1_of[pick]:.1f}; unconstrained optimum {opt_all} "
                  f"W1={w1_of[opt_all]:.1f})", flush=True)

        # ---- verdict: pick, R1 (+R1_c), RLL vs the kludge ----------------
        block["verdict"] = {"pick": compare_to_rdr(block, fig_rung)}
        for n in ("R1", "R1" + CONSTRAINED_SUFFIX, RLL):
            if n in block["rungs"] and n != fig_rung:
                block["verdict"][n] = compare_to_rdr(block, n)
        for k, v in block["verdict"].items():
            print(f"[{akey} verdict {k}] {v['statement']}", flush=True)

        # horizon sensitivity and the w_s question, in numbers
        def _hs(n):
            e = block["rungs"][n]["eval_q2500"]
            return dict(eta_la_m=block["rungs"][n]["extra_cfg"].get("eta_la"),
                        kappa_c=block["rungs"][n]["kappa_c"],
                        kappa_r=block["rungs"][n]["kappa_r"], W1=e["W1"],
                        wake=e["wake"]["wake_mean_vehkm"],
                        ds_max_s=e["ds_max_s_vehkm"],
                        ds_stuck=e["ds_stuck"]["frac"],
                        startup_wave_ms=e["startup_wave"].get("v_recede_cav_ms"),
                        ll_in_queue_frac=e["ll_budget"].get("in_queue_frac"),
                        admissible=e["admissible"]["ok"])
        block["horizon_sensitivity"] = {
            n: _hs(n) for n in ("R2", "R1", "R3", "R2_c", "R1_c", "R3_c")
            if n in block["rungs"]}
        ws_rows = {n: _hs(n) for n in ("R1", "R4", "R1_c", "R4_c")
                   if n in block["rungs"]}
        best_ws = min((n for n in ws_rows if n.startswith("R1")
                       and ws_rows[n]["admissible"]),
                      key=lambda n: ws_rows[n]["W1"], default=None)
        best_nows = min((n for n in ws_rows if n.startswith("R4")
                         and ws_rows[n]["admissible"]),
                        key=lambda n: ws_rows[n]["W1"], default=None)
        block["ws_needed"] = dict(
            rows=ws_rows, best_admissible_with_ws=best_ws,
            best_admissible_without_ws=best_nows,
            verdict=("no admissible rung on one side; compare unconstrained "
                     f"W1: R1 {ws_rows['R1']['W1']:.1f} vs R4 "
                     f"{ws_rows['R4']['W1']:.1f}"
                     if best_ws is None or best_nows is None else
                     (f"w_s still needed: {best_ws} W1 "
                      f"{ws_rows[best_ws]['W1']:.1f} < {best_nows} W1 "
                      f"{ws_rows[best_nows]['W1']:.1f}"
                      if ws_rows[best_ws]["W1"] < ws_rows[best_nows]["W1"]
                      else f"w_s NOT needed on W1: {best_nows} W1 "
                      f"{ws_rows[best_nows]['W1']:.1f} <= {best_ws} W1 "
                      f"{ws_rows[best_ws]['W1']:.1f}")))
        print(f"[{akey} w_s?] {block['ws_needed']['verdict']}", flush=True)

        # ---- zero-refit q2000 transfer: pick, every admissible candidate,
        #      RLL and RDR (same contrast) --------------------------------
        regr_p, meas_p, cfg_p, kc_p, kr_p = kept[fig_rung]
        refs_t = L.data_refs(A, UC, QIN_TRANSFER)
        block["data_refs_q2000"] = refs_t
        block["transfer_q2000"] = {}
        for n in ([fig_rung] + [c for c in adm if c != fig_rung]
                  + [RLL, RDR]):
            _, _, cfg_n, kc_n, kr_n = kept[n]
            ev_t, _, _ = evaluate(A, QIN_TRANSFER, kc_n, kr_n, cfg_n, refs_t)
            block["transfer_q2000"][n] = ev_t
            print(f"[{akey}] transfer q2000 {n:5s}: W1={ev_t['W1']:.2f} "
                  f"RMSE={ev_t['RMSE']:.2f} e_s={ev_t['e_s']:+.1%} "
                  f"om={ev_t['omega_err']:+.1%} "
                  f"wake={ev_t['wake']['wake_mean_vehkm']:.1f} "
                  f"stuckf={_nan(ev_t['ds_stuck']['frac']):.4f} "
                  f"ds|s|={ev_t['ds_max_s_vehkm']:.3f} "
                  f"Ns700={ev_t['queue']['N_s_700_veh']:.1f} "
                  f"rel={_fmt(ev_t['startup_wave']['mass'].get('released_frac'), '.0%')}"
                  f" adm={ev_t['admissible']['ok']}"
                  + ("  <- pick" if n == fig_rung else ""), flush=True)
        block["pick_transfer_q2000"] = dict(
            rung=fig_rung, flagged_not_admissible=(pick is None),
            **block["transfer_q2000"][fig_rung])

        # ---- figures for the pick ----------------------------------------
        regr_d, _, _, _, _ = kept[RDR]
        rdr_lbl = f"{RDR}: cap + w_s + downstream_release (E8 κ, kludge)"
        win_lbl = ("E9 " + rung_label(fig_rung, cfg_p)
                   + ("" if pick is not None else " [NOT admissible]"))
        fig_tag = ("smoke_" if smoke else "") + tag
        fig_profiles_e9(tag, meas_p, regr_d, rdr_lbl, regr_p, win_lbl,
                        OUT_E9, fname_tag=fig_tag)
        fig_heat3_e9(tag, meas_p, regr_d, rdr_lbl, regr_p, win_lbl, OUT_E9,
                     fname_tag=fig_tag)
        block["figures"] = dict(
            profiles=f"fig_profiles_e9_{fig_tag}.png",
            heat3=f"fig_heat3_e9_{fig_tag}.png",
            labels=dict(data="SUMO (rep mean)", gray_dashed=rdr_lbl,
                        blue=win_lbl))
        # if the winner carries no leader-loss term, also draw the best
        # admissible LL rung so the general term itself can be inspected
        if cfg_p.get("eta_la") is None:
            ll_adm = [n for n in adm
                      if block["rungs"][n]["extra_cfg"].get("eta_la")
                      is not None]
            if ll_adm:
                n_ll = min(ll_adm, key=w1_of.get)
                regr_l, meas_l, cfg_l, _, _ = kept[n_ll]
                ll_lbl = "E9 " + rung_label(n_ll, cfg_l) + " (best admissible LL)"
                fig_profiles_e9(tag, meas_l, regr_d, rdr_lbl, regr_l, ll_lbl,
                                OUT_E9, fname_tag=fig_tag + "_bestLL")
                fig_heat3_e9(tag, meas_l, regr_d, rdr_lbl, regr_l, ll_lbl,
                             OUT_E9, fname_tag=fig_tag + "_bestLL")
                block["figures"]["bestLL"] = dict(
                    rung=n_ll,
                    profiles=f"fig_profiles_e9_{fig_tag}_bestLL.png",
                    heat3=f"fig_heat3_e9_{fig_tag}_bestLL.png",
                    blue=ll_lbl)
        results[akey] = block
        print_table(akey, block)

    # ---- overall statement ----------------------------------------------
    lines = []
    for akey in ("A1", "A10"):
        b = results[akey]
        lines.append(f"{akey}: pick={b['pick']} -> "
                     f"{b['verdict']['pick']['statement']}")
        if RLL in b["verdict"]:
            lines.append(f"{akey}: like-for-like swap -> "
                         f"{b['verdict'][RLL]['statement']}")
        hs = b["horizon_sensitivity"]
        lines.append(f"{akey}: horizon sensitivity (W1 / max ds s / adm): "
                     + ", ".join(f"{n} eta={v['eta_la_m']:g}: {v['W1']:.1f} / "
                                 f"{v['ds_max_s']:.2f} / "
                                 f"{'yes' if v['admissible'] else 'no'}"
                                 for n, v in hs.items())
                     + f"; w_s: {b['ws_needed']['verdict']}")
        tq = b["pick_transfer_q2000"]
        lines.append(f"{akey}: q2000 transfer of {tq['rung']}: W1 "
                     f"{tq['W1']:.1f} (RDR "
                     f"{b['transfer_q2000'][RDR]['W1']:.1f}), e_s "
                     f"{tq['e_s']:+.1%}, max ds s {tq['ds_max_s_vehkm']:.2f} "
                     f"veh/km, admissible={tq['admissible']['ok']}")
    results["_meta"]["overall_statement"] = lines
    results["_meta"]["runtime_s"] = round(time.time() - t_start, 1)
    out_path = OUT_E9 / ("ladder_smoke.json" if smoke else "ladder.json")
    out_path.write_text(json.dumps(results, indent=2, default=_json_default))
    print("\n===== does the general leader-loss term reproduce the kludge? "
          "=====")
    for ln in lines:
        print(ln)
    print(f"\nwrote {out_path} ({results['_meta']['runtime_s']} s)")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="E9 leader-loss release ladder (state-only replacement "
                    "for downstream_release)")
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
