"""E10: attribution ladder for the state-only moving-bottleneck model.

Mladen's verdict on E8: downstream_release was a KLUDGE -- a label
conversion keyed to x_cav, i.e. scenario information inside a source term.
A general model's source terms must be state functions; the scenario is
then instantiated on top.  The E9 ratio-form leader-loss term was the first
attempt (out/e9/ladder.json): state-only, but it could not reproduce the
kludge without a constrained refit and released inside the queue body.

E10 replaces it by two ingredients, both in solver.py (E10 state, 24/24
tests):

  (I)  s-impermeable moving-bottleneck interface (SimConfig.s_impermeable):
       catch & release DEFINES release as becoming free, so only free
       vehicles overtake the bottleneck -- F^s = 0 at the CAV's downstream
       face(s) while the bottleneck is active (t in [t_slow, t_fast], u_cav
       < v_f), then the DM-G cap acts on what remains.  An interface flux
       constraint of the same category as the cap, not a source term.
  (II) connectivity leader-loss (eta_la, ll_mode="connect", tau_ll):
       mu_ll = (1 - cover)/tau_ll with cover the LEAST fixed point of
       cover_j = 1 if an A vehicle is within the look-ahead, else
       min{1, dx sum_{k=j+1..j+n} s_k cover_k}: coverage propagates
       backward from a slow leader through contiguous s and nowhere else.
       In simulate() the A vehicle anchors coverage only while class A is
       slow (u_s(t) < v_f) -- the same class-speed input that enters
       Delta v, no position information -- so after t_fast the queue is
       leaderless and dissolves at 1/tau_ll: the start-up wave.

Ladder (u15 q2500, lf, W1 metric, A in {1, 10}); every rung's (kappa_c,
kappa_r) is refit with the E7 protocol (e7_wasserstein.fit_field: 6x6 log
grid + Nelder-Mead maxfev 40 at dt_fit=1, production re-eval dt=0.5; A=10
kc_grid=logspace(-2.5,-0.5,6); fixed knobs via the extra_cfg passthrough,
NOT the optimization vector):

  R0   cap 2000 veh/h + w_s=0.6w                     leaky reference
  R1   R0 + s_impermeable                            interface alone
  R2   R1 + connect-LL (eta_la 200 m, tau_ll 3 s)    THE E10 MODEL
  R3   R0 + connect-LL only (no impermeable face)    attribution: LL alone?
  R4   R2 without w_s                                is w_s still needed?
  R5   R2 with tau_ll = 10 s                         sensitivity
  R6   R2 with eta_la = 400 m                        sensitivity
  RDR  cap + w_s + downstream_release, E8 final kappas
       (out/e8/final_config.json) re-evaluated -- reference only, never a
       winner; its W1 must reproduce final_config.json exactly

Per rung (production dt=0.5): kappas, W1, RMSE, e_s, omega_err, wake density
(data A1 58.6 / A10 55.4), downstream stuck fraction at t=700, max
downstream s (E8 definition: all snapshots with the CAV on road, data cells
fully beyond the CAV's solver cell), waviness, wedge, post-release
rarefaction growth, s-layer extent, N_s(700), N_s(1000), the E9 mass-based
start-up-wave number (v = -(dN_s/dt)/max_x s over [760, 900] s and the
fraction of N_s released there), plus two E10 additions:

  ds split     the E8 max-downstream-s split into the slow window
               [t_slow, t_fast] (the bottleneck is active: with the
               impermeable face this is 0 EXACTLY by construction) and the
               post-t_fast window (the CAV drives at v_cav_free = 27.78 m/s
               < v_f = 27.94 m/s, so an UNRELEASED s-platoon rides past it
               on the s-class free branch and is counted as downstream s --
               the signature of a missing release mechanism, not of a
               leak through the interface); both also beyond the CAV's
               two-cell discrete footprint.
  LL budget    where the connectivity release acts (saved states, 10 s
               cadence, exact per-cell decay 1 - exp(-mu dt) over the
               cadence, an estimate): slow window in-queue (j <= CAV cell)
               vs downstream, and post-t_fast; the in-queue slow-window
               release vs the kappa_r release over the same cells.

Admissibility (E9 rule, no constrained-refit fallback): ds_stuck(700) <=
0.01 AND max downstream s <= 1 veh/km.  With the impermeable interface the
primary fits should be admissible by construction; if one is not, the ds
split says which window fails and why.  Degenerate rungs (N_s(700) < 1
veh: no stuck population, not a catch & release model) are flagged and
excluded from the winner, as in E9.  Winner per A = min W1 among
admissible non-degenerate rungs (RDR is reference only).  Zero-refit q2000
transfer of every rung and RDR, admissibility re-checked there.  Figures
for the winners: fig_profiles_e10_{tag}.png (t=300/500/700/850; data
black, RDR gray dashed, winner blue with s/f fills, actual labels in the
title) and fig_heat3_e10_{tag}.png; if the winner is not R2 the E10 model
itself is drawn as well (suffix _R2).

Three questions, answered explicitly (JSON _meta.answers and stdout):
  (i)   does R1 alone eliminate the downstream plume and the plug?
  (ii)  does the E10 model (R2) match or beat the RDR kludge on W1, wake,
        rarefaction, e_s and omega?
  (iii) transfer admissibility at q2000.

CLI:  python3 e10_ladder.py --run     -> out/e10/ladder.json + figures
      python3 e10_ladder.py --smoke   -> out/e10/ladder_smoke.json (tiny grid)
"""

from __future__ import annotations

import argparse
import json
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import e7_wasserstein as e7
import e8_ladder as L              # eval_rung / data_refs / diagnostics
import e9_ladder as e9             # startup_wave / admissibility / _raw_sim
import ev4_compare as ev4
from loader import CELL_LEN
from solver import leader_loss_rate_connect
from solver import speed as fd_speed

HERE = Path(__file__).parent
OUT_E10 = HERE / "out" / "e10"
E8_FINAL_CONFIG = HERE / "out" / "e8" / "final_config.json"

UC = 15.0                      # [m/s]   focus CAV slow speed
QIN_FIT = 2500.0               # [veh/h] fit inflow
QIN_TRANSFER = 2000.0          # [veh/h] zero-refit transfer inflow
FORM = "lf"
METRIC = "w1"
QXI = 2000.0 / 3600.0          # [veh/s] one-lane-blocked capacity cap
WS_FRAC_FIX = 0.6
W_S = WS_FRAC_FIX * ev4.W      # [m/s]   stuck-class wave speed
ETA_MAIN = 200.0               # [m]     connectivity look-ahead
ETA_SENS = 400.0               # [m]     sensitivity
TAU_MAIN = 3.0                 # [s]     connect-form release time scale
TAU_SENS = 10.0                # [s]     sensitivity
SOLVER_DX = L.SOLVER_DX
KC_GRID_A10 = np.logspace(-2.5, -0.5, 6)   # E6/E-V3 magnitude regime
KR_GRID = np.logspace(-4.0, -0.5, 6)       # fit_field default
NM_MAXFEV = 40

DS_STUCK_MAX = e9.DS_STUCK_MAX             # 0.01
DS_MAX_S_VEHKM = e9.DS_MAX_S_VEHKM         # 1 veh/km
DEGENERATE_NS_VEH = e9.DEGENERATE_NS_VEH   # 1 veh
T_SLOW, T_FAST = ev4.T_SLOW, ev4.T_FAST
FOOTPRINT_CELLS = 2            # CAV linear split over two solver cells

# verdict tolerances (rung vs RDR): 'match' if within, 'beat' if better
TOL_W1_REL = 0.05
TOL_WAKE_VEHKM = 3.0
TOL_RARE_M = 300.0             # three data cells
TOL_ES = 0.05
TOL_OMEGA = 0.05

RHO_CRIT = L.RHO_CRIT
X_LO = L.X_LO
X_UP = L.X_UP
PROFILE_SNAPS = (300.0, 500.0, 700.0, 850.0)

_LL = dict(eta_la=ETA_MAIN, ll_mode="connect", tau_ll=TAU_MAIN)
RUNGS = (
    # (name, extra_cfg, description)
    ("R0", {"q_xi_max": QXI, "w_s": W_S},
     "cap 2000 + w_s=0.6w (leaky reference)"),
    ("R1", {"q_xi_max": QXI, "w_s": W_S, "s_impermeable": True},
     "R0 + s_impermeable (interface constraint alone)"),
    ("R2", {"q_xi_max": QXI, "w_s": W_S, "s_impermeable": True, **_LL},
     f"R1 + connect-LL (eta_la {ETA_MAIN:g} m, tau_ll {TAU_MAIN:g} s): "
     "the E10 model"),
    ("R3", {"q_xi_max": QXI, "w_s": W_S, **_LL},
     "R0 + connect-LL only, no impermeable face (does LL alone suffice?)"),
    ("R4", {"q_xi_max": QXI, "s_impermeable": True, **_LL},
     "R2 without w_s (is w_s still needed?)"),
    ("R5", {"q_xi_max": QXI, "w_s": W_S, "s_impermeable": True,
            **dict(_LL, tau_ll=TAU_SENS)},
     f"R2 with tau_ll = {TAU_SENS:g} s (sensitivity)"),
    ("R6", {"q_xi_max": QXI, "w_s": W_S, "s_impermeable": True,
            **dict(_LL, eta_la=ETA_SENS)},
     f"R2 with eta_la = {ETA_SENS:g} m (sensitivity)"),
)
E10_MODEL = "R2"
RDR = "RDR"
RDR_CFG = {"q_xi_max": QXI, "w_s": W_S, "downstream_release": True}
RDR_DESC = ("cap + w_s=0.6w + downstream_release, E8 final kappas "
            "re-evaluated (NOT refit; the kludge; reference only, never a "
            "winner)")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fmt(v, spec):
    return "n/a" if v is None else format(v, spec)


def _nan(v):
    return float("nan") if v is None else v


def rung_label(name, cfg):
    parts = ["cap"]
    if cfg.get("w_s") is not None:
        parts.append(f"w_s={WS_FRAC_FIX:g}w")
    if cfg.get("s_impermeable"):
        parts.append("s-impermeable")
    if cfg.get("downstream_release"):
        parts.append("downstream_release")
    if cfg.get("eta_la") is not None:
        parts.append(f"connect-LL(eta {cfg['eta_la']:g} m, tau "
                     f"{cfg.get('tau_ll', TAU_MAIN):g} s)")
    return f"{name}: " + " + ".join(parts)


# ---------------------------------------------------------------------------
# E10 diagnostics
# ---------------------------------------------------------------------------

def ds_max_s_split(regr):
    """The E8 max-downstream-s (data cells with lower edge beyond the
    CAV's solver cell, all snapshots with the CAV on road) split into the
    slow window [T_SLOW, T_FAST] and the post-t_fast window, each also
    beyond the CAV's FOOTPRINT_CELLS-cell discrete support."""
    tt = np.asarray(regr["tt"], float)
    out = {}
    for key, lo, hi in (("slow", T_SLOW, T_FAST), ("post", T_FAST, np.inf)):
        worst = worst_fp = 0.0
        n = 0
        for i in range(len(tt)):
            t = tt[i]
            inside = (lo <= t <= hi) if key == "slow" else (t > lo)
            xc = float(regr["x_cav"][i])
            if not inside or not np.isfinite(xc):
                continue
            n += 1
            j_cav = np.floor(xc / SOLVER_DX)
            sel = X_LO >= (j_cav + 1.0) * SOLVER_DX
            sel_fp = X_LO >= (j_cav + FOOTPRINT_CELLS) * SOLVER_DX
            if sel.any():
                worst = max(worst, float(np.max(np.abs(regr["s"][i, sel]))))
            if sel_fp.any():
                worst_fp = max(worst_fp,
                               float(np.max(np.abs(regr["s"][i, sel_fp]))))
        out[key] = dict(max_s_vehkm=worst, max_s_beyond_footprint_vehkm=worst_fp,
                        n_snapshots=n)
    out["definition"] = (
        "E8 ds_max_s restricted to t in [t_slow, t_fast] ('slow', the "
        "bottleneck is active) and t > t_fast ('post'); "
        "'beyond_footprint' excludes data cells overlapping the CAV's "
        f"{FOOTPRINT_CELLS}-cell linear-split support as well")
    return out


def ll_budget_connect(uc, qin, kc, kr, extra_cfg):
    """Where the connectivity leader-loss release acts.  Saved states at
    the 10 s cadence (an estimate): per cell the exact decay over the
    cadence, s dx (1 - exp(-mu_ll dt_save)), with mu_ll from
    leader_loss_rate_connect on the saved (a, s) and the simulate() anchor
    rule (A anchors only while u_s(t) < v_f).  Split: slow window
    [T_SLOW, T_FAST] in-queue (j <= CAV cell) vs strictly downstream, and
    post-t_fast, which is exact (N_s(t_fast) - N_s(t_end): with Delta v = 0
    after t_fast the leader-loss term is the only s-sink); the kappa_r
    release over the slow-window in-queue cells for comparison."""
    eta = extra_cfg.get("eta_la")
    if eta is None:
        return dict(applies=False, status="no leader-loss term in this rung")
    tau = float(extra_cfg.get("tau_ll", TAU_MAIN))
    res = e9._raw_sim(uc, qin, kc, kr, extra_cfg)
    dx = float(res.x[1] - res.x[0])
    save_dt = float(np.median(np.diff(res.t)))
    parts = dict(slow_in_queue=0.0, slow_downstream=0.0, post_fast=0.0)
    kr_slow_in_queue = 0.0
    ns_tfast = None
    for i in range(res.t.size):
        t = float(res.t[i])
        if t < T_SLOW:
            continue
        a_i, f_i, s_i = res.a[i], res.f[i], res.s[i]
        slow = t <= T_FAST
        a_anchor = a_i if slow else np.zeros_like(a_i)
        mu = leader_loss_rate_connect(a_anchor, s_i, dx, float(eta), tau)
        rel = s_i * dx * (1.0 - np.exp(-mu * save_dt))       # [veh]
        if slow:
            xc = float(res.x_cav[i])
            if not np.isfinite(xc):
                continue
            jc = int(xc // dx)
            parts["slow_in_queue"] += float(rel[:jc + 1].sum())
            parts["slow_downstream"] += float(rel[jc + 1:].sum())
            rho = a_i + f_i + s_i
            dv = np.maximum(fd_speed(rho, ev4.V_F, ev4.W, ev4.P) - uc, 0.0)
            mu_kr = kr * np.maximum(ev4.P - rho, 0.0) * dv
            kr_slow_in_queue += float(
                (s_i * dx * (1.0 - np.exp(-mu_kr * save_dt)))[:jc + 1].sum())
            if ns_tfast is None or t <= T_FAST:
                ns_tfast = float(res.N_s[i])
        else:
            parts["post_fast"] += float(rel.sum())
    # After t_fast, Delta v = 0 (u_s = v_f) so the leader-loss term is the
    # ONLY s-sink: the post-fast release is exactly N_s(t_fast) - N_s(end)
    # (telescoping; the s-platoon is far from the road end).  The cadence
    # estimate above misses a queue that dissolves between two saves
    # (tau_ll = 3 s vs 10 s cadence), so it is kept only for the record.
    post_fast_estimate = parts["post_fast"]
    i_f = int(np.argmin(np.abs(res.t - T_FAST)))
    parts["post_fast"] = float(max(res.N_s[i_f] - res.N_s[-1], 0.0))
    total = sum(parts.values())
    frac = {k: (v / total if total > 0.0 else None) for k, v in parts.items()}
    return dict(
        applies=True, eta_la_m=float(eta), tau_ll_s=tau,
        save_cadence_s=save_dt, total_ll_release_veh=total,
        parts_veh=parts, parts_frac=frac,
        post_fast_cadence_estimate_veh=post_fast_estimate,
        post_fast_rule=("exact: N_s(t_fast) - N_s(t_end), the LL term being "
                        "the only s-sink after t_fast; slow-window parts are "
                        "the 10 s-cadence estimate"),
        kappa_r_release_slow_in_queue_veh=kr_slow_in_queue,
        ll_over_kappa_r_slow_in_queue=(
            parts["slow_in_queue"] / kr_slow_in_queue
            if kr_slow_in_queue > 0.0 else None),
        N_s_at_t_fast_veh=ns_tfast,
        post_fast_release_over_N_s_tfast=(
            parts["post_fast"] / ns_tfast if ns_tfast else None),
        definition=ll_budget_connect.__doc__)


def queue_dissolution(regr):
    """Queue dissolution after t_fast from the saved N_s (10 s cadence):
    N_s at t_fast, and the time after t_fast at which N_s first drops
    below 50 % / 1 % of N_s(t_fast) (None if never within the horizon;
    the cadence is the resolution, so 10 s means 'by the first save').
    The E9 mass-based start-up-wave number assumes a front receding
    through the platoon over [760, 900] s; a queue that dissolves
    uniformly at 1/tau_ll within a few tau_ll of t_fast is gone before
    that window starts, which makes the E9 number ill-conditioned (N_s and
    max_x s both ~0 there) -- these times are the robust statement."""
    tt = np.asarray(regr["tt"], float)
    ns = np.asarray(regr["N_s"], float)
    i_f = L._snap(tt, T_FAST)
    n0 = float(ns[i_f])
    out = dict(N_s_t_fast_veh=n0, t_half_s=None, t_99_s=None,
               N_s_800_veh=float(ns[L._snap(tt, 800.0)]),
               cadence_s=float(np.median(np.diff(tt))),
               definition=queue_dissolution.__doc__)
    if n0 <= 0.0:
        out["status"] = "no queue at t_fast"
        return out
    for key, frac in (("t_half_s", 0.5), ("t_99_s", 0.01)):
        idx = np.where((tt > tt[i_f]) & (ns < frac * n0))[0]
        out[key] = float(tt[idx[0]] - tt[i_f]) if idx.size else None
    out["status"] = "ok"
    return out


def evaluate(A, qin, kc, kr, extra_cfg, refs):
    ev, regr, meas = L.eval_rung(A, UC, qin, kc, kr, extra_cfg)
    tt = regr["tt"]
    ev["startup_wave"] = e9.startup_wave(regr, extra_cfg)
    ev["dissolution"] = queue_dissolution(regr)
    ev["ds_split"] = ds_max_s_split(regr)
    ev["ll_budget"] = ll_budget_connect(UC, qin, kc, kr, extra_cfg)
    i700 = L._snap(tt, 700.0)
    i1000 = L._snap(tt, 1000.0)
    ns700 = float(regr["N_s"][i700])
    ev["queue"] = dict(
        N_s_700_veh=ns700, N_s_1000_veh=float(regr["N_s"][i1000]),
        N_s_end_veh=float(regr["N_s"][-1]),
        n_caught_data_700_veh=float(meas["n_caught_mean"][i700]),
        degenerate=bool(ns700 < DEGENERATE_NS_VEH),
        degenerate_rule=f"N_s(700) < {DEGENERATE_NS_VEH:g} veh: no stuck "
                        "population (not a catch & release model)")
    wd = refs["waviness"]["std_vehkm"]
    ws = ev["waviness_sim"]["std_vehkm"]
    ev["waviness_rel_data"] = (ws / wd if (ws is not None and wd) else None)
    ev["rarefaction_data_growth_m"] = refs["rarefaction"]["width_growth_m"]
    ev["wake_data_vehkm"] = refs["wake"]["wake_mean_vehkm"]
    ev["admissible"] = e9.admissibility(ev)
    return ev, regr, meas


# ---------------------------------------------------------------------------
# fit: the E7 protocol, nothing else
# ---------------------------------------------------------------------------

def fit_rung(A, extra_cfg, smoke):
    fk = dict(kr_grid=KR_GRID, maxfev=NM_MAXFEV)
    if smoke:
        fk = dict(kc_grid=np.array([0.05, 0.2]), kr_grid=np.array([0.002, 0.02]),
                  maxfev=2)
    elif A == 10.0:
        fk["kc_grid"] = KC_GRID_A10      # A=1 keeps fit_field's default grid
    return e7.fit_field(A, UC, QIN_FIT, form=FORM, metric=METRIC, extra=None,
                        dt_fit=1.0, extra_cfg=(extra_cfg or None), **fk)


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------

def _cmp(better_if_lower_a, better_if_lower_d, tol):
    """'beat' if the rung's error is lower than RDR's, 'match' if within
    tol of it, else 'worse'."""
    if better_if_lower_a < better_if_lower_d - 1e-12:
        return "beat"
    return "match" if abs(better_if_lower_a - better_if_lower_d) <= tol \
        else "worse"


def compare_to_rdr(block, name):
    """Component-wise rung vs RDR on W1, wake, rarefaction, e_s, omega."""
    a = block["rungs"][name]["eval_q2500"]
    d = block["rungs"][RDR]["eval_q2500"]
    wake_data = a["wake_data_vehkm"]
    rare_data = a["rarefaction_data_growth_m"]
    comp = dict(
        W1=dict(rung=a["W1"], RDR=d["W1"], rel=a["W1"] / d["W1"] - 1.0,
                verdict=_cmp(a["W1"], d["W1"], TOL_W1_REL * d["W1"])),
        wake_vehkm=dict(rung=a["wake"]["wake_mean_vehkm"],
                        RDR=d["wake"]["wake_mean_vehkm"], data=wake_data,
                        verdict=_cmp(abs(a["wake"]["wake_mean_vehkm"] - wake_data),
                                     abs(d["wake"]["wake_mean_vehkm"] - wake_data),
                                     TOL_WAKE_VEHKM)),
        rarefaction_growth_m=dict(
            rung=a["rarefaction"]["width_growth_m"],
            RDR=d["rarefaction"]["width_growth_m"], data=rare_data,
            verdict=_cmp(abs(a["rarefaction"]["width_growth_m"] - rare_data),
                         abs(d["rarefaction"]["width_growth_m"] - rare_data),
                         TOL_RARE_M)),
        e_s=dict(rung=a["e_s"], RDR=d["e_s"],
                 verdict=_cmp(abs(a["e_s"]), abs(d["e_s"]), TOL_ES)),
        omega_err=dict(rung=a["omega_err"], RDR=d["omega_err"],
                       verdict=_cmp(abs(a["omega_err"]), abs(d["omega_err"]),
                                    TOL_OMEGA)))
    verdicts = {k: v["verdict"] for k, v in comp.items()}
    n_beat = sum(v == "beat" for v in verdicts.values())
    n_worse = sum(v == "worse" for v in verdicts.values())
    ma, md = a["startup_wave"]["mass"], d["startup_wave"]["mass"]
    out = dict(
        rung=name, admissible=a["admissible"]["ok"],
        degenerate=a["queue"]["degenerate"], components=comp,
        verdicts=verdicts, n_beat=n_beat, n_match=5 - n_beat - n_worse,
        n_worse=n_worse, matches_or_beats_all=bool(n_worse == 0),
        ds_stuck_frac=dict(rung=a["ds_stuck"]["frac"], RDR=d["ds_stuck"]["frac"]),
        ds_max_s_vehkm=dict(rung=a["ds_max_s_vehkm"], RDR=d["ds_max_s_vehkm"]),
        startup_wave_mass_ms=dict(rung=ma.get("v_startup_mass_ms"),
                                  RDR=md.get("v_startup_mass_ms")),
        released_frac_760_900=dict(rung=ma.get("released_frac"),
                                   RDR=md.get("released_frac")),
        N_s_1000_veh=dict(rung=a["queue"]["N_s_1000_veh"],
                          RDR=d["queue"]["N_s_1000_veh"]),
        tolerances=dict(W1_rel=TOL_W1_REL, wake_vehkm=TOL_WAKE_VEHKM,
                        rarefaction_m=TOL_RARE_M, e_s=TOL_ES,
                        omega_err=TOL_OMEGA))
    out["statement"] = (
        f"{name} vs RDR: W1 {a['W1']:.1f} vs {d['W1']:.1f} "
        f"({comp['W1']['rel']:+.1%}: {verdicts['W1']}); wake "
        f"{a['wake']['wake_mean_vehkm']:.1f} vs {d['wake']['wake_mean_vehkm']:.1f} "
        f"veh/km (data {wake_data:.1f}: {verdicts['wake_vehkm']}); rarefaction "
        f"growth {a['rarefaction']['width_growth_m']:+.0f} vs "
        f"{d['rarefaction']['width_growth_m']:+.0f} m (data {rare_data:+.0f}: "
        f"{verdicts['rarefaction_growth_m']}); e_s {a['e_s']:+.1%} vs "
        f"{d['e_s']:+.1%} ({verdicts['e_s']}); omega err {a['omega_err']:+.1%} "
        f"vs {d['omega_err']:+.1%} ({verdicts['omega_err']}); downstream: "
        f"stuck {a['ds_stuck']['frac']:.4f} (RDR {d['ds_stuck']['frac']:.4f}), "
        f"max s {a['ds_max_s_vehkm']:.2f} (RDR {d['ds_max_s_vehkm']:.2f}) "
        f"-> {'ADMISSIBLE' if out['admissible'] else 'NOT admissible'}; "
        f"start-up wave {_fmt(ma.get('v_startup_mass_ms'), '.2f')} m/s, "
        f"{_fmt(ma.get('released_frac'), '.0%')} of N_s released over "
        f"[760,900] s (RDR {_fmt(md.get('v_startup_mass_ms'), '.2f')} m/s, "
        f"{_fmt(md.get('released_frac'), '.0%')}); queue gone (1%) "
        f"{_fmt(a['dissolution']['t_99_s'], '.0f')} s after t_fast (RDR "
        f"{_fmt(d['dissolution']['t_99_s'], '.0f')}); N_s(1000) "
        f"{a['queue']['N_s_1000_veh']:.1f} vs {d['queue']['N_s_1000_veh']:.1f} "
        f"veh => {n_beat} beat / {5 - n_beat - n_worse} match / {n_worse} "
        f"worse of 5 -> " + ("MATCHES OR BEATS the kludge on all five"
                             if n_worse == 0 else
                             "does NOT match the kludge on "
                             + ", ".join(k for k, v in verdicts.items()
                                         if v == "worse")))
    return out


def answer_r1(block):
    """(i) does R1 alone eliminate the downstream plume and the plug?"""
    r0, r1 = (block["rungs"][n]["eval_q2500"] for n in ("R0", "R1"))
    sp1, sp0 = r1["ds_split"], r0["ds_split"]
    plume_slow_gone = bool(sp1["slow"]["max_s_vehkm"] == 0.0
                           and r1["ds_stuck"]["frac"] == 0.0)
    post = sp1["post"]["max_s_vehkm"]
    rel1 = r1["startup_wave"]["mass"].get("released_frac")
    out = dict(
        plume_slow_window=dict(
            R0_max_ds_s_vehkm=sp0["slow"]["max_s_vehkm"],
            R1_max_ds_s_vehkm=sp1["slow"]["max_s_vehkm"],
            R0_ds_stuck_700=r0["ds_stuck"]["frac"],
            R1_ds_stuck_700=r1["ds_stuck"]["frac"],
            eliminated=plume_slow_gone),
        post_t_fast=dict(
            R0_max_ds_s_vehkm=sp0["post"]["max_s_vehkm"],
            R1_max_ds_s_vehkm=post,
            R1_N_s_700=r1["queue"]["N_s_700_veh"],
            R1_N_s_1000=r1["queue"]["N_s_1000_veh"],
            R1_released_frac_760_900=rel1,
            R1_startup_wave_ms=r1["startup_wave"]["mass"].get("v_startup_mass_ms"),
            R1_t_99_s=r1["dissolution"]["t_99_s"]),
        plug=dict(
            R0=dict(omega_err=r0["omega_err"], e_s=r0["e_s"],
                    W1=r0["W1"], wake=r0["wake"]["wake_mean_vehkm"]),
            R1=dict(omega_err=r1["omega_err"], e_s=r1["e_s"],
                    W1=r1["W1"], wake=r1["wake"]["wake_mean_vehkm"])),
        R1_admissible=r1["admissible"]["ok"],
        R1_admissible_rule_failure=[k for k, ok in
                                    (("ds_stuck", r1["admissible"]["ds_stuck_ok"]),
                                     ("ds_max_s", r1["admissible"]["ds_max_s_ok"]))
                                    if not ok])
    out["statement"] = (
        f"R1 (interface alone): slow-window downstream s "
        f"{sp1['slow']['max_s_vehkm']:.3f} veh/km (R0 "
        f"{sp0['slow']['max_s_vehkm']:.2f}), ds_stuck(700) "
        f"{r1['ds_stuck']['frac']:.4f} (R0 {r0['ds_stuck']['frac']:.4f}) -> the "
        f"plume while the bottleneck is active is "
        f"{'ELIMINATED (exactly, by construction)' if plume_slow_gone else 'NOT eliminated'}; "
        f"the plug (leaked s throttling the bottleneck) is gone with it: omega err "
        f"{r1['omega_err']:+.1%} (R0 {r0['omega_err']:+.1%}), e_s {r1['e_s']:+.1%} "
        f"(R0 {r0['e_s']:+.1%}), W1 {r1['W1']:.1f} (R0 {r0['W1']:.1f}), wake "
        f"{r1['wake']['wake_mean_vehkm']:.1f} (R0 {r0['wake']['wake_mean_vehkm']:.1f}). "
        f"BUT after t_fast: max downstream s {post:.2f} veh/km, N_s(700) "
        f"{r1['queue']['N_s_700_veh']:.1f} -> N_s(1000) {r1['queue']['N_s_1000_veh']:.1f} "
        f"veh, released {_fmt(max(rel1, 0.0) if rel1 is not None else None, '.0%')} "
        f"over [760,900] s, queue gone (1%) {_fmt(r1['dissolution']['t_99_s'], '.0f')} s "
        f"after t_fast: with no release "
        f"mechanism (Delta v = 0 after t_fast) the platoon never dissolves and "
        f"rides past the free-driving CAV, so R1 is "
        f"{'admissible' if out['R1_admissible'] else 'NOT admissible'}"
        + (f" (fails {out['R1_admissible_rule_failure']} on the post-t_fast "
           "window, not on the interface)" if not out["R1_admissible"] else ""))
    return out


# ---------------------------------------------------------------------------
# figures (winners): data black, RDR gray dashed, winner blue + s/f fills
# ---------------------------------------------------------------------------

def fig_profiles_e10(tag, meas, rdr_regr, rdr_lbl, win_regr, win_lbl,
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
                     + ("  (post-release)" if t_snap > T_FAST else ""),
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
    fig.savefig(out_dir / f"fig_profiles_e10_{fname_tag or tag}.png", dpi=160)
    plt.close(fig)


def fig_heat3_e10(tag, meas, rdr_regr, rdr_lbl, win_regr, win_lbl, out_dir,
                  fname_tag=None):
    fields = [(np.mean(meas["rho"], axis=0), f"{tag} — SUMO (rep mean)"),
              (rdr_regr["rho_tot"], rdr_lbl),
              (win_regr["rho_tot"], win_lbl)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    tt = win_regr["tt"]
    for ax, (rho, name) in zip(axes, fields):
        im = ax.imshow(rho, origin="lower", aspect="auto",
                       extent=[CELL_LEN / 1000, 30.0,
                               tt[0] / 60, tt[-1] / 60],
                       cmap="turbo", vmin=0, vmax=90)
        ax.plot(win_regr["x_cav"] / 1000, tt / 60, "w-", lw=1.5)
        ax.set_title(textwrap.fill(name, 46), fontsize=8.5)
        ax.set_xlabel("x [km]")
        ax.set_xlim(0, 20)
    axes[0].set_ylabel("t [min]")
    fig.colorbar(im, ax=axes, label=r"$\rho$ [veh/km]", shrink=0.9)
    fig.savefig(out_dir / f"fig_heat3_e10_{fname_tag or tag}.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# ladder table
# ---------------------------------------------------------------------------

def _row(name, r, mark=""):
    ev = r["eval_q2500"]
    suw = ev["startup_wave"]["mass"]
    sp = ev["ds_split"]
    return (f"{name:4s}{mark:1s} {r['kappa_c']:9.3e} {r['kappa_r']:9.3e} "
            f"{ev['W1']:6.1f} {ev['RMSE']:5.2f} {ev['e_s']:+7.1%} "
            f"{ev['omega_err']:+7.1%} "
            f"{ev['wake']['wake_mean_vehkm']:5.1f} "
            f"{_nan(ev['ds_stuck']['frac']):7.4f} "
            f"{ev['ds_max_s_vehkm']:6.2f} "
            f"{sp['slow']['max_s_vehkm']:6.2f} "
            f"{sp['post']['max_s_vehkm']:6.2f} "
            f"{_nan(ev['waviness_sim']['std_vehkm']):5.2f} "
            f"{_nan(ev['wedge']['frac']):5.3f} "
            f"{ev['rarefaction']['width_growth_m']:+6.0f} "
            f"{ev['s_layer']['max_m']:6.0f} "
            f"{ev['queue']['N_s_700_veh']:5.1f} "
            f"{ev['queue']['N_s_1000_veh']:6.1f} "
            f"{_nan(suw.get('v_startup_mass_ms')):6.2f} "
            f"{100.0 * max(_nan(suw.get('released_frac')), 0.0):5.0f} "
            f"{_fmt(ev['dissolution']['t_99_s'], '3.0f'):>4s} "
            f"{('deg' if ev['queue']['degenerate'] else 'yes' if ev['admissible']['ok'] else 'NO'):>3s}")


def print_table(akey, block):
    refs = block["data_refs_q2500"]
    print(f"\n===== E10 ladder {akey} (u{UC:g} q{QIN_FIT:g} {FORM}, W1 fit) "
          "=====")
    print(f"data refs: wake={refs['wake']['wake_mean_vehkm']:.1f} veh/km, "
          f"waviness={refs['waviness']['std_vehkm']:.2f} veh/km, "
          f"wedge_frac={refs['wedge']['frac']:.3f}, "
          f"rarefaction_growth={refs['rarefaction']['width_growth_m']:+.0f} m")
    print(f"{'rung':5s} {'kappa_c':>9s} {'kappa_r':>9s} {'W1':>6s} "
          f"{'RMSE':>5s} {'e_s':>7s} {'om_err':>7s} {'wake':>5s} "
          f"{'stuckf':>7s} {'ds|s|':>6s} {'dsslow':>6s} {'dspost':>6s} "
          f"{'wavi':>5s} {'wedge':>5s} {'rare_m':>6s} {'slayer':>6s} "
          f"{'Ns700':>5s} {'Ns1000':>6s} {'suw_m':>6s} {'rel%':>5s} "
          f"{'t99':>4s} {'adm':>3s}")
    for name, *_ in RUNGS:
        mark = "*" if name == block["winner"] else ""
        print(_row(name, block["rungs"][name], mark))
    print(_row(RDR, block["rungs"][RDR]))
    print("  ds|s| = max downstream s, all snapshots (E8 admissibility "
          "metric); dsslow / dspost = the same restricted to [t_slow,t_fast] "
          "/ t > t_fast [veh/km];\n  suw_m = E9 mass-based start-up-wave "
          "speed -(dN_s/dt)/max_x s over [760,900] s [m/s]; rel% = share of "
          "N_s released there (ill-conditioned when the queue is gone "
          "before 760 s); t99 = time after t_fast for N_s to fall below 1% "
          "of N_s(t_fast) [s, 10 s cadence; n/a = never];\n  adm = "
          "ds_stuck<=0.01 AND ds|s|<=1 (deg = N_s(700) < 1 veh, excluded); "
          "* = winner; RDR = E8 final kappas, not refit")


def _print_eval(ev, extra=""):
    suw = ev["startup_wave"]["mass"]
    sp = ev["ds_split"]
    print(f"[eval dt=0.5] W1={ev['W1']:.2f} RMSE={ev['RMSE']:.2f} "
          f"e_s={ev['e_s']:+.1%} om={ev['omega_err']:+.1%} "
          f"wake={ev['wake']['wake_mean_vehkm']:.1f} "
          f"stuckf={_nan(ev['ds_stuck']['frac']):.4f} "
          f"ds|s|={ev['ds_max_s_vehkm']:.3f} "
          f"(slow {sp['slow']['max_s_vehkm']:.3f} / post "
          f"{sp['post']['max_s_vehkm']:.3f}) "
          f"rare={ev['rarefaction']['width_growth_m']:+.0f} m "
          f"Ns700={ev['queue']['N_s_700_veh']:.1f} "
          f"Ns1000={ev['queue']['N_s_1000_veh']:.1f} "
          f"suw_m={_fmt(suw.get('v_startup_mass_ms'), '.2f')} m/s "
          f"rel={_fmt(suw.get('released_frac'), '.0%')} "
          f"t99={_fmt(ev['dissolution']['t_99_s'], '.0f')} s "
          f"adm={ev['admissible']['ok']}{extra}", flush=True)


# ---------------------------------------------------------------------------
# main program
# ---------------------------------------------------------------------------

def run(smoke=False):
    t_start = time.time()
    OUT_E10.mkdir(parents=True, exist_ok=True)
    e8_final = json.loads(E8_FINAL_CONFIG.read_text())

    results = {"_meta": dict(
        scenario=f"u{UC:g} q{QIN_FIT:g} form={FORM} metric={METRIC}",
        protocol=("every rung: e7_wasserstein.fit_field, 6x6 log grid + "
                  "Nelder-Mead maxfev 40 at dt_fit=1, production re-eval "
                  "dt=0.5; fixed knobs via extra_cfg passthrough (NOT "
                  "fitted); A=10 kc_grid=logspace(-2.5,-0.5,6); NO "
                  "constrained-refit fallback; RDR = out/e8/final_config.json "
                  "kappas re-evaluated with downstream_release, NOT refit"),
        rungs=({n: dict(extra_cfg=dict(x), description=d) for n, x, d in RUNGS}
               | {RDR: dict(extra_cfg=dict(RDR_CFG), description=RDR_DESC)}),
        e10_model=E10_MODEL,
        downstream_release_policy=("used ONLY in the RDR reference row; every "
                                   "R-rung is state-only (interface flux "
                                   "constraint + connectivity leader-loss)"),
        knobs=dict(q_xi_max_vehs=QXI, q_xi_max_vehh=QXI * 3600.0,
                   ws_frac=WS_FRAC_FIX, w_s_ms=W_S, w_ms=float(ev4.W),
                   eta_la_main_m=ETA_MAIN, eta_la_sens_m=ETA_SENS,
                   tau_ll_main_s=TAU_MAIN, tau_ll_sens_s=TAU_SENS,
                   v_cav_free_ms=27.78, v_f_ms=float(ev4.V_F)),
        dt_fit=1.0, dt_production=e7.DT_PRODUCTION,
        admissibility_rule=(f"ds_stuck(t=700) <= {DS_STUCK_MAX:g} AND max "
                            f"downstream s <= {DS_MAX_S_VEHKM:g} veh/km (E8 "
                            "definition: all snapshots with the CAV on road, "
                            "data cells fully beyond the CAV's solver cell); "
                            "no constrained-refit fallback"),
        winner_rule=("min W1 (dt=0.5, q2500) among admissible rungs that are "
                     f"not degenerate (N_s(700) < {DEGENERATE_NS_VEH:g} veh); "
                     "RDR is reference only; if none is admissible winner="
                     "None and figures use the min-W1 rung, flagged"),
        verdict_rule=("per component vs RDR: 'beat' = lower error than RDR, "
                      "'match' = within tolerance, else 'worse'; errors: W1 "
                      "(rel tol 5%), |wake - data| (3 veh/km), |rarefaction "
                      "growth - data| (300 m), |e_s| (5 pp), |omega_err| "
                      "(5 pp)"),
        diagnostics=dict(
            e8=("wake, ds_stuck, waviness, wedge, rarefaction, s_layer, "
                "ds_max_s: e8_ladder.eval_rung (imported)"),
            startup_wave=("e9_ladder.startup_wave (imported); 'mass' block "
                          "= the E9 mass-based number; NOTE: ill-conditioned "
                          "for rungs whose queue dissolves before 760 s "
                          "(see 'dissolution')"),
            dissolution=queue_dissolution.__doc__,
            ds_split=ds_max_s_split.__doc__,
            ll_budget=ll_budget_connect.__doc__,
            queue="N_s at t=700, 1000 s [veh] (regrid_sim N_s)"),
        notes=[
            ("The E8 admissibility metric 'max downstream s' spans ALL "
             "snapshots with the CAV on road, including t > t_fast when the "
             "CAV drives at v_cav_free = 27.78 m/s < v_f = 27.94 m/s as an "
             "ordinary vehicle; s still present then rides past it on the "
             "s-class free branch and is counted.  ds_split separates that "
             "window from the active-bottleneck window, where the "
             "impermeable face gives 0 exactly."),
            ("The E9 mass-based start-up-wave number (-(dN_s/dt)/max_x s "
             "over [760, 900] s) is ill-conditioned for rungs whose queue "
             "dissolves within a few tau_ll of t_fast (before the window); "
             "'dissolution' (t_half, t_99 after t_fast) is the robust "
             "statement.  The connect-LL with the class-A speed anchor "
             "releases the whole leaderless queue uniformly at 1/tau_ll, a "
             "relaxation, not a front receding at a wave speed."),
            ("Rungs whose queue never dissolves (R0, R1: Delta v = 0 after "
             "t_fast, no release mechanism) keep N_s(1000) >= N_s(700).")],
        smoke=smoke)}

    for A in (1.0, 10.0):
        akey = f"A{A:g}"
        tag = f"A{A:g}_u{UC:g}_q{QIN_FIT:g}"
        refs = L.data_refs(A, UC, QIN_FIT)
        block = dict(tag=tag, data_refs_q2500=refs, rungs={})
        kept = {}

        # ---- RDR reference: E8 final kappas, re-evaluated, NOT refit ----
        e8k = e8_final[tag]
        kc8, kr8 = float(e8k["kappa_c"]), float(e8k["kappa_r"])
        w1_e8 = float(e8k["summary"]["W1"])
        print(f"\n=== {akey} {RDR}: {RDR_DESC} ===", flush=True)
        ev, regr, meas = evaluate(A, QIN_FIT, kc8, kr8, RDR_CFG, refs)
        repro = bool(abs(ev["W1"] - w1_e8) <= 1e-6 * max(1.0, w1_e8))
        block["rungs"][RDR] = dict(
            metric=METRIC, description=RDR_DESC, extra_cfg=dict(RDR_CFG),
            kappa_c=kc8, kappa_r=kr8, refit=False,
            source=str(E8_FINAL_CONFIG.relative_to(HERE)),
            e8_final_W1=w1_e8, reproduces_e8_final=repro, eval_q2500=ev)
        kept[RDR] = (regr, meas, RDR_CFG, kc8, kr8)
        _print_eval(ev, f" | E8 final_config W1 {w1_e8:.2f}, reproduced={repro}")
        if not repro:
            print(f"[{akey}] WARNING: RDR re-evaluation does not reproduce "
                  f"out/e8/final_config.json W1 ({ev['W1']:.4f} vs "
                  f"{w1_e8:.4f}); the solver's default path may have changed",
                  flush=True)

        # ---- rungs: fit_field as prescribed, then production evaluation --
        for name, cfg, desc in RUNGS:
            print(f"\n=== {akey} {name}: {desc} ===", flush=True)
            t0 = time.time()
            fit = fit_rung(A, cfg, smoke)
            ev, regr, meas = evaluate(A, QIN_FIT, fit["kappa_c"],
                                      fit["kappa_r"], cfg, refs)
            consistent = bool(abs(ev["W1"] - fit["at_production"]["w1"])
                              <= 1e-9 * max(1.0, ev["W1"]))
            block["rungs"][name] = dict(
                metric=METRIC, description=desc, extra_cfg=dict(cfg),
                label=rung_label(name, cfg),
                kappa_c=fit["kappa_c"], kappa_r=fit["kappa_r"], refit=True,
                fit_objective_dtfit=fit["objective"],
                fit_grid_objective=fit["grid_objective"],
                n_sim_fit=fit["n_sim"], n_polish=fit["n_polish"],
                fit_at_production=fit["at_production"],
                eval_consistent_with_fit=consistent,
                fit_runtime_s=round(time.time() - t0, 1),
                eval_q2500=ev)
            kept[name] = (regr, meas, cfg, fit["kappa_c"], fit["kappa_r"])
            _print_eval(ev, f" | fit {fit['n_sim']} sims, "
                            f"{time.time() - t0:.0f} s")
            if not consistent:
                print(f"[{akey} {name}] WARNING: production W1 differs "
                      "between fit_field re-eval and eval_rung", flush=True)

        # ---- winner: admissible (and non-degenerate), then min W1 --------
        prim = [n for n, *_ in RUNGS]
        w1_of = {n: block["rungs"][n]["eval_q2500"]["W1"] for n in prim}
        degen = [n for n in prim
                 if block["rungs"][n]["eval_q2500"]["queue"]["degenerate"]]
        adm_all = [n for n in prim
                   if block["rungs"][n]["eval_q2500"]["admissible"]["ok"]]
        adm = [n for n in adm_all if n not in degen]
        inadm = [n for n in prim if n not in adm_all]
        opt_all = min(prim, key=w1_of.get)
        winner = min(adm, key=w1_of.get) if adm else None
        fig_rung = winner if winner is not None else opt_all
        block["admissible_rungs"] = adm_all
        block["degenerate_rungs"] = degen
        block["inadmissible_rungs"] = {
            n: [k for k, ok in
                (("ds_stuck", block["rungs"][n]["eval_q2500"]["admissible"]["ds_stuck_ok"]),
                 ("ds_max_s", block["rungs"][n]["eval_q2500"]["admissible"]["ds_max_s_ok"]))
                if not ok] for n in inadm}
        block["w1_optimal_unconstrained"] = dict(rung=opt_all, W1=w1_of[opt_all])
        block["winner"] = winner
        block["figure_rung"] = fig_rung
        block["winner_rule"] = (
            f"admissible {adm_all}, degenerate {degen}, candidates {adm}; "
            + (f"winner = {winner} (W1={w1_of[winner]:.1f}); unconstrained "
               f"min-W1 rung = {opt_all} (W1={w1_of[opt_all]:.1f}, "
               f"admissible={opt_all in adm_all})" if winner is not None
               else f"NO ADMISSIBLE RUNG; figures/transfer use {opt_all} "
                    "(flagged)"))
        print(f"\n[{akey}] {block['winner_rule']}", flush=True)
        impermeable_inadm = [n for n in inadm
                             if block["rungs"][n]["extra_cfg"].get("s_impermeable")]
        block["impermeable_rungs_not_admissible"] = {
            n: dict(fails=block["inadmissible_rungs"][n],
                    ds_split=block["rungs"][n]["eval_q2500"]["ds_split"])
            for n in impermeable_inadm}
        if impermeable_inadm:
            for n in impermeable_inadm:
                sp = block["rungs"][n]["eval_q2500"]["ds_split"]
                print(f"[{akey}] NOTE: impermeable rung {n} is NOT admissible "
                      f"(fails {block['inadmissible_rungs'][n]}): max ds s slow "
                      f"window {sp['slow']['max_s_vehkm']:.3f}, post-t_fast "
                      f"{sp['post']['max_s_vehkm']:.3f} veh/km", flush=True)

        # ---- verdicts -----------------------------------------------------
        block["verdict"] = {n: compare_to_rdr(block, n) for n in prim}
        for n in (E10_MODEL, fig_rung):
            print(f"[{akey} verdict] {block['verdict'][n]['statement']}",
                  flush=True)
        block["answer_i"] = answer_r1(block)
        print(f"[{akey} (i)] {block['answer_i']['statement']}", flush=True)

        # w_s and sensitivity, in numbers
        def _hs(n):
            e = block["rungs"][n]["eval_q2500"]
            return dict(kappa_c=block["rungs"][n]["kappa_c"],
                        kappa_r=block["rungs"][n]["kappa_r"], W1=e["W1"],
                        wake=e["wake"]["wake_mean_vehkm"],
                        e_s=e["e_s"], omega_err=e["omega_err"],
                        rarefaction_growth_m=e["rarefaction"]["width_growth_m"],
                        ds_max_s=e["ds_max_s_vehkm"],
                        ds_stuck=e["ds_stuck"]["frac"],
                        N_s_700=e["queue"]["N_s_700_veh"],
                        N_s_1000=e["queue"]["N_s_1000_veh"],
                        startup_wave_mass_ms=e["startup_wave"]["mass"].get(
                            "v_startup_mass_ms"),
                        released_frac=e["startup_wave"]["mass"].get(
                            "released_frac"),
                        t_99_s=e["dissolution"]["t_99_s"],
                        t_half_s=e["dissolution"]["t_half_s"],
                        admissible=e["admissible"]["ok"],
                        degenerate=e["queue"]["degenerate"])
        r2, r4 = _hs("R2"), _hs("R4")
        block["ws_needed"] = dict(
            R2_with_ws=r2, R4_without_ws=r4,
            verdict=(f"w_s still needed: R2 W1 {r2['W1']:.1f} < R4 W1 "
                     f"{r4['W1']:.1f} (wake {r2['wake']:.1f} vs {r4['wake']:.1f}"
                     f", data {refs['wake']['wake_mean_vehkm']:.1f})"
                     if r2["W1"] < r4["W1"] and r2["admissible"] else
                     f"w_s NOT needed on W1: R4 W1 {r4['W1']:.1f} <= R2 W1 "
                     f"{r2['W1']:.1f} (R4 admissible={r4['admissible']})"
                     if r4["admissible"] else
                     f"R4 not admissible (W1 {r4['W1']:.1f} vs R2 {r2['W1']:.1f})"))
        block["sensitivity"] = {n: _hs(n) for n in ("R2", "R5", "R6")}
        block["attribution"] = {n: _hs(n) for n in ("R0", "R1", "R2", "R3")}
        print(f"[{akey} w_s?] {block['ws_needed']['verdict']}", flush=True)
        print(f"[{akey} sensitivity] "
              + "; ".join(f"{n} ({block['rungs'][n]['label'].split(': ')[1]}): "
                          f"W1 {v['W1']:.1f}, wake {v['wake']:.1f}, "
                          f"rare {v['rarefaction_growth_m']:+.0f} m, t99 "
                          f"{_fmt(v['t_99_s'], '.0f')} s, ds post "
                          f"{block['rungs'][n]['eval_q2500']['ds_split']['post']['max_s_vehkm']:.2f}, "
                          f"adm={v['admissible']}"
                          for n, v in block["sensitivity"].items()), flush=True)

        # ---- zero-refit q2000 transfer: every rung + RDR -----------------
        refs_t = L.data_refs(A, UC, QIN_TRANSFER)
        block["data_refs_q2000"] = refs_t
        block["transfer_q2000"] = {}
        for n in prim + [RDR]:
            _, _, cfg_n, kc_n, kr_n = kept[n]
            ev_t, _, _ = evaluate(A, QIN_TRANSFER, kc_n, kr_n, cfg_n, refs_t)
            block["transfer_q2000"][n] = ev_t
            print(f"[{akey}] transfer q2000 {n:4s}: W1={ev_t['W1']:.2f} "
                  f"RMSE={ev_t['RMSE']:.2f} e_s={ev_t['e_s']:+.1%} "
                  f"om={ev_t['omega_err']:+.1%} "
                  f"wake={ev_t['wake']['wake_mean_vehkm']:.1f} "
                  f"stuckf={_nan(ev_t['ds_stuck']['frac']):.4f} "
                  f"ds|s|={ev_t['ds_max_s_vehkm']:.3f} "
                  f"(slow {ev_t['ds_split']['slow']['max_s_vehkm']:.3f} / post "
                  f"{ev_t['ds_split']['post']['max_s_vehkm']:.3f}) "
                  f"Ns700={ev_t['queue']['N_s_700_veh']:.1f} "
                  f"rel={_fmt(ev_t['startup_wave']['mass'].get('released_frac'), '.0%')}"
                  f" adm={ev_t['admissible']['ok']}"
                  + ("  <- winner" if n == fig_rung else ""), flush=True)
        tw = block["transfer_q2000"][fig_rung]
        block["winner_transfer_q2000"] = dict(
            rung=fig_rung, flagged_not_admissible_at_q2500=(winner is None),
            W1=tw["W1"], RMSE=tw["RMSE"], e_s=tw["e_s"],
            omega_err=tw["omega_err"], wake=tw["wake"]["wake_mean_vehkm"],
            wake_data=refs_t["wake"]["wake_mean_vehkm"],
            ds_stuck=tw["ds_stuck"]["frac"], ds_max_s=tw["ds_max_s_vehkm"],
            admissible=tw["admissible"]["ok"],
            RDR_W1=block["transfer_q2000"][RDR]["W1"],
            RDR_e_s=block["transfer_q2000"][RDR]["e_s"])

        # ---- figures: winner (and R2 if the winner is not R2) -----------
        regr_d, _, _, _, _ = kept[RDR]
        rdr_lbl = f"{RDR}: cap + w_s + downstream_release (E8 κ, kludge)"
        fig_tag = ("smoke_" if smoke else "") + tag
        block["figures"] = {}
        draw = [(fig_rung, fig_tag, "" if winner is not None
                 else " [NOT admissible]")]
        if fig_rung != E10_MODEL:
            draw.append((E10_MODEL, fig_tag + "_" + E10_MODEL,
                         " (the E10 model"
                         + ("" if E10_MODEL in adm_all else ", NOT admissible")
                         + ")"))
        for n, ftag, suffix in draw:
            regr_n, meas_n, cfg_n, _, _ = kept[n]
            lbl = "E10 " + rung_label(n, cfg_n) + suffix
            fig_profiles_e10(tag, meas_n, regr_d, rdr_lbl, regr_n, lbl,
                             OUT_E10, fname_tag=ftag)
            fig_heat3_e10(tag, meas_n, regr_d, rdr_lbl, regr_n, lbl, OUT_E10,
                          fname_tag=ftag)
            block["figures"][n] = dict(
                profiles=f"fig_profiles_e10_{ftag}.png",
                heat3=f"fig_heat3_e10_{ftag}.png",
                labels=dict(data="SUMO (rep mean)", gray_dashed=rdr_lbl,
                            blue=lbl))
        results[akey] = block
        print_table(akey, block)

    # ---- the three answers ----------------------------------------------
    answers = {}
    for akey in ("A1", "A10"):
        b = results[akey]
        v2 = b["verdict"][E10_MODEL]
        tw = b["winner_transfer_q2000"]
        answers[akey] = dict(
            i=b["answer_i"]["statement"],
            ii=v2["statement"],
            iii=(f"winner {tw['rung']} at q2000 (zero refit): W1 {tw['W1']:.1f} "
                 f"(RDR {tw['RDR_W1']:.1f}), e_s {tw['e_s']:+.1%} (RDR "
                 f"{tw['RDR_e_s']:+.1%}), omega err {tw['omega_err']:+.1%}, "
                 f"wake {tw['wake']:.1f} (data {tw['wake_data']:.1f}), ds_stuck "
                 f"{_nan(tw['ds_stuck']):.4f}, max ds s {tw['ds_max_s']:.2f} "
                 f"veh/km -> {'ADMISSIBLE' if tw['admissible'] else 'NOT admissible'}"
                 + (" [winner flagged: not admissible at q2500]"
                    if tw["flagged_not_admissible_at_q2500"] else "")),
            winner=b["winner"], winner_W1=(
                b["rungs"][b["winner"]]["eval_q2500"]["W1"]
                if b["winner"] else None),
            R1_admissible=b["rungs"]["R1"]["eval_q2500"]["admissible"]["ok"],
            R1_plume_slow_window_eliminated=b["answer_i"]["plume_slow_window"]["eliminated"],
            R2_admissible=b["rungs"]["R2"]["eval_q2500"]["admissible"]["ok"],
            R2_matches_or_beats_RDR_all5=v2["matches_or_beats_all"],
            R2_component_verdicts=v2["verdicts"],
            winner_transfer_q2000_admissible=tw["admissible"])
    results["_meta"]["answers"] = answers
    results["_meta"]["overall_statement"] = [
        f"{akey} ({k}): {answers[akey][k]}"
        for akey in ("A1", "A10") for k in ("i", "ii", "iii")]
    results["_meta"]["runtime_s"] = round(time.time() - t_start, 1)
    out_path = OUT_E10 / ("ladder_smoke.json" if smoke else "ladder.json")
    out_path.write_text(json.dumps(results, indent=2, default=e9._json_default))
    print("\n===== E10 answers =====")
    for ln in results["_meta"]["overall_statement"]:
        print(ln)
    print(f"\nwrote {out_path} ({results['_meta']['runtime_s']} s)")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="E10 attribution ladder: s-impermeable interface + "
                    "connectivity leader-loss vs the downstream_release kludge")
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
