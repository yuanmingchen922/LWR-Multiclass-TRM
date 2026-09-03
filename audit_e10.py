"""E10 audit: independent probes of the E10 solver changes in solver.py --
(I) the s-impermeable moving-bottleneck interface (cfg.s_impermeable) and
(II) the connectivity leader-loss rate (ll_mode="connect", tau_ll).

Run: python3 audit_e10.py          (exit code 0 == all probes passed;
                                    ISSUES are listed separately at the end)

solver.py / test_solver.py / e9_ladder.py are NOT modified.  The pre-change
solver is git HEAD:solver.py (the E8 solver; the E9 eta_la solver was never
committed and is not on disk -- out/e10_ref_A1.npz carries only its sha256).

Probes (numbering follows the audit brief):

P1  suite green: pytest test_solver.py collects and passes 24 tests.
P2  new-fields-off bit-identity vs git HEAD across ALL SimResult fields:
    (a) all-legacy, (b) E8 hybrid (cap + w_s), each with the E10 fields at
    their defaults AND with s_impermeable=False, ll_mode="connect",
    tau_ll=0.7, eta_la=None spelled out (connect/tau inert without eta_la);
    (c) hybrid + eta_la=200 ratio mode vs an independent composition
    [HEAD transport_step + E10 leader_loss_rate + reaction_exact(mu_extra)]
    of the E9 scheme (all fields), also with tau_ll=9 (inert in ratio
    mode).  Stored references: out/ll_ref_A1.npz is HEAD's output (sha +
    arrays); out/e10_ref_A1.npz (E9 solver, different sha) equals HEAD's
    output on the hybrid config -> E9's eta_la=None path == E8.  Also:
    transport_step (s_impermeable=False, any t, cap/w_s on/off) and
    reaction_exact (mu_extra None / 0) bit-identical to HEAD on random
    states.  Finally the audit's own replica of simulate() is shown to be
    bit-faithful to sol.simulate on the full E10 config (used by P3/P5).
P3  s-impermeable semantics on a manually constructed state (s > 0 in the
    CAV cell and downstream): one transport_step at t = 500 (footprint
    {157,158}, cap cell 158: one face) and t = 501 (footprint {158,159}:
    cap face AND footprint front face).  Interface fluxes are captured
    exactly (the module's `np.diff` is intercepted in memory) and cross-
    checked against an independent re-implementation of the min-flux +
    interface logic (bit-identical f_new, s_new).  F_s == 0.0 exactly at
    the zeroed faces, F_s bit-identical elsewhere, F_f bit-identical at
    every face other than the cap face; per-cell accounting: the s that
    legacy transports across the face is stored in the upstream cell
    exactly (to 1e-16), total mass conserved; outside [t_slow, t_fast]
    (and t=None, and s_impermeable=False) the step is bit-identical to
    HEAD's.  Step invariants (f, s >= 0, rho <= P) on random near-jam
    states.  Plus two findings: (F1) the DM-G violation test keeps the
    trace f_j + s_j although s no longer passes -> the cap is disengaged
    by u_cav s_j (state B: legacy cap binds, E10 cap does not, F^f - u f_j
    exceeds omega_max); quantified on the A1 run with an in-memory
    corrected step.  (F2) k_front = max(j, last nonzero of a) is the
    GLOBAL last A cell, not the CAV footprint: a second A vehicle
    downstream zeroes F_s at every face in between.
P4  connectivity LL: an explicit BACKWARD recursion (cover_j from
    cover_{j+1..j+n}) vs leader_loss_rate_connect on 50 random states
    (multiple/split A vehicles, gaps, tapers, sparse s, road-end contact)
    to 1e-14 (and bitwise).  The brief asks to show that iterating from
    cover = 1 wrongly covers a disconnected platoon: on a finite road with
    empty padding the map is strictly upper-triangular, the fixed point is
    UNIQUE (the head of an unanchored platoon has an empty window, so
    cover_head = 0 for any start), and Jacobi from 1 converges to the same
    point -- only an early-stopped iteration covers the platoon.  (F3) the
    docstring's "cover = 1 would also be self-consistent" is false.  (F4)
    the 500-sweep cap: with n = 1 (eta_la = dx) coverage propagates one
    cell per sweep, so a queue longer than 500 cells anchored at its head
    is wrongly released at its tail (demonstrated); the exact answer is the
    one-pass backward recursion.
P5  anchoring rule 'a anchors only while u_s < v_f': static check of the
    connect block in simulate() (inputs u_s, cfg.v_f, a_star, s, cfg
    constants; no x_cav / cav_position / t); dynamically: on the A1 E10
    run N_s(1000) ~ 0 after t_fast, in-queue mu_ll == 0 exactly on every
    slow-phase save (independent backward cover), and toggling u_s on a
    frozen slow-phase state flips the queue from mu = 0 to mu = 1/tau.
P6  purity: AST + code-object inspection of both LL functions (no
    x_cav / schedule / t / cfg names; numpy + builtins only; no closure).
P7  invariants under stress (q_in = 2900/3600, u_xi = 10) with ALL E10
    knobs on (s_impermeable + connect LL + cap + w_s): ledger 1e-10,
    f, s >= 0, rho <= P, no NaN in any field, for A1/A10 kappas, eta_la
    50/200/1000, tau_ll 0.5/30, dx 25, + downstream_release, no cap,
    no w_s, ratio + s_impermeable, u_xi = 5; np.seterr(raise) throughout.
P8  exact-update property with mu_extra from the connect rule vs a
    2000-step RK4 of the frozen ODE, on frozen states of the A1 E10 run at
    t = 500 (slow phase) and t = 760 (post-t_fast release), dt = 0.5 and
    5 s, to 1e-10; p conserved, positivity; random state with connect
    mu_extra on gamma / 'af' paths.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import solver as sol  # noqa: E402  (the changed, working-tree E10 solver)

_params = json.loads((HERE / "out" / "params.json").read_text())
V_F = _params["v_f_kmh"] / 3.6
W = _params["w_kmh"] / 3.6
P = _params["P_vehkm"] / 1000.0

ALL_FIELDS = ("t", "x", "a", "f", "s", "x_cav", "omega", "N_s",
              "denied_inflow", "injected", "outflowed", "on_road")

FAILURES: list[str] = []
ISSUES: list[str] = []


def check(ok: bool, msg: str) -> None:
    if ok:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def issue(msg: str) -> None:
    print(f"  ISSUE {msg}")
    ISSUES.append(msg)


def bit_equal(a, b) -> bool:
    """Bitwise equality (NaN-safe: compares raw bytes)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def getf(res, name):
    return res[name] if isinstance(res, dict) else getattr(res, name)


def compare_all_fields(res_a, res_b, what: str) -> None:
    bad = [n for n in ALL_FIELDS if not bit_equal(getf(res_a, n), getf(res_b, n))]
    check(not bad, f"{what}: all {len(ALL_FIELDS)} SimResult fields bit-identical"
                   + (f" (differ: {bad})" if bad else ""))


# --------------------------------------------------------------------------
# Pre-change solver reconstructed from git HEAD (E8)
# --------------------------------------------------------------------------

def load_head_solver():
    src = subprocess.run(["git", "show", "HEAD:solver.py"], cwd=HERE,
                         capture_output=True, text=True, check=True).stdout
    sha = hashlib.sha256(src.encode()).hexdigest()
    path = HERE / "out" / "_solver_head_e10_audit.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("solver_head_e10", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["solver_head_e10"] = mod
    spec.loader.exec_module(mod)
    fields = mod.SimConfig.__dataclass_fields__
    for name in ("eta_la", "s_impermeable", "ll_mode", "tau_ll"):
        assert name not in fields, f"HEAD solver unexpectedly has {name}"
    assert "downstream_release" in fields, "HEAD solver lacks the E8 flag"
    assert not hasattr(mod, "leader_loss_rate")
    assert not hasattr(mod, "leader_loss_rate_connect")
    return mod, sha


# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------

def kw_A1(**kw):
    """E8 final hybrid A1 config (out/e8/final_config.json; test _ll_cfg):
    cap 2000 veh/h + w_s = 0.6 w + W1-fitted kappas, downstream_release OFF."""
    d = dict(v_f=V_F, w=W, P=P, q_in=2500.0 / 3600.0, u_xi=15.0,
             kappa_c=5.32e-2, kappa_r=1.65e-3, capture_form="lf",
             dt=0.5, save_every=20, w_s=0.6 * W,
             q_xi_max=2000.0 / 3600.0, downstream_release=False)
    d.update(kw)
    return d


def kw_A10(**kw):
    return kw_A1(kappa_c=1.66e-1, kappa_r=1.40e-2, **kw)


def kw_legacy(**kw):
    d = dict(v_f=V_F, w=W, P=P, q_in=2500.0 / 3600.0, u_xi=15.0,
             kappa_c=0.026, kappa_r=3e-5)
    d.update(kw)
    return d


E10_KNOBS = dict(s_impermeable=True, eta_la=200.0, ll_mode="connect",
                 tau_ll=3.0)


def kw_E10(**kw):
    """A1 hybrid + ALL E10 knobs (the t24 config)."""
    d = kw_A1(**E10_KNOBS)
    d.update(kw)
    return d


def strip_head(kw):
    """kwargs acceptable to HEAD's SimConfig (drop E9/E10 fields)."""
    return {k: v for k, v in kw.items()
            if k not in ("eta_la", "s_impermeable", "ll_mode", "tau_ll")}


# --------------------------------------------------------------------------
# Interface-flux capture (in memory; solver.py untouched) and an
# independent re-implementation of transport_step's fluxes
# --------------------------------------------------------------------------

class _DiffRecorder:
    """Stand-in for a solver module's `np`: records the argument of every
    np.diff call (transport_step calls np.diff(F_f) then np.diff(F_s)) and
    delegates everything else to numpy."""

    def __init__(self):
        self.rec: list[np.ndarray] = []

    def __getattr__(self, name):
        if name == "diff":
            def diff(x, *a, **k):
                self.rec.append(np.array(x, dtype=float, copy=True))
                return np.diff(x, *a, **k)
            return diff
        return getattr(np, name)


def step_with_fluxes(mod, f, s, a, c_f, c_s, cfg, t):
    """(f_new, s_new, q_adm, q_out), F_f, F_s of mod.transport_step."""
    rec = _DiffRecorder()
    saved = mod.np
    mod.np = rec
    try:
        out = mod.transport_step(f, s, a, c_f, c_s, cfg, t=t)
    finally:
        mod.np = saved
    assert mod.np is np and len(rec.rec) == 2
    return out, rec.rec[0], rec.rec[1]


def fluxes_indep(f, s, a, c_f, c_s, cfg, t, s_imp: bool, cap: bool,
                 cap_trace: str = "auto", footprint: str = "contiguous"):
    """Independent re-implementation of transport_step (docstring
    semantics).  cap_trace: 'fs' = the solver's violation test with the
    trace f_j + s_j; 'f' = the trace of the passing class only.
    footprint: 'global' = the solver's k_front = max(j, last nonzero of a);
    'contiguous' = the contiguous run of a > 0 starting at cell j."""
    f = np.asarray(f, float)
    s = np.asarray(s, float)
    a = np.asarray(a, float)
    nx = f.size
    lam = cfg.dt / cfg.dx
    rho = a + f + s
    r_safe = np.maximum(rho, 1e-300)
    pi_f = np.where(rho > 0.0, f / r_safe, 0.0)
    pi_s = np.where(rho > 0.0, s / r_safe, 0.0)
    C_f = c_f * cfg.w * cfg.P / (c_f + cfg.w)
    D_f = np.minimum(c_f * rho, C_f)
    S_f = np.minimum(C_f, cfg.w * (cfg.P - rho))
    if cfg.w_s is None and cfg.P_s is None:
        C_s = c_s * cfg.w * cfg.P / (c_s + cfg.w)
        D_s = np.minimum(c_s * rho, C_s)
        S_s = np.minimum(C_s, cfg.w * (cfg.P - rho))
    else:
        w_s = cfg.w if cfg.w_s is None else cfg.w_s
        P_s = cfg.P if cfg.P_s is None else cfg.P_s
        C_s = c_s * w_s * P_s / (c_s + w_s)
        D_s = np.minimum(c_s * rho, C_s)
        S_s = np.maximum(np.minimum(C_s, w_s * (P_s - rho)), 0.0)
    F_f = np.empty(nx + 1)
    F_s = np.empty(nx + 1)
    F_f[1:-1] = pi_f[:-1] * np.minimum(D_f[:-1], S_f[1:])
    F_s[1:-1] = pi_s[:-1] * np.minimum(D_s[:-1], S_s[1:])
    F_f[0] = min(cfg.q_in, S_f[0])
    F_s[0] = 0.0
    F_f[-1] = pi_f[-1] * D_f[-1]
    F_s[-1] = pi_s[-1] * D_s[-1]
    info = dict(j=None, zeroed=[], cap_test=False, cap_hit=False,
                scale=1.0, omega_max=None, F_tot_pre=None)
    if (s_imp or cap) and t is not None and cfg.t_slow <= t <= cfg.t_fast:
        x_cav = sol.cav_position(cfg, t)
        u_cav = cfg.u_xi
        if np.isfinite(x_cav) and u_cav < cfg.v_f:
            j = min(int(x_cav / cfg.dx), nx - 1)
            info["j"] = j
            if s_imp:
                if footprint == "global":
                    ja = np.flatnonzero(a)
                    k_front = max(j, int(ja[-1])) if ja.size else j
                else:
                    k_front = j
                    while k_front + 1 < nx and a[k_front + 1] > 0.0:
                        k_front += 1
                for face in range(j + 1, k_front + 2):
                    F_s[face] = 0.0
                info["zeroed"] = list(range(j + 1, k_front + 2))
            if cap:
                sigma_xi = cfg.beta * cfg.w * cfg.P / (cfg.v_f + cfg.w)
                omega_max = max(cfg.q_xi_max - u_cav * sigma_xi, 0.0)
                info["omega_max"] = omega_max
                F_tot = F_f[j + 1] + F_s[j + 1]
                info["F_tot_pre"] = F_tot
                if cap_trace == "auto":
                    trace = f[j] if s_imp else (f[j] + s[j])
                else:
                    trace = (f[j] + s[j]) if cap_trace == "fs" else f[j]
                if F_tot - u_cav * trace > omega_max:
                    info["cap_test"] = True
                    rho_hat = omega_max / (cfg.v_f - u_cav)
                    cap_abs = omega_max + u_cav * rho_hat
                    if F_tot > cap_abs:
                        scale = cap_abs / F_tot
                        F_f[j + 1] *= scale
                        F_s[j + 1] *= scale
                        info["cap_hit"] = True
                        info["scale"] = scale
    f_new = f - lam * np.diff(F_f)
    s_new = s - lam * np.diff(F_s)
    return (f_new, s_new, F_f[0], F_f[-1] + F_s[-1]), F_f, F_s, info


# --------------------------------------------------------------------------
# Replica of simulate() with pluggable transport / leader-loss pieces
# --------------------------------------------------------------------------

def ll_connect_rule(a_star, s, u_s, cfg):
    """The E10 anchoring rule re-stated: A anchors only while u_s < v_f."""
    a_anchor = a_star if u_s < cfg.v_f else np.zeros_like(a_star)
    return sol.leader_loss_rate_connect(a_anchor, s, cfg.dx, cfg.eta_la,
                                        cfg.tau_ll)


def ll_ratio_rule(a_star, s, u_s, cfg):
    c_wave = cfg.w if cfg.w_s is None else cfg.w_s
    return sol.leader_loss_rate(a_star, s, cfg.dx, cfg.eta_la, c_wave)


def replica_simulate(cfg, transport_fn, transport_cfg, ll_fn=None,
                     hook=None) -> dict:
    """Mirror of sol.simulate: transport (pluggable) -> leader-loss rate
    (pluggable, None = legacy) -> reaction_exact(mu_extra) -> optional
    downstream_release.  hook(n, t, f, s, a, u_s) sees the PRE-transport
    state."""
    nx = int(round(cfg.L_road / cfg.dx))
    x = (np.arange(nx) + 0.5) * cfg.dx
    f = np.zeros(nx) if cfg.f0 is None else np.array(cfg.f0, float)
    s = np.zeros(nx) if cfg.s0 is None else np.array(cfg.s0, float)
    n_steps = int(round(cfg.t_end / cfg.dt))
    injected = denied = outflowed = 0.0
    saves: list[tuple] = []

    def _save(step: int) -> None:
        t = step * cfg.dt
        a = sol.cav_density(cfg, t, nx)
        rho = a + f + s
        jw = sol._cav_weights(cfg, t, nx)
        u_s = sol.u_s_of_t(cfg, t)
        if jw is None:
            xc, om = np.nan, 0.0
        else:
            j, wr = jw
            xc = sol.cav_position(cfg, t)
            f_c = (1.0 - wr) * f[j] + wr * f[j + 1]
            rho_c = (1.0 - wr) * rho[j] + wr * rho[j + 1]
            v_c = float(sol.speed(rho_c, cfg.v_f, cfg.w, cfg.P))
            om = f_c * max(v_c - u_s, 0.0)
            if (cfg.q_xi_max is not None
                    and cfg.t_slow <= t <= cfg.t_fast and u_s < cfg.v_f):
                sigma_xi = cfg.beta * cfg.w * cfg.P / (cfg.v_f + cfg.w)
                om = min(om, max(cfg.q_xi_max - u_s * sigma_xi, 0.0))
        saves.append((t, a, f.copy(), s.copy(), xc, om))

    for n in range(n_steps):
        t = n * cfg.dt
        if n % cfg.save_every == 0:
            _save(n)
        u_s = sol.u_s_of_t(cfg, t)
        a = sol.cav_density(cfg, t, nx)
        if hook is not None:
            hook(n, t, f, s, a, u_s)
        f, s, q_adm, q_out = transport_fn(f, s, a, cfg.v_f, u_s,
                                          transport_cfg, t=t)
        injected += q_adm * cfg.dt
        denied += (cfg.q_in - q_adm) * cfg.dt
        outflowed += q_out * cfg.dt
        a_star = sol.cav_density(cfg, t + cfg.dt, nx)
        rho_star = a_star + f + s
        dv = np.maximum(sol.speed(rho_star, cfg.v_f, cfg.w, cfg.P) - u_s, 0.0)
        mu_ll = None if ll_fn is None else ll_fn(a_star, s, u_s, cfg)
        f, s = sol.reaction_exact(f, s, a_star, rho_star, dv, cfg.kappa_c,
                                  cfg.kappa_r, cfg.P, cfg.dt,
                                  cfg.capture_form, cfg.gamma, mu_extra=mu_ll)
        if cfg.downstream_release:
            x_c = sol.cav_position(cfg, t + cfg.dt)
            if np.isfinite(x_c):
                j_cav = int(x_c // cfg.dx)
                f[j_cav + 1:] += s[j_cav + 1:]
                s[j_cav + 1:] = 0.0
    _save(n_steps)
    return dict(
        t=np.array([sv[0] for sv in saves]), x=x,
        a=np.array([sv[1] for sv in saves]),
        f=np.array([sv[2] for sv in saves]),
        s=np.array([sv[3] for sv in saves]),
        x_cav=np.array([sv[4] for sv in saves]),
        omega=np.array([sv[5] for sv in saves]),
        N_s=np.array([np.sum(sv[3]) * cfg.dx for sv in saves]),
        denied_inflow=denied, injected=injected, outflowed=outflowed,
        on_road=float(np.sum(f + s) * cfg.dx))


# --------------------------------------------------------------------------
# Independent connectivity coverage (explicit backward recursion)
# --------------------------------------------------------------------------

def cover_backward(a, s, dx: float, n: int) -> np.ndarray:
    """cover_j = 1 if a_k > 0 for some k in j..j+n, else min{1, dx sum_{k=
    j+1..j+n} s_k cover_k}; cells beyond the road end are empty.  Because
    cover_j depends only on cover_k with k > j, ONE backward pass gives
    the fixed point (it is unique)."""
    nx = s.size
    cover = np.zeros(nx)
    for j in range(nx - 1, -1, -1):
        anchored = False
        for k in range(j, min(j + n, nx - 1) + 1):
            if a[k] > 0.0:
                anchored = True
                break
        if anchored:
            cover[j] = 1.0
            continue
        win = 0.0
        for k in range(j + 1, min(j + n, nx - 1) + 1):
            win += s[k] * cover[k]
        cover[j] = min(1.0, dx * win)
    return cover


def kleene(a, s, dx: float, n: int, start: float, max_sweeps: int,
           tol: float = 1e-14):
    """Jacobi/Kleene sweeps of the coverage map from cover = start;
    returns (cover, sweeps used, converged)."""
    nx = s.size
    zeros_n = np.zeros(n)
    a_pad = np.concatenate([a, zeros_n])
    anchored = a > 0.0
    for k in range(1, n + 1):
        anchored |= a_pad[k:k + nx] > 0.0
    cover = np.full(nx, float(start))
    for m in range(1, max_sweeps + 1):
        src_pad = np.concatenate([s * cover, zeros_n])
        win = np.zeros(nx)
        for k in range(1, n + 1):
            win += src_pad[k:k + nx]
        new = np.where(anchored, 1.0, np.minimum(1.0, dx * win))
        delta = float(np.max(np.abs(new - cover)))
        cover = new
        if delta < tol:
            return cover, m, True
    return cover, max_sweeps, False


# --------------------------------------------------------------------------
# P1: suite green
# --------------------------------------------------------------------------

def probe_suite():
    print("[P1] test suite (pytest test_solver.py)")
    r = subprocess.run([sys.executable, "-m", "pytest", "test_solver.py",
                        "-q", "-p", "no:cacheprovider"],
                       cwd=HERE, capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    print(f"  pytest: {tail}")
    check(r.returncode == 0, "pytest exit code 0")
    check("24 passed" in r.stdout, "exactly 24 tests passed")
    check("failed" not in r.stdout and "error" not in r.stdout.lower(),
          "no failures / errors reported")


# --------------------------------------------------------------------------
# P2: new-fields-off bit-identity vs git HEAD (all fields)
# --------------------------------------------------------------------------

def probe_bit_identity(sol_head, head_sha):
    print("[P2] new-fields-off bit-identity vs git-HEAD (E8) solver, ALL fields")
    cur_sha = hashlib.sha256((HERE / "solver.py").read_bytes()).hexdigest()
    print(f"  HEAD solver sha256 {head_sha[:12]}..., working tree {cur_sha[:12]}...")
    check(cur_sha != head_sha, "working-tree solver differs from HEAD (E10 uncommitted)")

    # (a), (b): E10 defaults and E10 fields spelled out (inert without eta_la)
    for label, kw in (("all-legacy", kw_legacy()),
                      ("A1 hybrid (cap + w_s)", kw_A1())):
        res_head = sol_head.simulate(sol_head.SimConfig(**kw))
        res_def = sol.simulate(sol.SimConfig(**kw))
        res_x = sol.simulate(sol.SimConfig(**kw, s_impermeable=False,
                                           ll_mode="connect", tau_ll=0.7,
                                           eta_la=None))
        compare_all_fields(res_def, res_head, f"(P2 {label}) E10 defaults vs HEAD")
        compare_all_fields(res_x, res_head,
                           f"(P2 {label}) s_imp=False, ll_mode=connect, tau_ll=0.7, "
                           "eta_la=None vs HEAD")

    # (c): hybrid + eta_la=200 ratio mode vs an independent composition of
    # the E9 scheme from HEAD's transport_step (no s_impermeable code path
    # exists there) + E10's leader_loss_rate + reaction_exact(mu_extra)
    kw_r = kw_A1(eta_la=200.0)
    cfg_r = sol.SimConfig(**kw_r)
    cfg_r_head = sol_head.SimConfig(**strip_head(kw_r))
    res_r = sol.simulate(cfg_r)
    res_comp = replica_simulate(cfg_r, sol_head.transport_step, cfg_r_head,
                                ll_fn=ll_ratio_rule)
    compare_all_fields(res_r, res_comp,
                       "(P2c) hybrid + eta_la=200 ratio: E10 simulate vs "
                       "[HEAD transport + leader_loss_rate + reaction(mu_extra)]")
    res_r2 = sol.simulate(sol.SimConfig(**kw_A1(eta_la=200.0, ll_mode="ratio",
                                                s_impermeable=False, tau_ll=9.0)))
    compare_all_fields(res_r2, res_r, "(P2c) ratio mode: tau_ll=9 inert")
    # sanity: the ratio run really differs from HEAD's eta_la=None run
    res_head_A1 = sol_head.simulate(sol_head.SimConfig(**kw_A1()))
    check(not bit_equal(res_r.s, res_head_A1.s),
          "(P2c) ratio run differs from the eta_la=None run (term is live)")

    # stored references
    ref = np.load(HERE / "out" / "ll_ref_A1.npz")
    check(str(ref["solver_sha256"]) == head_sha,
          "out/ll_ref_A1.npz solver_sha256 == sha256(git HEAD:solver.py)")
    for name in ref.files:
        if name != "solver_sha256":
            check(bit_equal(ref[name], getattr(res_head_A1, name)),
                  f"out/ll_ref_A1.npz['{name}'] == HEAD output")
    ref10 = np.load(HERE / "out" / "e10_ref_A1.npz")
    sha10 = str(ref10["solver_sha256"])
    check(sha10 not in (head_sha, cur_sha),
          f"out/e10_ref_A1.npz was produced by a third solver (E9, sha "
          f"{sha10[:12]}...), not HEAD nor the working tree")
    for name in ref10.files:
        if name != "solver_sha256":
            check(bit_equal(ref10[name], getattr(res_head_A1, name)),
                  f"out/e10_ref_A1.npz['{name}'] (E9 solver, eta_la=None) == "
                  "HEAD output -> E9's eta_la=None path == E8")
    print("  note: the E9 solver source is not on disk (never committed); the "
          "ratio path is therefore checked by composition (P2c), not by diff")

    # transport_step (s_impermeable=False) and reaction_exact vs HEAD on
    # random states, any t, cap / w_s on and off
    rng = np.random.default_rng(3)
    n_ok = n_tot = 0
    for kw in (kw_A1(), kw_A1(q_xi_max=None), kw_A1(w_s=None), kw_legacy()):
        cfg_n = sol.SimConfig(**kw)
        cfg_h = sol_head.SimConfig(**strip_head(kw))
        for t in (None, 100.0, 500.0, 501.0, 800.0):
            for _ in range(5):
                nx = 600
                f = rng.uniform(0.0, 0.1, nx)
                s = rng.uniform(0.0, 0.1, nx)
                a = (sol.cav_density(cfg_n, t, nx) if t is not None
                     else np.where(rng.uniform(size=nx) < 0.01,
                                   rng.uniform(0.0, 0.02, nx), 0.0))
                u_s = sol.u_s_of_t(cfg_n, t) if t is not None else cfg_n.u_xi
                on = sol.transport_step(f, s, a, cfg_n.v_f, u_s, cfg_n, t=t)
                oh = sol_head.transport_step(f, s, a, cfg_h.v_f, u_s, cfg_h, t=t)
                n_tot += 1
                n_ok += int(all(bit_equal(x, y) for x, y in zip(on, oh)))
    check(n_ok == n_tot, f"transport_step (s_impermeable=False) == HEAD on "
                         f"{n_ok}/{n_tot} random states (t None/100/500/501/800; "
                         "cap, no cap, no w_s, legacy)")
    n = 800
    f = rng.uniform(0.0, 0.12, n)
    s = rng.uniform(0.0, 0.12, n)
    a = np.where(rng.uniform(size=n) < 0.05, rng.uniform(0.0, 0.02, n), 0.0)
    rho = a + f + s
    dv = rng.uniform(-2.0, 13.0, n)
    ok = True
    for form, gamma in (("lf", None), ("af", None), ("lf", 0.37)):
        fh, sh = sol_head.reaction_exact(f, s, a, rho, dv, 5.32e-2, 1.65e-3, P,
                                         0.5, form, gamma)
        for mx in (None, np.zeros(n)):
            fn, sn = sol.reaction_exact(f, s, a, rho, dv, 5.32e-2, 1.65e-3, P,
                                        0.5, form, gamma, mu_extra=mx)
            ok &= bit_equal(fh, fn) and bit_equal(sh, sn)
    check(ok, "reaction_exact(mu_extra=None / 0) == HEAD (lf, af, gamma)")

    # replica fidelity on the full E10 config (used by P3 / P5)
    cfg_e = sol.SimConfig(**kw_E10())
    res_e = sol.simulate(cfg_e)
    res_rep = replica_simulate(cfg_e, sol.transport_step, cfg_e,
                               ll_fn=ll_connect_rule)
    compare_all_fields(res_rep, res_e,
                       "(P2 fidelity) audit replica of simulate() with the "
                       "re-stated anchoring rule == sol.simulate on the full "
                       "E10 config")
    return res_e, cfg_e


# --------------------------------------------------------------------------
# P3: s-impermeable interface semantics
# --------------------------------------------------------------------------

def manual_state_A(nx=600):
    """Queue (60 veh/km s) from cell 140 through the CAV cell 158, an
    s plume (10 veh/km) strictly downstream 159..170, free background."""
    f = np.full(nx, 0.020)
    s = np.zeros(nx)
    s[140:159] = 0.060
    s[159:171] = 0.010
    f[140:159] = 0.010
    return f, s


def manual_state_B(nx=600):
    """Free-flow cap-binding state: in the CAV cell f = 30, s = 10 veh/km
    (a = 0.84/dx there at t = 500), light traffic ahead."""
    f = np.full(nx, 0.020)
    s = np.zeros(nx)
    f[150:159] = 0.030
    s[150:159] = 0.010
    s[159:165] = 0.004
    return f, s


def probe_s_impermeable(sol_head):
    print("[P3] s-impermeable interface: one transport_step on manual states")
    kw = kw_A1(s_impermeable=True)
    cfg_i = sol.SimConfig(**kw)                       # s_imp + cap
    cfg_i_nocap = sol.SimConfig(**kw_A1(s_impermeable=True, q_xi_max=None))
    cfg_h = sol_head.SimConfig(**strip_head(kw))      # HEAD: cap only
    cfg_h_nocap = sol_head.SimConfig(**strip_head(kw_A1(q_xi_max=None)))
    cfg_off = sol.SimConfig(**kw_A1())                # E10 with s_imp off
    nx = int(round(cfg_i.L_road / cfg_i.dx))
    lam = cfg_i.dt / cfg_i.dx
    u = cfg_i.u_xi

    for t in (500.0, 501.0):
        a = sol.cav_density(cfg_i, t, nx)
        j = int(sol.cav_position(cfg_i, t) / cfg_i.dx)
        supp = np.flatnonzero(a).tolist()
        k_front = max(j, supp[-1])
        faces = list(range(j + 1, k_front + 2))
        print(f"  t = {t:g}: x_cav = {sol.cav_position(cfg_i, t):.1f} m, cap cell j = {j}, "
              f"footprint supp(a) = {supp}, expected zeroed faces {faces}")
        if t == 500.0:
            check(supp == [157, 158] and faces == [159],
                  "t=500: footprint {157,158} ends at the cap cell -> cap face only")
        else:
            check(supp == [158, 159] and faces == [159, 160],
                  "t=501: footprint {158,159} straddles past the cap cell -> "
                  "cap face + footprint front face")
        for sname, state in (("A queue+plume", manual_state_A(nx)),
                             ("B free/cap-binding", manual_state_B(nx))):
            f, s = state
            u_s = sol.u_s_of_t(cfg_i, t)
            (fi, si, qi, oi), Ff_i, Fs_i = step_with_fluxes(
                sol, f, s, a, cfg_i.v_f, u_s, cfg_i, t)
            (fl, sl, ql, ol), Ff_l, Fs_l = step_with_fluxes(
                sol_head, f, s, a, cfg_h.v_f, u_s, cfg_h, t)
            (fo, so, qo, oo), Ff_o, Fs_o = step_with_fluxes(
                sol, f, s, a, cfg_off.v_f, u_s, cfg_off, t)
            tag = f"t={t:g} state {sname}"
            # legacy path of the E10 solver == HEAD
            check(all(bit_equal(x, y) for x, y in
                      ((fo, fl), (so, sl), (Ff_o, Ff_l), (Fs_o, Fs_l)))
                  and qo == ql and oo == ol,
                  f"{tag}: E10 step with s_impermeable=False == HEAD step (bitwise)")
            # independent re-implementation reproduces the E10 step bitwise
            (fx, sx, qx, ox), Ff_x, Fs_x, info = fluxes_indep(
                f, s, a, cfg_i.v_f, u_s, cfg_i, t, s_imp=True, cap=True)
            check(bit_equal(fx, fi) and bit_equal(sx, si) and qx == qi and ox == oi
                  and bit_equal(Ff_x, Ff_i) and bit_equal(Fs_x, Fs_i),
                  f"{tag}: independent flux re-implementation == E10 step "
                  f"(f_new, s_new, fluxes bitwise); zeroed faces {info['zeroed']}")
            check(info["zeroed"] == faces, f"{tag}: zeroed faces == {faces}")
            # F_s == 0.0 exactly at the zeroed faces, bit-identical elsewhere
            check(all(Fs_i[k] == 0.0 for k in faces),
                  f"{tag}: F_s == 0.0 exactly at faces {faces} "
                  f"(legacy F_s there {[round(float(Fs_l[k]), 5) for k in faces]} veh/s)")
            check(all(Fs_l[k] > 0.0 for k in faces),
                  f"{tag}: legacy F_s > 0 at those faces (the constraint is live)")
            others = [k for k in range(nx + 1) if k not in faces]
            check(bit_equal(Fs_i[others], Fs_l[others]),
                  f"{tag}: F_s bit-identical at all {len(others)} other faces")
            # F_f: bit-identical at every face other than the cap face
            not_cap = [k for k in range(nx + 1) if k != j + 1]
            check(bit_equal(Ff_i[not_cap], Ff_l[not_cap]),
                  f"{tag}: F_f bit-identical at all {len(not_cap)} faces other than the cap face")
            _, _, _, info_l = fluxes_indep(f, s, a, cfg_h.v_f, u_s, cfg_h, t,
                                           s_imp=False, cap=True)
            pass_i = Ff_i[j + 1] - u * f[j]
            print(f"    cap face {j + 1}: legacy F_f {Ff_l[j + 1]:.5f} (cap hit "
                  f"{info_l['cap_hit']}, scale {info_l['scale']:.4f}) vs E10 F_f "
                  f"{Ff_i[j + 1]:.5f} (cap hit {info['cap_hit']}); omega_max "
                  f"{info['omega_max']:.5f}; E10 passing flow F^f - u f_j = "
                  f"{pass_i:.5f} veh/s; code test uses F^f - u (f_j + s_j) = "
                  f"{Ff_i[j + 1] - u * (f[j] + s[j]):.5f}")
            if Ff_i[j + 1] == Ff_l[j + 1]:
                check(True, f"{tag}: F_f unchanged at the cap face too (cap inactive in both)")
            else:
                check(info_l["cap_hit"] or info["cap_hit"],
                      f"{tag}: F_f differs at the cap face ONLY because the cap acted "
                      f"differently (legacy hit {info_l['cap_hit']}, E10 hit {info['cap_hit']})")
            # no-cap variant: F_f bit-identical everywhere
            (fi2, si2, _, _), Ff_i2, Fs_i2 = step_with_fluxes(
                sol, f, s, a, cfg_i.v_f, u_s, cfg_i_nocap, t)
            (_, _, _, _), Ff_l2, Fs_l2 = step_with_fluxes(
                sol_head, f, s, a, cfg_h.v_f, u_s, cfg_h_nocap, t)
            check(bit_equal(Ff_i2, Ff_l2) and all(Fs_i2[k] == 0.0 for k in faces)
                  and bit_equal(Fs_i2[others], Fs_l2[others]),
                  f"{tag}: cap OFF: F_f bit-identical at ALL faces, F_s zeroed at {faces} only")
            # per-cell accounting (cap-off pair, so only the s faces differ):
            # the s legacy moves across each zeroed face stays upstream
            ds = si2 - sl_nocap(sol_head, f, s, a, cfg_h, u_s, t, cfg_h_nocap)
            exp = np.zeros(nx)
            for k in faces:                     # face k sits between cells k-1 | k
                exp[k - 1] += lam * Fs_l2[k]
                exp[k] -= lam * Fs_l2[k]
            err = np.max(np.abs(ds - exp))
            moved = sum(float(Fs_l2[k]) for k in faces) * cfg_i.dt
            check(err <= 1e-16,
                  f"{tag}: per-cell accounting: s_new(imp) - s_new(legacy) == "
                  f"+lam F_s^leg at the upstream cell / -lam F_s^leg at the "
                  f"downstream cell of each zeroed face, max err {err:.1e} "
                  f"(moved from flux to storage: {moved:.6f} veh)")
            check(np.all(ds[[k for k in range(nx) if k not in
                             set(faces) | {k - 1 for k in faces}]] == 0.0),
                  f"{tag}: every other cell's s_new bit-unchanged")
            mass_in = np.sum(f + s) * cfg_i.dx + cfg_i.dt * qi
            mass_out = np.sum(fi + si) * cfg_i.dx + cfg_i.dt * oi
            check(abs(mass_in - mass_out) <= 1e-12 * mass_in,
                  f"{tag}: mass conserved by the step (rel {abs(mass_in - mass_out) / mass_in:.1e})")
            check(fi.min() >= 0.0 and si.min() >= 0.0
                  and (a + fi + si).max() <= P + 1e-12,
                  f"{tag}: f, s >= 0 and rho <= P after the step")

    # F1: the DM-G violation test keeps the trace f_j + s_j
    t = 500.0
    a = sol.cav_density(cfg_i, t, nx)
    j = int(sol.cav_position(cfg_i, t) / cfg_i.dx)
    f, s = manual_state_B(nx)
    u_s = sol.u_s_of_t(cfg_i, t)
    (fi, si, _, _), Ff_i, Fs_i = step_with_fluxes(sol, f, s, a, cfg_i.v_f, u_s, cfg_i, t)
    _, Ff_l, Fs_l, info_l = fluxes_indep(f, s, a, cfg_h.v_f, u_s, cfg_h, t,
                                         s_imp=False, cap=True)
    _, Ff_c, Fs_c, info_c = fluxes_indep(f, s, a, cfg_i.v_f, u_s, cfg_i, t,
                                         s_imp=True, cap=True, cap_trace="f")
    om = info_l["omega_max"]
    pass_i = Ff_i[j + 1] - u * f[j]
    print(f"  [F1] state B at t = 500, cap cell {j}: f_j = {f[j] * 1e3:.0f}, s_j = "
          f"{s[j] * 1e3:.0f} veh/km, omega_max = {om:.4f} veh/s")
    print(f"       legacy: F_tot = {info_l['F_tot_pre']:.4f}, test F_tot - u (f_j + s_j) = "
          f"{info_l['F_tot_pre'] - u * (f[j] + s[j]):.4f} > omega_max -> cap hit "
          f"{info_l['cap_hit']}, F_f -> {Ff_l[j + 1]:.4f}")
    print(f"       E10 s_imp: F_tot = F^f = {Ff_i[j + 1]:.4f}, code test F^f - u (f_j + s_j) = "
          f"{Ff_i[j + 1] - u * (f[j] + s[j]):.4f} -> no cap; passing flow F^f - u f_j = "
          f"{pass_i:.4f} > omega_max by {pass_i - om:+.4f} veh/s")
    print(f"       corrected trace (f_j only): cap hit {info_c['cap_hit']}, F_f -> "
          f"{Ff_c[j + 1]:.4f}, passing flow {Ff_c[j + 1] - u * f[j]:.4f} <= omega_max")
    if pass_i > om + 1e-12 and info_l["cap_hit"] and Ff_i[j + 1] == Ff_l[j + 1] / info_l["scale"]:
        issue("F1 transport_step (solver.py L654-655): with s_impermeable the DM-G "
              "violation test still subtracts u_cav (f_j + s_j) although F^s = 0 "
              "at that face (s does not pass): the test is loosened by u_cav s_j, "
              "so the cap can stay off while the passing flow F^f - u_cav f_j "
              f"exceeds omega_max (state B: {pass_i:.4f} > {om:.4f} veh/s; legacy "
              "caps the same state).  Consistent test: F^f - u_cav f_j > omega_max.")

    # F2: k_front = max(j, last nonzero of a) is the GLOBAL last A cell
    f, s = manual_state_A(nx)
    s[200:320] = 0.03                                   # s between the two vehicles
    a2 = a.copy()
    a2[300] = 1.0 / cfg_i.dx                            # a second A vehicle downstream
    (_, _, _, _), Ff_2, Fs_2 = step_with_fluxes(sol, f, s, a2, cfg_i.v_f, u_s, cfg_i, t)
    zero_faces = np.flatnonzero(Fs_2[j + 1:302] == 0.0) + j + 1
    _, _, Fs_2c, info_2c = fluxes_indep(f, s, a2, cfg_i.v_f, u_s, cfg_i, t,
                                        s_imp=True, cap=True, footprint="contiguous")
    print(f"  [F2] second A vehicle at cell 300 (s = 30 veh/km on 200..319): "
          f"solver zeroes F_s at {zero_faces.size} faces {int(zero_faces[0])}.."
          f"{int(zero_faces[-1])} (contiguous-footprint reading: faces {info_2c['zeroed']})")
    if zero_faces.size > len(info_2c["zeroed"]):
        issue("F2 transport_step (solver.py L648-649): k_front = max(j, ja[-1]) takes "
              "the LAST nonzero cell of a on the whole road, not the CAV's "
              "contiguous footprint; with a second A vehicle downstream every "
              f"face between them is made s-impermeable ({zero_faces.size} faces "
              "here).  Harmless for the single exogenous CAV of simulate(), but "
              "the function is documented as general (supp(a)).")

    # outside the window / t=None / off: bit-identical to HEAD
    f, s = manual_state_A(nx)
    s = np.maximum(s, 0.030)                # s > 0 everywhere: F_s at the CAV face is live
    ok_bit = ok_live = True
    for t in (None, 100.0, 249.5, 750.5, 900.0):
        a = sol.cav_density(cfg_i, t, nx) if t is not None else np.zeros(nx)
        if t is None:
            a[158] = 1.0 / cfg_i.dx
        u_s = sol.u_s_of_t(cfg_i, t) if t is not None else cfg_i.u_xi
        (fi, si, qi, oi), Ff_i, Fs_i = step_with_fluxes(sol, f, s, a, cfg_i.v_f, u_s, cfg_i, t)
        (fl, sl, ql, ol), Ff_l, Fs_l = step_with_fluxes(sol_head, f, s, a, cfg_h.v_f, u_s, cfg_h, t)
        ok_bit &= (bit_equal(fi, fl) and bit_equal(si, sl) and qi == ql and oi == ol
                   and bit_equal(Ff_i, Ff_l) and bit_equal(Fs_i, Fs_l))
        # and the s flux at the CAV face is NOT zero there (constraint inactive)
        if t is not None and np.isfinite(sol.cav_position(cfg_i, t)):
            jj = int(sol.cav_position(cfg_i, t) / cfg_i.dx)
            ok_live &= Fs_i[jj + 1] > 0.0
    check(ok_bit, "s_impermeable=True at t = None / 100 / 249.5 / 750.5 / 900: step "
                  "(f_new, s_new, fluxes, boundary fluxes) bit-identical to HEAD")
    check(ok_live, "... and F_s > 0 at the CAV face there (constraint inactive outside "
                   "[t_slow, t_fast])")
    # boundary of the window: active at t_slow and t_fast (inclusive)
    ok = True
    for t in (cfg_i.t_slow, cfg_i.t_fast):
        a = sol.cav_density(cfg_i, t, nx)
        jj = int(sol.cav_position(cfg_i, t) / cfg_i.dx)
        f2 = np.full(nx, 0.02)
        s2 = np.full(nx, 0.03)
        (_, _, _, _), _, Fs_i = step_with_fluxes(sol, f2, s2, a, cfg_i.v_f, cfg_i.u_xi, cfg_i, t)
        ok &= Fs_i[jj + 1] == 0.0
    check(ok, "active at t = t_slow and t = t_fast (inclusive, as the cap)")

    # step invariants on random near-jam states (s_imp + cap, t = 500 / 501)
    rng = np.random.default_rng(21)
    worst = -np.inf
    ok = True
    for t in (500.0, 501.0):
        a = sol.cav_density(cfg_i, t, nx)
        for _ in range(100):
            f = rng.uniform(0.0, 0.08, nx)
            s = rng.uniform(0.0, 0.08, nx)
            lo, hi = rng.integers(120, 160), rng.integers(160, 200)
            room = P - a[lo:hi]
            f[lo:hi] = rng.uniform(0.0, 0.5, hi - lo) * room
            s[lo:hi] = (room - f[lo:hi]) * rng.uniform(0.9, 1.0, hi - lo)
            fi, si, _, _ = sol.transport_step(f, s, a, cfg_i.v_f, cfg_i.u_xi, cfg_i, t=t)
            rho_n = a + fi + si
            worst = max(worst, float(rho_n.max() - P))
            ok &= fi.min() >= 0.0 and si.min() >= 0.0 and rho_n.max() <= P + 1e-12
    check(ok, f"200 random near-jam states (rho -> P at the CAV): f, s >= 0, rho <= P "
              f"after the s-impermeable step (max rho - P = {worst:.1e})")


def sl_nocap(sol_head, f, s, a, cfg_h, u_s, t, cfg_h_nocap):
    """Legacy (HEAD) s_new with the cap off."""
    _, sl, _, _ = sol_head.transport_step(f, s, a, cfg_h.v_f, u_s, cfg_h_nocap, t=t)
    return sl


def probe_cap_trace_bias(cfg_e, res_e):
    """[F1] quantified on the A1 E10 run: per-step comparison of the code's
    violation test with the passing-class test, and an in-memory corrected
    step's effect on the run."""
    print("[P3b] F1 quantified on the A1 E10 run (hybrid + s_imp + connect)")
    u = cfg_e.u_xi
    rows = []
    ds_viol = [0, 0, 0]          # steps checked, violations, straddle steps

    def hook(n, t, f, s, a, u_s):
        if not (cfg_e.t_slow <= t <= cfg_e.t_fast):
            return
        _, Ff, Fs, info = fluxes_indep(f, s, a, cfg_e.v_f, u_s, cfg_e, t,
                                       s_imp=True, cap=True)
        j = info["j"]
        if j is None:
            return
        # s strictly downstream of the CAV footprint must be exactly 0 at
        # EVERY step (the saves only ever see x_cav in the left half of a cell)
        k_front = max(j, int(np.flatnonzero(a)[-1]))
        ds_viol[0] += 1
        ds_viol[1] += int(np.any(s[k_front + 1:] != 0.0))
        ds_viol[2] += int(k_front > j)
        om = info["omega_max"]
        # pre-cap passing flow (F_s = 0 at this face, so F_tot_pre == F^f pre-cap)
        pass_flow = info["F_tot_pre"] - u * f[j]
        code_fire = info["cap_test"]
        true_fire = pass_flow > om
        rows.append((t, j, f[j], s[j], Ff[j + 1], pass_flow - om, code_fire,
                     true_fire, info["cap_hit"]))

    res_chk = replica_simulate(cfg_e, sol.transport_step, cfg_e,
                               ll_fn=ll_connect_rule, hook=hook)
    check(bit_equal(res_chk["s"], res_e.s) and bit_equal(res_chk["f"], res_e.f),
          "diagnostic replica reproduces the run bitwise")
    check(ds_viol[0] > 900 and ds_viol[1] == 0,
          f"s == 0.0 exactly strictly downstream of the CAV footprint at every one of "
          f"the {ds_viol[0]} slow-phase steps ({ds_viol[2]} of them with the footprint "
          "straddling past the cap cell, which no save ever shows)")
    rows_a = np.array([(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8])
                       for r in rows], float)
    code_fire = rows_a[:, 6] > 0
    true_fire = rows_a[:, 7] > 0
    disagree = true_fire & ~code_fire
    exc = np.maximum(rows_a[:, 5], 0.0)
    print(f"  slow-phase steps: {len(rows)}; code test fires {int(code_fire.sum())}, "
          f"cap actually scales {int(rows_a[:, 8].sum())}; passing-class test "
          f"F^f - u f_j > omega_max (pre-cap flux) fires {int(true_fire.sum())}; "
          f"disagreement (should cap, code does not) {int(disagree.sum())} steps")
    check(np.all(true_fire[code_fire]), "code-fire steps are a subset of passing-test-fire steps")
    if disagree.any():
        exc = np.where(disagree, exc, 0.0)
        i = int(np.argmax(exc))
        print(f"  max excess of the passing flow over omega_max: {exc.max():.4f} veh/s "
              f"at t = {rows_a[i, 0]:.1f} (f_j {rows_a[i, 2] * 1e3:.1f}, s_j "
              f"{rows_a[i, 3] * 1e3:.1f} veh/km); mean excess over disagreeing steps "
              f"{exc[disagree].mean():.4f} veh/s; disagreeing steps span t = "
              f"{rows_a[disagree, 0].min():.0f}..{rows_a[disagree, 0].max():.0f}")

    # in-memory corrected step: same scheme, violation test with the trace f_j
    def transport_corrected(f, s, a, c_f, c_s, cfg, t=None):
        out, _, _, _ = fluxes_indep(f, s, a, c_f, c_s, cfg, t,
                                    s_imp=cfg.s_impermeable,
                                    cap=cfg.q_xi_max is not None, cap_trace="f")
        return out

    def transport_same(f, s, a, c_f, c_s, cfg, t=None):
        out, _, _, _ = fluxes_indep(f, s, a, c_f, c_s, cfg, t,
                                    s_imp=cfg.s_impermeable,
                                    cap=cfg.q_xi_max is not None, cap_trace="auto")
        return out

    res_same = replica_simulate(cfg_e, transport_same, cfg_e, ll_fn=ll_connect_rule)
    compare_all_fields(res_same, res_e, "independent transport (trace f_j + s_j) "
                                        "reproduces sol.simulate on the E10 config")
    res_c = replica_simulate(cfg_e, transport_corrected, cfg_e, ll_fn=ll_connect_rule)

    def summary(r):
        i7 = int(np.argmin(np.abs(getf(r, "t") - 700.0)))
        xc = getf(r, "x_cav")[i7]
        x = getf(r, "x")
        m = (x >= xc - 1000.0) & (x <= xc - 200.0)
        rho = getf(r, "a")[i7] + getf(r, "f")[i7] + getf(r, "s")[i7]
        Ns = getf(r, "N_s")
        return (Ns.max(), Ns[i7], rho[m].mean() * 1e3, getf(r, "omega")[i7],
                getf(r, "on_road"), getf(r, "outflowed"))
    s0, s1 = summary(res_e), summary(res_c)
    print(f"  E10 as is  : max N_s {s0[0]:.2f}, N_s(700) {s0[1]:.2f} veh, wake rho(700) "
          f"{s0[2]:.2f} veh/km, omega(700) {s0[3]:.4f}, outflowed {s0[5]:.2f} veh")
    print(f"  trace f_j  : max N_s {s1[0]:.2f}, N_s(700) {s1[1]:.2f} veh, wake rho(700) "
          f"{s1[2]:.2f} veh/km, omega(700) {s1[3]:.4f}, outflowed {s1[5]:.2f} veh")
    rel = abs(s1[0] - s0[0]) / max(s0[0], 1e-9)
    print(f"  -> max N_s changes by {100 * rel:.2f}% with the consistent test "
          f"(impact of F1 on the A1 production config)")
    return rel


# --------------------------------------------------------------------------
# P4: connectivity coverage vs an explicit backward recursion
# --------------------------------------------------------------------------

def random_connect_state(rng, nx, dx):
    a = np.zeros(nx)
    s = np.zeros(nx)
    for _ in range(int(rng.integers(0, 4))):
        j = int(rng.integers(0, nx - 1))
        wr = float(rng.uniform(0.0, 1.0)) if rng.uniform() < 0.7 else 0.0
        a[j] += (1.0 - wr) / dx
        a[j + 1] += wr / dx
    for _ in range(int(rng.integers(1, 5))):
        L = int(rng.integers(1, 40))
        j0 = int(rng.integers(0, nx - L + 1))
        kind = rng.choice(["flat", "taper", "sparse", "noise"])
        if kind == "flat":
            s[j0:j0 + L] = rng.uniform(0.02, 0.1)
        elif kind == "taper":
            s[j0:j0 + L] = np.linspace(rng.uniform(0.03, 0.1),
                                       rng.uniform(0.001, 0.02), L)
        elif kind == "sparse":
            s[j0:j0 + L] = rng.uniform(0.001, 0.015, L)
        else:
            s[j0:j0 + L] = rng.uniform(0.0, 0.08, L) * (rng.uniform(size=L) < 0.6)
    return a, s


def probe_connect_fixed_point():
    print("[P4] connectivity coverage: solver vs explicit backward recursion")
    rng = np.random.default_rng(1234)
    dx = 50.0
    tau = 3.0
    worst = 0.0
    n_bit = 0
    n_states = 50
    max_sweeps = 0
    n_frac = 0
    for i in range(n_states):
        nx = int(rng.integers(200, 601))
        eta = float(rng.choice([50.0, 100.0, 200.0, 400.0]))
        n = max(1, int(round(eta / dx)))
        a, s = random_connect_state(rng, nx, dx)
        mu_sol = sol.leader_loss_rate_connect(a, s, dx, eta, tau)
        cov = cover_backward(a, s, dx, n)
        mu_ind = (1.0 - cov) / tau
        d = float(np.max(np.abs(mu_sol - mu_ind)))
        worst = max(worst, d)
        n_bit += int(bit_equal(mu_sol, mu_ind))
        _, sweeps, conv = kleene(a, s, dx, n, 0.0, 500)
        max_sweeps = max(max_sweeps, sweeps)
        n_frac += int(np.sum((cov > 0.0) & (cov < 1.0)))
        assert conv
    check(worst <= 1e-14,
          f"{n_states} random states (nx 200-600, n = 1/2/4/8, 0-3 split A vehicles, "
          f"flat/taper/sparse/noisy platoons, gaps): max |mu_sol - mu_backward| = "
          f"{worst:.1e} <= 1e-14; bitwise equal on {n_bit}/{n_states}; "
          f"{n_frac} partially covered cells exercised; solver Kleene sweeps <= {max_sweeps}")

    # from cover = 1: the fixed point is unique, so Jacobi from 1 converges
    # to the same coverage; only early stopping covers the platoon
    nx, eta, n = 100, 200.0, 4
    a = np.zeros(nx)
    s = np.zeros(nx)
    s[40:60] = 0.04
    cov0 = cover_backward(a, s, dx, n)
    c1, m1, conv1 = kleene(a, s, dx, n, 1.0, 500)
    c1_early, _, _ = kleene(a, s, dx, n, 1.0, 1)
    print(f"  disconnected platoon (cells 40..59, a = 0): backward recursion cover "
          f"= {cov0[40:60].max():.0f} on all platoon cells; Jacobi from cover=1 "
          f"converges in {m1} sweeps to max cover {c1[40:60].max():.0f} (same); "
          f"stopped after ONE sweep it still covers {int(np.sum(c1_early[40:60] == 1.0))}/20 cells")
    check(np.all(c1 == cov0) and conv1,
          "Jacobi from cover=1 converges to the SAME fixed point as from 0 "
          "(unique: the platoon head has an empty window, so cover_head = 0 "
          "for any start, and it propagates backward)")
    check(np.all(c1_early[40:59] == 1.0) and c1_early[59] == 0.0,
          "an early-stopped from-1 iteration (1 sweep) wrongly covers 19/20 "
          "platoon cells -- the only sense in which 'from 1' is wrong")
    # self-consistency of cover = 1 on the platoon: evaluate the map once
    c_map, _, _ = kleene(a, s, dx, n, 1.0, 1)
    check(c_map[59] == 0.0,
          "cover = 1 is NOT a fixed point on a disconnected platoon (the map sends "
          "the head cell to 0): the docstring's 'cover = 1 would also be "
          "self-consistent' is false on a finite road with empty padding")
    issue("F3 leader_loss_rate_connect docstring (solver.py L453-464) / module docstring (L126-146): "
          "the coverage map is strictly upper-triangular (cover_j depends on "
          "cover_{k>j} only; padding beyond the road end is empty), so its fixed "
          "point is UNIQUE and equals a one-pass backward recursion; 'cover = 1 "
          "would also be self-consistent' / 'LEAST fixed point' is a false "
          "statement of the model (harmless numerically).")

    # sweep cap: n = 1 propagates one cell per sweep
    nx = 600
    eta = 50.0
    n = 1
    a = np.zeros(nx)
    s = np.zeros(nx)
    a[nx - 1] = 1.0 / dx
    s[:nx - 1] = 0.04
    mu_sol = sol.leader_loss_rate_connect(a, s, dx, eta, tau)
    cov = cover_backward(a, s, dx, n)
    _, sweeps, conv = kleene(a, s, dx, n, 0.0, 500)
    wrong = int(np.sum((mu_sol > 0.0) & (cov == 1.0)))
    print(f"  600-cell queue anchored at its head, eta_la = dx (n = 1): backward "
          f"recursion cover = 1 everywhere; solver (500-sweep cap, converged = "
          f"{conv}) returns mu = 1/tau on {wrong} tail cells")
    check(np.all(cov[:nx - 1] == 1.0), "backward recursion: anchored 600-cell queue fully covered")
    if wrong > 0:
        issue("F4 leader_loss_rate_connect (solver.py L502-510, Kleene loop 'for _ in "
              "range(500)'): coverage propagates <= n cells per sweep, so an "
              f"anchored queue longer than 500 n cells is truncated ({wrong} tail "
              "cells released at 1/tau for n = 1 on a 600-cell road; needs a "
              ">= 25 km queue at dx = 50, eta_la = 50 -- not reached in the E10 "
              "scenarios).  A one-pass backward recursion is exact, tolerance-free "
              "and O(nx n).")
    # realistic horizon: n = 4 needs only nx/4 sweeps
    _, sweeps4, conv4 = kleene(a, s, dx, 4, 0.0, 500)
    check(conv4, f"same queue with eta_la = 200 (n = 4) converges in {sweeps4} sweeps")

    # the brief's synthetic cases via the backward recursion (cross-check)
    nx = 100
    a = np.zeros(nx)
    s = np.zeros(nx)
    s[40:60] = np.linspace(0.040, 0.005, 20)
    a[60] = 1.0 / dx
    cov = cover_backward(a, s, dx, 4)
    check(np.all(cov[40:60] == 1.0), "tapered anchored queue: cover == 1 exactly (backward)")
    a2 = np.zeros(nx)
    s2 = np.zeros(nx)
    s2[30:45] = 0.04
    s2[50:60] = 0.04
    a2[60] = 1.0 / dx
    cov = cover_backward(a2, s2, dx, 4)
    check(np.all(cov[50:60] == 1.0) and np.all(cov[30:45] == 0.0),
          "5-cell gap: front part covered, rear part 0 (backward)")


# --------------------------------------------------------------------------
# P5: anchoring rule 'a anchors only while u_s < v_f'
# --------------------------------------------------------------------------

def _attr_chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def probe_anchoring(res_e, cfg_e):
    print("[P5] anchoring rule in simulate(): static + dynamic")
    src = (HERE / "solver.py").read_text()
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    sim = fns["simulate"]
    blocks = [n for n in ast.walk(sim) if isinstance(n, ast.If)
              and "eta_la" in (ast.get_source_segment(src, n.test) or "")]
    check(len(blocks) == 1, "exactly one 'if cfg.eta_la is not None' block in simulate()")
    blk = blocks[0]
    names = {n.id for n in ast.walk(blk) if isinstance(n, ast.Name)}
    attrs = {_attr_chain(n) for n in ast.walk(blk) if isinstance(n, ast.Attribute)}
    attrs.discard(None)
    allowed_names = {"cfg", "mu_ll", "a_anchor", "a_star", "s", "u_s", "np",
                     "c_wave", "leader_loss_rate", "leader_loss_rate_connect"}
    allowed_attrs = {"cfg.eta_la", "cfg.ll_mode", "cfg.v_f", "cfg.dx", "cfg.tau_ll",
                     "cfg.w", "cfg.w_s", "np.zeros_like"}
    check(names <= allowed_names, f"eta_la block names: {sorted(names)}")
    check(attrs <= allowed_attrs, f"eta_la block attributes: {sorted(attrs)} "
                                  "(cfg constants + np.zeros_like only)")
    forbidden = {"x_cav", "x_c", "xc", "cav_position", "_cav_weights",
                 "cav_density", "t", "t_slow", "t_fast", "t_enter", "n", "j"}
    check(not (names & forbidden), "eta_la block references no position / time / "
                                   "schedule name (x_cav, cav_position, t, ...)")
    conn = [n for n in ast.walk(blk) if isinstance(n, ast.If)
            and "connect" in (ast.get_source_segment(src, n.test) or "")]
    check(len(conn) == 1, "one ll_mode == 'connect' branch")
    cmp_nodes = [n for st in conn[0].body for n in ast.walk(st) if isinstance(n, ast.IfExp)]
    check(len(cmp_nodes) == 1 and
          ast.get_source_segment(src, cmp_nodes[0].test) == "u_s < cfg.v_f",
          f"anchoring condition is literally 'u_s < cfg.v_f' (line {cmp_nodes[0].lineno if cmp_nodes else '?'})")
    # u_s in simulate is the class-A speed schedule u_s_of_t(cfg, t)
    assigns = [n for n in ast.walk(sim) if isinstance(n, ast.Assign)
               and any(isinstance(t_, ast.Name) and t_.id == "u_s" for t_ in n.targets)]
    srcs = {ast.get_source_segment(src, n.value) for n in assigns}
    check(srcs == {"u_s_of_t(cfg, t)"},
          f"u_s is bound only as u_s_of_t(cfg, t) (the class-A speed input that "
          f"also enters Delta v): {sorted(srcs)}")
    print("  note: u_s(t) is the prescribed class-A speed (u_xi inside [t_slow, "
          "t_fast], v_f outside) -- a time-only class input, no position.")

    # dynamic: queue released after t_fast, in-queue mu == 0 in the slow phase
    n_look = max(1, int(round(cfg_e.eta_la / cfg_e.dx)))
    dx = cfg_e.dx
    Ns = res_e.N_s
    i_max = int(np.argmax(Ns))

    def at(tt):
        return Ns[int(np.argmin(np.abs(res_e.t - tt)))]
    print(f"  N_s: max {Ns[i_max]:.2f} veh at t = {res_e.t[i_max]:g}; "
          f"N_s(700) {at(700):.2f}, (750) {at(750):.2f}, (760) {at(760):.2f}, "
          f"(770) {at(770):.2f}, (780) {at(780):.2f}, (800) {at(800):.2f}, "
          f"(1000) {Ns[-1]:.4f} veh")
    check(Ns[-1] < 0.05 * Ns.max() and Ns[-1] < 0.5,
          f"queue released after t_fast: N_s(1000) = {Ns[-1]:.4f} veh "
          f"({Ns[-1] / Ns.max():.1e} of max)")
    # e-fold: after t_fast + eta/u the whole queue is unanchored: N_s decays
    # at exp(-t/tau) -- check the decade after 760
    i760 = int(np.argmin(np.abs(res_e.t - 760.0)))
    i770 = int(np.argmin(np.abs(res_e.t - 770.0)))
    ratio = Ns[i770] / Ns[i760] if Ns[i760] > 0 else np.nan
    print(f"  N_s(770)/N_s(760) = {ratio:.4f} vs exp(-10/tau) = {np.exp(-10.0 / cfg_e.tau_ll):.4f} "
          "(pure relabelling at 1/tau once unanchored)")
    n_cells = n_saves = 0
    mu_worst = 0.0
    for i in range(res_e.t.size):
        t = res_e.t[i]
        if not (cfg_e.t_slow <= t <= cfg_e.t_fast) or not np.isfinite(res_e.x_cav[i]):
            continue
        a_i, s_i = res_e.a[i], res_e.s[i]
        cov = cover_backward(a_i, s_i, dx, n_look)
        mu = (1.0 - cov) / cfg_e.tau_ll
        mu_sol = sol.leader_loss_rate_connect(a_i, s_i, dx, cfg_e.eta_la, cfg_e.tau_ll)
        assert bit_equal(mu, mu_sol)
        kf = max(int(res_e.x_cav[i] / dx), int(np.flatnonzero(a_i)[-1]))
        k, inq = kf, []
        while k >= 0 and (s_i[k] > 0.020 or a_i[k] > 0.0):
            if s_i[k] > 0.020:
                inq.append(k)
            k -= 1
        if inq:
            n_saves += 1
            n_cells += len(inq)
            mu_worst = max(mu_worst, float(np.max(mu[inq])))
    check(n_cells > 50 and mu_worst == 0.0,
          f"slow phase: mu_ll == 0.0 exactly on all {n_cells} attached-queue cells "
          f"over {n_saves} saves (independent backward cover)")
    # toggle u_s on a frozen slow-phase state
    i500 = int(np.argmin(np.abs(res_e.t - 500.0)))
    a_i, s_i = res_e.a[i500], res_e.s[i500]
    q = s_i > 0.020
    mu_slow = ll_connect_rule(a_i, s_i, cfg_e.u_xi, cfg_e)
    mu_free = ll_connect_rule(a_i, s_i, cfg_e.v_f, cfg_e)
    check(np.all(mu_slow[q] == 0.0) and np.all(mu_free[q] == 1.0 / cfg_e.tau_ll),
          f"frozen t=500 state ({int(q.sum())} queue cells): u_s = u_xi -> mu = 0 "
          f"exactly; u_s = v_f (same a, same s) -> mu = 1/tau exactly on every queue cell")
    # after t_fast the CAV is still on the road with a > 0: the rule (not the
    # position) is what unanchors -- a_star is nonzero there
    i760 = int(np.argmin(np.abs(res_e.t - 760.0)))
    live = res_e.s[i760] > 1e-6
    mu760 = ll_connect_rule(res_e.a[i760], res_e.s[i760], cfg_e.v_f, cfg_e)
    check(np.isfinite(res_e.x_cav[i760]) and res_e.a[i760].max() > 0.0 and live.sum() > 5
          and np.all(mu760[live] == 1.0 / cfg_e.tau_ll),
          f"t = 760: CAV still on road (a > 0, x_cav = {res_e.x_cav[i760]:.0f} m) but u_s = v_f "
          f"-> all {int(live.sum())} cells with s > 0 released at exactly 1/tau")


# --------------------------------------------------------------------------
# P6: purity of both leader-loss functions
# --------------------------------------------------------------------------

def probe_purity():
    print("[P6] purity: static inspection of leader_loss_rate / _connect")
    src = (HERE / "solver.py").read_text()
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    scenario = {"cfg", "t", "x_cav", "x_c", "xc", "cav_position", "_cav_weights",
                "cav_density", "u_s", "u_s_of_t", "t_slow", "t_fast", "t_enter",
                "v_cav_free", "j_cav", "u_xi", "schedule", "simulate"}
    for name, params_exp in (("leader_loss_rate", ["a", "s", "dx", "eta_la", "c_wave"]),
                             ("leader_loss_rate_connect", ["a", "s", "dx", "eta_la", "tau_ll"])):
        fn = fns[name]
        params = [x.arg for x in fn.args.args]
        check(params == params_exp and fn.args.vararg is None and fn.args.kwarg is None
              and not fn.args.kwonlyargs,
              f"{name}({', '.join(params)}): state arrays + constants, no *args/**kwargs")
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        bad = (names | attrs) & scenario
        check(not bad, f"{name}: no scenario / time / cfg name (offenders {sorted(bad)})")
        body_src = "\n".join(ast.get_source_segment(src, st) or "" for st in fn.body[1:])
        check("cav" not in body_src.lower() and "cfg" not in body_src and "schedule" not in body_src,
              f"{name}: body text contains no 'cav' / 'cfg' / 'schedule'")
        code = getattr(sol, name).__code__
        check(code.co_freevars == (), f"{name}: no closure variables")
        glob = set(code.co_names)
        allowed = {"np", "asarray", "size", "max", "int", "round", "float", "concatenate",
                   "zeros", "range", "where", "minimum", "maximum", "abs"}
        check(glob <= allowed, f"{name}: global names numpy/builtins only: {sorted(glob)}")
        print(f"  {name}: lines {fn.lineno}-{fn.end_lineno}")
    # translation invariance of the connect rate (no positional dependence)
    rng = np.random.default_rng(5)
    a, s = random_connect_state(rng, 400, 50.0)
    mu = sol.leader_loss_rate_connect(a, s, 50.0, 200.0, 3.0)
    sh = 37
    mu_sh = sol.leader_loss_rate_connect(np.roll(a, sh), np.roll(s, sh), 50.0, 200.0, 3.0)
    check(bit_equal(mu_sh[sh:], mu[:400 - sh]) and mu.max() > 0.0,
          "connect rate: translating the state by 37 cells shifts mu bit-identically")


# --------------------------------------------------------------------------
# P7: invariants under stress, all E10 knobs on
# --------------------------------------------------------------------------

def invariants(res, cfg, what: str, expect_queue: bool = True) -> None:
    balance = res.on_road + res.outflowed
    rel = abs(res.injected - balance) / max(res.injected, 1.0)
    rho = res.a + res.f + res.s
    bad = []
    for name in ALL_FIELDS:
        v = np.asarray(getattr(res, name), float)
        if name == "x_cav":
            exp_nan = np.array([not np.isfinite(sol.cav_position(cfg, t)) for t in res.t])
            if not np.array_equal(np.isnan(v), exp_nan):
                bad.append("x_cav NaN pattern")
        elif not np.all(np.isfinite(v)):
            bad.append(name)
    print(f"  {what}: ledger {rel:.1e}, min f {res.f.min():.1e}, min s {res.s.min():.1e}, "
          f"max rho-P {rho.max() - P:.1e}, N_s max {res.N_s.max():.1f}, end {res.N_s[-1]:.3f} veh")
    check(not bad, f"{what}: no NaN/inf in any field (x_cav NaN only off-road)")
    check(rel < 1e-10, f"{what}: mass ledger closes to 1e-10")
    check(res.f.min() >= 0.0 and res.s.min() >= 0.0, f"{what}: f, s >= 0")
    check(rho.max() <= P + 1e-12, f"{what}: rho <= P")
    if expect_queue:
        check(res.N_s.max() > 5.0, f"{what}: a queue formed (N_s max > 5 veh)")
    else:
        print(f"  note  {what}: N_s max {res.N_s.max():.2f} veh -- with n = 1 the connect "
              "coverage threshold is one slow vehicle per dx (20 veh/km), so a queue "
              "never bootstraps at eta_la = dx (parameter behaviour, not an invariant)")


def probe_invariants():
    print("[P7] invariants under stress (q_in = 2900/3600, u_xi = 10), all E10 knobs on")
    st = dict(q_in=2900.0 / 3600.0, u_xi=10.0)
    cases = (
        ("A1 s_imp+connect+cap+w_s", kw_E10(**st)),
        ("A10 s_imp+connect+cap+w_s", kw_A10(**E10_KNOBS, **st)),
        ("A1 eta50 (n=1)", kw_E10(eta_la=50.0, **st)),
        ("A1 eta1000 (n=20)", kw_E10(eta_la=1000.0, **st)),
        ("A1 tau_ll=0.5", kw_E10(tau_ll=0.5, **st)),
        ("A1 tau_ll=30", kw_E10(tau_ll=30.0, **st)),
        ("A1 dx25/dt0.25", kw_E10(dx=25.0, dt=0.25, save_every=40, **st)),
        ("A1 +downstream_release", kw_E10(downstream_release=True, **st)),
        ("A1 no cap", kw_E10(q_xi_max=None, **st)),
        ("A1 no w_s (road branch)", kw_E10(w_s=None, **st)),
        ("A1 ratio + s_imp", kw_E10(ll_mode="ratio", **st)),
        ("A1 u_xi=5", kw_E10(q_in=2900.0 / 3600.0, u_xi=5.0)),
        ("A1 s_imp only (no LL)", kw_A1(s_impermeable=True, **st)),
    )
    for what, kw in cases:
        cfg = sol.SimConfig(**kw)
        invariants(sol.simulate(cfg), cfg, what, expect_queue="eta50" not in what)


# --------------------------------------------------------------------------
# P8: exact-update property with connect mu_extra
# --------------------------------------------------------------------------

def rk4_frozen(f0, s0, sigma, mu_tot, T: float, n_rk: int = 2000):
    h = T / n_rk
    f = np.array(f0, float)
    s = np.array(s0, float)

    def rhs(ff, ss):
        d = mu_tot * ss - sigma * ff
        return d, -d

    for _ in range(n_rk):
        k1f, k1s = rhs(f, s)
        k2f, k2s = rhs(f + 0.5 * h * k1f, s + 0.5 * h * k1s)
        k3f, k3s = rhs(f + 0.5 * h * k2f, s + 0.5 * h * k2s)
        k4f, k4s = rhs(f + h * k3f, s + h * k3s)
        f = f + h / 6.0 * (k1f + 2 * k2f + 2 * k3f + k4f)
        s = s + h / 6.0 * (k1s + 2 * k2s + 2 * k3s + k4s)
    return f, s


def probe_exact_update(res_e, cfg_e):
    print("[P8] exact update with connect mu_extra vs 2000-step RK4")
    kc, kr = cfg_e.kappa_c, cfg_e.kappa_r
    for t_probe in (500.0, 760.0):
        i = int(np.argmin(np.abs(res_e.t - t_probe)))
        a, f, s = res_e.a[i], res_e.f[i], res_e.s[i]
        rho = a + f + s
        u_s = sol.u_s_of_t(cfg_e, res_e.t[i])
        dv = np.maximum(sol.speed(rho, V_F, W, P) - u_s, 0.0)
        mu_ll = ll_connect_rule(a, s, u_s, cfg_e)
        sigma = kc * (a + s) * dv
        mu_tot = kr * np.maximum(P - rho, 0.0) * dv + mu_ll
        active = int(np.sum((mu_ll > 0.0) & (s > 1e-6)))
        print(f"  frozen t = {res_e.t[i]:.0f}: u_s = {u_s:.1f}, N_s = {res_e.N_s[i]:.2f} veh, "
              f"cells with mu_ll > 0 and s > 0: {active}, max mu_ll {mu_ll.max():.4f}, "
              f"max sigma {sigma.max():.4f}, max mu_tot {mu_tot.max():.4f} 1/s")
        check(active > 0, f"t = {res_e.t[i]:.0f}: connect term live somewhere")
        for T in (cfg_e.dt, 5.0):
            fe, se = sol.reaction_exact(f, s, a, rho, dv, kc, kr, P, T,
                                        cfg_e.capture_form, cfg_e.gamma, mu_extra=mu_ll)
            fr, sr = rk4_frozen(f, s, sigma, mu_tot, T)
            err = max(np.max(np.abs(fe - fr)), np.max(np.abs(se - sr)))
            p_err = np.max(np.abs((fe + se) - (f + s)))
            check(err < 1e-10, f"t = {res_e.t[i]:.0f}, T = {T}: exact vs RK4 max err {err:.2e}")
            check(p_err <= 1e-16 * max(1.0, np.max(f + s)) and fe.min() >= 0.0 and se.min() >= 0.0,
                  f"t = {res_e.t[i]:.0f}, T = {T}: p conserved ({p_err:.1e}), positive")
    rng = np.random.default_rng(11)
    nx = 600
    f = rng.uniform(0.0, 0.1, nx)
    a, s = random_connect_state(rng, nx, 50.0)
    s = np.minimum(s + rng.uniform(0.0, 0.05, nx), 0.12)
    rho = a + f + s
    dv = rng.uniform(0.0, 13.0, nx)
    mu_x = sol.leader_loss_rate_connect(a, s, 50.0, 200.0, 0.8)
    for form, gamma in (("lf", None), ("af", None), ("lf", 0.4)):
        ell = (a + gamma * s) if gamma is not None else ((a + s) if form == "lf" else a)
        sigma = 0.03 * ell * dv
        mu_tot = 0.01 * np.maximum(P - rho, 0.0) * dv + mu_x
        for T in (0.5, 5.0):
            fe, se = sol.reaction_exact(f, s, a, rho, dv, 0.03, 0.01, P, T, form, gamma,
                                        mu_extra=mu_x)
            fr, sr = rk4_frozen(f, s, sigma, mu_tot, T)
            err = max(np.max(np.abs(fe - fr)), np.max(np.abs(se - sr)))
            check(err < 1e-10 and np.max(np.abs((fe + se) - (f + s))) <= 1e-16
                  and fe.min() >= 0.0 and se.min() >= 0.0,
                  f"random state, connect mu_extra (tau 0.8), {form}, gamma={gamma}, "
                  f"T = {T}: max err {err:.2e}, p conserved, positive")


# --------------------------------------------------------------------------

def main() -> int:
    np.seterr(invalid="raise", divide="raise", over="raise", under="ignore")
    probe_suite()
    sol_head, head_sha = load_head_solver()
    res_e, cfg_e = probe_bit_identity(sol_head, head_sha)
    probe_s_impermeable(sol_head)
    probe_cap_trace_bias(cfg_e, res_e)
    probe_connect_fixed_point()
    probe_anchoring(res_e, cfg_e)
    probe_purity()
    probe_invariants()
    probe_exact_update(res_e, cfg_e)
    print()
    if ISSUES:
        print(f"ISSUES ({len(ISSUES)}):")
        for m in ISSUES:
            print(f"  - {m}")
    if FAILURES:
        print(f"FAILED probes ({len(FAILURES)}):")
        for m in FAILURES:
            print(f"  - {m}")
        return 1
    print("ALL PROBES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
