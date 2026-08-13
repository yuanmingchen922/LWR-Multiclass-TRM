"""Invariant-domain-preserving splitting scheme for the three-mode LWR
transition model (Multi-class LWR Equations.tex).

Model (tex eqs. a/f/s-equation): modes A (CAV, density a), B_f (free cars,
density f) and B_s (synchronized cars, density s), all in veh/m, with
rho = a + f + s and shared triangular FD parameters (w, P):

    Q_c(r) = min{c r, w (P - r)}                    (eq. intrinsic-fd)
    C_c    = c w P / (c + w)                        (eq. critical-capacity)
    D_c(r) = min{c r, C_c},  S_c(r) = min{C_c, w (P - r)}
                                                    (eq. intrinsic-demand-supply)
    V_c(r) = min{c, w (P / r - 1)}  (r > 0), else c (eq. intrinsic-speed-explicit)

Scheme (tex section "Invariant-domain-preserving splitting scheme"): per step,
first the conservative Godunov/CTM transport update (eq. transport-update)
with the class-specific min-flux

    F^m_{j+1/2} = pi_j^m min{D_{c_m}(rho_j), S_{c_m}(rho_{j+1})}   (eq. min-flux)

with pi_j^m = rho_j^m / rho_j (0 at vacuum, eq. cell-fractions); then the exact
reaction substep (eq. exact-reaction-update) with per-cell coefficients frozen
at the substep start.  Under the CFL condition dt max{v_f, w} / dx <= 1
(eq. CFL) the transport keeps f, s >= 0 and rho <= P, and the reaction step
conserves p = f + s exactly.

Deviation from the fully coupled scheme: a(x, t) is EXOGENOUS here.  The CAV
is one vehicle placed on the cell(s) straddling its prescribed trajectory
x_cav(t) (linear split by position, a = weight / dx).  a still occupies road
space -- it enters rho, hence reduces supply and takes its fraction pi^a of
interface capacity -- but its own flux share is simply not applied: a is
re-prescribed from the schedule each step.  Because a is bounded by 1/dx and
the simulated densities stay far from P, the discrete invariant domain is
preserved in practice (verified in test_solver.py).

Class speeds: c_f = v_f (constant); c_s = c_a = u_s(t), the CAV reference
speed schedule, u_s(t) = u_xi for t in [t_slow, t_fast] and v_f otherwise.

Units: SI throughout (veh/m, m/s, veh/s).  kappa_c, kappa_r are per-vehicle
[1/veh] and unit-consistent in SI without conversion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Triangular fundamental diagram (SI units)
# --------------------------------------------------------------------------

def capacity(c: float, w: float, P: float) -> float:
    """C_c = c w P / (c + w)  [veh/s]  (tex eq. critical-capacity)."""
    return c * w * P / (c + w)


def demand(rho, c: float, w: float, P: float):
    """D_c(r) = min{c r, C_c}  (tex eq. intrinsic-demand-supply)."""
    return np.minimum(c * np.asarray(rho, float), capacity(c, w, P))


def supply(rho, c: float, w: float, P: float):
    """S_c(r) = min{C_c, w (P - r)}  (tex eq. intrinsic-demand-supply)."""
    return np.minimum(capacity(c, w, P), w * (P - np.asarray(rho, float)))


def speed(rho, c: float, w: float, P: float):
    """V_c(r) = min{c, w (P/r - 1)} for r > 0, else c (tex eq.
    intrinsic-speed-explicit)."""
    rho = np.asarray(rho, float)
    # floor guards against overflow of P/rho at denormal densities; the min
    # with c makes the result exact for any rho below ~1e-300
    r_safe = np.maximum(rho, 1e-300)
    return np.where(rho > 0.0, np.minimum(c, w * (P / r_safe - 1.0)), c)


# --------------------------------------------------------------------------
# Configuration and result containers
# --------------------------------------------------------------------------

@dataclass
class SimConfig:
    v_f: float                    # fast free-flow speed [m/s]
    w: float                      # backward wave speed [m/s]
    P: float                      # jam density [veh/m]
    q_in: float                   # background inflow demand at x = 0 [veh/s]
    u_xi: float                   # CAV reference speed during slowdown [m/s]
    kappa_c: float                # capture coefficient [1/veh]
    kappa_r: float                # release coefficient [1/veh]
    capture_form: str = "lf"      # "lf": ell = a + s (baseline); "af": ell = a
    dx: float = 50.0              # cell length [m]
    dt: float = 1.0               # time step [s]
    L_road: float = 30000.0       # road length [m]
    t_end: float = 1000.0         # simulation horizon [s]
    t_enter: float = 100.0        # CAV enters x = 0 [s]
    t_slow: float = 250.0         # slowdown window start [s]
    t_fast: float = 750.0         # slowdown window end [s]
    v_cav_free: float = 27.78     # CAV speed outside the slowdown window [m/s]
    save_every: int = 10          # save cadence [steps]
    f0: np.ndarray | None = None  # optional initial f (default: empty road)
    s0: np.ndarray | None = None  # optional initial s (default: empty road)
    q_xi_max: float | None = None  # one-lane-blocked capacity [veh/s];
    #                                None = capacity cap disabled (legacy)
    beta: float = 0.5             # open-lane fraction of the blocked FD


@dataclass
class SimResult:
    t: np.ndarray          # (n_save,) saved times [s]
    x: np.ndarray          # (nx,) cell centers [m]
    a: np.ndarray          # (n_save, nx) CAV density [veh/m]
    f: np.ndarray          # (n_save, nx) free-car density [veh/m]
    s: np.ndarray          # (n_save, nx) synchronized-car density [veh/m]
    x_cav: np.ndarray      # (n_save,) CAV position [m], nan when off road
    omega: np.ndarray      # (n_save,) f_cav * (V_{v_f}(rho_cav) - u_s)_+ [veh/s]
    N_s: np.ndarray        # (n_save,) queued vehicles sum(s) dx [veh]
    denied_inflow: float   # cumulative inflow denied by supply [veh]
    injected: float        # cumulative admitted inflow at x = 0 [veh]
    outflowed: float       # cumulative outflow at x = L [veh]
    on_road: float         # final passenger-car mass sum(f + s) dx [veh]


# --------------------------------------------------------------------------
# Prescribed CAV schedule
# --------------------------------------------------------------------------

def u_s_of_t(cfg: SimConfig, t: float) -> float:
    """Slow reference speed u_s(t): u_xi inside [t_slow, t_fast], else v_f."""
    return cfg.u_xi if cfg.t_slow <= t <= cfg.t_fast else cfg.v_f


def cav_position(cfg: SimConfig, t: float) -> float:
    """Analytic CAV trajectory; nan before entry and after leaving the road.

    The CAV drives at v_cav_free except during [t_slow, t_fast], where it
    drives at u_xi.  Position is the integral of that speed from t_enter.
    """
    if t < cfg.t_enter:
        return np.nan
    dur_slow = max(0.0, min(t, cfg.t_fast) - max(cfg.t_enter, cfg.t_slow))
    x = cfg.v_cav_free * ((t - cfg.t_enter) - dur_slow) + cfg.u_xi * dur_slow
    return x if x <= cfg.L_road else np.nan


def _cav_weights(cfg: SimConfig, t: float, nx: int):
    """(left cell j, right weight wr) of the linear split of the CAV between
    the two cell centers straddling x_cav(t); None when the CAV is off road."""
    x = cav_position(cfg, t)
    if not np.isfinite(x):
        return None
    xi = x / cfg.dx - 0.5          # fractional index in cell-center coords
    if xi <= 0.0:
        return 0, 0.0
    if xi >= nx - 1:
        return nx - 2, 1.0
    j = int(np.floor(xi))
    return j, xi - j


def cav_density(cfg: SimConfig, t: float, nx: int) -> np.ndarray:
    """Exogenous a(x, t): one vehicle split linearly over the straddling
    cells, a = weight / dx [veh/m]; zero when the CAV is off the road."""
    a = np.zeros(nx)
    jw = _cav_weights(cfg, t, nx)
    if jw is not None:
        j, wr = jw
        a[j] = (1.0 - wr) / cfg.dx
        a[j + 1] += wr / cfg.dx
    return a


# --------------------------------------------------------------------------
# Exact reaction substep (tex eq. exact-reaction-update)
# --------------------------------------------------------------------------

def reaction_exact(f, s, a, rho, dv, kappa_c: float, kappa_r: float,
                   P: float, dt: float, capture_form: str = "lf"):
    """Exact update of the frozen-coefficient reaction ODE over dt.

    With sigma = kappa_c ell dv, mu = kappa_r (P - rho)_+ dv, gamma = sigma
    + mu and p = f + s (all frozen at substep start),

        f* = (mu/gamma) p + (f - (mu/gamma) p) exp(-gamma dt),  s* = p - f*,

    and f* = f, s* = s where gamma = 0 (tex eq. exact-reaction-update).
    ell = a + s ("lf" baseline: any slow-mode vehicle captures) or ell = a
    ("af": strict tex form, eq. per-capita-rates).  Conserves p exactly and
    keeps f*, s* >= 0 (tex Prop. reaction-positivity).
    """
    f = np.asarray(f, float)
    s = np.asarray(s, float)
    ell = (a + s) if capture_form == "lf" else np.asarray(a, float)
    dv = np.maximum(np.asarray(dv, float), 0.0)
    sigma = kappa_c * ell * dv
    mu = kappa_r * np.maximum(P - np.asarray(rho, float), 0.0) * dv
    gamma = sigma + mu
    p = f + s
    g_safe = np.where(gamma > 0.0, gamma, 1.0)
    f_eq = mu / g_safe * p
    f_new = np.where(gamma > 0.0,
                     f_eq + (f - f_eq) * np.exp(-gamma * dt),
                     f)
    return f_new, p - f_new


# --------------------------------------------------------------------------
# Godunov/CTM transport substep (tex eqs. min-flux, transport-update)
# --------------------------------------------------------------------------

def transport_step(f, s, a, c_f: float, c_s: float, cfg: SimConfig,
                   t: float | None = None):
    """One conservative transport update of f and s.

    Interior fluxes: F^m_{j+1/2} = pi_j^m min{D_{c_m}(rho_j),
    S_{c_m}(rho_{j+1})} with rho including the exogenous a (a reduces supply
    and takes its share pi^a of interface capacity; that share is simply not
    applied since a is prescribed).  Left boundary: pure-f inflow
    F^f = min{q_in, S_{v_f}(rho_1)}, F^s = 0.  Right boundary: free outflow
    F^m = pi^m D_{c_m}(rho_nx).

    Moving-bottleneck capacity cap (Delle Monache--Goatin flux constraint):
    when cfg.q_xi_max is not None, t is given, t in [t_slow, t_fast] and the
    CAV is on the road, the flux past the CAV in its own frame is bounded,

        F^f + F^s - u_cav rho_j <= omega_max,
        omega_max = (q_xi_max - u_cav sigma_xi)_+,
        sigma_xi  = beta w P / (v_f + w),

    with sigma_xi the critical density of the one-lane-blocked flux Q_xi
    (open-lane fraction beta) and u_cav = u_xi.  The bound is enforced at
    the single interface j+1/2 (downstream face of the CAV cell j): the
    violation test uses the CAV-cell traffic density f_j + s_j as the
    upstream trace, and a binding constraint caps the face at the road-frame
    flux of the free wake state the bottleneck emits,

        cap_abs = omega_max + u_cav rho_hat,  rho_hat = omega_max/(v_f - u_cav)

    (the Godunov trace at a constrained face is the downstream wake state;
    capping at omega_max + u_cav rho_j with the cell average instead is
    bistable on a fixed grid -- the cap inflates with the local density and
    free flow never transitions to the congested branch, verified in the
    E-V4b build).  Both class fluxes are scaled by the same factor, so the
    update stays conservative.  With q_xi_max=None (default) the scheme is
    bit-identical to the uncapped one.

    Returns (f_new, s_new, q_admitted, q_out) with boundary fluxes in veh/s.
    """
    lam = cfg.dt / cfg.dx
    rho = a + f + s
    r_safe = np.maximum(rho, 1e-300)   # overflow guard at denormal densities
    pi_f = np.where(rho > 0.0, f / r_safe, 0.0)
    pi_s = np.where(rho > 0.0, s / r_safe, 0.0)

    D_f = demand(rho, c_f, cfg.w, cfg.P)
    S_f = supply(rho, c_f, cfg.w, cfg.P)
    D_s = demand(rho, c_s, cfg.w, cfg.P)
    S_s = supply(rho, c_s, cfg.w, cfg.P)

    nx = f.size
    F_f = np.empty(nx + 1)
    F_s = np.empty(nx + 1)
    F_f[1:-1] = pi_f[:-1] * np.minimum(D_f[:-1], S_f[1:])
    F_s[1:-1] = pi_s[:-1] * np.minimum(D_s[:-1], S_s[1:])
    F_f[0] = min(cfg.q_in, S_f[0])
    F_s[0] = 0.0
    F_f[-1] = pi_f[-1] * D_f[-1]
    F_s[-1] = pi_s[-1] * D_s[-1]

    if (cfg.q_xi_max is not None and t is not None
            and cfg.t_slow <= t <= cfg.t_fast):
        x_cav = cav_position(cfg, t)
        u_cav = cfg.u_xi
        if np.isfinite(x_cav) and u_cav < cfg.v_f:
            j = min(int(x_cav / cfg.dx), nx - 1)   # cell containing x_cav
            sigma_xi = cfg.beta * cfg.w * cfg.P / (cfg.v_f + cfg.w)
            omega_max = max(cfg.q_xi_max - u_cav * sigma_xi, 0.0)
            F_tot = F_f[j + 1] + F_s[j + 1]
            if F_tot - u_cav * (f[j] + s[j]) > omega_max:   # constraint hit
                rho_hat = omega_max / (cfg.v_f - u_cav)     # free wake state
                cap_abs = omega_max + u_cav * rho_hat
                if F_tot > cap_abs:
                    scale = cap_abs / F_tot
                    F_f[j + 1] *= scale
                    F_s[j + 1] *= scale

    f_new = f - lam * np.diff(F_f)
    s_new = s - lam * np.diff(F_s)
    return f_new, s_new, F_f[0], F_f[-1] + F_s[-1]


# --------------------------------------------------------------------------
# Full splitting scheme
# --------------------------------------------------------------------------

def simulate(cfg: SimConfig) -> SimResult:
    """Run the transport-reaction splitting scheme over [0, t_end]."""
    cfl = cfg.dt * max(cfg.v_f, cfg.w) / cfg.dx
    assert cfl <= 1.0 + 1e-12, (
        f"CFL condition violated: dt*max(v_f, w)/dx = {cfl:.3f} > 1 "
        "(tex eq. CFL)")

    nx = int(round(cfg.L_road / cfg.dx))
    x = (np.arange(nx) + 0.5) * cfg.dx
    f = np.zeros(nx) if cfg.f0 is None else np.array(cfg.f0, float)
    s = np.zeros(nx) if cfg.s0 is None else np.array(cfg.s0, float)
    assert f.size == nx and s.size == nx, "f0/s0 must have nx cells"

    n_steps = int(round(cfg.t_end / cfg.dt))
    injected = denied = outflowed = 0.0
    saves: list[tuple] = []

    def _save(step: int) -> None:
        t = step * cfg.dt
        a = cav_density(cfg, t, nx)
        rho = a + f + s
        jw = _cav_weights(cfg, t, nx)
        u_s = u_s_of_t(cfg, t)
        if jw is None:
            xc, om = np.nan, 0.0
        else:
            j, wr = jw
            xc = cav_position(cfg, t)
            f_c = (1.0 - wr) * f[j] + wr * f[j + 1]
            rho_c = (1.0 - wr) * rho[j] + wr * rho[j + 1]
            v_c = float(speed(rho_c, cfg.v_f, cfg.w, cfg.P))
            om = f_c * max(v_c - u_s, 0.0)
            # Under the DM-G cap the CAV-frame passing flux is bounded by
            # omega_max at all times; the cell-average diagnostic straddles
            # the bottleneck discontinuity and must be clipped accordingly.
            if (cfg.q_xi_max is not None
                    and cfg.t_slow <= t <= cfg.t_fast and u_s < cfg.v_f):
                sigma_xi = cfg.beta * cfg.w * cfg.P / (cfg.v_f + cfg.w)
                om = min(om, max(cfg.q_xi_max - u_s * sigma_xi, 0.0))
        saves.append((t, a, f.copy(), s.copy(), xc, om))

    for n in range(n_steps):
        t = n * cfg.dt
        if n % cfg.save_every == 0:
            _save(n)
        u_s = u_s_of_t(cfg, t)          # frozen for the whole step
        a = cav_density(cfg, t, nx)
        f, s, q_adm, q_out = transport_step(f, s, a, cfg.v_f, u_s, cfg, t=t)
        injected += q_adm * cfg.dt
        denied += (cfg.q_in - q_adm) * cfg.dt
        outflowed += q_out * cfg.dt
        # Reaction on the transported state; the prescribed a at t + dt plays
        # the role of the transported a* of the tex scheme.
        a_star = cav_density(cfg, t + cfg.dt, nx)
        rho_star = a_star + f + s
        dv = np.maximum(speed(rho_star, cfg.v_f, cfg.w, cfg.P) - u_s, 0.0)
        f, s = reaction_exact(f, s, a_star, rho_star, dv, cfg.kappa_c,
                              cfg.kappa_r, cfg.P, cfg.dt, cfg.capture_form)
    _save(n_steps)

    t_arr = np.array([sv[0] for sv in saves])
    return SimResult(
        t=t_arr,
        x=x,
        a=np.array([sv[1] for sv in saves]),
        f=np.array([sv[2] for sv in saves]),
        s=np.array([sv[3] for sv in saves]),
        x_cav=np.array([sv[4] for sv in saves]),
        omega=np.array([sv[5] for sv in saves]),
        N_s=np.array([np.sum(sv[3]) * cfg.dx for sv in saves]),
        denied_inflow=denied,
        injected=injected,
        outflowed=outflowed,
        on_road=float(np.sum(f + s) * cfg.dx),
    )
