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

Structural knobs (all default to None = exact legacy behavior, bit-for-bit):

* gamma: continuous capture-agent weight.  When set, the reaction layer uses
  ell = a + gamma s, overriding capture_form (gamma = 1 == "lf", gamma = 0 ==
  "af"); intermediate gamma interpolates how strongly already-caught cars
  recruit further captures.

* w_s, P_s: s-class congested-branch parameters.  When set, the s-class
  TRANSPORT demand/supply (hence its speed law min{c_s, w_s (P_s / rho - 1)},
  still evaluated at the TOTAL rho) uses (w_s or w, P_s or P), while the
  f-class and the reaction-layer Delta v = [V_{v_f}(rho; w, P) - u_s]_+ keep
  the shared road FD.  Requires w_s <= w and P_s <= P (asserted at config
  time); see transport_step for the invariant-domain proof.

* downstream_release (bool, default False = legacy): the DEFINITIONAL
  downstream-release constraint -- a vehicle cannot be caught by a
  bottleneck that is BEHIND it.  When True, at every step while the CAV is
  on the road (x_cav finite), immediately after the reaction substep every
  cell strictly downstream of the CAV cell (j > j_cav, j_cav =
  int(x_cav // dx)) converts its s to f: f += s, s = 0.  It applies in all
  phases (slow and fast), carries no parameters, and conserves p = f + s
  trivially (per-cell f + s is untouched), so all reaction/transport
  invariants are unchanged.  Without it, kappa_r-released s that drifts
  past the CAV keeps exchanging with f downstream (wrong downstream
  density, an intermediate-density wedge travelling at u_s < v_f, and the
  R1 dispersion 'waviness' with growth rate Dv (kappa_c rho - kappa_r
  (P - rho)) > 0).  SUPERSEDED for new results by eta_la (below): it
  references the CAV position x_cav, i.e. scenario information rather than
  traffic state, which a general model's source terms must not do.  Kept
  for the record; independent of eta_la (both may be on).

* eta_la (float [m], default None = legacy, bit-for-bit): nonlocal
  LEADER-LOSS release, the general replacement for downstream_release.
  Being synchronized is a relation to a slow LEADER AHEAD: a stuck vehicle
  whose look-ahead window of length eta_la contains no slow vehicle has
  lost its leader and is released.  Per cell j, with n = max(1,
  round(eta_la / dx)) look-ahead cells j+1..j+n (cells beyond the road end
  count as empty),

      a_ahead = sum a[j+1..j+n],   s_ahead = (1/n) sum s[j+1..j+n],
      c_j     = 1                       if a_ahead > 0  (an A vehicle is
                                         lane-blocking: full coverage)
              = min{1, s_ahead / s_j}   otherwise,
      mu_ll_j = (c_wave / ell_eff) (1 - c_j),   c_wave = w_s (if set) or w,
      ell_eff = dx (n + 1) / 2   (mean look-ahead distance of the window,
                                  -> eta_la / 2 in the continuum),

  an ADDITIONAL per-capita release rate: the reaction substep uses
  mu_total = mu + mu_ll (the exact 2x2 update is unchanged, only the frozen
  coefficient changes).  It is evaluated on the post-transport state and is
  a function of the state arrays (a, s) ONLY -- nothing in it references
  x_cav or the CAV schedule.  No Delta v factor (leader loss is not an
  overtaking event), so it also acts after t_fast (u_s = v_f, Delta v = 0),
  where it turns the released queue into f from the head backward: the
  START-UP WAVE.  Rate scale (leader_loss_rate docstring): for a sharp-
  edged platoon the per-cell decay law telescopes and the release front
  recedes at exactly c_wave, independent of dx and n; the naive prefactor
  c_wave / eta_la would give c_wave (n + 1) / (2 n) (c_wave / 2 in the
  continuum).  Inside a queue (the CAV, or at least as much s, within
  eta_la ahead) mu_ll is exactly 0.  Conservation of p = f + s and the
  reaction positivity are untouched (mu_ll >= 0 is just a larger mu).

* s_impermeable (bool, default False = legacy, bit-for-bit): the
  s-IMPERMEABLE MOVING-BOTTLENECK INTERFACE.  Catch & release DEFINES
  release as becoming free: only free vehicles overtake the bottleneck,
  i.e. the synchronized class has zero flux RELATIVE to the moving
  bottleneck.  Discretely, F^s = 0 at the CAV's downstream face j + 1
  (j = the cell containing x_cav -- the Delle Monache--Goatin cap face)
  whenever the bottleneck is active (t in [t_slow, t_fast], x_cav finite,
  u_cav < v_f): the same face and the same activity condition as the DM-G
  cap, which then constrains F^f alone (F_tot = F^f there).  When the
  CAV's two-cell footprint extends past the cap cell (x_cav in the right
  half of cell j, a_{j+1} > 0) F^s = 0 is imposed at the footprint's front
  face j + 2 as well, so that s created by the reaction in the front
  straddle cell stays with the vehicle (transport_step).  An interface
  flux constraint AT the bottleneck, in the same category as the DM-G
  constraint -- not a state-independent label conversion (contrast
  downstream_release) -- and it removes the first-order-upwind s-front
  diffusion across the CAV cell (the E9 co-moving plume, ~21/12/7/4/2
  veh/km over the 5 cells ahead) at its source.  Independent of q_xi_max
  (works with the cap on or off); every transport invariant survives
  (transport_step).

* ll_mode ("ratio" | "connect", default "ratio" = the E9 form above,
  bit-for-bit) and tau_ll (float [s], default 3.0): the CONNECTIVITY form
  of the leader-loss rate, leader_loss_rate_connect.  A vehicle is
  synchronized only while it is CONNECTED through slow traffic to a
  bottleneck vehicle within the look-ahead horizon.  Coverage is the
  LEAST fixed point of

      cover_j = 1                                        if a_k > 0 for
                                                          some k in j..j+n
              = min{1, dx sum_{k=j+1..j+n} s_k cover_k}   otherwise,

  computed by Kleene iteration from cover = 0 (vectorized window sums;
  until max |delta| < 1e-14, cap 500 sweeps): coverage propagates BACKWARD
  from A vehicles through contiguous s and nowhere else.  A disconnected
  s-platoon (no A reachable) gets cover = 0 everywhere -- cover = 1 would
  also be self-consistent there, which is why the LEAST fixed point is
  the definition; a queue attached to a CAV gets cover = 1 everywhere
  (the CAV, or >= 1 connected slow vehicle, within eta_la ahead).  Rate
  A vehicles anchor only while class A is slow (u_s < v_f); at free speed
  they are ordinary vehicles and do not hold a queue.
  mu_ll = (1 - cover) / tau_ll, tau_ll a physical acceleration / headway
  time; no c_wave, no ell_eff, no Delta v.  Replaces the ratio form's
  failure mode (95-97% of its release INSIDE a queue whose s decreases
  toward the head).  After t_fast the head loses coverage once the CAV is
  > eta_la ahead, the platoon is then unanchored and relabelled free
  within ~tau_ll everywhere at once (the least fixed point drops to 0);
  dynamically neutral when w_s is None (for u_s = v_f the s and f speed
  laws coincide), while with w_s < w the post-departure dispersal follows
  the road FD instead of the s-branch.  DISCRETIZATION NOTE: the A term
  is a PRESENCE indicator over the cell itself and the n cells ahead, not
  the dx-weighted A mass ahead of the cell, because the exogenous CAV is a
  single vehicle linearly split over two cells and the queue head sits IN
  the CAV's own cell (the cap cell, against the impermeable face): with
  the mass form that cell sees only the fraction wr of the CAV 'ahead'
  (nothing when x_cav is in its left half) and is released at up to
  1/tau_ll -- the queue drains at its head (max N_s 0.25 veh instead of
  77 on the A1 hybrid).  The s term is the mass form exactly.  Requires
  eta_la (the horizon); a pure function of (a, s).

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
    gamma: float | None = None    # capture-agent weight: ell = a + gamma s,
    #                               overrides capture_form; None = legacy
    w_s: float | None = None      # s-class backward wave speed [m/s];
    #                               None = shared road w (legacy)
    P_s: float | None = None      # s-class jam density [veh/m];
    #                               None = shared road P (legacy)
    downstream_release: bool = False  # definitional constraint: s strictly
    #                               downstream of the CAV cell converts to f
    #                               after each reaction substep while the CAV
    #                               is on road; False = legacy, bit-for-bit.
    #                               SUPERSEDED by eta_la for new results.
    eta_la: float | None = None   # leader-loss look-ahead horizon [m]: extra
    #                               state-only release rate mu_ll (module
    #                               docstring); None = legacy, bit-for-bit
    s_impermeable: bool = False   # s-impermeable moving-bottleneck interface:
    #                               F^s = 0 at the CAV's downstream face while
    #                               the bottleneck is active (module
    #                               docstring); False = legacy, bit-for-bit
    ll_mode: str = "ratio"        # leader-loss form when eta_la is set:
    #                               "ratio" (E9 form, leader_loss_rate) or
    #                               "connect" (leader_loss_rate_connect)
    tau_ll: float = 3.0           # connect-form release time scale [s]

    def __post_init__(self) -> None:
        assert self.ll_mode in ("ratio", "connect"), (
            f"ll_mode = {self.ll_mode!r} must be 'ratio' or 'connect'")
        assert np.isfinite(self.tau_ll) and self.tau_ll > 0.0, (
            f"tau_ll = {self.tau_ll} must be a positive time scale [s]")
        if self.eta_la is not None:
            assert np.isfinite(self.eta_la) and self.eta_la > 0.0, (
                f"eta_la = {self.eta_la} must be a positive look-ahead "
                "distance [m] (None disables the leader-loss release)")
        if self.w_s is not None:
            assert self.w_s <= self.w, (
                f"w_s = {self.w_s} exceeds the road wave speed w = {self.w}: "
                "the transport invariant-domain bound rho <= P is only "
                "proved for w_s <= w (see transport_step)")
        if self.P_s is not None:
            assert self.P_s <= self.P, (
                f"P_s = {self.P_s} exceeds the road jam density P = "
                f"{self.P}: the transport invariant-domain bound rho <= P "
                "is only proved for P_s <= P (see transport_step)")
        if self.gamma is not None:
            assert self.gamma >= 0.0, (
                f"gamma = {self.gamma} < 0 makes ell = a + gamma s negative, "
                "voiding reaction positivity (tex Prop. reaction-positivity)")


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
    cum_cap: np.ndarray = None   # (n_save,) cumulative gross captures [veh]
    cum_rel: np.ndarray = None   # (n_save,) cumulative gross releases [veh]
    #                              (E13 counters; sum over cells of the exact
    #                              reaction integrals, see reaction_exact)


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
                   P: float, dt: float, capture_form: str = "lf",
                   gamma: float | None = None, mu_extra=None,
                   return_gross: bool = False):
    """Exact update of the frozen-coefficient reaction ODE over dt.

    With sigma = kappa_c ell dv, mu = kappa_r (P - rho)_+ dv [+ mu_extra],
    theta = sigma + mu and p = f + s (all frozen at substep start),

        f* = (mu/theta) p + (f - (mu/theta) p) exp(-theta dt),  s* = p - f*,

    and f* = f, s* = s where theta = 0 (tex eq. exact-reaction-update).
    Capture-agent density: ell = a + gamma s when gamma is not None (the
    continuous knob, OVERRIDING capture_form: gamma = 1 == "lf", gamma = 0
    == "af"); otherwise ell = a + s ("lf" baseline: any slow-mode vehicle
    captures) or ell = a ("af": strict tex form, eq. per-capita-rates).
    Conserves p exactly and keeps f*, s* >= 0 (tex Prop.
    reaction-positivity) for any ell >= 0, so gamma changes no invariant.
    mu_extra (optional, per-cell >= 0, [1/s]): an ADDITIONAL per-capita
    release rate added to the frozen mu (the leader-loss rate of
    leader_loss_rate); it carries no Delta v factor and only enlarges mu,
    so conservation and positivity are unchanged.  None (default) leaves
    the legacy update bit-identical.
    return_gross (E13): additionally return the GROSS capture and release
    integrals over the substep, C = int_0^dt sigma f(t') dt' and
    R = int_0^dt mu s(t') dt' [veh/m per cell], closed form
        C = sigma [f_eq dt + (f - f_eq) g],  R = mu [s_eq dt + (s - s_eq) g],
        g = (1 - exp(-theta dt)) / theta  (g = dt where theta = 0),
    with s_eq = p - f_eq; C - R equals the net change s* - s exactly, so
    the counters are consistent with the update to round-off.  The update
    itself is unchanged (same operations, bit-identical).
    """
    f = np.asarray(f, float)
    s = np.asarray(s, float)
    if gamma is not None:
        ell = np.asarray(a, float) + gamma * s
    else:
        ell = (a + s) if capture_form == "lf" else np.asarray(a, float)
    dv = np.maximum(np.asarray(dv, float), 0.0)
    sigma = kappa_c * ell * dv
    mu = kappa_r * np.maximum(P - np.asarray(rho, float), 0.0) * dv
    if mu_extra is not None:
        mu = mu + np.asarray(mu_extra, float)
    theta = sigma + mu
    p = f + s
    th_safe = np.where(theta > 0.0, theta, 1.0)
    f_eq = mu / th_safe * p
    f_new = np.where(theta > 0.0,
                     f_eq + (f - f_eq) * np.exp(-theta * dt),
                     f)
    if not return_gross:
        return f_new, p - f_new
    g = np.where(theta > 0.0, (1.0 - np.exp(-theta * dt)) / th_safe, dt)
    s_eq = p - f_eq
    cap = sigma * (f_eq * dt + (f - f_eq) * g)
    rel = mu * (s_eq * dt + (s - s_eq) * g)
    return f_new, p - f_new, cap, rel


# --------------------------------------------------------------------------
# Nonlocal leader-loss release (state-only; module docstring, eta_la)
# --------------------------------------------------------------------------

def leader_loss_rate(a, s, dx: float, eta_la: float, c_wave: float):
    """Per-capita leader-loss release rate mu_ll [1/s] per cell.

    A function of the state arrays (a, s) ONLY -- it must never reference
    x_cav or the CAV schedule.  With n = max(1, round(eta_la / dx))
    look-ahead cells j+1..j+n (cells beyond the road end count as empty),

        a_ahead_j = sum_{k=1..n} a_{j+k},   s_ahead_j = (1/n) sum_{k=1..n} s_{j+k},
        c_j       = 1 if a_ahead_j > 0 else min{1, s_ahead_j / max(s_j, 1e-12)},
        mu_ll_j   = (c_wave / ell_eff) (1 - c_j),   ell_eff = dx (n + 1) / 2.

    Coverage c_j: one A vehicle (lane-blocking moving bottleneck) ahead is
    full slow-leader coverage; otherwise coverage is the s in the window
    relative to the cell's own s (a queue is self-covering: at least as
    much s ahead gives c_j = 1 and mu_ll_j = 0.0 EXACTLY, as 1 - 1.0).  The
    s_j = 0 case is moot (nothing to release).

    Rate scale ell_eff (why not eta_la): for a sharp-edged platoon of
    density s0 the per-cell laws telescope,

        d/dt sum_j s_j dx = -(c_wave / ell_eff) dx sum_{j: s_j > s_ahead_j}
                            (s_j - s_ahead_j)
                          = -(c_wave / ell_eff) s0 dx sum_{m=0}^{n-1} (1 - m/n)
                          = -(c_wave / ell_eff) s0 dx (n + 1) / 2 = -c_wave s0,

    so the release front recedes at EXACTLY c_wave for any dx and n
    (verified 1.00 w at dx = 100/50/25/12.5 m, test t18); ell_eff is the
    mean look-ahead distance of the window, (1/n) sum_{k=1..n} k dx, and
    tends to eta_la / 2 in the continuum.  The naive prefactor c_wave /
    eta_la gives a grid-dependent front speed c_wave (n + 1) / (2 n)
    (c_wave / 2 in the continuum: 0.64 w at eta_la = 200 m, dx = 50 m).

    Window sums are accumulated sequentially (k = 1..n, starting from 0.0)
    so that an empty window is exactly 0 and the coverage test is
    reproducible with a plain sequential sum.
    """
    a = np.asarray(a, float)
    s = np.asarray(s, float)
    nx = s.size
    n = max(1, int(round(eta_la / dx)))
    a_pad = np.concatenate([a, np.zeros(n)])      # beyond the road end: empty
    s_pad = np.concatenate([s, np.zeros(n)])
    a_ahead = np.zeros(nx)
    s_sum = np.zeros(nx)
    for k in range(1, n + 1):
        a_ahead += a_pad[k:k + nx]
        s_sum += s_pad[k:k + nx]
    s_ahead = s_sum / n
    cov = np.where(a_ahead > 0.0, 1.0,
                   np.minimum(1.0, s_ahead / np.maximum(s, 1e-12)))
    ell_eff = dx * (n + 1) / 2.0
    return (c_wave / ell_eff) * (1.0 - cov)


def leader_loss_rate_connect(a, s, dx: float, eta_la: float, tau_ll: float):
    """Connectivity leader-loss rate mu_ll [1/s] per cell (ll_mode="connect").

    A pure function of the state arrays (a, s) ONLY -- it must never
    reference x_cav or the CAV schedule.  With n = max(1, round(eta_la /
    dx)) look-ahead cells (cells beyond the road end count as empty), the
    coverage is the (unique) fixed point of

        cover_j = 1                                      if a_k > 0 for some
                                                          k in j, j+1, .., j+n
                = min{1, dx sum_{k=j+1..j+n} s_k cover_k}  otherwise,

    i.e. a cell is covered if a bottleneck vehicle is in it or within the
    look-ahead, else to the extent that CONNECTED slow vehicles (s weighted
    by their own coverage) are within the look-ahead: coverage propagates
    BACKWARD from A through contiguous s and nowhere else.  The map is
    strictly upper-triangular (cover_j depends only on cells ahead, and
    the road end is empty), so its fixed point is UNIQUE; the Kleene
    iteration from cover = 0 reaches it after at most (queue length / n)
    sweeps (capped at 500 -- exceeded only by an anchored queue longer
    than 500 n cells, i.e. > 25 km at eta_la = dx).  A disconnected
    s-platoon (no A reachable) has cover = 0 everywhere (its head cell
    has an empty window, and the recursion propagates that backward); a
    queue attached to a CAV gets cover = 1 EXACTLY everywhere (an A within
    reach, or >= 1 connected slow vehicle within eta_la: the window mass
    is >= 1 and is clipped to 1.0); only a queue tail with < 1 connected
    slow vehicle within eta_la ahead is partially covered.

        mu_ll_j = (1 - cover_j) / tau_ll,

    tau_ll a physical acceleration / headway time.  No Delta v factor, no
    wave-speed prefactor.

    Discrete A term (module docstring, ll_mode): a PRESENCE indicator over
    the cell itself and the n cells ahead rather than the dx-weighted A
    mass ahead, because the exogenous CAV is one vehicle linearly split
    over two cells and the queue head is held in the CAV's own cell by the
    impermeable face; the mass form leaves that cell only fractionally
    covered (wr, or 0) and drains the queue at its head.  Any a > 0 counts:
    a is the density of one lane-blocking vehicle, and the indicator is
    exact (no (1 - wr)/dx + wr/dx roundoff at the head).

    Window sums are the sequential sum of the n shifted arrays (k = 1..n,
    starting from 0.0), as in leader_loss_rate: exact whenever the window
    holds a single nonzero entry and reproducible with a plain sequential
    sum, which cumulative-sum differences (S + x) - S are not.
    """
    a = np.asarray(a, float)
    s = np.asarray(s, float)
    nx = s.size
    n = max(1, int(round(eta_la / dx)))
    zeros_n = np.zeros(n)
    a_pad = np.concatenate([a, zeros_n])          # beyond the road end: empty
    anchored = a > 0.0                            # a bottleneck vehicle here
    for k in range(1, n + 1):                     # ... or within eta_la ahead
        anchored |= a_pad[k:k + nx] > 0.0
    cover = np.zeros(nx)
    for _ in range(500):
        src_pad = np.concatenate([s * cover, zeros_n])
        win = np.zeros(nx)
        for k in range(1, n + 1):
            win += src_pad[k:k + nx]
        new = np.where(anchored, 1.0, np.minimum(1.0, dx * win))
        delta = float(np.max(np.abs(new - cover)))
        cover = new
        if delta < 1e-14:
            break
    return (1.0 - cover) / tau_ll


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

    s-class congested branch (cfg.w_s / cfg.P_s): when either is set, the
    s-class demand/supply use (w_s or w, P_s or P) -- equivalently the
    s-class speed law becomes min{c_s, w_s (P_s / rho - 1)}, still evaluated
    at the TOTAL rho -- while the f-class keeps the shared road (w, P).
    Because rho can exceed P_s (only P bounds it), the raw s-supply
    w_s (P_s - rho) can go negative there; it is clipped at 0 on this branch
    so all class fluxes stay nonnegative.  Invariant-domain proof for
    w_s <= w, P_s <= P (asserted in SimConfig): (i) rho <= P: each interface
    flux obeys F^m_{j-1/2} <= pi^m_{j-1} S_m(rho_j), and for rho_j <= P both
    class supplies are dominated by the road supply -- S_f <= w (P - rho_j)
    directly, and S_s <= max{0, w_s (P_s - rho_j)} <= w (P - rho_j) since
    for rho_j <= P_s both factors are dominated (w_s <= w, P_s - rho_j <=
    P - rho_j) and for rho_j > P_s the clipped value is 0 -- so the total
    inflow is <= (pi^f + pi^s) w (P - rho_j) <= w (P - rho_j); with
    nonnegative outflow, rho_j + lam w (P - rho_j) <= P under lam w <= 1
    (eq. CFL, unchanged since w_s <= w).  (ii) f, s >= 0: all fluxes are
    nonnegative (supply clipped at 0), and each class outflow obeys
    F^m_{j+1/2} <= pi^m_j D_m(rho_j) <= pi^m_j c_m rho_j = c_m m_j, so
    m_new_j >= m_j (1 - lam c_m) >= 0 under lam c_m <= 1 (c_s <= v_f,
    unchanged).  With w_s=None and P_s=None (default) the legacy code path
    runs bit-identically.

    s-impermeable moving-bottleneck interface (cfg.s_impermeable): catch &
    release DEFINES release as becoming free, so only free vehicles
    overtake the bottleneck -- the synchronized class has zero flux
    relative to it.  Under the same activity condition as the cap (t in
    [t_slow, t_fast], x_cav finite, u_cav < v_f), F^s = 0 is imposed at
    the cap face j + 1 BEFORE the cap logic, so the cap then constrains
    F^f alone (F_tot = F^f at that face).  The CAV's discrete footprint
    supp(a) = {j', j'+1} (linear split, cav_density) extends past the cap
    cell when x_cav is in the right half of cell j (a_{j+1} > 0); the
    reaction (a_star) then creates s in the front straddle cell j + 1, so
    F^s = 0 is imposed at the footprint's front face j + 2 as well: s
    never crosses the bottleneck vehicle's support, and the front-straddle
    s becomes the queue head as the CAV enters that cell (without this
    face it leaks downstream at ~0.1 veh/km through the first-order upwind
    flux).  An interface constraint of the same category as the DM-G cap
    and independent of it (either alone, or both).  Invariants: zeroing an
    interface flux keeps every flux nonnegative and every inflow bounded
    by the receiving cell's supply, so the rho <= P and f, s >= 0
    arguments above hold unchanged (a zero outflow only strengthens m_new
    >= m (1 - lam c_m)), and the update stays conservative (interface
    fluxes telescope).  With s_impermeable=False (default) the code path
    is bit-identical.

    Returns (f_new, s_new, q_admitted, q_out) with boundary fluxes in veh/s.
    """
    lam = cfg.dt / cfg.dx
    rho = a + f + s
    r_safe = np.maximum(rho, 1e-300)   # overflow guard at denormal densities
    pi_f = np.where(rho > 0.0, f / r_safe, 0.0)
    pi_s = np.where(rho > 0.0, s / r_safe, 0.0)

    D_f = demand(rho, c_f, cfg.w, cfg.P)
    S_f = supply(rho, c_f, cfg.w, cfg.P)
    if cfg.w_s is None and cfg.P_s is None:      # legacy path, bit-identical
        # NOTE (E10): the branch is deliberately NOT gated by u_s < v_f: a
        # platoon released when the bottleneck deactivates keeps its slower
        # congested branch until it disperses (start-up-wave physics); gating
        # it off worsened the post-release W1 for every model (E10_results.md).
        D_s = demand(rho, c_s, cfg.w, cfg.P)
        S_s = supply(rho, c_s, cfg.w, cfg.P)
    else:
        w_s = cfg.w if cfg.w_s is None else cfg.w_s
        P_s = cfg.P if cfg.P_s is None else cfg.P_s
        D_s = demand(rho, c_s, w_s, P_s)
        # clip: rho may exceed P_s (only P bounds it), where the raw
        # s-supply goes negative; see the invariant-domain proof above
        S_s = np.maximum(supply(rho, c_s, w_s, P_s), 0.0)

    nx = f.size
    F_f = np.empty(nx + 1)
    F_s = np.empty(nx + 1)
    F_f[1:-1] = pi_f[:-1] * np.minimum(D_f[:-1], S_f[1:])
    F_s[1:-1] = pi_s[:-1] * np.minimum(D_s[:-1], S_s[1:])
    F_f[0] = min(cfg.q_in, S_f[0])
    F_s[0] = 0.0
    F_f[-1] = pi_f[-1] * D_f[-1]
    F_s[-1] = pi_s[-1] * D_s[-1]

    # Moving-bottleneck interface constraints at the CAV's downstream face
    # j + 1 (j = the cell containing x_cav), active while the bottleneck is:
    # t given and in [t_slow, t_fast], x_cav finite, u_cav < v_f.  Applied
    # in order: (i) s-impermeability (cfg.s_impermeable), (ii) the DM-G
    # capacity cap (cfg.q_xi_max) on what remains.  Either alone reproduces
    # its legacy path bit for bit; both off skips the block entirely.
    if ((cfg.q_xi_max is not None or cfg.s_impermeable) and t is not None
            and cfg.t_slow <= t <= cfg.t_fast):
        x_cav = cav_position(cfg, t)
        u_cav = cfg.u_xi
        if np.isfinite(x_cav) and u_cav < cfg.v_f:
            j = min(int(x_cav / cfg.dx), nx - 1)   # cell containing x_cav
            if cfg.s_impermeable:
                # (i) only free vehicles overtake the bottleneck: F^s = 0
                # at the cap face and, when the CAV's footprint supp(a)
                # extends past the cap cell, at the footprint's front face
                k_front = j                      # contiguous footprint of
                while k_front + 1 < nx and a[k_front + 1] > 0.0:  # THIS CAV
                    k_front += 1
                F_s[j + 1:k_front + 2] = 0.0
            if cfg.q_xi_max is not None:
                # (ii) DM-G cap; with (i) on, F_tot = F^f at this face
                sigma_xi = cfg.beta * cfg.w * cfg.P / (cfg.v_f + cfg.w)
                omega_max = max(cfg.q_xi_max - u_cav * sigma_xi, 0.0)
                F_tot = F_f[j + 1] + F_s[j + 1]
                # CAV-frame relative flux of what actually crosses the face:
                # with the s-impermeable interface only f passes, so the s
                # term must not enter the test (else a spurious -u_cav s_j
                # bias disengages the cap whenever the cap cell holds s).
                rho_pass = f[j] if cfg.s_impermeable else (f[j] + s[j])
                if F_tot - u_cav * rho_pass > omega_max:  # constraint hit
                    rho_hat = omega_max / (cfg.v_f - u_cav)    # free wake state
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
    cum_cap = cum_rel = 0.0
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
        saves.append((t, a, f.copy(), s.copy(), xc, om, cum_cap, cum_rel))

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
        # Nonlocal leader-loss release (eta_la): an additional frozen
        # per-capita release rate from the post-transport state (a_star,
        # s) only -- no Delta v factor, no x_cav.  None = legacy update.
        mu_ll = None
        if cfg.eta_la is not None:
            if cfg.ll_mode == "connect":
                # A vehicles anchor coverage only while class A is SLOW
                # (u_s < v_f): once the bottleneck vehicle drives at free
                # speed it is no longer a slow leader, so the queue behind
                # it is leaderless and is released.  Same class-speed input
                # that already enters Delta v; no position information.
                a_anchor = a_star if u_s < cfg.v_f else np.zeros_like(a_star)
                mu_ll = leader_loss_rate_connect(a_anchor, s, cfg.dx,
                                                 cfg.eta_la, cfg.tau_ll)
            else:                                   # "ratio": E9 form
                c_wave = cfg.w if cfg.w_s is None else cfg.w_s
                mu_ll = leader_loss_rate(a_star, s, cfg.dx, cfg.eta_la,
                                         c_wave)
        f, s, cap_step, rel_step = reaction_exact(
            f, s, a_star, rho_star, dv, cfg.kappa_c, cfg.kappa_r, cfg.P,
            cfg.dt, cfg.capture_form, cfg.gamma, mu_extra=mu_ll,
            return_gross=True)
        cum_cap += float(np.sum(cap_step)) * cfg.dx
        cum_rel += float(np.sum(rel_step)) * cfg.dx
        # Downstream-release constraint (definitional, zero parameters): a
        # vehicle cannot be caught by a bottleneck that is behind it, so s
        # strictly downstream of the CAV cell converts to f immediately.
        # Evaluated at t + dt, the time of the post-reaction state (and of
        # a_star), in all phases while the CAV is on the road.  Per-cell
        # p = f + s is untouched, so conservation is trivially exact.
        # Superseded for new results by eta_la (state-only); kept for the
        # record and independent of it (both may be on).
        if cfg.downstream_release:
            x_c = cav_position(cfg, t + cfg.dt)
            if np.isfinite(x_c):
                j_cav = int(x_c // cfg.dx)
                f[j_cav + 1:] += s[j_cav + 1:]
                s[j_cav + 1:] = 0.0
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
        cum_cap=np.array([sv[6] for sv in saves]),
        cum_rel=np.array([sv[7] for sv in saves]),
    )
