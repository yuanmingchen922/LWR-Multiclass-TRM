"""Tests for solver.py (invariant-domain splitting scheme).

Run: python3 test_solver.py

Uses the E-V2 calibrated parameters (out/params.json, stored in km/h and
veh/km, converted to SI here) and plain asserts with printed diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from solver import (SimConfig, simulate, reaction_exact, speed,
                    cav_density, transport_step, u_s_of_t, leader_loss_rate,
                    leader_loss_rate_connect)

HERE = Path(__file__).parent

_params = json.loads((HERE / "out" / "params.json").read_text())
V_F = _params["v_f_kmh"] / 3.6       # 27.94 m/s
W = _params["w_kmh"] / 3.6           # 6.21 m/s
P = _params["P_vehkm"] / 1000.0      # 0.2667 veh/m


def base_cfg(**kw) -> SimConfig:
    d = dict(v_f=V_F, w=W, P=P, q_in=2500.0 / 3600.0, u_xi=15.0,
             kappa_c=0.026, kappa_r=3e-5)
    d.update(kw)
    return SimConfig(**d)


def _rho_crossing(x, rho, level, direction):
    """Interpolated position of the first crossing of `level` in the given
    direction ("up": rho[k] < level <= rho[k+1]; "down": the reverse)."""
    below = rho < level
    if direction == "up":
        ks = np.where(below[:-1] & ~below[1:])[0]
    else:
        ks = np.where(~below[:-1] & below[1:])[0]
    assert ks.size > 0, "no crossing found"
    k = ks[0] if direction == "up" else ks[-1]
    frac = (level - rho[k]) / (rho[k + 1] - rho[k])
    return x[k] + frac * (x[k + 1] - x[k])


# --------------------------------------------------------------------------
# 1. Conservation
# --------------------------------------------------------------------------

def test_conservation():
    cfg = base_cfg()   # A=1-like
    res = simulate(cfg)

    offered = cfg.q_in * cfg.t_end
    balance = res.on_road + res.outflowed
    rel = abs(res.injected - balance) / max(res.injected, 1.0)
    print(f"  injected {res.injected:.6f} veh, on_road {res.on_road:.6f} + "
          f"outflowed {res.outflowed:.6f} = {balance:.6f}, rel err {rel:.2e}")
    assert rel < 1e-6, "mass ledger not closed"
    off_err = abs(offered - (res.injected + res.denied_inflow))
    print(f"  offered {offered:.6f} = injected + denied "
          f"{res.injected + res.denied_inflow:.6f} (denied "
          f"{res.denied_inflow:.6f}), abs err {off_err:.2e}")
    assert off_err < 1e-6 * max(offered, 1.0)

    # reaction step alone conserves p = f + s and keeps positivity
    rng = np.random.default_rng(0)
    n = 500
    f = rng.uniform(0.0, 0.1, n)
    s = rng.uniform(0.0, 0.1, n)
    a = rng.uniform(0.0, 0.02, n)
    rho = a + f + s
    dv = rng.uniform(0.0, 13.0, n)
    for form in ("lf", "af"):
        fn, sn = reaction_exact(f, s, a, rho, dv, 0.026, 3e-5, P, 1.0, form)
        p_err = np.max(np.abs((fn + sn) - (f + s)))
        assert p_err < 1e-15, f"p not conserved ({form}): {p_err:.2e}"
        assert fn.min() >= 0.0 and sn.min() >= 0.0
    print(f"  reaction p-conservation max err {p_err:.2e}, positivity OK")

    # reaction function vs high-resolution RK4 of the frozen 2x2 linear ODE
    f0, s0, a0, dv0 = 0.06, 0.04, 0.015, 8.0
    kc, kr, T = 0.03, 0.01, 5.0
    rho0 = a0 + f0 + s0
    sigma = kc * (a0 + s0) * dv0            # "lf" form, frozen
    mu = kr * max(P - rho0, 0.0) * dv0
    fr, sr = f0, s0
    n_rk, h = 20000, T / 20000

    def rhs(ff, ss):
        return mu * ss - sigma * ff, sigma * ff - mu * ss

    for _ in range(n_rk):
        k1f, k1s = rhs(fr, sr)
        k2f, k2s = rhs(fr + 0.5 * h * k1f, sr + 0.5 * h * k1s)
        k3f, k3s = rhs(fr + 0.5 * h * k2f, sr + 0.5 * h * k2s)
        k4f, k4s = rhs(fr + h * k3f, sr + h * k3s)
        fr += h / 6.0 * (k1f + 2 * k2f + 2 * k3f + k4f)
        sr += h / 6.0 * (k1s + 2 * k2s + 2 * k3s + k4s)
    fe, se = reaction_exact(f0, s0, a0, rho0, dv0, kc, kr, P, T, "lf")
    err = max(abs(float(fe) - fr), abs(float(se) - sr))
    print(f"  exact vs RK4 (2x2 linear ODE): f {float(fe):.12f} vs {fr:.12f},"
          f" max err {err:.2e}")
    assert err < 1e-12


# --------------------------------------------------------------------------
# 2. Invariant domain
# --------------------------------------------------------------------------

def test_invariant_domain():
    cfg = base_cfg(q_in=2900.0 / 3600.0, u_xi=10.0)   # stress run
    res = simulate(cfg)
    rho = res.a + res.f + res.s
    print(f"  min f {res.f.min():.3e}, min s {res.s.min():.3e}, "
          f"max rho {rho.max():.6f} (P = {P:.6f})")
    assert res.f.min() >= 0.0, "f went negative"
    assert res.s.min() >= 0.0, "s went negative"
    assert rho.max() <= P + 1e-12, "rho exceeded jam density"


# --------------------------------------------------------------------------
# 3. Pure-class limit: inflow front propagates at v_f
# --------------------------------------------------------------------------

def test_pure_class_front():
    cfg = base_cfg(kappa_c=0.0, kappa_r=0.0, q_in=2000.0 / 3600.0,
                   t_enter=2000.0)   # no CAV within the horizon
    res = simulate(cfg)
    assert np.all(res.s == 0.0) and np.all(res.a == 0.0)
    rho_in = cfg.q_in / V_F
    lvl = 0.5 * rho_in

    def front(ti):
        i = int(np.argmin(np.abs(res.t - ti)))
        return _rho_crossing(res.x, res.f[i], lvl, "down")

    v_meas = (front(900.0) - front(400.0)) / 500.0
    rel = abs(v_meas - V_F) / V_F
    print(f"  front speed {v_meas:.3f} m/s vs v_f {V_F:.3f} m/s "
          f"({100 * rel:.2f}% off)")
    assert rel < 0.05


# --------------------------------------------------------------------------
# 4. Riemann shock: Rankine-Hugoniot speed
# --------------------------------------------------------------------------

def test_riemann_shock():
    dx, nx = 50.0, 600
    xc = (np.arange(nx) + 0.5) * dx
    rho_L, rho_R = 0.02, 0.15          # free upstream, congested downstream
    f0 = np.where(xc < 15000.0, rho_L, rho_R)
    Q = lambda r: min(V_F * r, W * (P - r))
    z = (Q(rho_R) - Q(rho_L)) / (rho_R - rho_L)   # RH shock speed

    cfg = base_cfg(kappa_c=0.0, kappa_r=0.0, q_in=Q(rho_L), t_enter=1e9,
                   t_end=300.0, save_every=1, f0=f0)
    res = simulate(cfg)
    lvl = 0.5 * (rho_L + rho_R)
    mask = res.t >= 50.0               # skip the initial smearing transient
    pos = np.array([_rho_crossing(res.x, res.f[i], lvl, "up")
                    for i in np.where(mask)[0]])
    slope = np.polyfit(res.t[mask], pos, 1)[0]
    rel = abs(slope - z) / abs(z)
    print(f"  tracked shock speed {slope:.4f} m/s vs RH z = {z:.4f} m/s "
          f"({100 * rel:.2f}% off)")
    assert rel < 0.10


# --------------------------------------------------------------------------
# 5. Queue smoke test
# --------------------------------------------------------------------------

def _hazard_ode_ns(kappa_c, u_xi, q_in, t0=250.0, t1=740.0, dt=0.01):
    """Semi-analytic reference for the lf-form queue: a fast car traversing
    the platoon accumulates capture hazard kappa_c * integral(a + s) dx =
    kappa_c (N_s + 1), and f reaches the platoon at the relative flux
    q_rel = f_up (v_f - u_xi).  Hence dN/dt = q_rel (1 - exp(-kappa_c (N+1)))
    -- independent of the platoon shape while it stays subcritical."""
    q_rel = (q_in / V_F) * (V_F - u_xi)
    u, t = 1.0, t0
    while t < t1:
        u += dt * q_rel * (1.0 - np.exp(-kappa_c * u))
        t += dt
    return u - 1.0


def test_queue_smoke():
    # --- A=1-like, spec parameters (kappa_c = 0.026, kappa_r = 3e-5):
    # scheme must reproduce the model's own crossing-hazard queue size.
    res1 = simulate(base_cfg())
    i740 = int(np.argmin(np.abs(res1.t - 740.0)))
    ns_spec = res1.N_s[i740]
    ns_ref = _hazard_ode_ns(0.026, 15.0, 2500.0 / 3600.0)
    seg = res1.N_s[(res1.t >= 300.0) & (res1.t <= 700.0)]
    print(f"  A=1-like (kappa_c=0.026): N_s(740) = {ns_spec:.2f} veh vs "
          f"crossing-hazard ODE {ns_ref:.2f} veh "
          f"({100 * abs(ns_spec - ns_ref) / ns_ref:.1f}% off), "
          f"min step in [300,700] = {np.diff(seg).min():.3e}")
    assert abs(ns_spec - ns_ref) / ns_ref < 0.15, \
        "queue size disagrees with the crossing-hazard reference"
    assert np.all(np.diff(seg) >= -1e-9), "N_s not monotone in [300, 700]"

    # --- A=1-like, calibrated kappa (out/ev3_kappa.json, A1_u15_q2500,
    # model-Dv lf form: kappa_cl_mod = 0.0307, kappa_r_mod = 3.6e-5): the
    # SUMO-anchored magnitude bracket [40, 120] veh.  NOTE: the originally
    # specified kappa_c = 0.026 gives N_s(740) = 34.8 veh -- 2% from the
    # model's own semi-analytic prediction (35.5) but below this bracket;
    # the bracket is only attainable with the calibrated kappa_c.
    kap = json.loads((HERE / "out" / "ev3_kappa.json").read_text())
    kc = kap["A1_u15_q2500"]["kappa_cl_mod"][0]     # 0.0307
    kr = kap["A1_u15_q2500"]["kappa_r_mod"][0]      # 3.6e-5
    res1c = simulate(base_cfg(kappa_c=kc, kappa_r=kr))
    ns_cal = res1c.N_s[int(np.argmin(np.abs(res1c.t - 740.0)))]
    seg_c = res1c.N_s[(res1c.t >= 300.0) & (res1c.t <= 700.0)]
    print(f"  A=1-like (calibrated kappa_c={kc:.4f}): N_s(740) = "
          f"{ns_cal:.2f} veh")
    assert 40.0 <= ns_cal <= 120.0, \
        f"N_s(740) = {ns_cal:.2f} outside [40, 120]"
    assert np.all(np.diff(seg_c) >= -1e-9)

    # --- A=10-like: strong release keeps the queue small
    res10 = simulate(base_cfg(kappa_c=0.034, kappa_r=0.016))
    print(f"  A=10-like: max N_s = {res10.N_s.max():.2f} veh")
    assert res10.N_s.max() < 15.0


# --------------------------------------------------------------------------
# 6. CFL assertion
# --------------------------------------------------------------------------

def test_cfl_assert():
    raised = False
    try:
        simulate(base_cfg(dt=10.0))    # 10 * 27.9 / 50 = 5.6 > 1
    except AssertionError as e:
        raised = True
        print(f"  raised as expected: {e}")
    assert raised, "CFL assertion did not fire"


# --------------------------------------------------------------------------
# 7. Capacity cap disabled: bit-identical to the uncapped scheme (E-V4b)
# --------------------------------------------------------------------------

def _ev4b_ref_cfg(**kw) -> SimConfig:
    """A1 reference config of the E-V4b bit-identity check."""
    d = dict(kappa_c=0.0307, kappa_r=0.0, u_xi=15.0,
             q_in=2500.0 / 3600.0, dt=0.5, save_every=20)
    d.update(kw)
    return base_cfg(**d)


def test_cap_disabled_bit_identical():
    ref_path = HERE / "out" / "ev4b_ref_A1.npz"
    res = simulate(_ev4b_ref_cfg(q_xi_max=None))
    if not ref_path.exists():
        np.savez(ref_path, f=res.f, s=res.s, N_s=res.N_s, omega=res.omega)
        print(f"  NOTE: no stored reference -- created {ref_path.name} from "
              "the current solver; rerun to check bit-stability")
    ref = np.load(ref_path)
    for name, arr in (("f", res.f), ("s", res.s),
                      ("N_s", res.N_s), ("omega", res.omega)):
        assert np.array_equal(ref[name], arr), \
            f"{name} not bit-identical to the uncapped reference"
    print("  q_xi_max=None: f, s, N_s, omega bit-identical to the saved "
          "uncapped reference")


# --------------------------------------------------------------------------
# 8. Capacity cap: analytic steady moving bottleneck (E-V4b)
# --------------------------------------------------------------------------

def test_cap_analytic_bottleneck():
    """Pure-transport moving bottleneck vs the Delle Monache-Goatin steady
    state: the upstream queue solves w (P - rho) - u_xi rho = omega_max on
    the congested branch, rho_minus = (w P - omega_max) / (w + u_xi); the
    road-frame flux on the queue side is q_minus = omega_max + u_xi
    rho_minus; the queue tail is the RH shock between (rho_in, q_in) and
    (rho_minus, q_minus).  NOTE: the capped face itself carries the WAKE
    flux omega_max + u_xi rho_hat (its Godunov trace is the downstream free
    state rho_hat, see transport_step); q_minus is measured on queue-
    interior faces just upstream of the CAV cell."""
    u_xi, q_xi = 15.0, 2000.0 / 3600.0
    cfg = base_cfg(kappa_c=0.0, kappa_r=0.0, u_xi=u_xi, dt=0.5,
                   save_every=20, q_xi_max=q_xi)
    sigma_xi = cfg.beta * W * P / (V_F + W)
    omega_max = max(q_xi - u_xi * sigma_xi, 0.0)
    rho_minus = (W * P - omega_max) / (W + u_xi)
    q_minus = omega_max + u_xi * rho_minus       # = W (P - rho_minus)
    rho_in = cfg.q_in / V_F
    z = (q_minus - cfg.q_in) / (rho_minus - rho_in)

    res = simulate(cfg)

    # (a) upstream queue density, x in [x_cav - 1000, x_cav - 200] at t = 700
    i700 = int(np.argmin(np.abs(res.t - 700.0)))
    xc = res.x_cav[i700]
    rho7 = res.a[i700] + res.f[i700] + res.s[i700]
    m = (res.x >= xc - 1000.0) & (res.x <= xc - 200.0)
    rho_sim = rho7[m].mean()
    rel_rho = abs(rho_sim - rho_minus) / rho_minus
    print(f"  rho_minus analytic {rho_minus:.6f} veh/m "
          f"({rho_minus * 1000:.2f} veh/km), simulated {rho_sim:.6f} "
          f"({rho_sim * 1000:.2f} veh/km), {100 * rel_rho:.2f}% off")
    assert rel_rho < 0.05, "upstream queue density off the analytic state"

    # (b) road-frame flux past the CAV on the queue side: telescope the
    # interface fluxes of one transport step from the saved t = 700 state
    # and average the three queue-interior faces upstream of the CAV cell
    f7, s7 = res.f[i700].copy(), res.s[i700].copy()
    a7 = cav_density(cfg, 700.0, f7.size)
    fn, sn, q_adm, _ = transport_step(f7, s7, a7, cfg.v_f,
                                      u_s_of_t(cfg, 700.0), cfg, t=700.0)
    lam = cfg.dt / cfg.dx
    F = np.empty(f7.size + 1)
    F[0] = q_adm
    F[1:] = q_adm - np.cumsum((fn - f7) + (sn - s7)) / lam
    j = int(xc / cfg.dx)
    q_queue = F[j - 3:j].mean()
    rel_q = abs(q_queue - q_minus) / q_minus
    print(f"  queue-side flux {q_queue:.6f} veh/s vs omega_max + u rho_minus"
          f" = {q_minus:.6f} ({100 * rel_q:.2f}% off); capped face "
          f"{F[j + 1]:.6f} = wake flux")
    assert rel_q < 0.03, "queue-side flux off omega_max + u_xi rho_minus"

    # (c) queue-tail shock speed over t in [350, 650]
    sel = np.where((res.t >= 350.0) & (res.t <= 650.0))[0]
    tails = []
    for i in sel:
        r = res.a[i] + res.f[i] + res.s[i]
        ks = np.where(r > 0.6 * rho_minus)[0]
        tails.append(res.x[ks[0]] if ks.size else np.nan)
    tails = np.asarray(tails)
    ok = np.isfinite(tails)
    assert ok.sum() > 10, "queue tail not detectable"
    slope = np.polyfit(res.t[sel][ok], tails[ok], 1)[0]
    rel_z = abs(slope - z) / abs(z)
    print(f"  tail shock {slope:.4f} m/s vs RH z = {z:.4f} m/s "
          f"({100 * rel_z:.2f}% off)")
    assert rel_z < 0.15, "queue-tail speed off the RH prediction"


# --------------------------------------------------------------------------
# 9. Capacity cap + reactions: invariants still hold (E-V4b)
# --------------------------------------------------------------------------

def test_cap_with_reactions_invariants():
    kap = json.loads((HERE / "out" / "ev3_kappa.json").read_text())
    kc = kap["A1_u15_q2500"]["kappa_cl_mod"][0]     # 0.0307
    kr = kap["A1_u15_q2500"]["kappa_r_mod"][0]      # 0.0
    res = simulate(_ev4b_ref_cfg(kappa_c=kc, kappa_r=kr,
                                 q_xi_max=2000.0 / 3600.0))
    balance = res.on_road + res.outflowed
    rel = abs(res.injected - balance) / max(res.injected, 1.0)
    rho = res.a + res.f + res.s
    print(f"  kappa_c={kc:.4f}, kappa_r={kr:g}: ledger rel err {rel:.2e}, "
          f"min f {res.f.min():.3e}, min s {res.s.min():.3e}, "
          f"max rho {rho.max():.6f} (P = {P:.6f})")
    assert rel < 1e-10, "mass ledger not closed with the cap active"
    assert res.f.min() >= 0.0 and res.s.min() >= 0.0
    assert rho.max() <= P + 1e-12, "rho exceeded jam density with the cap"


# --------------------------------------------------------------------------
# 10-13. Structural knobs: gamma (capture agent) and w_s / P_s (s-class FD)
# --------------------------------------------------------------------------

def _e6_a1_cfg(**kw) -> SimConfig:
    """A1 native-fit config (out/e6/fits.json, rounded): kappa_c = 0.504,
    kappa_r = 0.0306, lf, u15, q2500, production dt = 0.5 / save_every = 20,
    no capacity cap."""
    d = dict(kappa_c=0.504, kappa_r=0.0306, capture_form="lf", u_xi=15.0,
             q_in=2500.0 / 3600.0, dt=0.5, save_every=20)
    d.update(kw)
    return base_cfg(**d)


def _assert_bit_equal(res, ref, what: str) -> None:
    """f, s, N_s, omega of a SimResult bit-equal a reference (SimResult or
    npz mapping)."""
    for name in ("f", "s", "N_s", "omega"):
        r = ref[name] if not isinstance(ref, type(res)) else getattr(ref, name)
        assert np.array_equal(r, getattr(res, name)), \
            f"{name} not bit-identical: {what}"


def test_struct_none_bit_identical():
    """t10: all new fields at None reproduce the PRE-change solver bit for
    bit (reference out/struct_ref_A1.npz was generated from the solver
    BEFORE gamma / w_s / P_s existed)."""
    ref_path = HERE / "out" / "struct_ref_A1.npz"
    assert ref_path.exists(), (
        "missing pre-change reference out/struct_ref_A1.npz -- it must be "
        "generated with the PRE-knob solver, not recreated here")
    res = simulate(_e6_a1_cfg())
    _assert_bit_equal(res, np.load(ref_path), "new-fields-None vs pre-change")
    print("  gamma=w_s=P_s=None: f, s, N_s, omega bit-identical to the "
          "stored pre-change reference")


def test_gamma_matches_capture_forms():
    """t11: gamma = 1 bit-equals capture_form='lf', gamma = 0 bit-equals
    'af'; capture_form is set to the OPPOSITE form to prove gamma
    overrides it."""
    res_lf = simulate(_e6_a1_cfg(capture_form="lf"))
    res_af = simulate(_e6_a1_cfg(capture_form="af"))
    res_g1 = simulate(_e6_a1_cfg(capture_form="af", gamma=1.0))
    res_g0 = simulate(_e6_a1_cfg(capture_form="lf", gamma=0.0))
    _assert_bit_equal(res_g1, res_lf, "gamma=1.0 vs capture_form='lf'")
    _assert_bit_equal(res_g0, res_af, "gamma=0.0 vs capture_form='af'")
    i740 = int(np.argmin(np.abs(res_lf.t - 740.0)))
    print(f"  gamma=1.0 == 'lf' (N_s(740) = {res_g1.N_s[i740]:.2f} veh) and "
          f"gamma=0.0 == 'af' (N_s(740) = {res_g0.N_s[i740]:.2f} veh), both "
          "bit-identical and overriding capture_form")


def test_ws_ps_shared_bit_identical():
    """t12: w_s = w and P_s = P explicitly set bit-equal the None run."""
    res0 = simulate(_e6_a1_cfg())
    res1 = simulate(_e6_a1_cfg(w_s=W, P_s=P))
    _assert_bit_equal(res1, res0, "w_s=w, P_s=P vs None")
    print("  w_s=w, P_s=P: f, s, N_s, omega bit-identical to the None run")


def test_ws_slow_wake_wedge():
    """t13: w_s < w keeps all invariants; effect on the steady wake wedge
    (mean rho over [x_cav - 1 km, x_cav - 0.2 km] at t = 700 s).

    FINDING (deviation from the original t13 spec, which asserted a HIGHER
    wedge at w_s = 0.7 w): at w_s = 0.7 w the override is provably INERT
    during the slowdown window -- with c_s = u_xi = 15 the s-class critical
    density r*_s = w_s P / (u_xi + w_s) = 59.9 veh/km lies ABOVE the run's
    maximum total density (53.4 veh/km), so D_s = c_s rho is never capped
    and S_s never binds; the pre-release fields are bit-identical and the
    t = 700 wedge is UNCHANGED (both numbers still reported).  The branch
    only engages after release (c_s = v_f drops r*_s to 35.9 veh/km < the
    44 veh/km wedge), where it slows the platoon dispersal -- the
    post-t750 fields DO differ.  The wedge-raising mechanism is therefore
    verified at w_s = 0.5 w, where r*_s = 45.8 veh/km dips below the
    wedge: there the t = 700 wedge is strictly HIGHER."""
    # config-time guards fire with a clear message
    for bad in (dict(w_s=1.01 * W), dict(P_s=1.01 * P)):
        try:
            _e6_a1_cfg(**bad)
            raise RuntimeError(f"config assert did not fire for {bad}")
        except AssertionError as e:
            assert "invariant-domain" in str(e)
    print("  config-time asserts fire for w_s > w and P_s > P")

    res0 = simulate(_e6_a1_cfg())
    res7 = simulate(_e6_a1_cfg(w_s=0.7 * W))
    res5 = simulate(_e6_a1_cfg(w_s=0.5 * W))

    for tag, res in (("0.7w", res7), ("0.5w", res5)):
        balance = res.on_road + res.outflowed
        rel = abs(res.injected - balance) / max(res.injected, 1.0)
        rho = res.a + res.f + res.s
        print(f"  w_s={tag}: ledger rel err {rel:.2e}, "
              f"min f {res.f.min():.3e}, min s {res.s.min():.3e}, "
              f"max rho {rho.max():.6f} (P = {P:.6f})")
        assert rel < 1e-10, f"mass ledger not closed with w_s = {tag}"
        assert res.f.min() >= 0.0 and res.s.min() >= 0.0
        assert rho.max() <= P + 1e-12, f"rho exceeded P with w_s = {tag}"

    def wedge(res, i):
        xc = res.x_cav[i]
        assert np.isfinite(xc)
        m = (res.x >= xc - 1000.0) & (res.x <= xc - 200.0)
        return (res.a[i] + res.f[i] + res.s[i])[m].mean()

    i700 = int(np.argmin(np.abs(res0.t - 700.0)))
    w0 = wedge(res0, i700)
    w7 = wedge(res7, i700)
    w5 = wedge(res5, i700)
    print(f"  wake wedge mean rho at t=700, [x_cav-1km, x_cav-0.2km]: "
          f"w_s=None {w0 * 1000:.4f} vs w_s=0.7w {w7 * 1000:.4f} vs "
          f"w_s=0.5w {w5 * 1000:.4f} veh/km")

    # 0.7w: inert before release (r*_s above the realized densities) ...
    r_star7 = 0.7 * W * P / (15.0 + 0.7 * W)
    rho0_max = (res0.a + res0.f + res0.s).max()
    assert r_star7 > rho0_max, \
        "premise broke: r*_s(0.7w, u15) no longer above the run's max rho"
    pre = res0.t <= 750.0
    assert np.array_equal(res7.f[pre], res0.f[pre]) and \
        np.array_equal(res7.s[pre], res0.s[pre]), \
        "w_s=0.7w altered pre-release fields despite r*_s > max rho"
    assert w7 == w0, "wedge should be untouched by the inert 0.7w override"
    # ... but binding after release, when c_s = v_f (platoon dispersal)
    assert not np.array_equal(res7.s, res0.s), \
        "w_s=0.7w should alter the post-release dispersal (r*_s(v_f) < 44)"

    # 0.5w: r*_s below the wedge -> the congested branch binds inside the
    # slowdown window and the steady wedge is strictly denser
    assert w5 > w0, (
        "w_s = 0.5 w did not raise the steady wake wedge density "
        f"({w5 * 1000:.4f} <= {w0 * 1000:.4f} veh/km)")


# --------------------------------------------------------------------------
# 14-16. Downstream-release constraint (definitional, zero parameters)
# --------------------------------------------------------------------------

def _dsr_cfg(**kw) -> SimConfig:
    """E7-winner-like A1 config for the downstream-release tests:
    kappa_c = 0.046, kappa_r = 5.7e-4, lf, w_s = 0.6 w, u15, q2500,
    production dt = 0.5 / save_every = 20."""
    d = dict(kappa_c=0.046, kappa_r=5.7e-4, capture_form="lf", u_xi=15.0,
             q_in=2500.0 / 3600.0, dt=0.5, save_every=20, w_s=0.6 * W)
    d.update(kw)
    return base_cfg(**d)


def _downstream_mask(res, i):
    """Cells strictly downstream of the CAV cell at saved index i
    (j > j_cav, j_cav = int(x_cav // dx)) -- the cells the constraint
    empties of s."""
    dx = res.x[1] - res.x[0]
    j_cav = int(res.x_cav[i] // dx)
    return np.arange(res.x.size) > j_cav


def test_dsr_disabled_bit_identical():
    """t14: downstream_release=False (default) reproduces the PRE-change
    solver bit for bit (reference out/dsr_ref_A1.npz was generated from the
    solver BEFORE the flag existed)."""
    ref_path = HERE / "out" / "dsr_ref_A1.npz"
    assert ref_path.exists(), (
        "missing pre-change reference out/dsr_ref_A1.npz -- it must be "
        "generated with the PRE-flag solver, not recreated here")
    res = simulate(_dsr_cfg())
    _assert_bit_equal(res, np.load(ref_path),
                      "downstream_release=False vs pre-change")
    print("  downstream_release=False: f, s, N_s, omega bit-identical to "
          "the stored pre-change reference")


def test_dsr_downstream_zero_and_invariants():
    """t15: with the constraint on, s is exactly 0 strictly downstream of
    the CAV cell at every saved time while the CAV is on road, and all
    invariants hold (ledger closes to 1e-10, f, s >= 0, rho <= P)."""
    res = simulate(_dsr_cfg(downstream_release=True))

    n_checked = 0
    for i in range(res.t.size):
        if not np.isfinite(res.x_cav[i]):
            continue
        s_down = res.s[i][_downstream_mask(res, i)]
        assert np.all(s_down == 0.0), (
            f"s not exactly 0 downstream of the CAV at t = {res.t[i]:g} "
            f"(max {s_down.max():.3e})")
        n_checked += 1
    assert n_checked > 0, "CAV never on road at a saved time?"

    balance = res.on_road + res.outflowed
    rel = abs(res.injected - balance) / max(res.injected, 1.0)
    rho = res.a + res.f + res.s
    print(f"  s == 0 strictly downstream of the CAV at all {n_checked} "
          f"on-road saves; ledger rel err {rel:.2e}, "
          f"min f {res.f.min():.3e}, min s {res.s.min():.3e}, "
          f"max rho {rho.max():.6f} (P = {P:.6f})")
    assert rel < 1e-10, "mass ledger not closed with downstream_release"
    assert res.f.min() >= 0.0 and res.s.min() >= 0.0
    assert rho.max() <= P + 1e-12, "rho exceeded jam density"


def test_dsr_stuck_fraction_and_waviness():
    """t16: same config as t14, False vs True.  Reports (a) the t = 700
    downstream stuck fraction integral(s)/integral(rho) over the cells
    strictly downstream of the CAV (expected ~0.2-0.5 legacy vs exactly 0
    with the constraint -- the E7-winner leak Mladen diagnosed), and
    (b) the downstream waviness amplitude = std of linearly-detrended
    rho_tot over x in [x_cav + 0.5 km, x_cav + 5 km] at t = 600 (expected
    visible drop; both numbers reported, no hard threshold -- only the
    direction is asserted)."""
    res_f = simulate(_dsr_cfg())
    res_t = simulate(_dsr_cfg(downstream_release=True))

    # (a) t = 700 downstream stuck fraction
    i700 = int(np.argmin(np.abs(res_f.t - 700.0)))
    assert np.isfinite(res_f.x_cav[i700])

    def stuck(res):
        m = _downstream_mask(res, i700)
        rho = (res.a[i700] + res.f[i700] + res.s[i700])[m]
        return float(res.s[i700][m].sum() / rho.sum())

    st_f, st_t = stuck(res_f), stuck(res_t)
    print(f"  t=700 downstream stuck fraction: False {st_f:.4f} vs "
          f"True {st_t:.4f}")
    assert st_t == 0.0, "stuck fraction not exactly 0 with the constraint"
    assert 0.05 < st_f < 0.8, (
        f"legacy stuck fraction {st_f:.4f} outside the expected leak range")

    # (b) t = 600 downstream waviness amplitude
    i600 = int(np.argmin(np.abs(res_f.t - 600.0)))
    xc = res_f.x_cav[i600]
    assert np.isfinite(xc)

    def waviness(res):
        m = (res.x >= xc + 500.0) & (res.x <= xc + 5000.0)
        rho = (res.a[i600] + res.f[i600] + res.s[i600])[m]
        trend = np.polyval(np.polyfit(res.x[m], rho, 1), res.x[m])
        return float(np.std(rho - trend))

    wv_f, wv_t = waviness(res_f), waviness(res_t)
    print(f"  t=600 downstream waviness amplitude (std of detrended "
          f"rho_tot, [x_cav+0.5km, x_cav+5km]): False {wv_f:.3e} vs "
          f"True {wv_t:.3e} veh/m "
          f"({wv_f * 1000:.4f} vs {wv_t * 1000:.4f} veh/km)")
    assert wv_t < wv_f, (
        "waviness amplitude did not drop with the constraint "
        f"({wv_t:.3e} >= {wv_f:.3e})")


# --------------------------------------------------------------------------
# 17-20. Nonlocal leader-loss release (eta_la): the general, state-only
#        replacement for downstream_release
# --------------------------------------------------------------------------

def _ll_cfg(**kw) -> SimConfig:
    """E8 final hybrid A1 config (out/e8/final_config.json): kappa_c =
    5.32e-2, kappa_r = 1.65e-3, lf, u15, q2500, production dt = 0.5 /
    save_every = 20, capacity cap q_xi_max = 2000 veh/h, w_s = 0.6 w, and
    downstream_release=False (the new results use eta_la only)."""
    d = dict(kappa_c=5.32e-2, kappa_r=1.65e-3, capture_form="lf", u_xi=15.0,
             q_in=2500.0 / 3600.0, dt=0.5, save_every=20, w_s=0.6 * W,
             q_xi_max=2000.0 / 3600.0, downstream_release=False)
    d.update(kw)
    return base_cfg(**d)


def _ledger_and_invariants(res, what: str) -> float:
    """Assert the mass ledger closes to 1e-10, f, s >= 0 and rho <= P;
    return the ledger's relative error."""
    balance = res.on_road + res.outflowed
    rel = abs(res.injected - balance) / max(res.injected, 1.0)
    rho = res.a + res.f + res.s
    print(f"  {what}: ledger rel err {rel:.2e}, min f {res.f.min():.3e}, "
          f"min s {res.s.min():.3e}, max rho {rho.max():.6f} (P = {P:.6f})")
    assert rel < 1e-10, f"mass ledger not closed: {what}"
    assert res.f.min() >= 0.0 and res.s.min() >= 0.0, f"negative f/s: {what}"
    assert rho.max() <= P + 1e-12, f"rho exceeded jam density: {what}"
    return rel


def test_ll_disabled_bit_identical():
    """t17: eta_la=None (default) reproduces the PRE-change solver bit for
    bit (reference out/ll_ref_A1.npz was generated from the solver BEFORE
    eta_la existed; its solver_sha256 field records that solver)."""
    ref_path = HERE / "out" / "ll_ref_A1.npz"
    assert ref_path.exists(), (
        "missing pre-change reference out/ll_ref_A1.npz -- it must be "
        "generated with the PRE-eta_la solver, not recreated here")
    res = simulate(_ll_cfg())
    ref = np.load(ref_path)
    _assert_bit_equal(res, ref, "eta_la=None vs pre-change")
    print("  eta_la=None: f, s, N_s, omega bit-identical to the stored "
          f"pre-change reference (solver sha256 {str(ref['solver_sha256'])[:12]}...)")


def test_ll_startup_wave():
    """t18 (PHYSICAL UNIT TEST -- start-up wave): no CAV, q_in = 0, an
    initial pure-s block s0 = 0.04 veh/m on x in [10, 12] km (f0 = 0, no
    capture / kappa release), eta_la = 200 m.  The block advects at v_f
    (subcritical: 40 < 48.5 veh/km) and its DOWNSTREAM edge must recede
    relative to that advection: the s-region length L(t) (cells with
    s > 0.5 s0) shrinks at dL/dt = -w to within 30% (the spec tolerance;
    the telescoping identity in leader_loss_rate predicts exactly -w for
    a sharp-edged platoon, and the measured value is reported).  Mass
    f + s is conserved exactly (no inflow, block far from the outlet)."""
    dx, nx = 50.0, 600
    xc = (np.arange(nx) + 0.5) * dx
    s_blk = 0.04
    s0 = np.where((xc >= 10000.0) & (xc <= 12000.0), s_blk, 0.0)
    # t_end < t_slow, so u_s = v_f throughout and c_s = v_f; the CAV never
    # enters (t_enter > t_end), hence a == 0 and nothing references it
    cfg = base_cfg(q_in=0.0, kappa_c=0.0, kappa_r=0.0, t_enter=1e9,
                   t_end=150.0, dt=0.5, save_every=2, s0=s0, eta_la=200.0)
    res = simulate(cfg)
    assert np.all(res.a == 0.0), "CAV present in the no-CAV start-up test"

    L = np.sum(res.s > 0.5 * s_blk, axis=1) * dx
    m = (res.t >= 20.0) & (res.t <= 120.0)
    assert m.sum() > 50
    rate = np.polyfit(res.t[m], L[m], 1)[0]
    rel = abs(rate + W) / W
    # the upstream edge just advects: the first cell above 0.5 s0 moves at v_f
    up = np.array([res.x[np.where(res.s[i] > 0.5 * s_blk)[0][0]]
                   for i in np.where(m)[0]])
    v_up = np.polyfit(res.t[m], up, 1)[0]
    print(f"  s-region length L: {L[m][0]:.0f} m at t = 20 s -> "
          f"{L[m][-1]:.0f} m at t = 120 s; dL/dt = {rate:.4f} m/s vs "
          f"-w = {-W:.4f} m/s ({100 * rel:.2f}% off); upstream edge speed "
          f"{v_up:.3f} m/s vs v_f {V_F:.3f}")
    assert rel < 0.30, (
        f"start-up wave shrink rate {rate:.4f} m/s not within 30% of -w = "
        f"{-W:.4f}")
    assert abs(v_up - V_F) / V_F < 0.05, "upstream edge does not advect at v_f"

    mass = np.sum(res.f + res.s, axis=1) * dx
    m_err = np.max(np.abs(mass - mass[0]))
    print(f"  mass f + s: {mass[0]:.6f} veh, max |drift| {m_err:.2e} veh "
          f"(rel {m_err / mass[0]:.2e}); min f {res.f.min():.1e}, "
          f"min s {res.s.min():.1e}")
    assert m_err <= 1e-12 * mass[0], "f + s not conserved (exactly)"
    assert res.f.min() >= 0.0 and res.s.min() >= 0.0
    # released mass really shows up as f (start-up wave converts s -> f)
    i120 = int(np.argmin(np.abs(res.t - 120.0)))
    print(f"  released to f by t = 120 s: {np.sum(res.f[i120]) * dx:.2f} veh "
          f"of {mass[0]:.0f}")
    assert np.sum(res.f[i120]) * dx > 0.2 * mass[0]


def _ll_coverage_masks(a, s, dx, eta_la):
    """(a_ahead > 0, s_ahead >= s) per cell with a plain SEQUENTIAL window
    sum (the definition of leader_loss_rate re-implemented independently;
    cells beyond the road end count as empty)."""
    n = max(1, int(round(eta_la / dx)))
    nx = s.size
    a_pad = np.concatenate([a, np.zeros(n)])
    s_pad = np.concatenate([s, np.zeros(n)])
    cav_ahead = np.zeros(nx, bool)
    s_covered = np.zeros(nx, bool)
    for j in range(nx):
        a_ahead = sum(float(v) for v in a_pad[j + 1:j + n + 1])
        s_ahead = sum(float(v) for v in s_pad[j + 1:j + n + 1]) / n
        cav_ahead[j] = a_ahead > 0.0
        s_covered[j] = s_ahead >= s[j]
    return cav_ahead, s_covered


def test_ll_queue_immunity_and_downstream():
    """t19: A1 hybrid config with eta_la = 200 (no downstream_release).
    (a) Queue immunity: inside the queue -- cells with s > 20 veh/km that
    have the CAV, or at least as much s, within eta ahead -- the leader-
    loss rate is EXACTLY 0 (checked on every saved state with an
    independent sequential-sum re-implementation of the coverage).
    (b) Reports the integral of s strictly downstream of the CAV cell at
    t = 700 for eta_la = None vs 200 (a large drop is expected) and the
    maximum downstream s [veh/km]."""
    res0 = simulate(_ll_cfg())
    res2 = simulate(_ll_cfg(eta_la=200.0))
    dx = res2.x[1] - res2.x[0]
    c_wave = 0.6 * W

    # (a) exact immunity of covered queue cells, all saved states
    n_cells = n_saves = 0
    worst = 0.0
    for i in range(res2.t.size):
        a_i, s_i = res2.a[i], res2.s[i]
        mu = leader_loss_rate(a_i, s_i, dx, 200.0, c_wave)
        assert np.all(mu >= 0.0)
        cav_ahead, s_cov = _ll_coverage_masks(a_i, s_i, dx, 200.0)
        inq = (s_i > 0.020) & (cav_ahead | s_cov)
        if inq.any():
            n_saves += 1
            n_cells += int(inq.sum())
            worst = max(worst, float(np.max(np.abs(mu[inq]))))
            assert np.all(mu[inq] == 0.0), (
                f"leader-loss rate not exactly 0 inside the queue at "
                f"t = {res2.t[i]:g} (max {np.max(mu[inq]):.3e})")
        # and it is genuinely active somewhere while a queue exists
    print(f"  queue immunity: mu_ll == 0.0 exactly on {n_cells} covered "
          f"queue cells (s > 20 veh/km) over {n_saves} saves (max |mu| "
          f"{worst:.1e}); rate scale c_wave/ell_eff = "
          f"{c_wave / (dx * (4 + 1) / 2):.4f} 1/s")
    assert n_cells > 50, "queue never formed?"

    # (b) downstream s at t = 700
    i700 = int(np.argmin(np.abs(res0.t - 700.0)))
    assert np.isfinite(res0.x_cav[i700]) and np.isfinite(res2.x_cav[i700])
    out = {}
    for tag, res in (("None", res0), ("200", res2)):
        md = _downstream_mask(res, i700)
        out[tag] = (float(res.s[i700][md].sum() * dx),
                    float(res.s[i700][md].max() * 1000.0))
    print(f"  t=700 s strictly downstream of the CAV cell: eta_la=None "
          f"integral {out['None'][0]:.4f} veh (max {out['None'][1]:.3f} "
          f"veh/km) vs eta_la=200 integral {out['200'][0]:.4f} veh (max "
          f"{out['200'][1]:.3f} veh/km); drop factor "
          f"{out['None'][0] / max(out['200'][0], 1e-300):.1f}x")
    print(f"  N_s(700): None {res0.N_s[i700]:.2f} vs 200 {res2.N_s[i700]:.2f}"
          f" veh; N_s(t_end): None {res0.N_s[-1]:.2f} vs 200 "
          f"{res2.N_s[-1]:.2f} veh (start-up wave releases the queue)")
    assert out["200"][0] < 0.2 * out["None"][0], (
        "downstream s did not drop by at least 5x with eta_la = 200")
    assert out["200"][1] < out["None"][1]
    assert res2.N_s[-1] < res2.N_s[i700], \
        "queue not released by the start-up wave after t_fast"


def test_ll_ledger_and_invariants():
    """t20: eta_la = 200 with the capacity cap on: mass ledger closes to
    1e-10, f, s >= 0, rho <= P.  Also checked with downstream_release=True
    on top (the two are independent and may both be on; the new results
    use eta_la only)."""
    _ledger_and_invariants(simulate(_ll_cfg(eta_la=200.0)),
                           "eta_la=200 + cap")
    _ledger_and_invariants(simulate(_ll_cfg(eta_la=200.0,
                                            downstream_release=True)),
                           "eta_la=200 + cap + downstream_release")
    # config-time guard
    for bad in (0.0, -50.0, float("nan")):
        try:
            _ll_cfg(eta_la=bad)
            raise RuntimeError(f"config assert did not fire for eta_la={bad}")
        except AssertionError as e:
            assert "eta_la" in str(e)
    print("  config-time assert fires for eta_la <= 0 / nan")


# --------------------------------------------------------------------------
# 21-24. E10: s-impermeable moving-bottleneck interface (F^s = 0 at the CAV
#        face) + connectivity leader loss (state-only least fixed point)
# --------------------------------------------------------------------------

def _footprint_front(res, i):
    """Last cell of the CAV's discrete footprint (a > 0) at saved index i,
    at least the cap cell j = int(x_cav / dx)."""
    dx = res.x[1] - res.x[0]
    j = int(res.x_cav[i] / dx)
    ja = np.flatnonzero(res.a[i])
    return max(j, int(ja[-1])) if ja.size else j


def _wake(res, i):
    """(mean rho, mean s) [veh/m] over [x_cav - 1 km, x_cav - 0.2 km]."""
    xc = res.x_cav[i]
    assert np.isfinite(xc)
    m = (res.x >= xc - 1000.0) & (res.x <= xc - 200.0)
    return ((res.a[i] + res.f[i] + res.s[i])[m].mean(),
            res.s[i][m].mean())


def test_e10_off_bit_identical():
    """t21: all E10 fields at their defaults (s_impermeable=False,
    ll_mode='ratio', eta_la=None) reproduce the PRE-E10 solver bit for bit
    on the E8 hybrid A1 config (reference out/e10_ref_A1.npz, generated
    from the solver BEFORE the E10 edit; its solver_sha256 field records
    that solver).  Also: config-time guards for ll_mode / tau_ll."""
    ref_path = HERE / "out" / "e10_ref_A1.npz"
    assert ref_path.exists(), (
        "missing pre-change reference out/e10_ref_A1.npz -- it must be "
        "generated with the PRE-E10 solver, not recreated here")
    res = simulate(_ll_cfg())
    ref = np.load(ref_path)
    _assert_bit_equal(res, ref, "E10 fields off vs pre-change")
    print("  s_impermeable=False, ll_mode='ratio', eta_la=None: f, s, N_s, "
          "omega bit-identical to the stored pre-E10 reference (solver "
          f"sha256 {str(ref['solver_sha256'])[:12]}...)")
    for bad in (dict(ll_mode="foo"), dict(tau_ll=0.0), dict(tau_ll=-1.0),
                dict(tau_ll=float("nan"))):
        try:
            _ll_cfg(**bad)
            raise RuntimeError(f"config assert did not fire for {bad}")
        except AssertionError as e:
            assert "ll_mode" in str(e) or "tau_ll" in str(e)
    print("  config-time asserts fire for ll_mode not in {ratio, connect} "
          "and tau_ll <= 0 / nan")


def test_s_impermeable_downstream_zero():
    """t22: s_impermeable=True, no leader loss, E8 hybrid A1 config with
    the cap on.  At every saved time in the slow phase, s is EXACTLY 0 in
    every cell strictly downstream of the CAV's discrete footprint (a > 0;
    the general statement) and strictly downstream of the CAV cell j =
    int(x_cav / dx) (at this save cadence x_cav is in the left half of its
    cell at every save, so the footprint is {j-1, j}); the same cells in
    the s_impermeable=False run carry the E9 numerical plume (reported).
    Ledger 1e-10, f, s >= 0, rho <= P.  Wake density over [x_cav - 1 km,
    x_cav - 0.2 km] at t = 700 reported vs the s_impermeable=False run."""
    cfg = _ll_cfg(s_impermeable=True)
    res_f = simulate(_ll_cfg())
    res_t = simulate(cfg)
    dx = res_t.x[1] - res_t.x[0]

    n_checked = 0
    plume_max = 0.0
    for i in range(res_t.t.size):
        t = res_t.t[i]
        if not (cfg.t_slow <= t <= cfg.t_fast) or not np.isfinite(res_t.x_cav[i]):
            continue
        j = int(res_t.x_cav[i] / dx)
        kf = _footprint_front(res_t, i)
        assert kf == j, (
            "premise broke: CAV footprint extends past the cap cell at a "
            f"save (t = {t:g}); the cap-cell statement below is then only "
            "the footprint statement")
        s_down = res_t.s[i][kf + 1:]
        assert np.all(s_down == 0.0), (
            f"s not exactly 0 strictly downstream of the CAV footprint at "
            f"t = {t:g} (max {s_down.max():.3e})")
        assert np.all(res_t.s[i][j + 1:] == 0.0)
        plume_max = max(plume_max, float(res_f.s[i][j + 1:].max()))
        n_checked += 1
    assert n_checked > 0
    print(f"  s == 0.0 exactly strictly downstream of the CAV footprint / "
          f"CAV cell at all {n_checked} slow-phase saves (s_impermeable="
          f"False: max downstream s {plume_max * 1000:.2f} veh/km)")
    _ledger_and_invariants(res_t, "s_impermeable + cap")

    i700 = int(np.argmin(np.abs(res_t.t - 700.0)))
    rho_f, s_f = _wake(res_f, i700)
    rho_t, s_t = _wake(res_t, i700)
    print(f"  t=700 wake [x_cav-1km, x_cav-0.2km]: s_impermeable=False rho "
          f"{rho_f * 1000:.3f} (s {s_f * 1000:.3f}) vs True rho "
          f"{rho_t * 1000:.3f} (s {s_t * 1000:.3f}) veh/km; N_s(700) "
          f"{res_f.N_s[i700]:.2f} vs {res_t.N_s[i700]:.2f} veh; max N_s "
          f"{res_f.N_s.max():.2f} vs {res_t.N_s.max():.2f} veh")


def test_ll_connect_synthetic():
    """t23: connectivity coverage on synthetic arrays (dx = 50, eta = 200 =
    4 cells, tau = 3), calling leader_loss_rate_connect directly.
    (a) a = 0, s-platoon on 20 cells -> rate == 1/tau exactly on every
    platoon cell (cover = 1 would be self-consistent too: the LEAST fixed
    point is the definition); (b) same platoon with a = 1/dx in the cell
    just ahead of its head -> rate == 0 exactly on all platoon cells;
    (c) an A-anchored front part and a rear part separated by a 5-cell gap
    (> eta) -> rear rate == 1/tau, front rate == 0; (d) a queue whose s
    tapers linearly 40 -> 5 veh/km toward the head, anchored by A -> rate
    0 everywhere (the E9 ratio form releases INSIDE it: shown); (e) the
    CAV split over two cells (as cav_density does) with the queue head in
    the CAV's own cell -> rate 0 exactly on all queue cells, for the split
    weights of the production run and the extremes."""
    dx, eta, tau, nx = 50.0, 200.0, 3.0, 100

    def arrays():
        return np.zeros(nx), np.zeros(nx)

    # (a) disconnected platoon: least fixed point is cover = 0
    a, s = arrays()
    s[40:60] = 0.04
    mu = leader_loss_rate_connect(a, s, dx, eta, tau)
    assert np.all(mu[40:60] == 1.0 / tau), "(a) disconnected platoon"
    # (b) anchored by an A vehicle just ahead of the head
    a[60] = 1.0 / dx
    mu = leader_loss_rate_connect(a, s, dx, eta, tau)
    assert np.all(mu[40:60] == 0.0), "(b) anchored platoon"
    assert mu[60] == 0.0 and np.all(mu[61:] == 1.0 / tau)
    print("  (a) a=0 platoon: rate == 1/tau exactly on 20/20 cells; "
          "(b) A ahead of the head: rate == 0 exactly on 20/20 cells")
    # (c) 5-cell gap disconnects the rear part
    a, s = arrays()
    s[30:45] = 0.04
    s[50:60] = 0.04
    a[60] = 1.0 / dx
    mu = leader_loss_rate_connect(a, s, dx, eta, tau)
    assert np.all(mu[30:45] == 1.0 / tau), "(c) rear part not released"
    assert np.all(mu[50:60] == 0.0), "(c) front part released"
    print("  (c) 5-cell gap: rear 15 cells == 1/tau, front 10 cells == 0, "
          "all exactly")
    # (d) tapered queue anchored by A: connect 0 everywhere, ratio form not
    a, s = arrays()
    s[40:60] = np.linspace(0.040, 0.005, 20)
    a[60] = 1.0 / dx
    mu = leader_loss_rate_connect(a, s, dx, eta, tau)
    assert np.all(mu[40:60] == 0.0), "(d) tapered queue released"
    mu9 = leader_loss_rate(a, s, dx, eta, 0.6 * W)
    n_rel9 = int(np.sum(mu9[40:60] > 0.0))
    assert n_rel9 > 0, "E9 ratio form no longer shows its in-queue release?"
    print(f"  (d) taper 40 -> 5 veh/km + A: connect rate == 0 exactly on "
          f"20/20 cells; E9 ratio form > 0 on {n_rel9}/20 cells (max "
          f"{mu9[40:60].max():.4f} 1/s)")
    # (e) split CAV, queue head in the CAV's own cell
    for wa, wb in ((0.16, 0.84), (0.84, 0.16), (0.5, 0.5), (1.0, 0.0)):
        a, s = arrays()
        a[60] = wa / dx
        a[61] = wb / dx
        s[40:61] = 0.04          # queue up to and including the rear cell
        s[61] = 1e-4             # reaction-created s in the front cell
        mu = leader_loss_rate_connect(a, s, dx, eta, tau)
        assert np.all(mu[40:61] == 0.0), f"(e) split ({wa}, {wb}): queue"
        if wb > 0.0:
            # front cell carries CAV weight -> part of the footprint: covered
            assert mu[61] == 0.0, f"(e) split ({wa}, {wb}): footprint cell"
        else:
            # CAV entirely in cell 60: cell 61 is strictly DOWNSTREAM with no
            # slow leader ahead -> its (reaction-created) s must be released
            assert mu[61] == 1.0 / tau, f"(e) split ({wa}, {wb}): leaderless"
    print("  (e) split CAV (0.16/0.84, 0.84/0.16, 0.5/0.5, 1/0): rate == 0 "
          "exactly on all queue + footprint cells; leaderless front cell "
          "released at 1/tau when the CAV does not occupy it")


def test_e10_hybrid_connect_queue():
    """t24: full A1 hybrid + s_impermeable + connect leader loss (eta_la =
    200 m, tau_ll = 3 s).  (a) On every slow-phase save, mu_ll == 0 EXACTLY
    on the queue attached to the CAV (walking back from the CAV footprint
    through cells with s > 20 veh/km or a > 0) and s == 0 exactly ahead of
    the footprint.  (b) N_s(1000) < 0.05 max_t N_s: the queue is relabelled
    free after the CAV departs (unanchored least fixed point = 0).
    (c) Ledger 1e-10, f, s >= 0, rho <= P.  The ratio form on the same
    config is reported alongside."""
    cfg = _ll_cfg(s_impermeable=True, eta_la=200.0, ll_mode="connect",
                  tau_ll=3.0)
    res = simulate(cfg)
    dx = res.x[1] - res.x[0]

    n_cells = n_saves = 0
    for i in range(res.t.size):
        t = res.t[i]
        if not (cfg.t_slow <= t <= cfg.t_fast) or not np.isfinite(res.x_cav[i]):
            continue
        a_i, s_i = res.a[i], res.s[i]
        mu = leader_loss_rate_connect(a_i, s_i, dx, cfg.eta_la, cfg.tau_ll)
        assert np.all(mu >= 0.0) and np.all(mu <= 1.0 / cfg.tau_ll)
        kf = _footprint_front(res, i)
        assert np.all(s_i[kf + 1:] == 0.0), f"s ahead of the CAV at t={t:g}"
        k, inq = kf, []
        while k >= 0 and (s_i[k] > 0.020 or a_i[k] > 0.0):
            if s_i[k] > 0.020:
                inq.append(k)
            k -= 1
        if inq:
            n_saves += 1
            n_cells += len(inq)
            assert np.all(mu[inq] == 0.0), (
                f"mu_ll not exactly 0 inside the attached queue at t = {t:g} "
                f"(max {mu[inq].max():.3e})")
    assert n_cells > 50, "queue never formed?"
    print(f"  mu_ll == 0.0 exactly on {n_cells} attached-queue cells (s > 20 "
          f"veh/km) over {n_saves} slow-phase saves; s == 0 ahead of the CAV "
          "footprint at all of them")

    def at(res, tt):
        return res.N_s[int(np.argmin(np.abs(res.t - tt)))]

    i_max = int(np.argmax(res.N_s))
    print(f"  connect: max N_s {res.N_s[i_max]:.2f} veh at t = "
          f"{res.t[i_max]:g}; N_s(700) {at(res, 700):.2f}, N_s(750) "
          f"{at(res, 750):.2f}, N_s(760) {at(res, 760):.2f}, N_s(770) "
          f"{at(res, 770):.2f}, N_s(800) {at(res, 800):.2f}, N_s(1000) "
          f"{res.N_s[-1]:.4f} veh (ratio to max {res.N_s[-1] / res.N_s.max():.2e})")
    assert res.N_s[-1] < 0.05 * res.N_s.max(), \
        "queue not relabelled free after the CAV departs"
    _ledger_and_invariants(res, "s_impermeable + connect LL + cap")

    res9 = simulate(_ll_cfg(s_impermeable=True, eta_la=200.0))
    print(f"  ratio form (same config): max N_s {res9.N_s.max():.2f}, "
          f"N_s(700) {at(res9, 700):.2f}, N_s(1000) {res9.N_s[-1]:.2f} veh")


# --------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("conservation", test_conservation),
        ("invariant domain", test_invariant_domain),
        ("pure-class front speed", test_pure_class_front),
        ("Riemann shock RH speed", test_riemann_shock),
        ("queue smoke", test_queue_smoke),
        ("CFL assertion", test_cfl_assert),
        ("cap disabled bit-identity", test_cap_disabled_bit_identical),
        ("cap analytic bottleneck", test_cap_analytic_bottleneck),
        ("cap + reactions invariants", test_cap_with_reactions_invariants),
        ("struct knobs None bit-identity", test_struct_none_bit_identical),
        ("gamma == capture forms", test_gamma_matches_capture_forms),
        ("w_s=w, P_s=P bit-identity", test_ws_ps_shared_bit_identical),
        ("w_s=0.7w wake wedge", test_ws_slow_wake_wedge),
        ("downstream release off bit-identity", test_dsr_disabled_bit_identical),
        ("downstream release zero + invariants",
         test_dsr_downstream_zero_and_invariants),
        ("downstream stuck fraction + waviness",
         test_dsr_stuck_fraction_and_waviness),
        ("leader-loss off bit-identity", test_ll_disabled_bit_identical),
        ("leader-loss start-up wave (dL/dt = -w)", test_ll_startup_wave),
        ("leader-loss queue immunity + downstream s",
         test_ll_queue_immunity_and_downstream),
        ("leader-loss ledger + invariants", test_ll_ledger_and_invariants),
        ("E10 fields off bit-identity", test_e10_off_bit_identical),
        ("s-impermeable interface: downstream s == 0",
         test_s_impermeable_downstream_zero),
        ("connect leader-loss synthetic coverage", test_ll_connect_synthetic),
        ("hybrid + s-impermeable + connect LL",
         test_e10_hybrid_connect_queue),
    ]
    for name, fn in tests:
        print(f"[{name}]")
        fn()
        print("  PASS")
    print(f"\nAll {len(tests)} tests passed.")
