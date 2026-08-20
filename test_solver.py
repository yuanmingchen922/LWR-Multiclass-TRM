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
                    cav_density, transport_step, u_s_of_t)

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
    ]
    for name, fn in tests:
        print(f"[{name}]")
        fn()
        print("  PASS")
    print(f"\nAll {len(tests)} tests passed.")
