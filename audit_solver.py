"""Adversarial numerical audit of analysis/solver.py (independent checks).

Run: python3 audit_solver.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ANALYSIS = Path("/Users/yuanmingchen/Desktop/Research/SUMO/analysis")
sys.path.insert(0, str(ANALYSIS))

from solver import (SimConfig, simulate, reaction_exact, transport_step,
                    cav_density, u_s_of_t, speed, capacity)

_params = json.loads((ANALYSIS / "out" / "params.json").read_text())
V_F = _params["v_f_kmh"] / 3.6
W = _params["w_kmh"] / 3.6
P = _params["P_vehkm"] / 1000.0

FAIL = []


def check(name, ok, msg):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}: {msg}")
    if not ok:
        FAIL.append((name, msg))


def base_cfg(**kw) -> SimConfig:
    d = dict(v_f=V_F, w=W, P=P, q_in=2500.0 / 3600.0, u_xi=15.0,
             kappa_c=0.026, kappa_r=3e-5)
    d.update(kw)
    return SimConfig(**d)


# ---------------------------------------------------------------- check 1
# Mass ledger, A=1-like.  Two layers:
#  (a) solver's own counters: injected = on_road + outflowed
#  (b) fully independent re-run: my own loop calling transport_step /
#      reaction_exact directly, verifying EVERY step that
#      d(mass) = (F_in - F_out) dt and that reaction conserves p.
def check1():
    cfg = base_cfg()
    res = simulate(cfg)
    bal = res.on_road + res.outflowed
    rel = abs(res.injected - bal) / max(res.injected, 1.0)
    check("1a ledger (solver counters)", rel < 1e-6,
          f"injected {res.injected:.9f} vs on_road+out {bal:.9f}, "
          f"rel {rel:.2e}")
    off = cfg.q_in * cfg.t_end
    rel2 = abs(off - (res.injected + res.denied_inflow)) / off
    check("1b offered = injected + denied", rel2 < 1e-9,
          f"rel {rel2:.2e} (denied {res.denied_inflow:.3e})")

    # independent per-step ledger
    nx = int(round(cfg.L_road / cfg.dx))
    f = np.zeros(nx)
    s = np.zeros(nx)
    n_steps = int(round(cfg.t_end / cfg.dt))
    worst_tr = worst_rx = 0.0
    inj = out = 0.0
    for n in range(n_steps):
        t = n * cfg.dt
        u_s = u_s_of_t(cfg, t)
        a = cav_density(cfg, t, nx)
        m0 = np.sum(f + s) * cfg.dx
        f, s, q_adm, q_out = transport_step(f, s, a, cfg.v_f, u_s, cfg)
        m1 = np.sum(f + s) * cfg.dx
        worst_tr = max(worst_tr, abs(m1 - m0 - (q_adm - q_out) * cfg.dt))
        inj += q_adm * cfg.dt
        out += q_out * cfg.dt
        a2 = cav_density(cfg, t + cfg.dt, nx)
        rho2 = a2 + f + s
        dv = np.maximum(speed(rho2, cfg.v_f, cfg.w, cfg.P) - u_s, 0.0)
        f, s = reaction_exact(f, s, a2, rho2, dv, cfg.kappa_c, cfg.kappa_r,
                              cfg.P, cfg.dt, cfg.capture_form)
        m2 = np.sum(f + s) * cfg.dx
        worst_rx = max(worst_rx, abs(m2 - m1))
    m_end = np.sum(f + s) * cfg.dx
    rel3 = abs(inj - out - m_end) / max(inj, 1.0)
    check("1c independent per-step transport ledger", worst_tr < 1e-10,
          f"worst per-step |dM - (Fin-Fout)dt| = {worst_tr:.2e} veh")
    check("1d independent per-step reaction p-conservation",
          worst_rx < 1e-12, f"worst |dM_reaction| = {worst_rx:.2e} veh")
    check("1e independent cumulative ledger", rel3 < 1e-6,
          f"rel {rel3:.2e}")
    # cross-check my re-run against simulate() itself
    d_inj = abs(inj - res.injected)
    d_end = np.max(np.abs((f + s) - (res.f[-1] + res.s[-1])))
    check("1f re-run matches simulate()", d_inj < 1e-9 and d_end < 1e-14,
          f"|d injected| {d_inj:.2e}, max |d state| {d_end:.2e}")


# ---------------------------------------------------------------- check 2
def check2():
    cfg = base_cfg(q_in=2900.0 / 3600.0, u_xi=10.0, kappa_c=0.1)
    res = simulate(cfg)
    rho = res.a + res.f + res.s
    ok = (res.f.min() >= -1e-12 and res.s.min() >= -1e-12
          and rho.max() <= P * (1.0 + 1e-12))
    check("2 invariant domain (stress)", ok,
          f"min f {res.f.min():.3e}, min s {res.s.min():.3e}, "
          f"max rho {rho.max():.6f} vs P {P:.6f} "
          f"(max rho/P = {rho.max() / P:.4f})")
    # also stress with the calibrated large-kappa A=10 pair at every save
    cfg2 = base_cfg(q_in=2900.0 / 3600.0, u_xi=10.0, kappa_c=0.1,
                    kappa_r=0.016, save_every=1)
    r2 = simulate(cfg2)
    rho2 = r2.a + r2.f + r2.s
    ok2 = (r2.f.min() >= -1e-12 and r2.s.min() >= -1e-12
           and rho2.max() <= P * (1.0 + 1e-12))
    check("2b invariant domain (stress + large kappa_r, every step)", ok2,
          f"min f {r2.f.min():.3e}, min s {r2.s.min():.3e}, "
          f"max rho {rho2.max():.6f}")


# ---------------------------------------------------------------- check 3
def check3():
    r1 = simulate(base_cfg(dt=1.0))
    r2 = simulate(base_cfg(dt=0.5))
    n1 = r1.N_s[int(np.argmin(np.abs(r1.t - 740.0)))]
    n2 = r2.N_s[int(np.argmin(np.abs(r2.t - 740.0)))]
    rel = abs(n1 - n2) / max(n2, 1e-12)
    check("3 dt-refinement N_s(740)", rel < 0.05,
          f"dt=1.0: {n1:.3f} veh, dt=0.5: {n2:.3f} veh, rel diff "
          f"{100 * rel:.2f}%")
    # bonus: dt=0.25 to see the convergence trend
    r3 = simulate(base_cfg(dt=0.25, save_every=20))
    n3 = r3.N_s[int(np.argmin(np.abs(r3.t - 740.0)))]
    print(f"        (dt=0.25: {n3:.3f} veh; 1.0->0.5 delta "
          f"{n2 - n1:+.4f}, 0.5->0.25 delta {n3 - n2:+.4f})")


# ---------------------------------------------------------------- check 4
# Closed form of the frozen 2x2 ODE: p = f + s const,
# f(t) = f_eq + (f0 - f_eq) e^{-(sigma+mu) t},  f_eq = mu p / (sigma + mu).
def check4():
    rng = np.random.default_rng(42)
    worst = 0.0
    for i in range(3):
        sigma = rng.uniform(0.01, 2.0)
        mu = rng.uniform(0.001, 1.0)
        f0 = rng.uniform(0.0, 0.1)
        s0 = rng.uniform(0.0, 0.1)
        T = rng.uniform(0.5, 20.0)
        # construct solver inputs that reproduce (sigma, mu) exactly:
        # "af" form -> ell = a; choose a, dv, then kappas by division.
        a = 0.01
        dv = 5.0
        rho = a + f0 + s0
        assert rho < P
        kc = sigma / (a * dv)
        kr = mu / ((P - rho) * dv)
        fe, se = reaction_exact(f0, s0, a, rho, dv, kc, kr, P, T, "af")
        g = sigma + mu
        f_eq = mu / g * (f0 + s0)
        f_ref = f_eq + (f0 - f_eq) * np.exp(-g * T)
        s_ref = (f0 + s0) - f_ref
        err = max(abs(float(fe) - f_ref), abs(float(se) - s_ref))
        worst = max(worst, err)
        print(f"        case {i}: sigma={sigma:.4f} mu={mu:.4f} "
              f"f0={f0:.4f} s0={s0:.4f} T={T:.2f}  err={err:.2e}")
        # "lf" form: ell = a + s0 frozen -> same closed form with
        # sigma_lf = kc2 * (a + s0) * dv
        kc2 = sigma / ((a + s0) * dv)
        fe2, se2 = reaction_exact(f0, s0, a, rho, dv, kc2, kr, P, T, "lf")
        err2 = max(abs(float(fe2) - f_ref), abs(float(se2) - s_ref))
        worst = max(worst, err2)
    check("4 reaction exactness vs closed form", worst < 1e-13,
          f"worst abs err {worst:.2e}")
    # degenerate corners: gamma = 0, and pure capture (mu = 0)
    fz, sz = reaction_exact(0.05, 0.02, 0.0, 0.07, 0.0, 0.1, 0.1, P, 1.0,
                            "af")
    # f is returned bit-exactly; s = (f+s) - f may differ by 1 ulp
    ok0 = float(fz) == 0.05 and abs(float(sz) - 0.02) < 1e-16
    fm, sm = reaction_exact(0.05, 0.02, 0.01, 0.08, 5.0, 0.1, 0.0, P, 3.0,
                            "af")
    f_ref = 0.05 * np.exp(-0.1 * 0.01 * 5.0 * 3.0)
    okm = abs(float(fm) - f_ref) < 1e-16
    check("4b degenerate corners (gamma=0, mu=0)", ok0 and okm,
          f"identity at gamma=0: {ok0}; pure-decay err "
          f"{abs(float(fm) - f_ref):.2e}")


# ---------------------------------------------------------------- check 5
def check5():
    # (a) algebraic: dv = (V(rho) - v_f)_+ must vanish for ALL rho in [0, P]
    rho = np.linspace(0.0, P, 200001)
    dv = np.maximum(speed(rho, V_F, W, P) - V_F, 0.0)
    check("5a dv == 0 identically when u_s = v_f", float(dv.max()) == 0.0,
          f"max dv over rho-grid = {dv.max():.3e}")
    # (b) full run with the slow window disabled: u_s(t) = v_f always, CAV
    # still on the road, kappa_c large -> s must remain EXACTLY zero.
    cfg = base_cfg(kappa_c=0.5, kappa_r=0.016, t_slow=1e9, t_fast=2e9,
                   save_every=1)
    res = simulate(cfg)
    smax = np.abs(res.s).max()
    check("5b no transitions outside slow window", smax == 0.0,
          f"max |s| over full run = {smax:.3e}")
    # (c) and f must match a kappa = 0 run bit-for-bit
    res0 = simulate(base_cfg(kappa_c=0.0, kappa_r=0.0, t_slow=1e9,
                             t_fast=2e9, save_every=1))
    dmax = np.abs(res.f - res0.f).max()
    check("5c f identical to kappa=0 run", dmax == 0.0,
          f"max |df| = {dmax:.3e}")


# ---------------------------------------------------------------- check 6
def check6():
    cfg = base_cfg(kappa_c=0.034, kappa_r=0.016, save_every=1)
    res = simulate(cfg)
    m = (res.t >= 260.0) & (res.t <= 740.0)
    cum = float(np.trapz(res.omega[m], res.t[m]))
    ok = 55.0 <= cum <= 220.0
    check("6 cumulative omega A=10-like in [260, 740]", ok,
          f"integral = {cum:.1f} veh vs measured 110 veh "
          f"(ratio {cum / 110.0:.2f})")
    # context: same integral for the A=1-like pair
    resA1 = simulate(base_cfg(save_every=1))
    mA1 = (resA1.t >= 260.0) & (resA1.t <= 740.0)
    cumA1 = float(np.trapz(resA1.omega[mA1], resA1.t[mA1]))
    print(f"        (A=1-like pair for context: {cumA1:.1f} veh)")


if __name__ == "__main__":
    for fn in (check1, check2, check3, check4, check5, check6):
        fn()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILURE(S):")
        for name, msg in FAIL:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    print("ALL AUDIT CHECKS PASSED")
