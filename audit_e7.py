"""Independent audit probes for the E7 structural knobs (gamma, w_s, P_s)
added to solver.py.  Does NOT modify solver.py / test_solver.py.

Run: python3 audit_e7.py          (the 13-test suite is run separately:
                                   python3 test_solver.py)

Probes
------
P2a  defaults-None vs the STORED pre-change reference out/struct_ref_A1.npz
     (independent load + bitwise compare, not reusing test code).
P2b  defaults-None vs a LIVE run of the pre-change solver taken from
     git HEAD:solver.py (commit with solver.py before the knobs existed):
     every SimResult array and scalar, bit for bit.  Also validates the
     stored npz provenance against the same live legacy run.
P2c  gamma=1.0 (with capture_form='af') vs capture_form='lf', and
     gamma=0.0 (with capture_form='lf') vs capture_form='af' -- all
     SimResult fields bit-identical, proving gamma overrides capture_form.
P3   w_s = 0.7 w, P_s = P stress run (q_in = 2900/3600, u_xi = 10, dt=0.5):
     - conservation ledger rel err < 1e-10;
     - invariant domain at EVERY substep (post-transport and post-reaction):
       f, s >= 0 and rho <= P + 1e-12;
     - the s-class interface flux, reconstructed from the conservative
       update (F_s[0] = 0, F_s[k+1] = F_s[k] + (s_k - s_k_new)/lam), never
       exceeds pi_s * D_s(rho; c_s, w_s, P_s) at any face of any step;
     - the manual step loop is cross-checked bit-identical against
       simulate() (same saved fields and ledger scalars), so the probe
       audits the real code path, not a reimplementation.
P3b  transport_step unit probe with P_s = 0.8 P and rho > P_s in a block of
     cells: the raw s-supply w_s (P_s - rho) goes negative there (clip
     branch actually exercised); all reconstructed fluxes stay >= 0, the
     flux bound holds, and f, s >= 0, rho <= P after the step.
P4   reaction layer w_s-independence: solver.speed is traced (module
     attribute swap inside this audit only) during single-step simulate()
     runs from a frozen state.  (i) Low-density frozen state, where the
     s-branch is provably inert: the dv call's (c, w, P) equal the ROAD
     (v_f, w, P) both with and without w_s, the traced rho_star and the
     resulting Delta v arrays are bit-identical across the two runs, and
     the full step output is bit-identical.  (ii) Dense frozen state, where
     the s-branch binds (outputs differ): the dv call STILL uses the road
     (v_f, w, P), i.e. w_s never leaks into the reaction layer.
P5   DM-G cap regression with new fields at None: re-run the capped
     scenario A1_u15_q2500 (q_xi_max = 2000 veh/h) through the ev4_compare
     pipeline and diff every metric digit-for-digit against the stored
     out/ev4b_q2000/metrics.json; additionally the capped run itself is
     compared bit-for-bit against the live pre-change solver.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import solver
from solver import SimConfig, simulate

_params = json.loads((HERE / "out" / "params.json").read_text())
V_F = _params["v_f_kmh"] / 3.6
W = _params["w_kmh"] / 3.6
P = _params["P_vehkm"] / 1000.0

FIELDS = ("t", "x", "a", "f", "s", "x_cav", "omega", "N_s")
SCALARS = ("denied_inflow", "injected", "outflowed", "on_road")

FAILURES: list[str] = []


def check(ok: bool, label: str) -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}")
    if not ok:
        FAILURES.append(label)


def diff_results(ra, rb) -> list[str]:
    """Names of SimResult fields that are NOT bit-identical."""
    bad = []
    for name in FIELDS:
        va = np.asarray(getattr(ra, name), float)
        vb = np.asarray(getattr(rb, name), float)
        if va.shape != vb.shape or not np.array_equal(va, vb, equal_nan=True):
            bad.append(name)
    for name in SCALARS:
        if getattr(ra, name) != getattr(rb, name):
            bad.append(name)
    return bad


def e6_a1_kwargs(**kw) -> dict:
    """A1 native-fit run (out/e6/fits.json rounded), production cadence."""
    d = dict(v_f=V_F, w=W, P=P, q_in=2500.0 / 3600.0, u_xi=15.0,
             kappa_c=0.504, kappa_r=0.0306, capture_form="lf",
             dt=0.5, save_every=20)
    d.update(kw)
    return d


# --------------------------------------------------------------------------
# legacy (pre-change) solver, live from git
# --------------------------------------------------------------------------

def load_legacy_solver():
    src = subprocess.run(["git", "show", "HEAD:solver.py"], cwd=HERE,
                         capture_output=True, text=True, check=True).stdout
    assert "w_s" not in src and "P_s" not in src \
        and "__post_init__" not in src, \
        "git HEAD:solver.py already contains the knobs -- wrong baseline"
    tmp = Path(tempfile.mkdtemp(prefix="e7audit_")) / "solver_legacy.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("solver_legacy_e7", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["solver_legacy_e7"] = mod    # dataclass resolves __module__
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# P2: bit-identity of the None defaults and of the gamma knob
# --------------------------------------------------------------------------

def probe_p2():
    print("[P2a] defaults-None vs stored pre-change reference npz")
    res0 = simulate(SimConfig(**e6_a1_kwargs()))
    ref = np.load(HERE / "out" / "struct_ref_A1.npz")
    for name in ("f", "s", "N_s", "omega"):
        check(np.array_equal(ref[name], getattr(res0, name)),
              f"defaults-None {name} == struct_ref_A1.npz")

    print("[P2b] defaults-None vs LIVE pre-change solver (git HEAD)")
    legacy = load_legacy_solver()
    res_leg = legacy.simulate(legacy.SimConfig(**e6_a1_kwargs()))
    bad = diff_results(res0, res_leg)
    check(not bad, f"all fields+scalars bit-identical to legacy run "
                   f"(diffs: {bad or 'none'})")
    for name in ("f", "s", "N_s", "omega"):
        check(np.array_equal(ref[name], getattr(res_leg, name)),
              f"stored npz provenance: legacy {name} == struct_ref_A1.npz")

    print("[P2c] gamma knob vs capture_form (opposite form set)")
    res_lf = res0                                          # capture_form='lf'
    res_af = simulate(SimConfig(**e6_a1_kwargs(capture_form="af")))
    res_g1 = simulate(SimConfig(**e6_a1_kwargs(capture_form="af", gamma=1.0)))
    res_g0 = simulate(SimConfig(**e6_a1_kwargs(capture_form="lf", gamma=0.0)))
    bad1 = diff_results(res_g1, res_lf)
    bad0 = diff_results(res_g0, res_af)
    check(not bad1, f"gamma=1.0 (form='af') == form='lf' (diffs: {bad1 or 'none'})")
    check(not bad0, f"gamma=0.0 (form='lf') == form='af' (diffs: {bad0 or 'none'})")
    check(diff_results(res_lf, res_af) != [],
          "sanity: 'lf' and 'af' runs actually differ")
    return legacy


# --------------------------------------------------------------------------
# P3: w_s = 0.7 w, P_s = P stress run -- ledger, invariants, flux bound
# --------------------------------------------------------------------------

def flux_recon(before, after, lam, left_bc):
    """Interface fluxes from the conservative update: F[0] = left_bc,
    F[k+1] = F[k] + (m_k - m_k_new)/lam."""
    F = np.empty(before.size + 1)
    F[0] = left_bc
    F[1:] = left_bc + np.cumsum(before - after) / lam
    return F


def probe_p3():
    print("[P3] w_s=0.7w, P_s=P stress run (q_in=2900/3600, u_xi=10)")
    cfg = SimConfig(v_f=V_F, w=W, P=P, q_in=2900.0 / 3600.0, u_xi=10.0,
                    kappa_c=0.026, kappa_r=3e-5, dt=0.5, save_every=1,
                    w_s=0.7 * W, P_s=P)
    res = simulate(cfg)

    nx = int(round(cfg.L_road / cfg.dx))
    lam = cfg.dt / cfg.dx
    n_steps = int(round(cfg.t_end / cfg.dt))
    f = np.zeros(nx)
    s = np.zeros(nx)
    inj = den = out = 0.0
    min_f = min_s = np.inf
    max_rho = -np.inf
    max_excess = -np.inf          # max over steps/faces of F_s - pi_s D_s
    min_Fs = np.inf
    max_qout_err = 0.0
    state_mismatch = 0

    for n in range(n_steps):
        t = n * cfg.dt
        if not (np.array_equal(f, res.f[n]) and np.array_equal(s, res.s[n])):
            state_mismatch += 1
        u_s = solver.u_s_of_t(cfg, t)
        a = solver.cav_density(cfg, t, nx)
        rho = a + f + s
        pi_s = np.where(rho > 0.0, s / np.maximum(rho, 1e-300), 0.0)
        D_s = solver.demand(rho, u_s, cfg.w_s, cfg.P_s)

        f_new, s_new, q_adm, q_out = solver.transport_step(
            f, s, a, cfg.v_f, u_s, cfg, t=t)

        F_s = flux_recon(s, s_new, lam, 0.0)
        F_f = flux_recon(f, f_new, lam, q_adm)
        max_excess = max(max_excess, float((F_s[1:] - pi_s * D_s).max()))
        min_Fs = min(min_Fs, float(F_s.min()))
        max_qout_err = max(max_qout_err, abs(F_f[-1] + F_s[-1] - q_out))

        f, s = f_new, s_new
        inj += q_adm * cfg.dt
        den += (cfg.q_in - q_adm) * cfg.dt
        out += q_out * cfg.dt

        a_star = solver.cav_density(cfg, t + cfg.dt, nx)
        rho_star = a_star + f + s
        min_f = min(min_f, f.min())            # post-transport
        min_s = min(min_s, s.min())
        max_rho = max(max_rho, rho_star.max())
        dv = np.maximum(solver.speed(rho_star, cfg.v_f, cfg.w, cfg.P) - u_s,
                        0.0)
        f, s = solver.reaction_exact(f, s, a_star, rho_star, dv, cfg.kappa_c,
                                     cfg.kappa_r, cfg.P, cfg.dt,
                                     cfg.capture_form, cfg.gamma)
        min_f = min(min_f, f.min())            # post-reaction
        min_s = min(min_s, s.min())

    check(state_mismatch == 0,
          "manual loop bit-identical to simulate() at every saved step")
    check(np.array_equal(f, res.f[-1]) and np.array_equal(s, res.s[-1]),
          "final state bit-identical to simulate()")
    check((inj, den, out) == (res.injected, res.denied_inflow,
                              res.outflowed),
          "ledger scalars bit-identical to simulate()")

    on_road = float(np.sum(f + s) * cfg.dx)
    rel = abs(inj - (on_road + out)) / max(inj, 1.0)
    print(f"    ledger: injected {inj:.6f} = on_road {on_road:.6f} + "
          f"outflowed {out:.6f} (rel err {rel:.2e}); denied {den:.6f}")
    check(rel < 1e-10, f"conservation ledger rel err {rel:.2e} < 1e-10")

    offered = cfg.q_in * cfg.t_end
    off_err = abs(offered - (inj + den))
    check(off_err < 1e-6 * offered, f"offered = injected + denied "
                                    f"(abs err {off_err:.2e})")

    print(f"    invariants over all {n_steps} steps (post-transport AND "
          f"post-reaction): min f {min_f:.3e}, min s {min_s:.3e}, "
          f"max rho {max_rho:.6f} (P = {P:.6f})")
    check(min_f >= 0.0, "f >= 0 at every substep")
    check(min_s >= 0.0, "s >= 0 at every substep")
    check(max_rho <= P + 1e-12, "rho <= P + 1e-12 at every substep")

    print(f"    s-class flux bound: max(F_s - pi_s D_s(w_s, P_s)) = "
          f"{max_excess:.3e} veh/s over all faces/steps; min F_s = "
          f"{min_Fs:.3e}; max |F_f[-1]+F_s[-1] - q_out| = {max_qout_err:.2e}")
    check(max_excess <= 1e-11,
          "s-class interface flux never exceeds pi_s * D_s(rho; c_s, w_s, P_s)")
    check(min_Fs >= -1e-11, "all reconstructed s-fluxes nonnegative")
    check(max_qout_err < 1e-12, "flux reconstruction closes on q_out")

    # context: realized wake density vs the s-class critical density
    r_star = float(cfg.w_s * cfg.P_s / (cfg.u_xi + cfg.w_s))
    rho_all = res.a + res.f + res.s
    print(f"    context: r*_s(0.7w, u10) = {r_star * 1000:.1f} veh/km, "
          f"realized max rho = {rho_all.max() * 1000:.1f} veh/km "
          f"(branch {'BINDS' if rho_all.max() > r_star else 'inert'})")


def probe_p3b():
    print("[P3b] transport_step clip branch (P_s = 0.8 P, rho > P_s block)")
    cfg = SimConfig(v_f=V_F, w=W, P=P, q_in=0.0, u_xi=10.0,
                    kappa_c=0.0, kappa_r=0.0, dt=0.5,
                    w_s=0.7 * W, P_s=0.8 * P)
    nx = 80
    lam = cfg.dt / cfg.dx
    f = np.full(nx, 0.005)
    s = np.full(nx, 0.02)
    s[30:50] = 0.25                       # rho = 0.255 < P, > P_s = 0.2133
    a = np.zeros(nx)
    rho = f + s
    assert rho.max() <= P, "probe precondition broke"
    c_s = 10.0
    raw_S = solver.supply(rho, c_s, cfg.w_s, cfg.P_s)
    check(float(raw_S.min()) < 0.0,
          f"raw s-supply goes negative (min {raw_S.min():.3e}) -> clip "
          "branch exercised")

    f_new, s_new, q_adm, q_out = solver.transport_step(
        f, s, a, cfg.v_f, c_s, cfg, t=None)
    F_s = flux_recon(s, s_new, lam, 0.0)
    pi_s = s / rho
    D_s = solver.demand(rho, c_s, cfg.w_s, cfg.P_s)
    check(float(F_s.min()) >= -1e-13, "all s-fluxes >= 0 despite raw S_s < 0")
    check(float((F_s[1:] - pi_s * D_s).max()) <= 1e-13,
          "flux bound pi_s * D_s(w_s, P_s) holds on the clip branch")
    check(f_new.min() >= 0.0 and s_new.min() >= 0.0,
          "f, s >= 0 after the clipped step")
    check(float((f_new + s_new).max()) <= P + 1e-12,
          "rho <= P after the clipped step")


# --------------------------------------------------------------------------
# P4: reaction layer never sees w_s / P_s
# --------------------------------------------------------------------------

def traced_single_step(cfg, f0, s0):
    """Run one simulate() step; capture every ARRAY call to solver.speed
    (i.e. the reaction dv computation -- the _save calls are scalar)."""
    calls = []
    orig = solver.speed

    def spy(rho, c, w, P):
        if np.asarray(rho).ndim == 1:
            calls.append((np.array(rho, float), float(c), float(w), float(P)))
        return orig(rho, c, w, P)

    solver.speed = spy
    try:
        res = simulate(cfg)
    finally:
        solver.speed = orig
    return res, calls


def probe_p4():
    print("[P4] reaction layer w_s-independence (traced solver.speed)")
    nx = 600
    x = np.arange(nx)
    common = dict(v_f=V_F, w=W, P=P, q_in=2500.0 / 3600.0, u_xi=10.0,
                  kappa_c=0.026, kappa_r=3e-5, dt=0.5, save_every=1,
                  t_enter=0.0, t_slow=0.0, t_fast=1e9, t_end=0.5)

    # (i) low-density frozen state: s-branch provably inert
    f0 = 0.020 * (1.0 + 0.5 * np.sin(x / 37.0))
    s0 = 0.015 * (1.0 + 0.5 * np.cos(x / 23.0))
    r_star = 0.7 * W * P / (10.0 + 0.7 * W)
    assert (f0 + s0).max() + 0.02 / 50.0 < r_star, "frozen state not inert"
    res_n, calls_n = traced_single_step(
        SimConfig(**common, f0=f0, s0=s0), f0, s0)
    res_w, calls_w = traced_single_step(
        SimConfig(**common, f0=f0, s0=s0, w_s=0.7 * W, P_s=P), f0, s0)
    check(len(calls_n) == 1 and len(calls_w) == 1,
          "exactly one array speed() call per step (the reaction dv)")
    (rho_n, c_n, w_n, P_n), (rho_w, c_w, w_w, P_w) = calls_n[0], calls_w[0]
    check((c_n, w_n, P_n) == (V_F, W, P),
          "dv call without w_s uses the road (v_f, w, P)")
    check((c_w, w_w, P_w) == (V_F, W, P),
          "dv call WITH w_s=0.7w, P_s=P still uses the road (v_f, w, P)")
    check(np.array_equal(rho_n, rho_w),
          "frozen state: rho_star fed to dv bit-identical with/without w_s")
    dv_n = np.maximum(solver.speed(rho_n, c_n, w_n, P_n) - 10.0, 0.0)
    dv_w = np.maximum(solver.speed(rho_w, c_w, w_w, P_w) - 10.0, 0.0)
    check(np.array_equal(dv_n, dv_w),
          "Delta v arrays bit-identical with and without w_s (frozen state)")
    bad = diff_results(res_n, res_w)
    check(not bad, f"inert regime: full step bit-identical (diffs: "
                   f"{bad or 'none'})")

    # (ii) dense frozen state: s-branch binds, transport differs, but the
    # reaction dv call still uses the road FD parameters
    s0d = np.full(nx, 0.02)
    s0d[100:300] = 0.12                    # rho ~ 0.13 > r*_s = 0.0808
    f0d = np.full(nx, 0.01)
    res_nd, calls_nd = traced_single_step(
        SimConfig(**common, f0=f0d, s0=s0d), f0d, s0d)
    res_wd, calls_wd = traced_single_step(
        SimConfig(**common, f0=f0d, s0=s0d, w_s=0.7 * W, P_s=P), f0d, s0d)
    check(diff_results(res_nd, res_wd) != [],
          "sanity: dense state -- w_s branch actually binds (outputs differ)")
    (_, c2, w2, P2) = calls_wd[0]
    check((c2, w2, P2) == (V_F, W, P),
          "dense state: reaction dv STILL uses road (v_f, w, P) under w_s")


# --------------------------------------------------------------------------
# P5: DM-G cap path regression vs out/ev4b_q2000/metrics.json
# --------------------------------------------------------------------------

def cmp_exact(got, want, path, errs):
    if isinstance(want, dict):
        if set(got) != set(want):
            errs.append(f"{path}: key sets differ ({set(got) ^ set(want)})")
            return
        for k in want:
            cmp_exact(got[k], want[k], f"{path}.{k}", errs)
    elif isinstance(want, list):
        if len(got) != len(want):
            errs.append(f"{path}: lengths differ")
            return
        for i, (g, v) in enumerate(zip(got, want)):
            cmp_exact(g, v, f"{path}[{i}]", errs)
    else:
        same = (got == want) or (isinstance(want, float) and
                                 np.isnan(want) and np.isnan(got))
        if not same:
            errs.append(f"{path}: {got!r} != stored {want!r}")


def probe_p5(legacy):
    print("[P5] DM-G cap regression (A1_u15_q2500, q_xi_max = 2000 veh/h)")
    import ev4_compare as ev4

    tag = "A1_u15_q2500"
    kap = json.loads((HERE / "out" / "ev3_kappa.json").read_text())
    kc = float(kap[tag]["kappa_cl_mod"][0])
    kr = float(kap[tag]["kappa_r_mod"][0])
    kwargs = dict(v_f=ev4.V_F, w=ev4.W, P=ev4.P, q_in=2500.0 / 3600.0,
                  u_xi=15.0, kappa_c=kc, kappa_r=kr, capture_form="lf",
                  dt=0.5, save_every=20, q_xi_max=2000.0 / 3600.0)
    res = simulate(SimConfig(**kwargs))

    res_leg = legacy.simulate(legacy.SimConfig(**kwargs))
    bad = diff_results(res, res_leg)
    check(not bad, f"capped run bit-identical to live pre-change solver "
                   f"(diffs: {bad or 'none'})")

    regr = ev4.regrid_sim(res)
    meas = ev4.load_measured(1.0, 15.0, 2500.0)
    met = ev4.metrics(regr, meas)
    got = dict(kappa_c=kc, kappa_r=kr, **met)
    stored = json.loads(
        (HERE / "out" / "ev4b_q2000" / "metrics.json").read_text())[tag]
    errs: list[str] = []
    cmp_exact(got, stored, tag, errs)
    for e in errs:
        print(f"    DIGIT DIFF {e}")
    check(not errs, f"all metrics digit-exact vs stored ev4b_q2000/"
                    f"metrics.json ({len(errs)} diffs)")
    print(f"    rho_rmse mean {met['rho_rmse']['mean']!r}")
    print(f"    Ns_mae   mean {met['Ns_mae']['mean']!r}")
    print(f"    omega    mean {met['omega_cum_rel_err']['mean']!r}")
    print(f"    e_s      mean {met['e_s']['mean']!r}")
    print(f"    cum_overtake_sim_veh {met['cum_overtake_sim_veh']!r}")


# --------------------------------------------------------------------------

def main():
    legacy = probe_p2()
    probe_p3()
    probe_p3b()
    probe_p4()
    probe_p5(legacy)
    print()
    if FAILURES:
        print(f"AUDIT: {len(FAILURES)} FAILURE(S):")
        for f_ in FAILURES:
            print(f"  - {f_}")
        sys.exit(1)
    print("AUDIT: all probes passed.")


if __name__ == "__main__":
    main()
