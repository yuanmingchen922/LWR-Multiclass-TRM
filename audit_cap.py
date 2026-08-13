"""Adversarial audit of the moving-bottleneck capacity cap in solver.py.

Run: python3 audit_cap.py    (from the analysis directory)

Independent probes of the E-V4b capacity-cap change (q_xi_max / beta in
SimConfig, capped face in transport_step), against the legacy behavior
reference q_xi_max=None:

  A1  full test_solver.py suite in a subprocess
  A2  bit-identity: q_xi_max=None twice (fresh runs, all SimResult fields);
      q_xi_max huge (armed, never binding) vs None; stored npz reference;
      optional legacy-module cross-check (AUDIT_LEGACY_SOLVER=/path/solver.py)
  A3  conservation ledger with the cap ON (A1 and A10 calibrated kappa),
      plus a single-step probe that the cap only rescales ONE interface flux
      (mass moved from flux to storage, never lost)
  A4  invariant domain with the cap ON under stress
      (q_in=2900/3600, u_xi=10, q_xi_max=1800/3600)
  A5  test-t8 analytic steady state recomputed independently (bisection of
      w (P - rho) - u_xi rho = omega_max, branch validity, tolerance margins)
  A6  cap OFF outside [t_slow, t_fast]: CAV-face flux untouched at t=200 s
      with q_xi_max set; positive control at t=400 s (binding, face capped)
  A7  omega_max sign guard: q_xi_max=500/3600 (omega_max clamps to 0),
      no negative fluxes, queue grows, no NaN

Plain asserts with printed diagnostics, mirroring test_solver.py style.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from solver import (SimConfig, simulate, transport_step, cav_density,
                    cav_position, u_s_of_t, capacity)

HERE = Path(__file__).parent

_cfg_fields = {fld.name for fld in dataclasses.fields(SimConfig)}
assert {"q_xi_max", "beta"} <= _cfg_fields, (
    "solver.py has no capacity cap (SimConfig lacks q_xi_max/beta): this "
    "audit targets the E-V4b capped build -- apply ev4b_solver.patch and "
    "ev4b_test_solver.patch first")

_params = json.loads((HERE / "out" / "params.json").read_text())
V_F = _params["v_f_kmh"] / 3.6       # 27.94 m/s
W = _params["w_kmh"] / 3.6           # 6.21 m/s
P = _params["P_vehkm"] / 1000.0      # 0.2667 veh/m

_FIELDS = ("t", "x", "a", "f", "s", "x_cav", "omega", "N_s",
           "denied_inflow", "injected", "outflowed", "on_road")


def base_cfg(**kw) -> SimConfig:
    d = dict(v_f=V_F, w=W, P=P, q_in=2500.0 / 3600.0, u_xi=15.0,
             kappa_c=0.026, kappa_r=3e-5)
    d.update(kw)
    return SimConfig(**d)


def ref_cfg(**kw) -> SimConfig:
    """A1 reference config of the E-V4b bit-identity check (test t7)."""
    d = dict(kappa_c=0.0307, kappa_r=0.0, u_xi=15.0,
             q_in=2500.0 / 3600.0, dt=0.5, save_every=20)
    d.update(kw)
    return base_cfg(**d)


def t8_cfg(**kw) -> SimConfig:
    """Pure-transport bottleneck config of test t8."""
    d = dict(kappa_c=0.0, kappa_r=0.0, u_xi=15.0, dt=0.5, save_every=20,
             q_xi_max=2000.0 / 3600.0)
    d.update(kw)
    return base_cfg(**d)


def _cmp_results(ra, rb, label: str) -> None:
    """Field-by-field bit comparison of two SimResult objects."""
    for name in _FIELDS:
        va, vb = getattr(ra, name), getattr(rb, name)
        if isinstance(va, np.ndarray):
            same = np.array_equal(va, vb, equal_nan=True)
        else:
            same = va == vb
        assert same, f"{label}: field '{name}' differs"
    print(f"  {label}: all {len(_FIELDS)} SimResult fields bit-identical")


def _face_fluxes(f, s, a, cfg: SimConfig, t: float):
    """One transport step plus the telescoped interface flux array
    F[0..nx] [veh/s] reconstructed from the conservative update."""
    fn, sn, q_adm, q_out = transport_step(f, s, a, cfg.v_f,
                                          u_s_of_t(cfg, t), cfg, t=t)
    lam = cfg.dt / cfg.dx
    F = np.empty(f.size + 1)
    F[0] = q_adm
    F[1:] = q_adm - np.cumsum((fn - f) + (sn - s)) / lam
    return fn, sn, q_adm, q_out, F


def _state_at(res, cfg: SimConfig, t: float):
    """Saved (f, s, a) at the save instant closest to t."""
    i = int(np.argmin(np.abs(res.t - t)))
    ti = float(res.t[i])
    assert abs(ti - t) < 1e-9, f"t={t} not on the save grid (nearest {ti})"
    return res.f[i].copy(), res.s[i].copy(), cav_density(cfg, ti, res.x.size)


# --------------------------------------------------------------------------
# A1. Full existing test suite
# --------------------------------------------------------------------------

def audit_suite():
    proc = subprocess.run([sys.executable, str(HERE / "test_solver.py")],
                          capture_output=True, text=True, cwd=HERE)
    tail = "\n".join(proc.stdout.strip().splitlines()[-3:])
    print(f"  exit code {proc.returncode}; tail:\n    "
          + tail.replace("\n", "\n    "))
    if proc.returncode != 0:
        print("  --- stderr ---\n" + proc.stderr[-2000:])
    assert proc.returncode == 0, "test_solver.py suite failed"
    assert "All 9 tests passed." in proc.stdout, \
        "suite did not report all 9 tests passing"


# --------------------------------------------------------------------------
# A2. Independent bit-identity of the cap-off path
# --------------------------------------------------------------------------

def audit_bit_identity():
    res_none = simulate(ref_cfg(q_xi_max=None))

    # (a) fresh rerun of the identical q_xi_max=None config
    _cmp_results(res_none, simulate(ref_cfg(q_xi_max=None)),
                 "None vs fresh None rerun")

    # (b) cap armed but never binding: the added block must not perturb a
    # single bit when the constraint test fails (q_xi_max huge)
    _cmp_results(res_none, simulate(ref_cfg(q_xi_max=1e9)),
                 "None vs armed-but-never-binding (q_xi_max=1e9)")

    # (c) stored uncapped reference (provenance: pre-change solver)
    ref_path = HERE / "out" / "ev4b_ref_A1.npz"
    assert ref_path.exists(), (
        "out/ev4b_ref_A1.npz missing -- generate it from the PRE-change "
        "solver (ev4b_make_ref.py) before trusting t7: without it the test "
        "self-seeds from the current build and passes trivially")
    ref = np.load(ref_path)
    for name in ("f", "s", "N_s", "omega"):
        assert np.array_equal(ref[name], getattr(res_none, name)), \
            f"{name} differs from the stored uncapped reference"
    print("  None run matches stored ev4b_ref_A1.npz (f, s, N_s, omega)")

    # (d) optional cross-check against a separate legacy solver module
    leg_path = os.environ.get("AUDIT_LEGACY_SOLVER", "")
    if leg_path and Path(leg_path).exists():
        spec = importlib.util.spec_from_file_location("legacy_solver",
                                                      leg_path)
        leg = importlib.util.module_from_spec(spec)
        sys.modules["legacy_solver"] = leg
        spec.loader.exec_module(leg)
        lcfg = leg.SimConfig(v_f=V_F, w=W, P=P, q_in=2500.0 / 3600.0,
                             u_xi=15.0, kappa_c=0.0307, kappa_r=0.0,
                             dt=0.5, save_every=20)
        _cmp_results(res_none, leg.simulate(lcfg),
                     "None vs legacy module " + Path(leg_path).name)
    else:
        print("  (legacy-module cross-check skipped: AUDIT_LEGACY_SOLVER "
              "not set)")


# --------------------------------------------------------------------------
# A3. Conservation ledger with the cap ON; cap only rescales one face
# --------------------------------------------------------------------------

def audit_ledger_cap_on():
    kap = json.loads((HERE / "out" / "ev3_kappa.json").read_text())
    for label in ("A1_u15_q2500", "A10_u15_q2500"):
        kc = kap[label]["kappa_cl_mod"][0]
        kr = kap[label]["kappa_r_mod"][0]
        cfg = ref_cfg(kappa_c=kc, kappa_r=kr, q_xi_max=2000.0 / 3600.0)
        res = simulate(cfg)
        rel = abs(res.injected - (res.on_road + res.outflowed)) \
            / max(res.injected, 1.0)
        offered = cfg.q_in * cfg.t_end
        off_err = abs(offered - (res.injected + res.denied_inflow))
        print(f"  {label} (kc={kc:.4f}, kr={kr:g}): injected "
              f"{res.injected:.6f} = on_road {res.on_road:.6f} + outflowed "
              f"{res.outflowed:.6f}, rel err {rel:.2e}; offered abs err "
              f"{off_err:.2e}")
        assert rel < 1e-10, f"{label}: ledger not closed to 1e-10 with cap on"
        assert off_err < 1e-10 * max(offered, 1.0)

    # single-step probe: exactly ONE interface flux changes, downward, and
    # the per-step mass identity holds for both the capped and uncapped step
    cfg = t8_cfg()
    res = simulate(cfg)
    f5, s5, a5 = _state_at(res, cfg, 500.0)
    fn_c, sn_c, qa_c, qo_c, F_c = _face_fluxes(f5, s5, a5, cfg, 500.0)
    cfg_off = dataclasses.replace(cfg, q_xi_max=None)
    fn_u, sn_u, qa_u, qo_u, F_u = _face_fluxes(f5, s5, a5, cfg_off, 500.0)

    dF = F_c - F_u
    changed = np.where(np.abs(dF) > 1e-15)[0]
    j = int(cav_position(cfg, 500.0) / cfg.dx)
    print(f"  single step at t=500: faces changed by the cap: "
          f"{changed.tolist()} (CAV face j+1 = {j + 1}), "
          f"dF = {dF[changed][0] if changed.size else 0.0:+.6f} veh/s")
    assert changed.size == 1 and changed[0] == j + 1, \
        "cap touched a face other than the CAV downstream face"
    assert dF[j + 1] < 0.0, "cap increased the face flux"
    for tag, (fn, sn, qa, qo) in (("capped", (fn_c, sn_c, qa_c, qo_c)),
                                  ("uncapped", (fn_u, sn_u, qa_u, qo_u))):
        gain = (np.sum(fn + sn) - np.sum(f5 + s5)) * cfg.dx
        err = abs(gain - (qa - qo) * cfg.dt)
        print(f"  {tag}: step mass gain {gain:.9f} veh vs (q_adm - q_out) dt "
              f"{(qa - qo) * cfg.dt:.9f}, abs err {err:.2e}")
        assert err < 1e-12, f"{tag} step lost mass"
    # mass withheld at the interior face j+1 stays in cell j and is missing
    # from cell j+1 in equal measure -- nothing lost, boundaries untouched
    lam = cfg.dt / cfg.dx
    dcell = (fn_c + sn_c) - (fn_u + sn_u)
    withheld = -dF[j + 1] * lam            # veh/m kept in cell j
    assert abs(dcell[j] - withheld) < 1e-15, "cell j did not store the mass"
    assert abs(dcell[j + 1] + withheld) < 1e-15, \
        "cell j+1 deficit does not match the withheld flux"
    others = np.delete(dcell, [j, j + 1])
    assert np.all(others == 0.0), "cap perturbed cells away from the face"
    assert qa_c == qa_u and qo_c == qo_u, "cap changed boundary fluxes"
    print(f"  withheld {withheld * cfg.dx:.9f} veh stayed in cell {j} and is "
          f"missing from cell {j + 1} exactly; all other cells and both "
          "boundary fluxes bit-identical")


# --------------------------------------------------------------------------
# A4. Invariant domain with the cap ON under stress
# --------------------------------------------------------------------------

def audit_invariant_stress():
    cfg = base_cfg(q_in=2900.0 / 3600.0, u_xi=10.0,
                   q_xi_max=1800.0 / 3600.0)
    res = simulate(cfg)
    rho = res.a + res.f + res.s
    for name in ("a", "f", "s", "omega", "N_s"):
        assert np.all(np.isfinite(getattr(res, name))), f"{name} has NaN/inf"
    print(f"  min f {res.f.min():.3e}, min s {res.s.min():.3e}, "
          f"max rho {rho.max():.6f} (P = {P:.6f}), max N_s "
          f"{res.N_s.max():.2f} veh, all fields finite")
    assert res.f.min() >= 0.0, "f went negative under stress with cap"
    assert res.s.min() >= 0.0, "s went negative under stress with cap"
    assert rho.max() <= P + 1e-12, "rho exceeded jam density with cap"


# --------------------------------------------------------------------------
# A5. Test-t8 analytic steady state, recomputed independently
# --------------------------------------------------------------------------

def audit_t8_analytic():
    u_xi, q_xi = 15.0, 2000.0 / 3600.0
    cfg = t8_cfg()
    sigma_xi = cfg.beta * W * P / (V_F + W)
    omega_max = max(q_xi - u_xi * sigma_xi, 0.0)

    # independent root of  w (P - rho) - u_xi rho = omega_max  by bisection
    g = lambda r: W * (P - r) - u_xi * r - omega_max
    lo, hi = 0.0, P
    assert g(lo) > 0.0 > g(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if g(mid) > 0.0 else (lo, mid)
    rho_bis = 0.5 * (lo + hi)
    rho_closed = (W * P - omega_max) / (W + u_xi)     # test t8's formula
    print(f"  rho_minus bisection {rho_bis:.12f} vs closed form "
          f"{rho_closed:.12f} (|diff| {abs(rho_bis - rho_closed):.2e})")
    assert abs(rho_bis - rho_closed) < 1e-12, "t8 closed form wrong"

    # branch validity and flux identities
    rho_crit = W * P / (V_F + W)
    rho_hat = omega_max / (V_F - u_xi)
    q_minus = omega_max + u_xi * rho_closed
    assert rho_closed > rho_crit, "rho_minus not on the congested branch"
    assert rho_hat < rho_crit, "wake state not on the free branch"
    assert abs(q_minus - W * (P - rho_closed)) < 1e-12
    assert abs(V_F * rho_hat - (omega_max + u_xi * rho_hat)) < 1e-12
    assert q_minus < capacity(V_F, W, P)
    rho_in = cfg.q_in / V_F
    assert cfg.q_in - u_xi * rho_in > omega_max, \
        "constraint would not bind -- t8 would measure free flow"
    print(f"  branches OK: rho_crit {rho_crit:.6f} < rho_minus "
          f"{rho_closed:.6f}, rho_hat {rho_hat:.6f} free; relative demand "
          f"{cfg.q_in - u_xi * rho_in:.4f} > omega_max {omega_max:.4f} "
          "(binding)")

    # rerun the t8 measurements and report margins against its tolerances
    res = simulate(cfg)
    i700 = int(np.argmin(np.abs(res.t - 700.0)))
    xc = res.x_cav[i700]
    rho7 = res.a[i700] + res.f[i700] + res.s[i700]
    m = (res.x >= xc - 1000.0) & (res.x <= xc - 200.0)
    rel_rho = abs(rho7[m].mean() - rho_closed) / rho_closed

    f7, s7, a7 = _state_at(res, cfg, 700.0)
    *_, F = _face_fluxes(f7, s7, a7, cfg, 700.0)
    j = int(xc / cfg.dx)
    rel_q = abs(F[j - 3:j].mean() - q_minus) / q_minus

    z = (q_minus - cfg.q_in) / (rho_closed - rho_in)
    sel = np.where((res.t >= 350.0) & (res.t <= 650.0))[0]
    tails = []
    for i in sel:
        r = res.a[i] + res.f[i] + res.s[i]
        ks = np.where(r > 0.6 * rho_closed)[0]
        tails.append(res.x[ks[0]] if ks.size else np.nan)
    tails = np.asarray(tails)
    ok = np.isfinite(tails)
    slope = np.polyfit(res.t[sel][ok], tails[ok], 1)[0]
    rel_z = abs(slope - z) / abs(z)

    for tag, rel, tol in (("rho_minus", rel_rho, 0.05),
                          ("q_minus", rel_q, 0.03),
                          ("tail shock", rel_z, 0.15)):
        print(f"  {tag}: rel err {100 * rel:.2f}% vs t8 tolerance "
              f"{100 * tol:.0f}% (margin x{tol / max(rel, 1e-12):.1f})")
        assert rel < tol, f"{tag} outside t8 tolerance"


# --------------------------------------------------------------------------
# A6. Cap OFF outside [t_slow, t_fast]
# --------------------------------------------------------------------------

def audit_cap_off_outside_window():
    cfg = t8_cfg()
    res = simulate(cfg)
    cfg_off = dataclasses.replace(cfg, q_xi_max=None)

    # t = 200 s: CAV on the road (entered t=100) but before t_slow=250
    f2, s2, a2 = _state_at(res, cfg, 200.0)
    fn_c, sn_c, qa_c, qo_c, F_c = _face_fluxes(f2, s2, a2, cfg, 200.0)
    fn_u, sn_u, qa_u, qo_u, F_u = _face_fluxes(f2, s2, a2, cfg_off, 200.0)
    j = int(cav_position(cfg, 200.0) / cfg.dx)
    assert F_c[j + 1] > 0.0, "no traffic at the CAV face -- probe is vacuous"
    assert (np.array_equal(fn_c, fn_u) and np.array_equal(sn_c, sn_u)
            and qa_c == qa_u and qo_c == qo_u), \
        "cap altered the transport step at t=200 (outside the window)"
    print(f"  t=200: CAV cell {j}, face flux {F_c[j + 1]:.6f} veh/s, "
          "bit-identical to the q_xi_max=None step (cap correctly off)")

    # positive control, t = 400 s (inside window, binding): the same probe
    # must detect the cap, otherwise the t=200 equality proves nothing
    f4, s4, a4 = _state_at(res, cfg, 400.0)
    *_, F_c4 = _face_fluxes(f4, s4, a4, cfg, 400.0)
    *_, F_u4 = _face_fluxes(f4, s4, a4, cfg_off, 400.0)
    j4 = int(cav_position(cfg, 400.0) / cfg.dx)
    print(f"  t=400 control: face {j4 + 1} capped {F_c4[j4 + 1]:.6f} < "
          f"uncapped {F_u4[j4 + 1]:.6f} veh/s")
    assert F_c4[j4 + 1] < F_u4[j4 + 1], \
        "positive control failed: cap not binding at t=400"


# --------------------------------------------------------------------------
# A7. omega_max sign guard (q_xi_max small)
# --------------------------------------------------------------------------

def audit_small_cap_guard():
    q_xi = 500.0 / 3600.0
    cfg = t8_cfg(q_xi_max=q_xi)
    sigma_xi = cfg.beta * W * P / (V_F + W)
    omega_max = max(q_xi - cfg.u_xi * sigma_xi, 0.0)
    print(f"  q_xi_max - u_xi sigma_xi = "
          f"{q_xi - cfg.u_xi * sigma_xi:.4f} veh/s -> omega_max clamps to "
          f"{omega_max:g} (full blockage)")
    assert omega_max == 0.0

    res = simulate(cfg)
    rho = res.a + res.f + res.s
    for name in ("f", "s", "N_s", "omega"):
        assert np.all(np.isfinite(getattr(res, name))), f"{name} has NaN"
    assert res.f.min() >= 0.0 and res.s.min() >= 0.0
    assert rho.max() <= P + 1e-12
    rel = abs(res.injected - (res.on_road + res.outflowed)) \
        / max(res.injected, 1.0)
    assert rel < 1e-10, "ledger not closed with omega_max = 0"

    # no negative interface fluxes; the CAV face is exactly zero
    f5, s5, a5 = _state_at(res, cfg, 500.0)
    *_, F = _face_fluxes(f5, s5, a5, cfg, 500.0)
    j = int(cav_position(cfg, 500.0) / cfg.dx)
    print(f"  t=500 step: min face flux {F.min():.3e} veh/s, CAV face "
          f"{F[j + 1]:.3e} (expected exactly 0), ledger rel err {rel:.2e}")
    assert F.min() >= -1e-13, "negative interface flux"
    assert abs(F[j + 1]) < 1e-13, "blocked CAV face flux not zero"

    # the queue behind the CAV grows monotonically through the window
    sel = np.where((res.t >= 300.0) & (res.t <= 700.0))[0]
    mass_up = []
    for i in sel:
        jc = int(res.x_cav[i] / cfg.dx)
        mass_up.append(np.sum(res.f[i][:jc + 1] + res.s[i][:jc + 1])
                       * cfg.dx)
    mass_up = np.asarray(mass_up)
    growth = np.diff(mass_up)
    rho_expect = W * P / (W + cfg.u_xi)     # queue state for omega_max = 0
    i700 = int(np.argmin(np.abs(res.t - 700.0)))
    xc = res.x_cav[i700]
    m = (res.x >= xc - 1000.0) & (res.x <= xc - 200.0)
    rho_q = (res.a[i700] + res.f[i700] + res.s[i700])[m].mean()
    print(f"  upstream mass 300->700 s: {mass_up[0]:.2f} -> "
          f"{mass_up[-1]:.2f} veh (min step {growth.min():.3e}); queue "
          f"density {rho_q:.6f} vs analytic wP/(w+u) = {rho_expect:.6f} "
          f"({100 * abs(rho_q - rho_expect) / rho_expect:.2f}% off)")
    assert np.all(growth > 0.0), "queue behind the blocked CAV not growing"
    assert abs(rho_q - rho_expect) / rho_expect < 0.05


# --------------------------------------------------------------------------

if __name__ == "__main__":
    audits = [
        ("A1 full test suite", audit_suite),
        ("A2 cap-off bit-identity", audit_bit_identity),
        ("A3 ledger with cap on", audit_ledger_cap_on),
        ("A4 invariant domain under stress", audit_invariant_stress),
        ("A5 t8 analytic steady state", audit_t8_analytic),
        ("A6 cap off outside window", audit_cap_off_outside_window),
        ("A7 omega_max sign guard", audit_small_cap_guard),
    ]
    for name, fn in audits:
        print(f"[{name}]")
        fn()
        print("  PASS")
    print(f"\nAll {len(audits)} audits passed.")
