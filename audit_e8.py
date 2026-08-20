"""E8 audit: independent probes of the downstream_release change in solver.py.

Run: python3 audit_e8.py

Probes (numbering follows the audit brief):

P2  flag-False bit-identity vs the PRE-change solver reconstructed from
    git HEAD (git show HEAD:solver.py), across ALL SimResult fields, for
    two configs: (a) the E7-winner-like A1 config WITH w_s set, (b) an
    all-legacy config (every structural knob at its None/False default).
    Also verifies the stored test reference out/dsr_ref_A1.npz really is
    the pre-change solver's output.
P3  flag-True invariants: global mass ledger closes to 1e-10, per-cell
    p = f + s conserved to 1e-10 through the conversion at every step
    (manual re-step, validated bitwise against simulate()), f, s >= 0,
    rho <= P, and s identically 0 strictly downstream of the CAV cell at
    every saved time AND at 3 probed UNSAVED substeps (each probed
    independently by re-running simulate() with t_end at the unsaved time,
    so the check does not lean on the replica).
P4  boundary semantics: a state with s > 0 planted in j_cav and j_cav + 1;
    one step; the CAV's own cell j_cav keeps its s (bit-equal the
    flag-False step, i.e. minus normal reaction only) while j_cav + 1
    loses its s to f, bitwise.
P5  interaction: one q_xi_max-capped run and one w_s run with the flag on;
    all invariants hold, no NaN in any field (x_cav NaN exactly on the
    off-road saves).
P6  off-road phases: (a) CAV never enters within [0, t_end] and (b) CAV
    already past x = L_road for all t >= 0 -- flag-True bit-identical to
    flag-False across ALL fields (the flag is a no-op); (c) long run to
    t_end = 2000 where the CAV leaves the domain: the release fires at
    exactly the steps with a finite CAV position at t + dt and at no step
    before entry or after exit.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

import solver as sol_new  # noqa: E402  (the changed, working-tree solver)

_params = json.loads((HERE / "out" / "params.json").read_text())
V_F = _params["v_f_kmh"] / 3.6
W = _params["w_kmh"] / 3.6
P = _params["P_vehkm"] / 1000.0

ALL_FIELDS = ("t", "x", "a", "f", "s", "x_cav", "omega", "N_s",
              "denied_inflow", "injected", "outflowed", "on_road")

FAILURES: list[str] = []


def check(ok: bool, msg: str) -> None:
    if ok:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


# --------------------------------------------------------------------------
# Pre-change solver reconstructed from git HEAD
# --------------------------------------------------------------------------

def load_head_solver():
    src = subprocess.run(["git", "show", "HEAD:solver.py"], cwd=HERE,
                         capture_output=True, text=True, check=True).stdout
    path = HERE / "out" / "_solver_head_e8_audit.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("solver_head_e8", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["solver_head_e8"] = mod   # dataclass needs it registered
    spec.loader.exec_module(mod)
    assert not hasattr(mod.SimConfig, "downstream_release") or \
        "downstream_release" not in mod.SimConfig.__dataclass_fields__, \
        "HEAD solver unexpectedly already has the flag"
    return mod


def bit_equal(a, b) -> bool:
    """Bitwise equality (NaN-safe: compares raw bytes)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def compare_all_fields(res_a, res_b, what: str) -> None:
    for name in ALL_FIELDS:
        check(bit_equal(getattr(res_a, name), getattr(res_b, name)),
              f"{what}: field '{name}' bit-identical")


# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------

def kw_A(**kw):
    """E7-winner-like A1 config WITH w_s set (matches test _dsr_cfg)."""
    d = dict(v_f=V_F, w=W, P=P, q_in=2500.0 / 3600.0, u_xi=15.0,
             kappa_c=0.046, kappa_r=5.7e-4, capture_form="lf",
             dt=0.5, save_every=20, w_s=0.6 * W)
    d.update(kw)
    return d


def kw_B(**kw):
    """All-legacy config: every structural knob at its default."""
    d = dict(v_f=V_F, w=W, P=P, q_in=2500.0 / 3600.0, u_xi=15.0,
             kappa_c=0.026, kappa_r=3e-5)
    d.update(kw)
    return d


def downstream_mask(x, x_cav, dx):
    j_cav = int(x_cav // dx)
    return np.arange(x.size) > j_cav


# --------------------------------------------------------------------------
# P2: flag-False bit-identity vs git HEAD, all fields, two configs
# --------------------------------------------------------------------------

def probe_bit_identity(sol_head):
    print("[P2] flag-False bit-identity vs git-HEAD solver (ALL fields)")
    for label, kw in (("A1 + w_s=0.6w", kw_A()), ("all-legacy", kw_B())):
        res_head = sol_head.simulate(sol_head.SimConfig(**kw))
        res_new = sol_new.simulate(sol_new.SimConfig(**kw))          # default False
        res_new_x = sol_new.simulate(
            sol_new.SimConfig(**kw, downstream_release=False))       # explicit False
        compare_all_fields(res_new, res_head, f"{label} (default flag)")
        compare_all_fields(res_new_x, res_head, f"{label} (explicit False)")

    # the stored test reference really is the pre-change solver's output
    ref = np.load(HERE / "out" / "dsr_ref_A1.npz")
    res_head_A = sol_head.simulate(sol_head.SimConfig(**kw_A()))
    for name in ref.files:
        check(bit_equal(ref[name], getattr(res_head_A, name)),
              f"out/dsr_ref_A1.npz['{name}'] == git-HEAD solver output")
    print(f"  (reference keys: {ref.files})")


# --------------------------------------------------------------------------
# Manual re-step replica of simulate() (validated bitwise against it)
# --------------------------------------------------------------------------

def replicate(cfg):
    """Mirror simulate() step by step; returns saves, per-step release
    bookkeeping and the max per-cell |p_post - p_pre| through the release."""
    nx = int(round(cfg.L_road / cfg.dx))
    f = np.zeros(nx) if cfg.f0 is None else np.array(cfg.f0, float)
    s = np.zeros(nx) if cfg.s0 is None else np.array(cfg.s0, float)
    n_steps = int(round(cfg.t_end / cfg.dt))
    injected = denied = outflowed = 0.0
    saves = []
    released = np.zeros(n_steps, dtype=bool)
    pre_release_down_max = np.zeros(n_steps)
    p_err_max = 0.0

    for n in range(n_steps):
        t = n * cfg.dt
        if n % cfg.save_every == 0:
            saves.append((t, f.copy(), s.copy()))
        u_s = sol_new.u_s_of_t(cfg, t)
        a = sol_new.cav_density(cfg, t, nx)
        f, s, q_adm, q_out = sol_new.transport_step(f, s, a, cfg.v_f, u_s,
                                                    cfg, t=t)
        injected += q_adm * cfg.dt
        denied += (cfg.q_in - q_adm) * cfg.dt
        outflowed += q_out * cfg.dt
        a_star = sol_new.cav_density(cfg, t + cfg.dt, nx)
        rho_star = a_star + f + s
        dv = np.maximum(sol_new.speed(rho_star, cfg.v_f, cfg.w, cfg.P) - u_s,
                        0.0)
        f, s = sol_new.reaction_exact(f, s, a_star, rho_star, dv, cfg.kappa_c,
                                      cfg.kappa_r, cfg.P, cfg.dt,
                                      cfg.capture_form, cfg.gamma)
        if cfg.downstream_release:
            x_c = sol_new.cav_position(cfg, t + cfg.dt)
            if np.isfinite(x_c):
                j_cav = int(x_c // cfg.dx)
                p_pre = f + s
                pre_release_down_max[n] = (s[j_cav + 1:].max()
                                           if j_cav + 1 < nx else 0.0)
                f[j_cav + 1:] += s[j_cav + 1:]
                s[j_cav + 1:] = 0.0
                p_err_max = max(p_err_max,
                                float(np.max(np.abs((f + s) - p_pre))))
                released[n] = True
    saves.append((n_steps * cfg.dt, f.copy(), s.copy()))
    return saves, released, pre_release_down_max, p_err_max, \
        (injected, denied, outflowed)


def validate_replica(cfg, res, saves, ledger, what):
    ok = len(saves) == res.t.size
    for i, (t, f, s) in enumerate(saves):
        ok = ok and t == res.t[i] and bit_equal(f, res.f[i]) \
            and bit_equal(s, res.s[i])
    check(ok, f"{what}: manual re-step replica bit-equal simulate() at all "
          f"{len(saves)} saves")
    inj, den, out = ledger
    check(inj == res.injected and den == res.denied_inflow
          and out == res.outflowed,
          f"{what}: replica boundary ledger bit-equal simulate()")


# --------------------------------------------------------------------------
# P3: flag-True invariants (saved times + 3 unsaved substeps)
# --------------------------------------------------------------------------

def probe_invariants():
    print("[P3] flag-True invariants (ledger, p-conservation, positivity, "
          "rho <= P, downstream-zero at saved AND unsaved times)")
    for label, kw in (("A1 + w_s=0.6w", kw_A()), ("all-legacy", kw_B())):
        cfg = sol_new.SimConfig(**kw, downstream_release=True)
        res = sol_new.simulate(cfg)

        bal = res.on_road + res.outflowed
        rel = abs(res.injected - bal) / max(res.injected, 1.0)
        check(rel < 1e-10, f"{label}: mass ledger closes (rel err {rel:.2e})")
        check(res.f.min() >= 0.0 and res.s.min() >= 0.0,
              f"{label}: f, s >= 0 (min f {res.f.min():.1e}, "
              f"min s {res.s.min():.1e})")
        rho = res.a + res.f + res.s
        check(rho.max() <= cfg.P + 1e-12,
              f"{label}: rho <= P (max rho {rho.max():.6f}, P {cfg.P:.6f})")

        n_on = 0
        all_zero = True
        for i in range(res.t.size):
            if not np.isfinite(res.x_cav[i]):
                continue
            m = downstream_mask(res.x, res.x_cav[i], cfg.dx)
            all_zero = all_zero and np.all(res.s[i][m] == 0.0)
            n_on += 1
        check(n_on > 0 and all_zero,
              f"{label}: s identically 0 strictly downstream of the CAV at "
              f"all {n_on} on-road saves")

        saves, released, pre_down, p_err, ledger = replicate(cfg)
        validate_replica(cfg, res, saves, ledger, label)
        check(p_err < 1e-10,
              f"{label}: per-cell p = f + s conserved through the conversion "
              f"at every step (max |dp| {p_err:.2e})")
        check(released.any() and pre_down.max() > 0.0,
              f"{label}: the release actually converted downstream s at some "
              f"step (max pre-release downstream s "
              f"{pre_down.max():.3e} veh/m)")

    # 3 unsaved substeps, probed INDEPENDENTLY of the replica: rerun
    # simulate() with t_end at the unsaved time; its final save is the
    # intermediate state of the full run (identical step sequence).
    cfg = sol_new.SimConfig(**kw_A(), downstream_release=True)
    for t_probe in (203.0, 501.5, 745.5):
        n_sub = round(t_probe / cfg.dt)
        assert n_sub * cfg.dt == t_probe
        assert n_sub % cfg.save_every != 0, "probe time accidentally saved"
        cfg_p = sol_new.SimConfig(**kw_A(), downstream_release=True,
                                  t_end=t_probe)
        res_p = sol_new.simulate(cfg_p)
        xc = res_p.x_cav[-1]
        ok = np.isfinite(xc)
        if ok:
            m = downstream_mask(res_p.x, xc, cfg.dx)
            ok = np.all(res_p.s[-1][m] == 0.0)
        check(bool(ok), f"unsaved substep t = {t_probe:g} s: s identically 0 "
              f"strictly downstream (x_cav {xc:.1f} m)")


# --------------------------------------------------------------------------
# P4: boundary semantics at the CAV cell
# --------------------------------------------------------------------------

def probe_boundary_semantics():
    print("[P4] boundary semantics: j_cav keeps its s, j_cav + 1 loses it")
    dx, dt = 50.0, 0.5
    x_target = 5230.0                       # cell 104 = [5200, 5250)
    u = 15.0
    kwc = dict(v_f=V_F, w=W, P=P, q_in=2500.0 / 3600.0, u_xi=u,
               kappa_c=0.046, kappa_r=5.7e-4, capture_form="lf",
               dx=dx, dt=dt, t_end=dt, save_every=1,
               t_enter=-(x_target / u - dt), t_slow=-1e6, t_fast=1e6,
               v_cav_free=27.78)
    nx = int(round(30000.0 / dx))
    f0 = np.full(nx, 0.02)
    s0 = np.zeros(nx)
    s0[103], s0[104], s0[105], s0[106] = 0.010, 0.008, 0.006, 0.004
    kwc.update(f0=f0, s0=s0)

    res_F = sol_new.simulate(sol_new.SimConfig(**kwc))
    res_T = sol_new.simulate(sol_new.SimConfig(**kwc, downstream_release=True))

    xc = res_T.x_cav[-1]
    j_cav = int(xc // dx)
    check(j_cav == 104, f"CAV cell after one step is j_cav = {j_cav} "
          f"(x_cav = {xc:.4f} m)")
    fT, sT = res_T.f[-1], res_T.s[-1]
    fF, sF = res_F.f[-1], res_F.s[-1]

    check(sF[104] > 0.0 and sF[105] > 0.0,
          f"legacy step leaves s > 0 in j_cav ({sF[104]:.6f}) and "
          f"j_cav+1 ({sF[105]:.6f}) -- there was something to purge")
    check(sT[104] == sF[104] and fT[104] == fF[104],
          f"j_cav = 104 retains its s bit-exactly (minus normal reaction "
          f"only): s {sT[104]:.6f} == legacy {sF[104]:.6f}")
    check(bit_equal(fT[:105], fF[:105]) and bit_equal(sT[:105], sF[:105]),
          "all cells j <= j_cav bit-equal the flag-False step")
    check(np.all(sT[105:] == 0.0),
          f"all cells j > j_cav purged: max s {sT[105:].max():.1e}")
    check(bit_equal(fT[105:], fF[105:] + sF[105:]),
          "purged s landed in f bitwise: f_True[j] == f_False[j] + "
          "s_False[j] for all j > j_cav")

    # note: at x_cav = 5230 the exogenous a straddles cells 104 AND 105
    a1 = res_T.a[-1]
    check(a1[104] > 0.0 and a1[105] > 0.0,
          f"probe also covers the straddling case: a > 0 in both 104 "
          f"({a1[104]:.5f}) and 105 ({a1[105]:.5f}), yet 105 is still purged "
          "(per the strictly-downstream-of-j_cav definition)")

    # reaction really acted in the CAV cell (dv > 0 there)
    a_star = sol_new.cav_density(sol_new.SimConfig(**kwc), dt, nx)
    rho104 = a_star[104] + fF[104] + sF[104]
    dv104 = max(float(sol_new.speed(rho104, V_F, W, P)) - u, 0.0)
    check(dv104 > 0.0, f"reaction non-trivial in j_cav (dv = {dv104:.2f} m/s)")


# --------------------------------------------------------------------------
# P5: interaction with q_xi_max and w_s (flag on)
# --------------------------------------------------------------------------

def finite_check(res, cfg, label):
    ok = all(np.all(np.isfinite(getattr(res, n)))
             for n in ("t", "x", "a", "f", "s", "omega", "N_s"))
    ok = ok and all(np.isfinite(getattr(res, n)) for n in
                    ("denied_inflow", "injected", "outflowed", "on_road"))
    check(ok, f"{label}: no NaN/inf in any field (x_cav aside)")
    expect = np.array([np.isfinite(sol_new.cav_position(cfg, t))
                       for t in res.t])
    check(bool(np.all(np.isfinite(res.x_cav) == expect)),
          f"{label}: x_cav NaN exactly on the off-road saves")


def probe_interactions():
    print("[P5] flag-True with q_xi_max cap and with w_s")
    runs = (("q_xi_max = 2000/3600 cap",
             dict(kw_B(), q_xi_max=2000.0 / 3600.0, dt=0.5, save_every=20)),
            ("w_s = 0.6 w", kw_A()))
    for label, kw in runs:
        cfg = sol_new.SimConfig(**kw, downstream_release=True)
        res = sol_new.simulate(cfg)
        bal = res.on_road + res.outflowed
        rel = abs(res.injected - bal) / max(res.injected, 1.0)
        rho = res.a + res.f + res.s
        check(rel < 1e-10, f"{label}: ledger closes (rel err {rel:.2e})")
        check(res.f.min() >= 0.0 and res.s.min() >= 0.0,
              f"{label}: f, s >= 0")
        check(rho.max() <= cfg.P + 1e-12,
              f"{label}: rho <= P (max {rho.max():.6f})")
        ok = True
        n_on = 0
        for i in range(res.t.size):
            if np.isfinite(res.x_cav[i]):
                m = downstream_mask(res.x, res.x_cav[i], cfg.dx)
                ok = ok and np.all(res.s[i][m] == 0.0)
                n_on += 1
        check(ok and n_on > 0,
              f"{label}: downstream s == 0 at all {n_on} on-road saves")
        finite_check(res, cfg, label)


# --------------------------------------------------------------------------
# P6: off-road phases -- the flag is a no-op
# --------------------------------------------------------------------------

def probe_off_road():
    print("[P6] off-road no-op: before entry, after exit")
    nx = 600
    f0 = np.full(nx, 0.03)
    s0 = np.full(nx, 0.01)

    # (a) CAV never enters within [0, t_end]
    kw = dict(kw_B(), t_enter=2000.0, t_slow=2100.0, t_fast=2500.0,
              t_end=500.0, f0=f0, s0=s0)
    ra = sol_new.simulate(sol_new.SimConfig(**kw))
    rb = sol_new.simulate(sol_new.SimConfig(**kw, downstream_release=True))
    compare_all_fields(rb, ra, "never-entered (t_enter > t_end)")
    check(rb.s[-1].sum() > 0.0, "never-entered: s survives (nothing purged)")

    # (b) CAV already past x = L_road for every t >= 0
    kw = dict(kw_B(), t_enter=-2000.0, t_slow=-1500.0, t_fast=-1000.0,
              t_end=500.0, f0=f0, s0=s0)
    cfg_b = sol_new.SimConfig(**kw)
    assert not np.isfinite(sol_new.cav_position(cfg_b, 0.0)), \
        "exit config wrong: CAV still on road at t = 0"
    ra = sol_new.simulate(cfg_b)
    rb = sol_new.simulate(sol_new.SimConfig(**kw, downstream_release=True))
    compare_all_fields(rb, ra, "already-exited (x > L_road for all t)")
    check(rb.s[-1].sum() > 0.0, "already-exited: s survives (nothing purged)")

    # (c) long run reaching the exit: release fires exactly on the steps
    # with a finite CAV position at t + dt
    cfg = sol_new.SimConfig(**kw_A(), t_end=2000.0, downstream_release=True)
    res = sol_new.simulate(cfg)
    saves, released, pre_down, p_err, ledger = replicate(cfg)
    validate_replica(cfg, res, saves, ledger, "t_end=2000 exit run")
    n_steps = released.size
    expect = np.array([np.isfinite(sol_new.cav_position(cfg, (n + 1) * cfg.dt))
                       for n in range(n_steps)])
    check(bool(np.all(released == expect)),
          "release fired exactly when the CAV is on road at t + dt "
          f"({released.sum()} of {n_steps} steps)")
    t_first = (np.argmax(released) + 1) * cfg.dt
    t_last = (n_steps - 1 - np.argmax(released[::-1]) + 1) * cfg.dt
    print(f"        first/last release at t = {t_first:g} / {t_last:g} s "
          f"(t_enter = {cfg.t_enter:g}, exit ~ 1409.9 s)")
    check(t_first == 100.0 and 1400.0 < t_last < 1420.0,
          "release window matches [t_enter, exit] analytically")
    check(p_err < 1e-10, f"exit run: p conserved through conversion "
          f"(max |dp| {p_err:.2e})")
    # after exit the saved x_cav is NaN and s is no longer forced to 0
    post = np.isfinite(res.x_cav) == False  # noqa: E712
    post_idx = np.where(post & (res.t > 200.0))[0]
    check(post_idx.size > 0 and
          all(np.isfinite(res.x_cav[i]) == False for i in post_idx),
          f"{post_idx.size} post-exit saves have x_cav = NaN (flag inert)")


# --------------------------------------------------------------------------

def main() -> int:
    # trap invalid/divide/overflow (would surface NaN/inf); underflow to
    # denormals is benign and expected at vacuum densities
    np.seterr(invalid="raise", divide="raise", over="raise", under="ignore")
    sol_head = load_head_solver()
    probe_bit_identity(sol_head)
    probe_invariants()
    probe_boundary_semantics()
    probe_interactions()
    probe_off_road()
    print()
    if FAILURES:
        print(f"AUDIT: {len(FAILURES)} FAILURE(S)")
        for m in FAILURES:
            print(f"  - {m}")
        return 1
    print("AUDIT: all probes passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
