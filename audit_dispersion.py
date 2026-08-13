"""Independent MATH audit of ev5_dispersion.py (E-V5 dispersion relation).

READ-ONLY companion: imports the module under audit, never modifies it or
its outputs.  Everything symbolic below is re-derived from scratch with
sympy from the model statement (Multi-class LWR Equations.tex), NOT from
the module's formulas:

  state (a, f, s), rho = a + f + s, flux q = (a u, f v, s u) with
  u = u_s constant in both smooth regimes and
    R1 free:      v = v_f,            Delta = v_f - u_s
    R2 two-speed: v = w (P/rho - 1),  Delta = v - u_s
  source R = (0, g, -g), g = mu s - sigma f, mu = kappa_r (P - rho) Delta,
  capture forms  sigma_af = kappa_c a Delta,
                 sigma_lf = kappa_c (a + s) Delta.

Checks
  A  exact sympy Jacobians q_rho, R_rho (sympy.jacobian, lambdified) vs
     the module's conv_jac / source_jac at 30 random admissible states
     (both a = 0 and a > 0), rtol 1e-8; phase_speeds cross-check.
  B  Lambda(k) = eig(R_rho - i k q_rho) with MY Jacobians on the module's
     kappa sets and k grid, on every scanned base state of every branch
     (superset of the required 5 per branch), vs
     out/ev5/dispersion_summary.json (atol 1e-10 1/s or rtol 1e-6); the
     absent lf_A10 mixed branch is re-verified as inadmissible.
  C  equilibrium branches symbolically: (i) a* = s* = 0 (trivial) and
     (ii) kappa_r (P - rho) = kappa_c f* (lf mixed) give R = 0 exactly;
     plus exact k-flatness of Re Lambda on the trivial branch (symbolic
     eigenvalues with symbolic k).
  D  k -> 0 limit (Lambda -> eig R_rho) and kappa = 0 limit
     (Re Lambda = 0 for all k; q_rho spectrum {u_s, u_s, v + f v'} real,
     verified symbolically).
  E  regime-boundary sanity: every base state (scan grids, rand_state,
     representative rho) lies strictly inside R1 or R2, off both kinks.

Run:  python3 audit_dispersion.py   (prints per-check PASS/FAIL, exits 1
                                     on any failure; writes nothing)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE))
import ev5_dispersion as EV5           # noqa: E402  module under audit

RTOL_JAC = 1e-8                        # check A
ATOL_DISP = 1e-10                      # check B [1/s]
RTOL_DISP = 1e-6                       # check B
N_RAND = 30                            # check A states
SEED = 987654321                       # independent of the module's seed

FAILURES: list[str] = []
DEV = {}                               # named max deviations for the report


def report(name, dev, tol, extra=""):
    DEV[name] = float(dev)
    status = "PASS" if dev <= tol else "FAIL"
    line = f"[{status}] {name}: max dev {dev:.3e} (tol {tol:g}) {extra}"
    print(line)
    if dev > tol:
        FAILURES.append(line)


def report_bool(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    line = f"[{status}] {name} {extra}"
    print(line)
    if not cond:
        FAILURES.append(line)


# ------------------------------------------------- independent parameters


def load_params_independent():
    d = json.loads((OUT / "params.json").read_text())

    def pick(si_key, kmh_key, bare_key, scale):
        if si_key in d:
            return d[si_key]
        if kmh_key in d:
            return d[kmh_key] / scale
        return d[bare_key]

    return (pick("v_f_ms", "v_f_kmh", "v_f", 3.6),
            pick("w_ms", "w_kmh", "w", 3.6),
            pick("P_vehm", "P_vehkm", "P", 1000.0))


V_F, W, P = load_params_independent()
U_XI = 15.0
RHO_CF = W * P / (V_F + W)
RHO_CS = W * P / (U_XI + W)
SUMM = json.loads((OUT / "ev5" / "dispersion_summary.json").read_text())
KTAB = json.loads((OUT / "ev3_kappa.json").read_text())
K_RADKM = np.logspace(-2.0, 3.0, 251)
K_SI = K_RADKM * 1e-3


# ------------------------------------------- from-scratch symbolic model

A_, F_, S_ = sp.symbols("a f s", real=True, nonnegative=True)
KC_, KR_ = sp.symbols("kappa_c kappa_r", real=True, nonnegative=True)
K_ = sp.Symbol("k", real=True)
VF_, W_, P_, US_ = sp.symbols("v_f w P u_s", positive=True)
RHO_ = A_ + F_ + S_
V_EXPR = {"R1": VF_ * sp.Integer(1), "R2": W_ * (P_ / RHO_ - 1)}
PSUBS = {VF_: V_F, W_: W, P_: P, US_: U_XI}


def sym_model(regime, form):
    """Flux vector q, source vector R and their exact Jacobians."""
    v = V_EXPR[regime]
    u = US_                              # u = u_s in both R1 and R2
    delta = v - US_ if regime == "R2" else VF_ - US_
    q = sp.Matrix([A_ * u, F_ * v, S_ * u])
    mu = KR_ * (P_ - RHO_) * delta
    ell = A_ if form == "af" else A_ + S_
    g = mu * S_ - KC_ * ell * delta * F_
    R = sp.Matrix([0, g, -g])
    return q, R, q.jacobian((A_, F_, S_)), R.jacobian((A_, F_, S_))


def lambdify_all():
    fns = {}
    for reg in ("R1", "R2"):
        v = V_EXPR[reg]
        fns[reg, "v"] = sp.lambdify(
            (A_, F_, S_), sp.Matrix([v, sp.diff(v, A_)]).subs(PSUBS),
            "numpy")
        for form in ("af", "lf"):
            q, R, qj, Rj = sym_model(reg, form)
            fns[reg, form] = dict(
                R=sp.lambdify((A_, F_, S_, KC_, KR_), R.subs(PSUBS),
                              "numpy"),
                qjac=sp.lambdify((A_, F_, S_), qj.subs(PSUBS), "numpy"),
                Rjac=sp.lambdify((A_, F_, S_, KC_, KR_), Rj.subs(PSUBS),
                                 "numpy"),
            )
    return fns


FNS = lambdify_all()


def regime_of(rho):
    if rho < RHO_CF:
        return "R1"
    if rho < RHO_CS:
        return "R2"
    return "R3"


def my_qjac(x):
    return np.asarray(FNS[regime_of(sum(x)), "lf"]["qjac"](*x), float)


def my_Rjac(x, kc, kr, form):
    return np.asarray(FNS[regime_of(sum(x)), form]["Rjac"](x[0], x[1],
                                                           x[2], kc, kr),
                      float)


def my_dispersion(x, kc, kr, form, k_si=K_SI):
    m0 = my_Rjac(x, kc, kr, form).astype(complex)
    aa = my_qjac(x)
    mats = m0[None, :, :] - 1j * k_si[:, None, None] * aa[None, :, :]
    return np.linalg.eigvals(mats)


def my_equilibria(rho, kc, kr, form):
    br = {"trivial": np.array([0.0, rho, 0.0])}
    if form == "lf":
        fst = (kr / kc) * (P - rho)
        br["mixed"] = (np.array([0.0, fst, rho - fst])
                       if 0.0 <= fst <= rho else None)
    return br


def reldev(mine, theirs):
    mine, theirs = np.asarray(mine, float), np.asarray(theirs, float)
    scale = max(np.abs(mine).max(), np.abs(theirs).max(), 1e-300)
    return float(np.abs(mine - theirs).max() / scale)


# ----------------------------------------------------- check 0: metadata


def check_meta():
    print("== check 0: parameter / kappa / grid provenance ==")
    m = SUMM["meta"]["params_si"]
    dev = max(abs(m["v_f_ms"] - V_F), abs(m["w_ms"] - W),
              abs(m["P_vehm"] - P), abs(m["u_xi_ms"] - U_XI))
    report("meta params vs params.json", dev, 0.0)
    report("rho_c_fast recomputed",
           abs(SUMM["meta"]["rho_c_fast_veh_per_km"] - RHO_CF * 1e3),
           1e-9)
    report("rho_c_slow recomputed",
           abs(SUMM["meta"]["rho_c_slow_veh_per_km"] - RHO_CS * 1e3),
           1e-9)
    exp = {
        "lf_A1": ("A1_u15_q2500", "kappa_cl_mod"),
        "lf_A10": ("A10_u15_q2500", "kappa_cl_mod"),
        "af_A1": ("A1_u15_q2500", "kappa_cA_mod"),
        "af_A10": ("A10_u15_q2500", "kappa_cA_mod"),
    }
    dev = 0.0
    for kname, (tag, ckey) in exp.items():
        ks = SUMM["meta"]["kappa_sets"][kname]
        dev = max(dev, abs(ks["kc"] - KTAB[tag][ckey][0]),
                  abs(ks["kr"] - KTAB[tag]["kappa_r_mod"][0]))
        dev = max(dev, abs(EV5.KSETS[kname]["kc"] - ks["kc"]),
                  abs(EV5.KSETS[kname]["kr"] - ks["kr"]))
    report("kappa sets vs ev3_kappa.json points", dev, 0.0)
    report("k grid (module vs independent logspace)",
           reldev(K_RADKM, EV5.K_RADKM), 1e-15)
    kg = SUMM["meta"]["k_grid_rad_per_km"]
    report_bool("k grid meta [k_min, k_max, n]",
                kg == [0.01, 1000.0, 251])


# --------------------------------------- check A: Jacobians vs the module


def rand_admissible(rng, i):
    if rng.random() < 0.5:
        rho = rng.uniform(0.05, 0.97) * RHO_CF
    else:
        rho = RHO_CF + (RHO_CS - RHO_CF) * rng.uniform(0.03, 0.97)
    wts = rng.dirichlet([1.0, 1.0, 1.0])
    if i % 2:                               # every other state has a = 0
        wts = np.array([0.0, wts[1], wts[2]]) / (wts[1] + wts[2])
    return rho * wts


def check_jacobians():
    print(f"== check A: sympy Jacobians vs module at {N_RAND} random "
          "admissible states (a = 0 and a > 0), rtol 1e-8 ==")
    rng = np.random.default_rng(SEED)
    worst_q = worst_R = worst_ps = worst_src = 0.0
    n_a0 = 0
    for i in range(N_RAND):
        x = rand_admissible(rng, i)
        n_a0 += x[0] == 0.0
        rho = float(sum(x))
        reg = regime_of(rho)
        # phase_speeds cross-check against my symbolic v, v'
        v, vp = np.asarray(FNS[reg, "v"](*x), float).ravel()
        delta = (v - U_XI) if reg == "R2" else (V_F - U_XI)
        deltap = vp if reg == "R2" else 0.0
        mine_ps = np.array([v, vp, U_XI, 0.0, delta, deltap])
        worst_ps = max(worst_ps, reldev(mine_ps,
                                        EV5.phase_speeds(rho)))
        worst_q = max(worst_q, reldev(my_qjac(x), EV5.conv_jac(x)))
        for kname, ks in EV5.KSETS.items():
            kc, kr, form = ks["kc"], ks["kr"], ks["form"]
            worst_R = max(worst_R, reldev(
                my_Rjac(x, kc, kr, form),
                EV5.source_jac(x, kc, kr, form)))
            mine_src = np.asarray(
                FNS[reg, form]["R"](x[0], x[1], x[2], kc, kr),
                float).ravel()
            worst_src = max(worst_src, reldev(
                mine_src, EV5.source(x, kc, kr, form)))
    report_bool(f"state mix: {N_RAND - n_a0} with a > 0, {n_a0} with "
                "a = 0", n_a0 >= 10)
    report("phase_speeds (v, v', u, u', Delta, Delta')", worst_ps,
           RTOL_JAC)
    report("q_rho: sympy vs conv_jac", worst_q, RTOL_JAC)
    report("R_rho: sympy vs source_jac (4 kappa sets)", worst_R,
           RTOL_JAC)
    report("R vector: sympy vs source (4 kappa sets)", worst_src,
           RTOL_JAC)


# ------------------------------- check B: dispersion vs the summary JSON


def module_grid(reg):
    if reg == "R1":
        return np.linspace(0.02, 0.995, 41) * RHO_CF
    return RHO_CF + (RHO_CS - RHO_CF) * np.linspace(0.005, 0.995, 41)


def check_dispersion_vs_json():
    print("== check B: Lambda(k) = eig(R_rho - i k q_rho) vs "
          "dispersion_summary.json (all scanned base states; "
          ">= 5 per branch) ==")
    worst_grid = worst_max = worst_src = worst_vs_mod = 0.0
    worst_k0 = 0.0
    n_states = 0
    for kname, ks in EV5.KSETS.items():
        kc, kr, form = ks["kc"], ks["kr"], ks["form"]
        branches = ("trivial", "mixed") if form == "lf" else ("trivial",)
        for reg in ("R1", "R2"):
            grid = module_grid(reg)
            for br in branches:
                ent = SUMM["results"][kname][reg][br]
                if not ent.get("exists"):
                    fst = (kr / kc) * (P - grid)
                    report_bool(
                        f"{kname} {reg} {br}: absence re-verified "
                        "(f* > rho on the whole grid)",
                        bool(np.all(fst > grid)))
                    continue
                worst_grid = max(worst_grid, reldev(
                    np.asarray(ent["rho_grid_veh_per_km"]) / 1e3, grid))
                ref = np.asarray(ent["max_growth_vs_rho_1_per_s"])
                ref_src = np.asarray(
                    ent["source_mode_growth_vs_rho_1_per_s"])
                mine = np.empty_like(ref)
                mine_src = np.empty_like(ref)
                mine_k0 = np.empty_like(ref)
                flat = True
                for j, rho in enumerate(grid):
                    x = my_equilibria(rho, kc, kr, form)[br]
                    assert regime_of(sum(x)) == reg
                    lam = my_dispersion(x, kc, kr, form)
                    re_k = lam.real.max(axis=1)
                    mine[j] = re_k.max()
                    flat &= bool(re_k.max() - re_k.min() < 1e-10)
                    mine_src[j] = np.trace(my_Rjac(x, kc, kr, form))
                    mine_k0[j] = np.linalg.eigvals(
                        my_Rjac(x, kc, kr, form)).real.max()
                    worst_vs_mod = max(worst_vs_mod, float(np.abs(
                        np.sort(lam.real, axis=1)
                        - np.sort(EV5.dispersion(x, kc, kr, form).real,
                                  axis=1)).max()))
                    n_states += 1
                worst_max = max(worst_max, float(
                    np.abs(mine - ref).max()))
                worst_src = max(worst_src, float(
                    np.abs(mine_src - ref_src).max()))
                worst_k0 = max(worst_k0, abs(
                    mine_k0.max() - ent["max_growth_at_k0_1_per_s"]))
                ok_ent = (np.all(np.abs(mine - ref)
                                 <= np.maximum(ATOL_DISP,
                                               RTOL_DISP * np.abs(ref)))
                          and abs(mine.max() - ent["max_growth_1_per_s"])
                          <= max(ATOL_DISP,
                                 RTOL_DISP
                                 * abs(ent["max_growth_1_per_s"]))
                          and flat == ent["k_flat"])
                report_bool(
                    f"{kname} {reg} {br}: 41 base states, max Re "
                    f"{mine.max():+.6e} vs JSON "
                    f"{ent['max_growth_1_per_s']:+.6e}, k_flat={flat}",
                    bool(ok_ent))
    report("scan grids: stored vs recomputed", worst_grid, 1e-12)
    report("max Re Lambda vs JSON (abs, 1/s)", worst_max, ATOL_DISP,
           f"over {n_states} base states x 251 k")
    report("source mode tr(R_rho) vs JSON (abs, 1/s)", worst_src,
           ATOL_DISP)
    report("max Re at k->0 vs JSON (abs, 1/s)", worst_k0, ATOL_DISP)
    report("sorted Re Lambda: mine vs module dispersion()",
           worst_vs_mod, 1e-10)


# ------------------------------ check C: equilibria + k-flat, symbolic


def check_equilibria_symbolic():
    print("== check C: equilibrium branches and exact k-flatness "
          "(symbolic) ==")
    rt = sp.Symbol("rho_t", positive=True)
    all_zero = True
    for reg in ("R1", "R2"):
        for form in ("af", "lf"):
            _, R, _, _ = sym_model(reg, form)
            g_triv = sp.simplify(R[1].subs({A_: 0, S_: 0, F_: rt}))
            all_zero &= g_triv == 0
    report_bool("(i) trivial branch a* = s* = 0: R = 0 exactly "
                "(both forms, R1 and R2)", all_zero)
    all_zero = True
    for reg in ("R1", "R2"):
        _, R, _, _ = sym_model(reg, "lf")
        fst = (KR_ / KC_) * (P_ - rt)
        g_mix = sp.simplify(R[1].subs({A_: 0, F_: fst, S_: rt - fst}))
        all_zero &= g_mix == 0
    report_bool("(ii) lf mixed branch kappa_r (P - rho) = kappa_c f*: "
                "R = 0 exactly (R1 and R2)", all_zero)
    # exact k-independence of Re Lambda on the trivial branch (R2, both
    # forms): spectrum must be {-i k u_s, -i k (v + f v'), tr(R_rho)}.
    okflat = True
    for form in ("af", "lf"):
        _, _, qj, Rj = sym_model("R2", form)
        sub = {A_: 0, S_: 0, F_: rt}
        Mk = (Rj - sp.I * K_ * qj).subs(sub)
        tr = sp.simplify(sp.trace(Rj.subs(sub)))
        n_zero = n_tr = 0
        for ev, mult in Mk.eigenvals().items():
            r = sp.simplify(sp.re(ev))
            if r == 0:
                n_zero += mult
            elif sp.simplify(r - tr) == 0:
                n_tr += mult
        okflat &= (n_zero == 2 and n_tr == 1)
    report_bool("trivial branch: Re Lambda(k) = {0, 0, tr(R_rho)} for "
                "symbolic k (R2, both forms) -- k-flat is EXACT", okflat)


# --------------------------- check D: k -> 0 and kappa = 0 limits


def check_limits():
    print("== check D: k -> 0 and kappa = 0 limits ==")
    worst0 = 0.0
    for kname, ks in EV5.KSETS.items():
        kc, kr, form = ks["kc"], ks["kr"], ks["form"]
        for rho in (0.6 * RHO_CF, 0.5 * (RHO_CF + RHO_CS)):
            for br, x in my_equilibria(rho, kc, kr, form).items():
                if x is None:
                    continue
                lam0 = np.sort(np.linalg.eigvals(
                    my_Rjac(x, kc, kr, form)).real)
                lam = my_dispersion(x, kc, kr, form,
                                    np.array([1e-12]))[0]
                worst0 = max(worst0,
                             float(np.abs(np.sort(lam.real)
                                          - lam0).max()),
                             float(np.abs(lam.imag).max()))
    report("k -> 0: Lambda -> eig(R_rho)", worst0, 1e-8)
    rng = np.random.default_rng(SEED + 1)
    worstk = 0.0
    for i in range(12):
        x = rand_admissible(rng, i)
        for form in ("af", "lf"):
            lam = my_dispersion(x, 0.0, 0.0, form)
            worstk = max(worstk, float(np.abs(lam.real).max()))
    report("kappa = 0: Re Lambda = 0 at every k", worstk, 1e-10)
    ok_real = True
    for reg in ("R1", "R2"):
        _, _, qj, _ = sym_model(reg, "lf")
        for ev in qj.eigenvals():
            ok_real &= sp.simplify(sp.im(ev)) == 0
    report_bool("q_rho spectrum real (symbolic) => kappa = 0 neutrality "
                "is exact", ok_real)


# --------------------------- check E: regime-boundary (kink) handling


def check_regime_margins():
    print("== check E: base states strictly inside R1 or R2 ==")
    margins = []
    for reg in ("R1", "R2"):
        g = module_grid(reg)
        assert all(regime_of(r) == reg for r in g)
        margins += [np.abs(g - RHO_CF).min(), np.abs(g - RHO_CS).min()]
    reps = [0.6 * RHO_CF, 0.5 * (RHO_CF + RHO_CS), 0.3 * RHO_CF,
            0.9 * RHO_CF] + list(RHO_CF + np.array([0.1, 0.5, 0.9])
                                 * (RHO_CS - RHO_CF))
    for r in reps:
        assert regime_of(r) in ("R1", "R2")
        margins += [abs(r - RHO_CF), abs(r - RHO_CS)]
    rng = np.random.default_rng(SEED + 2)
    for _ in range(2000):
        rho = float(sum(EV5.rand_state(rng)))
        assert regime_of(rho) in ("R1", "R2")
        margins += [abs(rho - RHO_CF), abs(rho - RHO_CS)]
    m = float(min(margins))
    report_bool(f"all scan/representative/rand_state rho off both kinks "
                f"(min margin {m * 1e3:.4f} veh/km = "
                f"{m / RHO_CF * 100:.2f}% of rho_c_fast)",
                m > 1e-4 * RHO_CF)
    v_lo = W * (P / (RHO_CF * 1.005) - 1.0)
    report_bool("classification convention: module phase_speeds at "
                "0.995 rho_c_fast / 1.005 rho_c_fast returns R1 / R2",
                EV5.phase_speeds(0.995 * RHO_CF)[1] == 0.0
                and abs(EV5.phase_speeds(1.005 * RHO_CF)[0] - v_lo)
                < 1e-12 and U_XI < v_lo < V_F)


# ------------------------------------------------------------------ main


def main():
    print(f"audit params SI: v_f={V_F:.6f} w={W:.6f} P={P:.6f} "
          f"u_xi={U_XI:g}")
    print(f"rho_c_fast={RHO_CF * 1e3:.4f} veh/km  "
          f"rho_c_slow={RHO_CS * 1e3:.4f} veh/km")
    check_meta()
    check_jacobians()
    check_dispersion_vs_json()
    check_equilibria_symbolic()
    check_limits()
    check_regime_margins()
    print("\n== deviation summary ==")
    for k, v in DEV.items():
        print(f"  {k}: {v:.3e}")
    if FAILURES:
        print(f"\nAUDIT FAIL ({len(FAILURES)} failing check(s)):")
        for m in FAILURES:
            print("  " + m)
        return 1
    print("\nAUDIT PASS: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
