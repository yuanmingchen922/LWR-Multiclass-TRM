"""E-V5: linear stability / dispersion relation of the calibrated model.

Two-class LWR transition model (see Multi-class LWR Equations.tex, sections
"Piecewise characteristic structure" and "Linear stability diagnostic").

State ordering rho_vec = (a, f, s), rho = a + f + s, flux
q = (a u, f v, s u) with v(rho) = min(v_f, w (P/rho - 1)) and
u(rho) = min(u_s, v(rho)).  Source R = (0, g, -g), g = mu s - sigma f,
mu = kappa_r (P - rho) Delta, Delta = v - u, and capture forms
  'af': sigma = kappa_c a Delta          (the .tex form)
  'lf': sigma = kappa_c (a + s) Delta    (calibrated baseline; the source
        Jacobian is derived here, NOT taken from the .tex gradients).

Smooth regimes about homogeneous base states with a* = 0, u_s = u_xi:
  R1 free          rho < rho_c_fast = w P / (v_f + w):  v = v_f, u = u_s,
                   v' = u' = 0, Delta = v_f - u_s, Delta' = 0.
  R2 two-speed     rho_c_fast < rho < rho_c_slow = w P / (u_s + w):
                   v = w (P/rho - 1) in (u_s, v_f), v' = -w P / rho^2,
                   u = u_s, u' = 0, Delta = v - u_s, Delta' = v'.
  R3 deep congested rho > rho_c_slow: v = u = w (P/rho - 1), Delta = 0,
                   the source is off and the system is transport-only;
                   all modes are neutrally stable (Re Lambda = 0), so no
                   scan is performed there.

Dispersion: perturbations ~ exp(i k x + lambda t) about a homogeneous
equilibrium give Lambda(k) = eig(R_rho - i k q_rho), with the convective
Jacobian q_rho = diag(u, v, u) + outer([a u', f v', s u'], [1, 1, 1])
(.tex eq. flux-Jacobian).  k in rad/m (SI) internally, reported in rad/km.

Run:  python3 ev5_dispersion.py
Outputs: out/ev5/dispersion_summary.json, out/ev5/fig_dispersion.png,
         out/ev5/fig_stability_map.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT_EV5 = OUT / "ev5"

# ------------------------------------------------------------- parameters


def load_params():
    """Per-key fallback: the real out/params.json mixes conventions
    (v_f_ms/w_ms in SI but only P_vehkm for the jam density)."""
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


V_F, W, P = load_params()
U_XI = 15.0                       # slow phase of the u15 scenarios [m/s]
assert U_XI < V_F
RHO_CF = W * P / (V_F + W)        # R1/R2 boundary (v leaves v_f)
RHO_CS = W * P / (U_XI + W)       # R2/R3 boundary (v reaches u_s)
assert RHO_CF < RHO_CS < P

_KTAB = json.loads((OUT / "ev3_kappa.json").read_text())

# kappa sets: point estimates ([0] of [point, lo, hi]) from E-V3
KSETS = {
    "lf_A1": dict(form="lf", tag="A1_u15_q2500",
                  kc=_KTAB["A1_u15_q2500"]["kappa_cl_mod"][0],
                  kr=_KTAB["A1_u15_q2500"]["kappa_r_mod"][0]),
    "lf_A10": dict(form="lf", tag="A10_u15_q2500",
                   kc=_KTAB["A10_u15_q2500"]["kappa_cl_mod"][0],
                   kr=_KTAB["A10_u15_q2500"]["kappa_r_mod"][0]),
    "af_A1": dict(form="af", tag="A1_u15_q2500",
                  kc=_KTAB["A1_u15_q2500"]["kappa_cA_mod"][0],
                  kr=_KTAB["A1_u15_q2500"]["kappa_r_mod"][0]),
    "af_A10": dict(form="af", tag="A10_u15_q2500",
                   kc=_KTAB["A10_u15_q2500"]["kappa_cA_mod"][0],
                   kr=_KTAB["A10_u15_q2500"]["kappa_r_mod"][0]),
}

K_RADKM = np.logspace(-2.0, 3.0, 251)     # rad/km
K_SI = K_RADKM * 1e-3                     # rad/m
TOL_STAB = 1e-8                           # 1/s: |Re| below this = roundoff

# ----------------------------------------------------------------- model


def phase_speeds(rho):
    """v, v', u, u', Delta, Delta' at scalar rho (SI).  Piecewise smooth
    with kinks at RHO_CF and RHO_CS; callers stay off the kinks."""
    v_cong = W * (P / rho - 1.0)
    if v_cong >= V_F:                          # R1 free
        return V_F, 0.0, U_XI, 0.0, V_F - U_XI, 0.0
    vp = -W * P / rho ** 2
    if v_cong > U_XI:                          # R2 two-speed congested
        return v_cong, vp, U_XI, 0.0, v_cong - U_XI, vp
    return v_cong, vp, v_cong, vp, 0.0, 0.0    # R3 deep congested


def source(rvec, kc, kr, form):
    a, f, s = rvec
    rho = a + f + s
    _, _, _, _, D, _ = phase_speeds(rho)
    mu = kr * (P - rho) * D
    ell = a if form == "af" else a + s
    g = mu * s - kc * ell * D * f
    return np.array([0.0, g, -g])


def source_jac(rvec, kc, kr, form):
    """Analytic R_rho = dR/d(a,f,s).  Every partial picks up the chain
    rule through rho = a + f + s (Delta(rho), (P - rho), ell)."""
    a, f, s = rvec
    rho = a + f + s
    _, _, _, _, D, Dp = phase_speeds(rho)
    mu = kr * (P - rho) * D
    mup = kr * (-D + (P - rho) * Dp)          # d(mu)/d(any component)
    if form == "af":
        ell, dell = a, np.array([1.0, 0.0, 0.0])
    else:
        ell, dell = a + s, np.array([1.0, 0.0, 1.0])
    sig = kc * ell * D
    dsig = kc * (dell * D + ell * Dp)         # d(sigma)/d(a,f,s)
    dg = s * mup - f * dsig
    dg[1] -= sig                              # direct d/df of (-sigma f)
    dg[2] += mu                               # direct d/ds of (mu s)
    return np.array([np.zeros(3), dg, -dg])


def conv_jac(rvec):
    """Flux Jacobian q_rho (.tex eq. flux-Jacobian):
    diag(u, v, u) + outer([a u', f v', s u'], [1, 1, 1])."""
    a, f, s = rvec
    rho = a + f + s
    v, vp, u, up, _, _ = phase_speeds(rho)
    return (np.diag([u, v, u])
            + np.outer([a * up, f * vp, s * up], np.ones(3)))


def dispersion(rvec, kc, kr, form, k_si=K_SI):
    """Lambda(k): eigenvalues of R_rho - i k q_rho, shape (nk, 3)."""
    m0 = source_jac(rvec, kc, kr, form).astype(complex)
    A = conv_jac(rvec)
    mats = m0[None, :, :] - 1j * k_si[:, None, None] * A[None, :, :]
    return np.linalg.eigvals(mats)


# ----------------------------------------------------- equilibrium branches


def equilibria(rho, kc, kr, form):
    """Homogeneous equilibria with a* = 0 at fixed rho.
    'lf': g = s Delta (kappa_r (P - rho) - kappa_c f) =>
      (i)  trivial  s* = 0, f* = rho
      (ii) mixed    f* = (kappa_r / kappa_c) (P - rho), s* = rho - f*,
           admissible iff 0 <= f* <= rho (transcritical exchange at
           rho_tc = P kappa_r / (kappa_r + kappa_c)).
    'af': sigma = kappa_c a Delta vanishes at a* = 0, so g = mu s and the
    only equilibrium with Delta > 0 is the trivial one (degenerate)."""
    br = {"trivial": np.array([0.0, rho, 0.0])}
    if form == "lf":
        f_star = (kr / kc) * (P - rho)
        if 0.0 <= f_star <= rho:
            br["mixed"] = np.array([0.0, f_star, rho - f_star])
        else:
            br["mixed"] = None
    return br


# ------------------------------------------------------- mandatory checks


def rand_state(rng):
    """Random admissible (a, f, s) with rho inside R1 or R2, off kinks."""
    if rng.random() < 0.5:
        rho = rng.uniform(0.10 * RHO_CF, 0.97 * RHO_CF)
    else:
        rho = RHO_CF + (RHO_CS - RHO_CF) * rng.uniform(0.03, 0.97)
    return rho * rng.dirichlet([1.0, 1.0, 1.0])


def check_source_jac_fd(n_states=20, h=1e-7, rtol=1e-6):
    """Analytic R_rho vs central finite differences of R(rho_vec)."""
    rng = np.random.default_rng(20260813)
    worst = 0.0
    for form in ("af", "lf"):
        for kname, ks in KSETS.items():
            if ks["form"] != form:
                continue
            for _ in range(n_states):
                x = rand_state(rng)
                ana = source_jac(x, ks["kc"], ks["kr"], form)
                fd = np.empty((3, 3))
                for j in range(3):
                    dx = np.zeros(3)
                    dx[j] = h
                    fd[:, j] = (source(x + dx, ks["kc"], ks["kr"], form)
                                - source(x - dx, ks["kc"], ks["kr"], form)
                                ) / (2.0 * h)
                scale = max(np.abs(ana).max(), np.abs(fd).max(), 1e-12)
                if not np.allclose(ana, fd, rtol=rtol, atol=1e-9 * scale):
                    raise AssertionError(
                        f"R_rho FD mismatch ({kname}, {form}) at {x}:\n"
                        f"analytic\n{ana}\nFD\n{fd}")
                big = np.abs(ana) > 1e-9 * scale
                if big.any():
                    rel = (np.abs(ana - fd)[big]
                           / np.abs(ana)[big]).max()
                    worst = max(worst, float(rel))
    print(f"[check] R_rho analytic vs FD ({n_states} states x 4 kappa "
          f"sets): worst rel err {worst:.2e} (rtol {rtol:g})  PASS")
    return worst


def check_k0_limit():
    """Lambda(k -> 0) must reduce to eig(R_rho)."""
    worst = 0.0
    for kname, ks in KSETS.items():
        for rho in (0.6 * RHO_CF, 0.5 * (RHO_CF + RHO_CS)):
            for br, x in equilibria(rho, ks["kc"], ks["kr"],
                                    ks["form"]).items():
                if x is None:
                    continue
                lam0 = np.sort(
                    np.linalg.eigvals(
                        source_jac(x, ks["kc"], ks["kr"],
                                   ks["form"])).real)
                lam = dispersion(x, ks["kc"], ks["kr"], ks["form"],
                                 np.array([1e-12]))[0]
                err = max(np.abs(np.sort(lam.real) - lam0).max(),
                          np.abs(lam.imag).max())
                worst = max(worst, float(err))
    assert worst < 1e-8, f"k->0 limit broken: {worst:.2e}"
    print(f"[check] k->0 limit -> eig(R_rho): worst err {worst:.2e}  PASS")
    return worst


def check_zero_kappa():
    """kappa_c = kappa_r = 0 must give Re Lambda = 0 for all k."""
    worst = 0.0
    for rho in (0.3 * RHO_CF, 0.9 * RHO_CF,
                RHO_CF + np.array([0.1, 0.5, 0.9]) * (RHO_CS - RHO_CF)):
        for r in np.atleast_1d(rho):
            for x in (np.array([0.0, r, 0.0]),
                      np.array([0.0, 0.3 * r, 0.7 * r]),
                      np.array([0.02 * r, 0.58 * r, 0.4 * r])):
                lam = dispersion(x, 0.0, 0.0, "lf")
                worst = max(worst, float(np.abs(lam.real).max()))
    assert worst < 1e-8, f"zero-kappa Re Lambda != 0: {worst:.2e}"
    print(f"[check] kappa=0 => Re Lambda = 0: max |Re| {worst:.2e}  PASS")
    return worst


def check_equilibria():
    """Branch states must satisfy R = 0; verify the mixed-branch algebra
    kappa_r (P - rho) = kappa_c f* by direct substitution."""
    worst = 0.0
    for kname, ks in KSETS.items():
        for rho in (0.6 * RHO_CF, 0.5 * (RHO_CF + RHO_CS)):
            for br, x in equilibria(rho, ks["kc"], ks["kr"],
                                    ks["form"]).items():
                if x is None:
                    continue
                g = np.abs(source(x, ks["kc"], ks["kr"], ks["form"])).max()
                worst = max(worst, float(g))
    assert worst < 1e-14, f"branch state not an equilibrium: {worst:.2e}"
    print(f"[check] equilibrium branches satisfy R = 0: max |g| "
          f"{worst:.2e}  PASS")
    return worst


# ------------------------------------------------------------------ scan


def scan_branch(ks, rho_grid, branch):
    """Max growth over k and eigenvalue index for each rho in rho_grid."""
    rows = []
    for rho in rho_grid:
        x = equilibria(rho, ks["kc"], ks["kr"], ks["form"]).get(branch)
        if x is None:
            rows.append(None)
            continue
        lam = dispersion(x, ks["kc"], ks["kr"], ks["form"])
        re_k = lam.real.max(axis=1)               # max over eigs, per k
        j = int(np.argmax(re_k))
        jac = source_jac(x, ks["kc"], ks["kr"], ks["form"])
        lam0 = np.linalg.eigvals(jac).real.max()
        # R_rho has rank <= 1, spectrum {0, 0, tr}: tr is the source-
        # active (relaxation) mode, the other two are the neutral
        # advective (a) and kinematic-wave (f) modes.
        rows.append(dict(rho=float(rho),
                         growth=float(re_k[j]),
                         k_radkm=float(K_RADKM[j]),
                         k_flat=bool(re_k.max() - re_k.min() < 1e-10),
                         growth_k0=float(lam0),
                         src=float(np.trace(jac))))
    return rows


def summarize(rows, branch):
    live = [r for r in rows if r is not None]
    if not live:
        return {"exists": False,
                "reason": "branch not admissible on this rho range "
                          "(f* = (kappa_r/kappa_c)(P - rho) > rho)"}
    best = max(live, key=lambda r: r["growth"])
    flat = all(r["k_flat"] for r in live)
    stable = bool(best["growth"] <= TOL_STAB)
    if stable:
        note = ("max Re attained by the always-neutral advective (a) "
                "and kinematic-wave (f) modes (Re = 0 at every k, so "
                "argmax rho/k are degenerate); the source-active mode "
                "tr(R_rho) is reported separately")
    else:
        note = ("growth carried by the source-active (capture) mode, "
                "advected at u_s; k-independent, so no wavelength is "
                "selected")
    return {
        "exists": True,
        "n_rho": len(live),
        "max_growth_1_per_s": best["growth"],
        "max_growth_at_k0_1_per_s": max(r["growth_k0"] for r in live),
        "argmax_k_rad_per_km": None if flat else best["k_radkm"],
        "k_flat": flat,
        "argmax_rho_veh_per_km": best["rho"] * 1000.0,
        "stable": stable,
        "source_mode_growth_max_1_per_s": max(r["src"] for r in live),
        "note": note,
        "rho_grid_veh_per_km": [r["rho"] * 1000.0 for r in live],
        "max_growth_vs_rho_1_per_s": [r["growth"] for r in live],
        "source_mode_growth_vs_rho_1_per_s": [r["src"] for r in live],
    }


# ---------------------------------------------------------------- figures


def fig_dispersion(path):
    panels = [("lf_A1", "trivial"), ("lf_A1", "mixed"),
              ("lf_A10", "trivial"), ("af_A1", "trivial"),
              ("af_A10", "trivial")]
    reps = [("R1", 0.6 * RHO_CF, "tab:blue"),
            ("R2", 0.5 * (RHO_CF + RHO_CS), "tab:red")]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex=True)
    for ax, (kname, br) in zip(axes.flat, panels):
        ks = KSETS[kname]
        for regime, rho, col in reps:
            x = equilibria(rho, ks["kc"], ks["kr"], ks["form"]).get(br)
            if x is None:
                continue
            lam = dispersion(x, ks["kc"], ks["kr"], ks["form"])
            re = np.sort(lam.real, axis=1)[:, ::-1]
            for l in range(3):
                ax.plot(K_RADKM, re[:, l], color=col,
                        ls=["-", "--", ":"][l], lw=1.4,
                        label=(f"{regime}, rho={rho * 1e3:.1f} veh/km"
                               if l == 0 else None))
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_xscale("log")
        ax.set_title(f"{kname}  ({ks['form']}), {br} branch", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    for ax in axes[1]:
        ax.set_xlabel("k [rad/km]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Re $\\lambda_l(k)$ [1/s]")
    ax = axes.flat[-1]
    ax.axis("off")
    ax.text(0.02, 0.95, "\n".join([
        "notes:",
        f"rho_c_fast = {RHO_CF * 1e3:.2f} veh/km",
        f"rho_c_slow = {RHO_CS * 1e3:.2f} veh/km  (u_xi = {U_XI:g} m/s)",
        "line styles: 3 eigenvalue branches (Re, sorted)",
        "lf_A10 mixed branch: not admissible in R1/R2",
        f"  (rho_tc = {P * KSETS['lf_A10']['kr'] / (KSETS['lf_A10']['kr'] + KSETS['lf_A10']['kc']) * 1e3:.2f}"
        f" veh/km > rho_c_slow)",
        "lf_A1 mixed branch = all captured (f* = 0, kappa_r = 0)",
        "af at a* = 0: sigma* = 0, degenerate (see JSON)",
        "R3 (rho > rho_c_slow): source off, purely convective,",
        "  Re Lambda = 0 (neutral) -- not scanned",
    ]), va="top", fontsize=9, family="monospace",
        transform=ax.transAxes)
    fig.suptitle("E-V5 dispersion relation Lambda(k) = "
                 "eig(R_rho - i k q_rho), homogeneous base states "
                 "(a* = 0, u_s = 15 m/s)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_stability_map(path, results):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    styles = {"lf_A1": ("tab:red", "-"), "lf_A10": ("tab:blue", "-"),
              "af_A1": ("tab:orange", "--"), "af_A10": ("tab:cyan", "--")}
    for ax, branch in zip(axes, ("trivial", "mixed")):
        for kname, per_reg in results.items():
            col, ls = styles[kname]
            xs, ys, zs = [], [], []
            for reg in ("R1", "R2"):
                ent = per_reg[reg].get(branch)
                if not ent or not ent.get("exists"):
                    continue
                xs += ent["rho_grid_veh_per_km"]
                ys += ent["max_growth_vs_rho_1_per_s"]
                zs += ent["source_mode_growth_vs_rho_1_per_s"]
            if xs:
                ax.plot(xs, ys, color=col, ls=ls, lw=1.6, label=kname)
                ax.plot(xs, zs, color=col, ls=":", lw=1.0,
                        label=f"{kname} source mode")
        ax.axhline(0.0, color="k", lw=0.6)
        for rc, nm in ((RHO_CF, "rho_c_fast"), (RHO_CS, "rho_c_slow")):
            ax.axvline(rc * 1e3, color="gray", ls=":", lw=1.0)
            ax.text(rc * 1e3, ax.get_ylim()[1], f" {nm}", fontsize=7,
                    va="top", color="gray", rotation=90)
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_xlabel("base density rho* [veh/km]")
        ax.set_title(f"{branch} branch", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, ncol=2)
    axes[0].set_ylabel("max$_{k,l}$ Re $\\lambda_l(k)$ [1/s]")
    fig.suptitle("E-V5 stability map (u_s = 15 m/s; solid: max over all "
                 "modes, dotted: source-active mode tr(R_rho); mixed "
                 "branch: lf only, absent for A10 in R1/R2)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main


def main():
    OUT_EV5.mkdir(parents=True, exist_ok=True)
    print(f"params SI: v_f={V_F:.4f} w={W:.4f} P={P:.5f} u_xi={U_XI:g}")
    print(f"rho_c_fast={RHO_CF * 1e3:.3f} veh/km  "
          f"rho_c_slow={RHO_CS * 1e3:.3f} veh/km")

    checks = {
        "source_jac_fd_worst_rel_err": check_source_jac_fd(),
        "k0_limit_worst_err": check_k0_limit(),
        "zero_kappa_max_abs_re": check_zero_kappa(),
        "equilibrium_max_abs_g": check_equilibria(),
    }

    grids = {
        "R1": np.linspace(0.02, 0.995, 41) * RHO_CF,
        "R2": RHO_CF + (RHO_CS - RHO_CF) * np.linspace(0.005, 0.995, 41),
    }
    results = {}
    for kname, ks in KSETS.items():
        results[kname] = {}
        branches = ("trivial", "mixed") if ks["form"] == "lf" \
            else ("trivial",)
        for reg, grid in grids.items():
            results[kname][reg] = {}
            for br in branches:
                ent = summarize(scan_branch(ks, grid, br), br)
                results[kname][reg][br] = ent
                if ent.get("exists"):
                    print(f"{kname:7s} {reg} {br:7s}: max Re = "
                          f"{ent['max_growth_1_per_s']:+.4e} 1/s at "
                          f"rho = {ent['argmax_rho_veh_per_km']:.2f} "
                          f"veh/km, "
                          + ("k-flat" if ent["k_flat"] else
                             f"k = {ent['argmax_k_rad_per_km']:.3g} "
                             f"rad/km")
                          + f", src mode <= "
                          f"{ent['source_mode_growth_max_1_per_s']:+.3e}"
                          f", stable = {ent['stable']}")
                else:
                    print(f"{kname:7s} {reg} {br:7s}: not admissible")

    # transcritical thresholds (lf): trivial/mixed exchange at
    # kappa_r (P - rho) = kappa_c rho
    rho_mid2 = 0.5 * (RHO_CF + RHO_CS)
    kr_hi_A1 = _KTAB["A1_u15_q2500"]["kappa_r_mod"][2]
    kc_A1 = KSETS["lf_A1"]["kc"]
    trans = {
        "formula": "mixed branch: f* = (kappa_r/kappa_c)(P - rho), "
                   "s* = rho - f*; exchange with the trivial branch at "
                   "rho_tc = P kappa_r / (kappa_r + kappa_c)",
        "lf_A1": {
            "kappa_r": KSETS["lf_A1"]["kr"],
            "kappa_c": kc_A1,
            "f_star_veh_per_km": 0.0,
            "rho_tc_veh_per_km": 0.0,
            "note": "kappa_r point estimate is exactly 0 (n_rel = 0 in "
                    "A1_u15_q2500), so f* = 0 for every rho: the mixed "
                    "branch is the all-captured state s* = rho and the "
                    "trivial branch is source-unstable wherever "
                    "Delta > 0.  With the CI upper bound kappa_r = "
                    f"{kr_hi_A1:.3e}, f* at mid-R2 (rho = "
                    f"{rho_mid2 * 1e3:.1f} veh/km) is "
                    f"{(kr_hi_A1 / kc_A1) * (P - rho_mid2) * 1e3:.4f} "
                    "veh/km, i.e. a fraction "
                    f"{(kr_hi_A1 / kc_A1) * (P - rho_mid2) / rho_mid2:.2e}"
                    " of rho -- tiny either way.",
        },
        "lf_A10": {
            "kappa_r": KSETS["lf_A10"]["kr"],
            "kappa_c": KSETS["lf_A10"]["kc"],
            "rho_tc_veh_per_km": P * KSETS["lf_A10"]["kr"]
            / (KSETS["lf_A10"]["kr"] + KSETS["lf_A10"]["kc"]) * 1e3,
            "note": "rho_tc exceeds rho_c_slow = "
                    f"{RHO_CS * 1e3:.2f} veh/km, so the mixed branch "
                    "does not exist anywhere in R1 u R2; the trivial "
                    "branch is source-stable there "
                    "(kappa_c rho < kappa_r (P - rho)).",
        },
    }

    g_a1 = max(results["lf_A1"][r]["trivial"]["max_growth_1_per_s"]
               for r in ("R1", "R2"))
    g_a10_src = max(
        results["lf_A10"][r]["trivial"]["source_mode_growth_max_1_per_s"]
        for r in ("R1", "R2"))
    interpretation = (
        "Only the lf A=1 trivial branch (free flow behind the CAV, no "
        f"release: kappa_r = 0) is linearly unstable: max Re lambda = "
        f"{g_a1:.4e} 1/s at the R1/R2 boundary (e-folding "
        f"{1.0 / g_a1:.0f} s vs the 500 s slow window, i.e. ~"
        f"{np.exp(500.0 * g_a1):.0f}x amplification).  The growth rate "
        "is exactly k-independent (the unstable s-mode advects at u_s "
        "and its growth kappa_c rho Delta does not involve k), so NO "
        "finite wavelength is selected: linear theory predicts one "
        "monotonically growing captured platoon for A=1 -- consistent "
        "with the single deep queue seen in SUMO -- and the stable "
        "mixed (all-captured) branch is its saturation.  For the "
        "calibrated A=10 kappas every mode of every admissible branch "
        "is stable at every scanned k: max Re lambda = 0 exactly, "
        "attained only by the always-neutral advective (a) and "
        "kinematic-wave (f) modes, while the source-active mode has "
        "Re = Delta (kappa_c rho - kappa_r (P - rho)) < 0 throughout "
        f"R1 u R2 (its supremum {g_a10_src:.3e} 1/s -> 0- only at the "
        "R2/R3 boundary where Delta -> 0): the homogeneous base states "
        "have NO linear instability, exactly the outcome anticipated "
        "by .tex item N1.  Consequently the "
        "observed A=10 stripes CANNOT be reproduced as a spontaneous "
        "(modulational) instability of a uniform state in this model; "
        "in the model they must be forced by the moving CAV "
        "inhomogeneity itself -- periodic capture-release cycling at "
        "the CAV position radiating kinematic waves -- so their "
        "spacing is set by the forcing (CAV speed, release rate), not "
        "by an intrinsic most-unstable wavenumber.  The af form is "
        "degenerate at a* = 0 (sigma* = 0): for A=1 (kappa_r = 0) the "
        "source Jacobian is nilpotent and all eigenvalues vanish "
        "(neutral, growth of s under a CAV-density perturbation is "
        "secular/algebraic, not exponential), and for A=10 the "
        "spectrum is {0, 0, -mu} (marginally stable); the dispersion "
        "diagnostic is therefore uninformative for af on the open "
        "road, which is honest cause to prefer the lf baseline here.  "
        "R3 (rho > rho_c_slow) is transport-only and neutrally stable."
    )

    summary = {
        "meta": {
            "model": "two-class LWR transition model, state (a, f, s), "
                     "Lambda(k) = eig(R_rho - i k q_rho)",
            "params_si": {"v_f_ms": V_F, "w_ms": W, "P_vehm": P,
                          "u_xi_ms": U_XI},
            "rho_c_fast_veh_per_km": RHO_CF * 1e3,
            "rho_c_slow_veh_per_km": RHO_CS * 1e3,
            "kappa_source": "out/ev3_kappa.json point estimates "
                            "(kappa_cl_mod / kappa_cA_mod / "
                            "kappa_r_mod [0])",
            "kappa_sets": {k: {kk: v[kk] for kk in
                               ("form", "tag", "kc", "kr")}
                           for k, v in KSETS.items()},
            "k_grid_rad_per_km": [float(K_RADKM[0]), float(K_RADKM[-1]),
                                  len(K_RADKM)],
            "stability_tol_1_per_s": TOL_STAB,
            "R3_statement": "deep congested rho > rho_c_slow: Delta = 0,"
                            " source off, transport-only; all modes "
                            "neutrally stable (Re Lambda = 0); "
                            "not scanned",
            "checks": checks,
        },
        "transcritical": trans,
        "results": results,
        "interpretation": interpretation,
    }
    (OUT_EV5 / "dispersion_summary.json").write_text(
        json.dumps(summary, indent=2))
    fig_dispersion(OUT_EV5 / "fig_dispersion.png")
    fig_stability_map(OUT_EV5 / "fig_stability_map.png", results)
    print(f"wrote {OUT_EV5 / 'dispersion_summary.json'}")
    print(f"wrote {OUT_EV5 / 'fig_dispersion.png'}")
    print(f"wrote {OUT_EV5 / 'fig_stability_map.png'}")


if __name__ == "__main__":
    main()
