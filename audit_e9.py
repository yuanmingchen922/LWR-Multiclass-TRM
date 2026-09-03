"""E9 audit: independent probes of the nonlocal LEADER-LOSS release (eta_la)
in solver.py -- the state-only replacement for downstream_release.

Run: python3 audit_e9.py          (exit code 0 == all probes passed)

Probes (numbering follows the audit brief):

P1  the test suite (test_solver.py) is green: 20 tests collected, 20 pass.
P2  eta_la=None bit-identity vs the PRE-change solver reconstructed from
    git HEAD (git show HEAD:solver.py), across ALL SimResult fields, for
    three configs: (a) the E8 final hybrid A1 config (cap + w_s), (b) an
    all-legacy config, (c) the E8 hybrid WITH downstream_release=True.
    Also: out/ll_ref_A1.npz really is the git-HEAD solver's output (sha256
    of the source and every stored array), and reaction_exact with
    mu_extra=None / mu_extra=0 is bit-identical to HEAD's reaction_exact.
P3  GENERALITY.  (a) static: AST inspection of leader_loss_rate and of the
    eta_la block in simulate(): the only inputs are the state arrays
    (a_star, s) and cfg constants (dx, eta_la, w, w_s); no x_cav,
    cav_position, _cav_weights, cav_density, t, u_s, t_slow/t_fast/t_enter
    anywhere on the code path; the function has no closure and its global
    names are numpy/builtins only.  (b) synthetic: a hand-built state
    (no schedule, no simulate) with TWO separated s-platoons and an A
    vehicle written by hand into the a array ahead of platoon 1: mu_ll is
    exactly 0 on platoon 1 (head included); on platoon 2 (nothing ahead)
    mu_ll = c_wave/ell_eff exactly at the head, a (n-k)/n ramp over the
    last n cells, exactly 0 in the interior, at the rear and behind it;
    the telescoping identity sum(mu s dx) = c_wave s0 holds to round-off;
    one reaction step with dv = 0 converts s -> f only on platoon 2's
    head; translation of the whole state shifts mu bit-identically
    (no positional dependence).  (c) informational: the leader-loss rate
    at the CAV's own straddling cells during the slow phase of the A1 run.
P4  grid independence of the t18-style start-up wave: dL/dt at dx = 50 /
    dt = 0.5 vs dx = 25 / dt = 0.25 (also 100/1.0 and 12.5/0.125) agree
    within 15% and sit at -w; the naive prefactor c_wave/eta_la (probed by
    monkeypatching the rate in memory, solver.py untouched) gives the
    grid-dependent c_wave (n+1)/(2n) the docstring predicts.
P5  invariants under stress (q_in = 2900/3600, u_xi = 10) with cap + w_s
    + eta_la: ledger 1e-10, f, s >= 0, rho <= P, no NaN in any field
    (x_cav NaN exactly on the off-road saves), for A1 and A10 kappas,
    eta_la = 50 (n = 1) / 200 / 1000 (n = 20), + downstream_release on
    top, and at dx = 25 / dt = 0.25; np.seterr(raise) active throughout.
P6  exact-update property with mu_total = mu + mu_ll: on a frozen state
    from the A1 eta_la=200 run (queue formed, CAV on road), reaction_exact
    vs a 2000-step RK4 of the frozen linear ODE, per cell, over dt = 0.5
    and 5.0 s, to 1e-10; p conserved and positivity kept; same on a random
    state with large mu_extra, gamma and 'af' paths.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

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
# Pre-change solver reconstructed from git HEAD
# --------------------------------------------------------------------------

def load_head_solver():
    src = subprocess.run(["git", "show", "HEAD:solver.py"], cwd=HERE,
                         capture_output=True, text=True, check=True).stdout
    sha = hashlib.sha256(src.encode()).hexdigest()
    path = HERE / "out" / "_solver_head_e9_audit.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("solver_head_e9", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["solver_head_e9"] = mod   # dataclass needs it registered
    spec.loader.exec_module(mod)
    fields = mod.SimConfig.__dataclass_fields__
    assert "eta_la" not in fields, "HEAD solver unexpectedly has eta_la"
    assert "downstream_release" in fields, "HEAD solver lacks E8 flag"
    assert not hasattr(mod, "leader_loss_rate")
    return mod, sha


# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------

def kw_A1(**kw):
    """E8 final hybrid A1 config (out/e8/final_config.json; test _ll_cfg):
    cap 2000 veh/h + w_s = 0.6 w + W1-fitted kappas, downstream_release
    OFF (the new results use eta_la only)."""
    d = dict(v_f=V_F, w=W, P=P, q_in=2500.0 / 3600.0, u_xi=15.0,
             kappa_c=5.32e-2, kappa_r=1.65e-3, capture_form="lf",
             dt=0.5, save_every=20, w_s=0.6 * W,
             q_xi_max=2000.0 / 3600.0, downstream_release=False)
    d.update(kw)
    return d


def kw_A10(**kw):
    """E8 final hybrid A10 kappas (same u15 / q2500 scenario)."""
    return kw_A1(kappa_c=1.66e-1, kappa_r=1.40e-2, **kw)


def kw_legacy(**kw):
    """All-legacy config: every structural knob at its default."""
    d = dict(v_f=V_F, w=W, P=P, q_in=2500.0 / 3600.0, u_xi=15.0,
             kappa_c=0.026, kappa_r=3e-5)
    d.update(kw)
    return d


# --------------------------------------------------------------------------
# P1: suite green
# --------------------------------------------------------------------------

def probe_suite():
    print("[P1] test suite (pytest test_solver.py)")
    r = subprocess.run([sys.executable, "-m", "pytest", "test_solver.py",
                        "-q", "-p", "no:cacheprovider"],
                       cwd=HERE, capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    print(f"  pytest: {tail}")
    check(r.returncode == 0, "pytest exit code 0")
    check("20 passed" in r.stdout, "exactly 20 tests passed")
    check("failed" not in r.stdout and "error" not in r.stdout.lower(),
          "no failures / errors reported")


# --------------------------------------------------------------------------
# P2: eta_la=None bit-identity vs git HEAD, all fields
# --------------------------------------------------------------------------

def probe_bit_identity(sol_head, head_sha):
    print("[P2] eta_la=None bit-identity vs git-HEAD solver (ALL fields)")
    configs = (("A1 hybrid (cap + w_s)", kw_A1()),
               ("all-legacy", kw_legacy()),
               ("A1 hybrid + downstream_release", kw_A1(downstream_release=True)))
    for label, kw in configs:
        res_head = sol_head.simulate(sol_head.SimConfig(**kw))
        res_new = sol_new.simulate(sol_new.SimConfig(**kw))     # default None
        res_new_x = sol_new.simulate(
            sol_new.SimConfig(**kw, eta_la=None))                # explicit None
        compare_all_fields(res_new, res_head, f"{label} (default)")
        compare_all_fields(res_new_x, res_head, f"{label} (explicit None)")

    # the stored test reference really is the pre-change solver's output
    ref = np.load(HERE / "out" / "ll_ref_A1.npz")
    check(str(ref["solver_sha256"]) == head_sha,
          f"out/ll_ref_A1.npz solver_sha256 == sha256(git HEAD:solver.py) "
          f"({head_sha[:12]}...)")
    res_head_A = sol_head.simulate(sol_head.SimConfig(**kw_A1()))
    for name in ref.files:
        if name == "solver_sha256":
            continue
        check(bit_equal(ref[name], getattr(res_head_A, name)),
              f"out/ll_ref_A1.npz['{name}'] == git-HEAD solver output")

    # reaction_exact: mu_extra=None and mu_extra=0 bit-identical to HEAD
    rng = np.random.default_rng(7)
    n = 800
    f = rng.uniform(0.0, 0.12, n)
    s = rng.uniform(0.0, 0.12, n)
    a = np.where(rng.uniform(size=n) < 0.05, rng.uniform(0.0, 0.02, n), 0.0)
    rho = a + f + s
    dv = rng.uniform(-2.0, 13.0, n)
    for form, gamma in (("lf", None), ("af", None), ("lf", 0.37)):
        fh, sh = sol_head.reaction_exact(f, s, a, rho, dv, 5.32e-2, 1.65e-3,
                                         P, 0.5, form, gamma)
        fn, sn = sol_new.reaction_exact(f, s, a, rho, dv, 5.32e-2, 1.65e-3,
                                        P, 0.5, form, gamma)
        fz, sz = sol_new.reaction_exact(f, s, a, rho, dv, 5.32e-2, 1.65e-3,
                                        P, 0.5, form, gamma,
                                        mu_extra=np.zeros(n))
        check(bit_equal(fh, fn) and bit_equal(sh, sn),
              f"reaction_exact({form}, gamma={gamma}) mu_extra=None == HEAD")
        check(bit_equal(fh, fz) and bit_equal(sh, sz),
              f"reaction_exact({form}, gamma={gamma}) mu_extra=0 == HEAD")


# --------------------------------------------------------------------------
# P3a: GENERALITY -- static inspection of the code path
# --------------------------------------------------------------------------

SCENARIO_NAMES = {"cfg", "t", "x_cav", "x_c", "xc", "cav_position",
                  "_cav_weights", "cav_density", "u_s", "u_s_of_t",
                  "t_slow", "t_fast", "t_enter", "v_cav_free", "j_cav",
                  "n_steps", "step", "f", "rho", "rho_star", "dv",
                  "q_in", "u_xi", "L_road", "t_end"}


def _attr_chain(node):
    """'cfg.eta_la' style dotted name for an Attribute node, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def probe_generality_static():
    print("[P3a] generality -- static inspection of the leader-loss path")
    src = (HERE / "solver.py").read_text()
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    fn = fns["leader_loss_rate"]

    params = [a.arg for a in fn.args.args]
    check(params == ["a", "s", "dx", "eta_la", "c_wave"],
          f"leader_loss_rate signature is state arrays + constants: {params}")
    check(fn.args.vararg is None and fn.args.kwarg is None
          and not fn.args.kwonlyargs,
          "leader_loss_rate has no *args/**kwargs back door")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    bad = (names | attrs) & SCENARIO_NAMES
    check(not bad, f"leader_loss_rate references no scenario/time name "
                   f"(offenders: {sorted(bad)})")
    allowed = set(params) | {"np", "max", "int", "round", "range", "float",
                             "nx", "n", "a_pad", "s_pad", "a_ahead", "s_sum",
                             "k", "s_ahead", "cov", "ell_eff"}
    check(names <= allowed,
          f"leader_loss_rate free names are numpy/builtins/locals only "
          f"(extra: {sorted(names - allowed)})")
    # body text (docstring stripped) has no 'cav' / 'x_cav' at all
    body_src = "\n".join(ast.get_source_segment(src, st) or ""
                         for st in fn.body[1:])
    check("cav" not in body_src.lower() and "x_cav" not in body_src,
          "leader_loss_rate body text contains no 'cav'")
    # runtime object: no closure, global names numpy/builtins only
    code = sol_new.leader_loss_rate.__code__
    check(code.co_freevars == (), "leader_loss_rate has no closure variables")
    glob = set(code.co_names)
    check(glob <= {"np", "asarray", "size", "max", "int", "round", "float",
                   "concatenate", "zeros", "range", "where", "minimum",
                   "maximum"},
          f"leader_loss_rate global names (numpy/builtins only): {sorted(glob)}")
    print(f"  line {fn.lineno}: def leader_loss_rate({', '.join(params)})")

    # the eta_la block in simulate()
    sim = fns["simulate"]
    blocks = [n for n in ast.walk(sim) if isinstance(n, ast.If)
              and "eta_la" in (ast.get_source_segment(src, n.test) or "")]
    check(len(blocks) == 1, "exactly one 'if cfg.eta_la is not None' block")
    blk = blocks[0]
    b_names = {n.id for n in ast.walk(blk) if isinstance(n, ast.Name)}
    b_attrs = {_attr_chain(n) for n in ast.walk(blk)
               if isinstance(n, ast.Attribute)}
    b_attrs.discard(None)
    check(b_names <= {"cfg", "mu_ll", "leader_loss_rate", "a_star", "s",
                      "c_wave"},
          f"eta_la block names: {sorted(b_names)}")
    check(b_attrs <= {"cfg.eta_la", "cfg.w", "cfg.w_s", "cfg.dx"},
          f"eta_la block cfg attributes: {sorted(b_attrs)} (constants only)")
    calls = [n for n in ast.walk(blk) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "leader_loss_rate"]
    check(len(calls) == 1, "leader_loss_rate called exactly once in simulate")
    args = [ast.get_source_segment(src, a) for a in calls[0].args]
    check(args == ["a_star", "s", "cfg.dx", "cfg.eta_la", "c_wave"]
          and not calls[0].keywords,
          f"call args (line {calls[0].lineno}): {args}")
    # mu_ll is consumed only by reaction_exact(mu_extra=...)
    uses = [n for n in ast.walk(sim) if isinstance(n, ast.Name)
            and n.id == "mu_ll" and isinstance(n.ctx, ast.Load)]
    kwuse = [n for n in ast.walk(sim) if isinstance(n, ast.keyword)
             and n.arg == "mu_extra"
             and isinstance(n.value, ast.Name) and n.value.id == "mu_ll"]
    check(len(uses) == 1 and len(kwuse) == 1,
          f"mu_ll consumed once, as reaction_exact(mu_extra=mu_ll) "
          f"(line {kwuse[0].lineno if kwuse else '?'})")
    # global text scan: every 'cav_position' / 'x_cav' use sits outside
    # leader_loss_rate and outside the eta_la block
    doc = fn.body[0]
    doc_lines = (set(range(doc.lineno, doc.end_lineno + 1))
                 if isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant)
                 else set())
    ll_lines = set(range(fn.lineno, fn.end_lineno + 1)) - doc_lines
    blk_lines = set(range(blk.lineno, blk.end_lineno + 1))
    hits = [i + 1 for i, line in enumerate(src.splitlines())
            if ("cav_position" in line or "x_cav" in line
                or "_cav_weights" in line)
            and (i + 1 in ll_lines or i + 1 in blk_lines)
            and not line.strip().startswith("#")]
    check(not hits, f"no cav_position/x_cav/_cav_weights code lines inside "
                    f"the leader-loss path (hits: {hits})")
    print(f"  eta_la block: lines {blk.lineno}-{blk.end_lineno}; "
          f"leader_loss_rate: lines {fn.lineno}-{fn.end_lineno}")
    print("  note: the A class enters ONLY through the state array a_star "
          "(the exogenous a at t + dt, the same array the reaction uses); "
          "the schedule is not consulted by the term itself")


# --------------------------------------------------------------------------
# P3b: GENERALITY -- synthetic two-platoon state, hand-placed A vehicle
# --------------------------------------------------------------------------

def probe_generality_synthetic():
    print("[P3b] generality -- synthetic two-platoon state, manual A vehicle")
    dx, nx, eta = 50.0, 400, 200.0
    n = max(1, int(round(eta / dx)))
    c_wave = 0.6 * W
    ell_eff = dx * (n + 1) / 2.0
    mu_max = c_wave / ell_eff
    s0 = 0.04
    a = np.zeros(nx)
    s = np.zeros(nx)
    f = np.zeros(nx)
    # platoon 1: cells 100..119, an A vehicle written BY HAND into cell 120
    # (the cell immediately ahead of the head: the CAV-at-queue-head layout)
    s[100:120] = s0
    a[120] = 1.0 / dx
    # platoon 2: cells 250..269, nothing ahead within eta (and beyond)
    s[250:270] = s0
    mu = sol_new.leader_loss_rate(a, s, dx, eta, c_wave)

    check(np.all(mu >= 0.0) and np.all(np.isfinite(mu)), "mu_ll >= 0, finite")
    check(mu[119] == 0.0, "platoon 1 head (A within eta ahead): mu_ll == 0.0 exactly")
    check(np.all(mu[100:120] == 0.0), "platoon 1, every cell: mu_ll == 0.0 exactly")
    check(mu[269] == mu_max,
          f"platoon 2 head (nothing ahead): mu_ll == c_wave/ell_eff = "
          f"{mu_max:.6f} 1/s exactly")
    ramp = mu_max * np.array([(n - k) / n for k in range(n)])[::-1]
    check(np.max(np.abs(mu[270 - n:270] - ramp)) <= 1e-15,
          f"platoon 2 last {n} cells: (n-k)/n ramp "
          f"{np.round(mu[270 - n:270] / mu_max, 3).tolist()} of c_wave/ell_eff")
    check(np.all(mu[250:270 - n] == 0.0),
          "platoon 2 interior (more than n cells behind the head): exactly 0")
    check(mu[250] == 0.0, "platoon 2 rear cell: exactly 0")
    check(np.all(mu[250 - n:250] == 0.0) and np.all(mu[100 - n:100] == 0.0),
          f"the {n} empty cells right behind each platoon (s = 0, s ahead): exactly 0")
    # empty cells whose whole window is empty (behind both platoons, the A
    # vehicle's own cell, the road ahead) carry mu_max -- but s = 0 there,
    # so the release is moot (checked in the reaction step below)
    idx = np.arange(nx)
    empty_win = (s == 0.0) & (a == 0.0) & ((idx < 100 - n) | ((idx > 120 + 0) & (idx < 250 - n)) | (idx > 269))
    check(np.all(mu[empty_win] == mu_max) and np.all(s[mu == mu_max] * (idx[mu == mu_max] != 269) == 0.0),
          f"cells with an empty window carry mu_max ({int(empty_win.sum())} cells) but s = 0 there (moot)")
    # telescoping identity: platoon 2 mass loss rate == c_wave s0
    loss = np.sum(mu[250:270] * s[250:270]) * dx
    check(abs(loss - c_wave * s0) <= 1e-14 * c_wave * s0,
          f"telescoping: sum(mu s dx) over platoon 2 = {loss:.10f} veh/s == "
          f"c_wave s0 = {c_wave * s0:.10f} (front speed exactly c_wave)")
    # the brief's expectation c_wave/eta vs the implemented c_wave/ell_eff
    print(f"  NOTE head rate = c_wave/ell_eff with ell_eff = dx (n+1)/2 = "
          f"{ell_eff:.1f} m, i.e. {mu_max / (c_wave / eta):.3f} x the brief's "
          f"c_wave/eta ({c_wave / eta:.6f} 1/s); continuum limit 2 c_wave/eta. "
          f"Documented in the leader_loss_rate docstring (telescoping); the "
          f"naive prefactor would give front speed c_wave (n+1)/(2n) = "
          f"{(n + 1) / (2 * n):.4f} c_wave -- checked in P4b.")

    # one reaction step, dv = 0 (post-t_fast: no kappa terms): only platoon
    # 2's last n cells lose s, p conserved bitwise, platoon 1 untouched
    dt = 0.5
    rho = a + f + s
    f1, s1 = sol_new.reaction_exact(f, s, a, rho, np.zeros(nx), 5.32e-2,
                                    1.65e-3, P, dt, "lf", None, mu_extra=mu)
    check(bit_equal(s1[100:120], s[100:120]) and bit_equal(f1[100:120], f[100:120]),
          "reaction step: platoon 1 bit-unchanged")
    check(np.all(s1[270 - n:270] < s0) and np.all(f1[270 - n:270] > 0.0),
          f"reaction step: platoon 2 last {n} cells release s -> f")
    check(bit_equal(s1[250:270 - n], s[250:270 - n]),
          "reaction step: platoon 2 interior bit-unchanged")
    check(np.all((f1 + s1) == (f + s)), "reaction step: p = f + s conserved bitwise")
    check(abs(s1[269] - s0 * np.exp(-mu_max * dt)) <= 1e-16,
          f"head cell decays exactly as s0 exp(-mu_max dt) "
          f"({s1[269]:.10f} vs {s0 * np.exp(-mu_max * dt):.10f})")

    # translation invariance: shift the entire state by 37 cells -> mu
    # shifts bit-identically (no positional / scenario dependence)
    sh = 37
    mu_sh = sol_new.leader_loss_rate(np.roll(a, sh), np.roll(s, sh), dx, eta, c_wave)
    check(bit_equal(mu_sh[sh:], mu[:nx - sh]),
          "translation of the state by 37 cells shifts mu_ll bit-identically")
    # removing the hand-placed A vehicle exposes platoon 1's head
    a0 = np.zeros(nx)
    mu_noA = sol_new.leader_loss_rate(a0, s, dx, eta, c_wave)
    check(mu_noA[119] == mu_max and bit_equal(mu_noA[250:270], mu[250:270]),
          "removing the A vehicle: platoon 1 head -> mu_max, platoon 2 unchanged")
    # a tiny A weight (as from a linear split) still counts as full coverage
    a_t = np.zeros(nx)
    a_t[120] = 1e-300
    check(sol_new.leader_loss_rate(a_t, s, dx, eta, c_wave)[119] == 0.0,
          "a = 1e-300 in the window counts as full coverage (strict > 0 test)")
    # A vehicle ahead but with a GAP (still within eta of the head): the
    # head is covered, but cells whose window does not reach the A cell see
    # the gap and are partially released -- report (behaviour, not a bug)
    a_g = np.zeros(nx)
    a_g[122] = 1.0 / dx          # 3 cells ahead of the head (gap of 2)
    mu_g = sol_new.leader_loss_rate(a_g, s, dx, eta, c_wave)
    leak = np.where(mu_g[100:120] > 0.0)[0] + 100
    print(f"  gap case (A 3 cells ahead of the head, gap 2 < n = {n}): head "
          f"mu = {mu_g[119]:.3e}; leaking cells {leak.tolist()} with mu/mu_max "
          f"= {np.round(mu_g[leak] / mu_max, 3).tolist()} (windows that end "
          f"before the A cell see the gap; head-adjacent cells are covered)")
    check(mu_g[119] == 0.0 and mu_g[118] == 0.0,
          "gap case: the 2 cells whose window (j+1..j+4) reaches cell 122 are covered")
    check(abs(mu_g[117] - 0.5 * mu_max) <= 1e-15 and abs(mu_g[116] - 0.25 * mu_max) <= 1e-15
          and mu_g[115] == 0.0,
          "gap case: cells 117/116 see the gap (0.5 / 0.25 mu_max), 115 fully covered")
    # rounding of the window length: Python round() is half-to-even
    n_vals = {eta: max(1, int(round(eta / dx))) for eta in (125.0, 175.0, 225.0)}
    print(f"  NOTE n = round(eta_la/dx) uses half-to-even rounding: "
          f"{ {k: v for k, v in n_vals.items()} } (eta 125 -> n 2, 175 -> 4, 225 -> 4)")
    # A vehicle beyond eta: no coverage from it
    a_f = np.zeros(nx)
    a_f[125] = 1.0 / dx          # 6 cells ahead of the head > n = 4
    check(sol_new.leader_loss_rate(a_f, s, dx, eta, c_wave)[119] == mu_max,
          "A vehicle beyond eta (6 cells ahead): head released at mu_max")
    # road-end padding: a platoon touching the last cell is released at its
    # head (beyond the road end counts as empty)
    s_e = np.zeros(nx)
    s_e[nx - 10:] = s0
    check(sol_new.leader_loss_rate(np.zeros(nx), s_e, dx, eta, c_wave)[nx - 1] == mu_max,
          "platoon touching the road end: last cell released at mu_max")
    # sub-cell horizons: eta < dx/2 -> n = 1 (no crash, one-cell window)
    mu1 = sol_new.leader_loss_rate(a, s, dx, 10.0, c_wave)
    check(mu1[269] == c_wave / dx and mu1[268] == 0.0,
          "eta_la = 10 m (< dx): n = 1, head rate c_wave/dx, cell behind 0")


# --------------------------------------------------------------------------
# P3c: informational -- mu_ll at the CAV's own cells in the A1 run
# --------------------------------------------------------------------------

def probe_cav_cells(res, cfg):
    print("[P3c] informational -- leader-loss rate at the CAV straddling cells")
    dx = cfg.dx
    c_wave = cfg.w_s
    ell_eff = dx * (round(cfg.eta_la / dx) + 1) / 2.0
    mu_max = c_wave / ell_eff
    rows = []
    for i in range(res.t.size):
        t = res.t[i]
        if not (cfg.t_slow <= t <= cfg.t_fast) or not np.isfinite(res.x_cav[i]):
            continue
        cells = np.where(res.a[i] > 0.0)[0]
        if cells.size == 0:
            continue
        mu = sol_new.leader_loss_rate(res.a[i], res.s[i], dx, cfg.eta_la, c_wave)
        jr, jf = cells.min(), cells.max()
        rows.append((t, jr, jf, res.a[i][jr] * dx, res.a[i][jf] * dx,
                     mu[jr] / mu_max, mu[jf] / mu_max,
                     res.s[i][jr] * 1e3, res.s[i][jf] * 1e3))
    print("     t    rear_cell  fwd_cell  w_rear  w_fwd  mu_rear/max  mu_fwd/max"
          "  s_rear  s_fwd [veh/km]")
    for r in rows[::5]:
        print(f"  {r[0]:6.0f}  {r[1]:8d}  {r[2]:8d}  {r[3]:6.3f} {r[4]:6.3f}"
              f"  {r[5]:10.3f}  {r[6]:10.3f}  {r[7]:7.2f} {r[8]:7.2f}")
    fwd = np.array([r[6] for r in rows])
    rear = np.array([r[5] for r in rows])
    s_r = np.array([r[7] for r in rows])
    s_f = np.array([r[8] for r in rows])
    print(f"  slow-phase saves: {len(rows)}; rear straddling cell mu_ll == 0 "
          f"on {np.sum(rear == 0.0)} (covered by the forward A weight); "
          f"forward straddling cell mu_ll/mu_max mean {fwd.mean():.3f} "
          f"(min {fwd.min():.3f}, max {fwd.max():.3f}; only diffused s in "
          f"j+2..j+n+1 covers it): s co-located with the A vehicle in its "
          f"forward cell (mean {s_f.mean():.2f} vs rear {s_r.mean():.2f} "
          f"veh/km) is released at ~{fwd.mean() * mu_max:.4f} 1/s (time scale "
          f"{1 / (fwd.mean() * mu_max):.0f} s) -- a sub-grid ambiguity of "
          f"'ahead' (window is strictly j+1..j+n, the own cell is not a "
          f"leader), vanishing as dx -> 0; downstream_release kept s in the "
          f"CAV cell j_cav.")
    check(np.all(rear == 0.0), "rear straddling cell always covered (mu_ll == 0)")
    check(np.all(fwd > 0.5), "forward straddling cell always > 0.5 mu_max (reported behaviour)")


# --------------------------------------------------------------------------
# P4: grid independence of the start-up wave
# --------------------------------------------------------------------------

def startup_wave(dx: float, dt: float, eta: float = 200.0):
    nx = int(round(30000.0 / dx))
    xc = (np.arange(nx) + 0.5) * dx
    s_blk = 0.04
    s0 = np.where((xc >= 10000.0) & (xc <= 12000.0), s_blk, 0.0)
    cfg = sol_new.SimConfig(v_f=V_F, w=W, P=P, q_in=0.0, u_xi=15.0,
                            kappa_c=0.0, kappa_r=0.0, t_enter=1e9,
                            t_end=150.0, dx=dx, dt=dt,
                            save_every=max(1, int(round(1.0 / dt))),
                            s0=s0, eta_la=eta)
    res = sol_new.simulate(cfg)
    assert np.all(res.a == 0.0)
    L = np.sum(res.s > 0.5 * s_blk, axis=1) * dx
    m = (res.t >= 20.0) & (res.t <= 120.0)
    rate = np.polyfit(res.t[m], L[m], 1)[0]
    # downstream edge position (first cell from the right above 0.5 s0)
    down = np.array([res.x[np.where(res.s[i] > 0.5 * s_blk)[0][-1]]
                     for i in np.where(m)[0]])
    v_down = np.polyfit(res.t[m], down, 1)[0]
    i120 = int(np.argmin(np.abs(res.t - 120.0)))
    mass = np.sum(res.f + res.s, axis=1) * dx
    rel_f = np.sum(res.f[i120]) * dx / mass[0]
    m_drift = np.max(np.abs(mass - mass[0])) / mass[0]
    return dict(rate=rate, v_down=v_down, rel_f=rel_f, m_drift=m_drift,
                fmin=res.f.min(), smin=res.s.min(), nx=nx)


def probe_grid_independence():
    print("[P4] grid independence of the start-up wave (t18-style)")
    out = {}
    for dx, dt in ((100.0, 1.0), (50.0, 0.5), (25.0, 0.25), (12.5, 0.125)):
        r = startup_wave(dx, dt)
        out[dx] = r
        print(f"  dx = {dx:5.1f} m, dt = {dt:5.3f} s (n = {round(200 / dx)}, "
              f"nx = {r['nx']}): dL/dt = {r['rate']:8.4f} m/s = "
              f"{-r['rate'] / W:.4f} w; downstream edge {r['v_down']:.3f} m/s "
              f"(v_f - w = {V_F - W:.3f}); released to f by 120 s "
              f"{100 * r['rel_f']:.1f}%; mass drift {r['m_drift']:.1e}; "
              f"min f {r['fmin']:.1e}, min s {r['smin']:.1e}")
        check(r["m_drift"] <= 1e-12 and r["fmin"] >= 0.0 and r["smin"] >= 0.0,
              f"dx = {dx}: mass exact, positivity")
    r50, r25 = out[50.0]["rate"], out[25.0]["rate"]
    diff = abs(r50 - r25) / abs(r25)
    check(diff < 0.15,
          f"dL/dt at dx = 50 vs 25 agree within 15% ({100 * diff:.2f}%)")
    for dx in (100.0, 50.0, 25.0, 12.5):
        check(abs(out[dx]["rate"] + W) / W < 0.15,
              f"dx = {dx}: dL/dt within 15% of -w ({100 * abs(out[dx]['rate'] + W) / W:.2f}% off)")
    d_rel = abs(out[50.0]["rel_f"] - out[25.0]["rel_f"]) / out[25.0]["rel_f"]
    check(d_rel < 0.15,
          f"released fraction at 120 s: dx = 50 {100 * out[50.0]['rel_f']:.1f}% "
          f"vs dx = 25 {100 * out[25.0]['rel_f']:.1f}% ({100 * d_rel:.1f}% apart)")

    # P4b: the naive prefactor c_wave/eta_la, probed by rescaling the rate
    # IN MEMORY (solver.py untouched, restored afterwards)
    print("[P4b] naive prefactor c_wave/eta_la (in-memory monkeypatch)")
    orig = sol_new.leader_loss_rate

    def naive(a, s, dx, eta_la, c_wave):
        n = max(1, int(round(eta_la / dx)))
        return orig(a, s, dx, eta_la, c_wave) * (dx * (n + 1) / 2.0) / eta_la

    try:
        sol_new.leader_loss_rate = naive
        for dx, dt in ((50.0, 0.5), (25.0, 0.25)):
            n = round(200 / dx)
            r = startup_wave(dx, dt)
            pred = (n + 1) / (2 * n)
            print(f"  dx = {dx:5.1f}: dL/dt = {-r['rate'] / W:.4f} w "
                  f"(docstring prediction (n+1)/(2n) = {pred:.4f} w)")
            check(abs(-r["rate"] / W - pred) < 0.08,
                  f"naive prefactor at dx = {dx}: front speed ~ (n+1)/(2n) w, "
                  f"i.e. grid-dependent and NOT w")
    finally:
        sol_new.leader_loss_rate = orig
    check(sol_new.leader_loss_rate is orig, "leader_loss_rate restored")


# --------------------------------------------------------------------------
# P5: invariants under stress
# --------------------------------------------------------------------------

def invariants(res, cfg, what: str) -> None:
    balance = res.on_road + res.outflowed
    rel = abs(res.injected - balance) / max(res.injected, 1.0)
    rho = res.a + res.f + res.s
    for name in ALL_FIELDS:
        v = np.asarray(getattr(res, name), float)
        if name == "x_cav":
            exp_nan = np.array([not np.isfinite(sol_new.cav_position(cfg, t))
                                for t in res.t])
            check(np.array_equal(np.isnan(v), exp_nan),
                  f"{what}: x_cav NaN exactly on the off-road saves "
                  f"({int(exp_nan.sum())} of {v.size})")
        else:
            check(np.all(np.isfinite(v)), f"{what}: no NaN/inf in '{name}'")
    print(f"  {what}: ledger rel err {rel:.2e}, min f {res.f.min():.2e}, "
          f"min s {res.s.min():.2e}, max rho {rho.max():.6f} (P {P:.6f}), "
          f"N_s max {res.N_s.max():.1f} veh, N_s end {res.N_s[-1]:.2f} veh")
    check(rel < 1e-10, f"{what}: mass ledger closes to 1e-10")
    check(res.f.min() >= 0.0 and res.s.min() >= 0.0, f"{what}: f, s >= 0")
    check(rho.max() <= P + 1e-12, f"{what}: rho <= P")
    check(res.N_s.max() > 5.0, f"{what}: a queue actually formed (N_s max > 5 veh)")


def probe_invariants():
    print("[P5] invariants under stress (q_in = 2900/3600, u_xi = 10)")
    stress = dict(q_in=2900.0 / 3600.0, u_xi=10.0)
    cases = (
        ("A1 cap+w_s+eta200", kw_A1(eta_la=200.0, **stress)),
        ("A10 cap+w_s+eta200", kw_A10(eta_la=200.0, **stress)),
        ("A1 cap+w_s+eta50 (n=1)", kw_A1(eta_la=50.0, **stress)),
        ("A1 cap+w_s+eta1000 (n=20)", kw_A1(eta_la=1000.0, **stress)),
        ("A1 cap+w_s+eta200+downstream_release", kw_A1(eta_la=200.0,
                                                       downstream_release=True,
                                                       **stress)),
        ("A1 cap+w_s+eta200 dx25/dt0.25", kw_A1(eta_la=200.0, dx=25.0,
                                                dt=0.25, save_every=40,
                                                **stress)),
        ("A1 no-cap legacy-w +eta200", kw_A1(eta_la=200.0, q_xi_max=None,
                                             w_s=None, **stress)),
    )
    for what, kw in cases:
        cfg = sol_new.SimConfig(**kw)
        res = sol_new.simulate(cfg)
        invariants(res, cfg, what)
    # config-time guard
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        try:
            sol_new.SimConfig(**kw_A1(eta_la=bad))
            check(False, f"config assert fires for eta_la = {bad}")
        except AssertionError as e:
            check("eta_la" in str(e), f"config assert fires for eta_la = {bad}")


# --------------------------------------------------------------------------
# P6: exact-update property with mu_total = mu + mu_ll
# --------------------------------------------------------------------------

def rk4_frozen(f0, s0, sigma, mu_tot, T: float, n_rk: int = 2000):
    """Vectorised classical RK4 of f' = mu_tot s - sigma f, s' = -f'."""
    h = T / n_rk
    f = np.array(f0, float)
    s = np.array(s0, float)

    def rhs(ff, ss):
        d = mu_tot * ss - sigma * ff
        return d, -d

    for _ in range(n_rk):
        k1f, k1s = rhs(f, s)
        k2f, k2s = rhs(f + 0.5 * h * k1f, s + 0.5 * h * k1s)
        k3f, k3s = rhs(f + 0.5 * h * k2f, s + 0.5 * h * k2s)
        k4f, k4s = rhs(f + h * k3f, s + h * k3s)
        f = f + h / 6.0 * (k1f + 2 * k2f + 2 * k3f + k4f)
        s = s + h / 6.0 * (k1s + 2 * k2s + 2 * k3s + k4s)
    return f, s


def probe_exact_update(res, cfg):
    print("[P6] exact-update property with mu_total = mu + mu_ll (2000-step RK4)")
    kc, kr = cfg.kappa_c, cfg.kappa_r
    c_wave = cfg.w if cfg.w_s is None else cfg.w_s
    for t_probe in (500.0, 900.0):
        i = int(np.argmin(np.abs(res.t - t_probe)))
        a, f, s = res.a[i], res.f[i], res.s[i]
        rho = a + f + s
        u_s = sol_new.u_s_of_t(cfg, res.t[i])
        dv = np.maximum(sol_new.speed(rho, V_F, W, P) - u_s, 0.0)
        mu_ll = sol_new.leader_loss_rate(a, s, cfg.dx, cfg.eta_la, c_wave)
        sigma = kc * (a + s) * dv
        mu_tot = kr * np.maximum(P - rho, 0.0) * dv + mu_ll
        active = int(np.sum((mu_ll > 0.0) & (s > 1e-6)))
        print(f"  frozen state t = {res.t[i]:.0f} s: u_s = {u_s:.1f}, "
              f"N_s = {res.N_s[i]:.1f} veh, cells with mu_ll > 0 and s > 0: "
              f"{active}, max mu_ll {mu_ll.max():.4f}, max sigma {sigma.max():.4f}, "
              f"max mu_tot {mu_tot.max():.4f} 1/s")
        check(active > 0, f"t = {res.t[i]:.0f}: leader-loss term active somewhere")
        for T in (cfg.dt, 5.0):
            fe, se = sol_new.reaction_exact(f, s, a, rho, dv, kc, kr, P, T,
                                            cfg.capture_form, cfg.gamma,
                                            mu_extra=mu_ll)
            fr, sr = rk4_frozen(f, s, sigma, mu_tot, T)
            err = max(np.max(np.abs(fe - fr)), np.max(np.abs(se - sr)))
            p_err = np.max(np.abs((fe + se) - (f + s)))
            change = np.max(np.abs(fe - f))
            check(err < 1e-10,
                  f"t = {res.t[i]:.0f}, T = {T}: exact vs RK4 max err {err:.2e} "
                  f"(max |f* - f| = {change:.3e})")
            check(p_err <= 1e-16 * max(1.0, np.max(f + s)),
                  f"t = {res.t[i]:.0f}, T = {T}: p conserved ({p_err:.1e})")
            check(fe.min() >= 0.0 and se.min() >= 0.0,
                  f"t = {res.t[i]:.0f}, T = {T}: positivity")
        # mu_extra=None vs mu_extra=mu_ll differ where mu_ll s > 0 (term live)
        f_n, _ = sol_new.reaction_exact(f, s, a, rho, dv, kc, kr, P, cfg.dt,
                                        cfg.capture_form, cfg.gamma)
        f_l, _ = sol_new.reaction_exact(f, s, a, rho, dv, kc, kr, P, cfg.dt,
                                        cfg.capture_form, cfg.gamma,
                                        mu_extra=mu_ll)
        live = (mu_ll * s) > 1e-12
        check(np.all(f_l[live] > f_n[live]) and bit_equal(f_l[~live & (s == 0.0)],
                                                          f_n[~live & (s == 0.0)]),
              f"t = {res.t[i]:.0f}: mu_ll strictly increases f where mu_ll s > 0 "
              f"({int(live.sum())} cells), no effect where s = 0")

    # random frozen state, large mu_extra, gamma and 'af' paths
    rng = np.random.default_rng(11)
    n = 600
    f = rng.uniform(0.0, 0.1, n)
    s = rng.uniform(0.0, 0.1, n)
    a = rng.uniform(0.0, 0.02, n)
    rho = a + f + s
    dv = rng.uniform(0.0, 13.0, n)
    mu_x = rng.uniform(0.0, 1.0, n)
    for form, gamma in (("lf", None), ("af", None), ("lf", 0.4)):
        ell = (a + gamma * s) if gamma is not None else ((a + s) if form == "lf" else a)
        sigma = 0.03 * ell * dv
        mu_tot = 0.01 * np.maximum(P - rho, 0.0) * dv + mu_x
        for T in (0.5, 5.0):
            fe, se = sol_new.reaction_exact(f, s, a, rho, dv, 0.03, 0.01, P, T,
                                            form, gamma, mu_extra=mu_x)
            fr, sr = rk4_frozen(f, s, sigma, mu_tot, T)
            err = max(np.max(np.abs(fe - fr)), np.max(np.abs(se - sr)))
            check(err < 1e-10,
                  f"random state ({form}, gamma={gamma}), T = {T}, mu_extra up "
                  f"to 1/s: exact vs RK4 max err {err:.2e}")
            check(np.max(np.abs((fe + se) - (f + s))) <= 1e-16
                  and fe.min() >= 0.0 and se.min() >= 0.0,
                  f"random state ({form}, gamma={gamma}), T = {T}: p conserved, positive")


# --------------------------------------------------------------------------

def main() -> int:
    # trap invalid/divide/overflow (would surface NaN/inf); underflow to
    # denormals is benign and expected at vacuum densities
    np.seterr(invalid="raise", divide="raise", over="raise", under="ignore")
    probe_suite()
    sol_head, head_sha = load_head_solver()
    probe_bit_identity(sol_head, head_sha)
    probe_generality_static()
    probe_generality_synthetic()
    cfg_ll = sol_new.SimConfig(**kw_A1(eta_la=200.0))
    res_ll = sol_new.simulate(cfg_ll)
    probe_cav_cells(res_ll, cfg_ll)
    probe_grid_independence()
    probe_invariants()
    probe_exact_update(res_ll, cfg_ll)
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
