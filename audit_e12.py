#!/usr/bin/env python3
"""audit_e12.py -- independent numerical audit of the E12 experiment
(E12_plan.md section 4).  Re-computes selected numbers of out/e12/ladder.json,
hypothesis.json and summary.md from the underlying modules and checks the
purity of every fitted configuration.  Prints one PASS/FAIL line per check;
exit code 0 iff every check passes.  Read-only: no output file is touched.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "out" / "e12"
REL = 1e-9
A_KEYS = [f"A{a}" for a in range(1, 11)]
KEY_FIT, KEY_TR = "u15_q2500", "u15_q2000"
SCEN_KEYS = ("u15_q2500", "u15_q2000", "u20_q2000", "u20_q2500")
RESULTS = []


def check(name, ok, msg=""):
    ok = bool(ok)
    RESULTS.append((name, ok, msg))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)
    return ok


def close(a, b, rel=REL):
    a, b = float(a), float(b)
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)


def rel_err(a, b):
    a, b = float(a), float(b)
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


L = json.loads((OUT / "ladder.json").read_text())
H = json.loads((OUT / "hypothesis.json").read_text())

import e12_assertiveness as e12          # noqa: E402  (runs the default guard)
import e7_ablation as e7a                # noqa: E402
import e7_wasserstein as e7              # noqa: E402
import ev4_compare as ev4                # noqa: E402
import solver                            # noqa: E402


# ---------------------------------------------------------------------------
# (1) purity
# ---------------------------------------------------------------------------

def check_purity():
    n, bad = 0, []
    for k in A_KEYS:
        for name, f in L[k]["fits"].items():
            cfg = e12.cr_config(15, 2500, f["kappa_c"], f["kappa_r"],
                                f["gamma"], f["ws_frac"], dt=0.5)
            n += 1
            pure = (cfg.q_xi_max is None and cfg.downstream_release is False
                    and cfg.eta_la is None and cfg.s_impermeable is False)
            if not pure:
                bad.append((k, name))
    check("1a purity of every fitted SimConfig (10 A x C1..C4)", not bad,
          f"{n} configs rebuilt at dt=0.5, violations={bad}")
    pat = re.compile(r"downstream_release=True|s_impermeable=True|eta_la=\s*[0-9.]")
    ctx = re.compile(r"PURE_OFF|assert_pure|test|purity")
    hits, n_lines = [], 0
    for fn in ("e12_assertiveness.py", "e12_figures.py"):
        for i, line in enumerate((HERE / fn).read_text().splitlines(), 1):
            n_lines += 1
            if pat.search(line) and not ctx.search(line):
                hits.append(f"{fn}:{i}: {line.strip()}")
    check("1b forbidden knob strings absent from e12_*.py", not hits,
          f"{n_lines} lines scanned, hits={hits}")
    a_set = sorted((k for k in L if k != "_meta"), key=lambda s: int(s[1:]))
    check("1c ladder.json A set == A1..A10", a_set == A_KEYS
          and L["_meta"]["A_levels"] == list(range(1, 11)),
          f"keys={a_set}, meta A_levels={L['_meta']['A_levels']}")
    ok, seen = True, []
    for m in ("M_c", "M_r", "M_c_ws", "M_r_ws"):
        ok &= sorted(H[m]["ladder"]) == sorted(A_KEYS)
        seen.append(m)
    ok &= sorted(H["M_both"]["ladder_kc"]) == sorted(A_KEYS)
    ok &= sorted(H["M_both"]["ladder_kr"]) == sorted(A_KEYS)
    for m in ("M_none", "M_c", "M_r", "M_both", "M_none_ws", "M_c_ws", "M_r_ws"):
        ok &= sorted(H[m]["per_A_W1_dt1"]) == sorted(A_KEYS)
        ok &= sorted(H[m]["per_A_W1_dt05"]) == sorted(A_KEYS)
    check("1d hypothesis.json ladders / per-A tables keyed A1..A10", ok,
          "ladders of M_c, M_r, M_c_ws, M_r_ws, M_both(kc,kr) + per_A_W1 of 7 models")
    cl = {k: L[k]["classical"][KEY_FIT]["model"] for k in A_KEYS}
    check("1e classical block is the only capped model", all(v == "classical"
          for v in cl.values()) and L["_meta"]["forbidden_knobs"] == dict(
              q_xi_max="off", downstream_release="off", eta_la="off",
              s_impermeable="off"), f"meta forbidden_knobs={L['_meta']['forbidden_knobs']}")


# ---------------------------------------------------------------------------
# (2) reproduction of classical and winner W1 at dt = 0.5
# ---------------------------------------------------------------------------

def classical_independent(qin):
    cfg = solver.SimConfig(v_f=ev4.V_F, w=ev4.W, P=ev4.P, q_in=qin / 3600.0,
                           u_xi=15, kappa_c=0, kappa_r=0, capture_form="lf",
                           dt=0.5, save_every=20, q_xi_max=2000.0 / 3600.0)
    return ev4.regrid_sim(solver.simulate(cfg))


def check_reproduction():
    for A in (1, 5, 10):
        k = f"A{A}"
        win = L[k]["winner"]
        f = L[k]["fits"][win]
        for qin in (2500, 2000):
            sk = f"u15_q{qin}"
            tt, rho_mean = e7.load_rho_mean(A, 15, qin)
            regr = classical_independent(qin)
            w1 = e7.w1_mean(regr["rho_tot"], rho_mean, tt)
            st = L[k]["classical"][sk]["W1"]
            check(f"2 classical W1 A={A} {sk}", close(w1, st),
                  f"recomputed {w1:.9f} stored {st:.9f} rel {rel_err(w1, st):.1e}")
            regr = e7.run_sim(15, qin, f["kappa_c"], f["kappa_r"], "lf",
                              gamma=f["gamma"], ws_frac=f["ws_frac"], dt=0.5)
            w1 = e7.w1_mean(regr["rho_tot"], rho_mean, tt)
            st = f["eval"][sk]["W1"]
            check(f"2 winner {win} W1 A={A} {sk}", close(w1, st),
                  f"kc={f['kappa_c']:.4e} kr={f['kappa_r']:.4e} gamma={f['gamma']} "
                  f"ws={f['ws_frac']} recomputed {w1:.9f} stored {st:.9f} "
                  f"rel {rel_err(w1, st):.1e}")


# ---------------------------------------------------------------------------
# (3) C1 grid landscape (dt = 1)
# ---------------------------------------------------------------------------

def check_grid():
    rng = np.random.default_rng(0)
    for A in (1, 10):
        k = f"A{A}"
        g = L[k]["grid_C1"]
        kc_g, kr_g, W = g["kc_grid"], g["kr_grid"], np.asarray(g["W1"], float)
        tt, rho_mean = e7.load_rho_mean(A, 15, 2500)
        idx = rng.choice(W.size, size=6, replace=False)
        worst, cells = 0.0, []
        for flat in idx:
            i, j = divmod(int(flat), len(kr_g))
            regr = e7.run_sim(15, 2500, kc_g[i], kr_g[j], "lf", dt=1.0)
            w1 = e7.w1_mean(regr["rho_tot"], rho_mean, tt)
            worst = max(worst, rel_err(w1, W[i, j]))
            cells.append(f"({i},{j}) {w1:.4f}|{W[i, j]:.4f}")
        check(f"3 grid_C1 6 random cells A={A}", worst <= REL,
              f"max rel err {worst:.1e}; " + ", ".join(cells))
        obj = L[k]["fits"]["C1"]["fit"]["objective_dt1"]
        gobj = L[k]["fits"]["C1"]["fit"]["grid_objective_dt1"]
        check(f"3 C1 objective_dt1 <= grid min A={A}", obj <= g["min"] + 1e-9,
              f"objective_dt1 {obj:.6f} grid min {g['min']:.6f}")
        check(f"3 grid min consistency A={A}", close(g["min"], W.min())
              and close(gobj, g["min"]),
              f"stored min {g['min']:.6f} W.min {W.min():.6f} "
              f"fit.grid_objective_dt1 {gobj:.6f}")


# ---------------------------------------------------------------------------
# (4) winner rule
# ---------------------------------------------------------------------------

def check_winner():
    ok, msg = True, []
    for k in A_KEYS:
        rows = {}
        for name, f in L[k]["fits"].items():
            ev = f["eval"][KEY_FIT]
            rows[name] = dict(W1=ev["W1"], d1=dict(wake_mean_vehkm=ev["wake"]),
                              d2=ev["d2"])
        w, rule = e7a.pick_winner(rows)
        same = (w == L[k]["winner"]) and (rule == L[k]["winner_rule"])
        ok &= same
        msg.append(f"{k}:{w}{'' if same else '!=' + L[k]['winner']}")
    check("4 e7_ablation.pick_winner re-applied, winner + rule string", ok,
          " ".join(msg))


# ---------------------------------------------------------------------------
# (5) hypothesis test
# ---------------------------------------------------------------------------

def _spear(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(np.asarray(x, float), np.asarray(y, float))
    return float(r[0]), float(r[1])


def check_hypothesis():
    models = ("M_none", "M_c", "M_r", "M_both", "M_none_ws", "M_c_ws", "M_r_ws")
    for dt_key, per_key in (("totals_dt1", "per_A_W1_dt1"),
                            ("totals_dt05", "per_A_W1_dt05")):
        worst, parts = 0.0, []
        for m in models:
            s = sum(H[m][per_key][k] for k in A_KEYS)
            worst = max(worst, rel_err(s, H[dt_key][m]))
            parts.append(f"{m}={H[dt_key][m]:.4f}")
        check(f"5 {dt_key} == sum of {per_key} (7 models)", worst <= REL,
              f"max rel err {worst:.1e}; " + ", ".join(parts))
    T = H["totals_dt1"]
    tol = 0.005
    pairs = (("M_none", "M_c"), ("M_c", "M_both"), ("M_none", "M_r"),
             ("M_r", "M_both"), ("M_none_ws", "M_c_ws"), ("M_none_ws", "M_r_ws"))
    ok, exact, parts = True, True, []
    for a, b in pairs:
        ok &= T[a] >= T[b] * (1 - tol)
        exact &= T[a] >= T[b]
        parts.append(f"{a} {T[a]:.2f} >= {b} {T[b]:.2f}")
    check("5 nestedness of sum W1 (dt=1), tolerance 0.5 %", ok,
          f"exact(no tol)={exact}; " + "; ".join(parts))
    A = np.arange(1, 11, dtype=float)
    kc = np.array([L[k]["fits"]["C1"]["kappa_c"] for k in A_KEYS])
    kr = np.array([L[k]["fits"]["C1"]["kappa_r"] for k in A_KEYS])
    todo = {("C1", "kappa_c"): kc, ("C1", "kappa_r"): kr, ("C1", "ratio"): kc / kr,
            ("M_c", "kappa_c"): [H["M_c"]["ladder"][k] for k in A_KEYS],
            ("M_r", "kappa_r"): [H["M_r"]["ladder"][k] for k in A_KEYS]}
    for sk in ("u15_q2500", "u15_q2000"):
        ev = H["events"][sk]
        for q in ("kappa_c", "kappa_r"):
            todo[(f"events_{sk}", q)] = [ev[k][q][0] for k in A_KEYS]
    worst, parts = 0.0, []
    for (lad, q), y in todo.items():
        rho, p = _spear(A, y)
        st = H["trends"][lad][q]
        worst = max(worst, rel_err(rho, st["rho"]), rel_err(p, st["p"]))
        parts.append(f"{lad}.{q} rho={rho:+.4f} p={p:.4f}")
        if st["n"] != 10:
            worst = 1.0
    check("5 Spearman(A, ladder) recomputed: C1 kc/kr/ratio, M_c, M_r, events",
          worst <= REL, f"max rel err {worst:.1e}; " + "; ".join(parts))
    kr_sh = H["M_c"]["shared"]["kr"]
    zs = np.linspace(-2.5, 2.0, 41)
    scan = np.array([e12.w1_one(3, 10.0 ** z, kr_sh) for z in zs])
    st3 = H["M_c"]["per_A_W1_dt1"]["A3"]
    check("5 M_c A=3: stored W1 <= 1.01 x 41-point log10 kappa_c scan minimum",
          st3 <= scan.min() * 1.01,
          f"stored {st3:.4f} scan min {scan.min():.4f} at log10 kc="
          f"{zs[int(np.argmin(scan))]:+.3f} (kr shared {kr_sh:.4e}), "
          f"scan max {scan.max():.2f}")
    w1_3 = e12.w1_one(3, H["M_c"]["ladder"]["A3"], kr_sh)
    check("5 M_c A=3: per_A_W1_dt1 == w1_one(3, ladder.A3, shared.kr)",
          close(w1_3, st3), f"recomputed {w1_3:.9f} stored {st3:.9f} "
          f"rel {rel_err(w1_3, st3):.1e} (kc={H['M_c']['ladder']['A3']:.4e})")


# ---------------------------------------------------------------------------
# (6) event metrics of the A = 5 winner
# ---------------------------------------------------------------------------

def check_metrics():
    k = "A5"
    win = L[k]["winner"]
    f = L[k]["fits"][win]
    regr = e7.run_sim(15, 2500, f["kappa_c"], f["kappa_r"], "lf",
                      gamma=f["gamma"], ws_frac=f["ws_frac"], dt=0.5)
    met = ev4.metrics(regr, e12.load_measured_any(5, 15, 2500))
    ev = f["eval"][KEY_FIT]
    got = dict(e_s=met["e_s"]["mean"], omega_err=met["omega_cum_rel_err"]["mean"],
               rho_rmse=met["rho_rmse"]["mean"], Ns_mae=met["Ns_mae"]["mean"])
    worst = max(rel_err(v, ev[q]) for q, v in got.items())
    check(f"6 A=5 winner {win} e_s / omega_err (+ rho_rmse, Ns_mae) via ev4.metrics",
          worst <= REL, f"max rel err {worst:.1e}; " + ", ".join(
              f"{q} {v:.6f}|{ev[q]:.6f}" for q, v in got.items()))


# ---------------------------------------------------------------------------
# (7) summary.md and figures
# ---------------------------------------------------------------------------

def _section(text, head, nxt):
    i = text.index(head)
    j = text.index(nxt, i)
    return text[i:j]


def check_summary():
    text = (OUT / "summary.md").read_text()
    sec1 = _section(text, "## 1.", "## 2.")
    rows = [ln for ln in sec1.splitlines()
            if ln.startswith("| ") and ln.split("|")[1].strip().isdigit()]
    n_cells, bad = 0, []
    for ln in rows:
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        k = f"A{c[0]}"
        f = L[k]["fits"][L[k]["winner"]]
        if c[1] != L[k]["winner"]:
            bad.append(f"{k} winner {c[1]}")
        for sk, cell in zip(SCEN_KEYS, c[6:10]):
            mark = "(worse)" in cell
            cl_s, cr_s = cell.replace("**(worse)**", "").split("/")
            cl, cr = L[k]["classical"][sk]["W1"], f["eval"][sk]["W1"]
            n_cells += 2
            if float(cl_s) != float(f"{cl:.1f}") or float(cr_s) != float(f"{cr:.1f}"):
                bad.append(f"{k} {sk} {cell} vs {cl:.1f}/{cr:.1f}")
            if mark != (cr > cl):
                bad.append(f"{k} {sk} worse-mark {mark}")
    check("7 summary section 1: W1 cl/CR x 4 scenarios x 10 A at 0.1 rounding",
          len(rows) == 10 and not bad, f"{len(rows)} rows, {n_cells} numbers, bad={bad}")
    sec5 = _section(text, "## 5.", "Verdict:")
    rows = [ln for ln in sec5.splitlines() if ln.startswith("| M_")]
    bad, seen = [], []
    for ln in rows:
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        m = c[0]
        seen.append(m)
        if int(c[3]) != H[m]["n_params"]:
            bad.append(f"{m} n_params {c[3]}")
        if float(c[4]) != float(f"{H['totals_dt1'][m]:.1f}"):
            bad.append(f"{m} dt1 {c[4]} vs {H['totals_dt1'][m]:.1f}")
        if float(c[5]) != float(f"{H['totals_dt05'][m]:.1f}"):
            bad.append(f"{m} dt05 {c[5]} vs {H['totals_dt05'][m]:.1f}")
    check("7 summary section 5: totals dt=1 / dt=0.5 / n_params vs hypothesis.json",
          sorted(seen) == sorted(H["totals_dt1"]) and not bad,
          f"models={seen}, bad={bad}")
    pngs = sorted(OUT.glob("fig_e12_*.png"))
    bad = []
    for p in pngs:
        sz = p.stat().st_size
        if not p.with_suffix(".pdf").exists() or sz <= 20_000:
            bad.append(f"{p.name} ({sz} B, pdf={p.with_suffix('.pdf').exists()})")
    check("7 figures: every fig_e12_*.png has a .pdf twin and is > 20 kB",
          len(pngs) >= 8 and not bad,
          f"{len(pngs)} png: " + ", ".join(f"{p.name} {p.stat().st_size // 1024} kB"
                                           for p in pngs) + f"; bad={bad}")


def main():
    for fn in (check_purity, check_reproduction, check_grid, check_winner,
               check_hypothesis, check_metrics, check_summary):
        try:
            fn()
        except Exception as exc:                   # a crash is a failed check
            check(f"{fn.__name__} raised", False, f"{type(exc).__name__}: {exc}")
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed"
          + ("" if n_fail == 0 else f", {n_fail} FAILED:"))
    for name, ok, _ in RESULTS:
        if not ok:
            print(f"  FAIL {name}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
