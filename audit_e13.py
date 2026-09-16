"""Independent numerical audit of E13 v2 (out/e13).  Standalone; touches no
output.  Exit code 0 iff every check passes.  One line per check."""
from __future__ import annotations
import glob, json, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e13_joint as e13
import e12_assertiveness as e12
import e7_wasserstein as e7
import ev4_compare as ev4

OUT = HERE / "out" / "e13"
L = json.loads((OUT / "ladder13.json").read_text())
SEL = json.loads((OUT / "lambda_star.json").read_text())
H = json.loads((OUT / "hypothesis13.json").read_text())
X = json.loads((OUT / "extras13.json").read_text())
PRE = json.loads((OUT / "prehull13.json").read_text())
SUMMARY = (OUT / "summary13.md").read_text()
AKEYS = [f"A{a}" for a in range(1, 11)]
PLAN_LAMBDAS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)
LAM_S = float(SEL["lambda_star"])
UC, QIN = 15.0, 2500.0
RESULTS = []


def check(name, ok, detail=""):
    ok = bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    RESULTS.append((name, ok, detail))
    return ok


def rd(a, b):
    """relative difference with an absolute floor of 1 (== abs diff for |b| < 1)."""
    if a is None or b is None:
        return 0.0 if a is b else np.inf
    return abs(float(a) - float(b)) / max(1.0, abs(float(b)))


def penalty(p, wN=1.0, wO=1.0, wR=1.0):
    return wN * p["Ns_mae"] / p["N_0"] + wO * abs(p["omega_err"]) / 0.2 + wR * p["eps_R"]


def my_parts(regr, A):
    """Metrics of a regridded run recomputed from first principles."""
    k = f"A{A:g}"
    ref = L[k]["ref"]
    tt, rho_mean_e7 = e7.load_rho_mean(A, UC, QIN)
    rho_mean_data = e13._data(A)[1]
    meas = e12.load_measured_any(A, UC, QIN)
    met = ev4.metrics(regr, meas)
    om = met["omega_cum_rel_err"]["mean"]
    om = 0.0 if om is None or not np.isfinite(om) else float(om)
    i0 = int(np.argmin(np.abs(tt - 260.0))); i1 = int(np.argmin(np.abs(tt - 740.0)))
    C = float(regr["cum_cap"][i1] - regr["cum_cap"][i0])
    R = float(regr["cum_rel"][i1] - regr["cum_rel"][i0])
    eps_R = abs(np.log10((R + 1) / (ref["R_d"] + 1))) + abs(np.log10((C + 1) / (ref["C_d"] + 1)))
    w1 = float(e7.w1_mean(regr["rho_tot"], rho_mean_data, tt))
    w1_e7 = float(e7.w1_mean(regr["rho_tot"], rho_mean_e7, tt))
    return dict(W1=w1, W1_e7=w1_e7, W1_rel=w1 / L[k]["W1_cl"], Ns_mae=float(met["Ns_mae"]["mean"]),
                omega_err=om, C_m=C, R_m=R, eps_R=float(eps_R), N_0=ref["N_0"])


def winner(k, lam):
    blk = L[k]["lambdas"][e13.lam_key(lam)]
    return blk["fits"][blk["winner"]], blk


# ---------------------------------------------------------------------------
# (1) PURITY
# ---------------------------------------------------------------------------
def check_purity():
    n, bad, disc = 0, [], []
    for k in AKEYS:
        for lk, blk in L[k]["lambdas"].items():
            for cname, fit in blk["fits"].items():
                cfg = e12.cr_config(UC, QIN, fit["kappa_c"], fit["kappa_r"], fit["gamma"],
                                    fit["ws_frac"], dt=0.5)
                n += 1
                if not (cfg.q_xi_max is None and cfg.downstream_release is False
                        and cfg.eta_la is None and cfg.s_impermeable is False
                        and cfg.kappa_c > 0 and cfg.kappa_r > 0):
                    bad.append((k, lk, cname))
                want_g = cname in ("C2", "C4"); want_w = cname in ("C3", "C4")
                if (fit["gamma"] is not None) != want_g or (fit["ws_frac"] is not None) != want_w:
                    disc.append((k, lk, cname))
    check("1a purity of every stored config (q_xi_max None, DR False, eta_la None, s_imp False)",
          n == 480 and not bad, f"{n} configs rebuilt, {len(bad)} violations {bad[:3]}")
    check("1b config discipline (gamma only in C2/C4, ws_frac only in C3/C4)", not disc,
          f"{len(disc)} deviations {disc[:3]}")
    A_set = [k for k in L if k != "_meta"]
    check("1c A set == A1..A10", A_set == AKEYS and L["_meta"]["A_levels"] == list(range(1, 11)),
          f"{A_set}")
    lam_ok = all(list(L[k]["lambdas"]) == [e13.lam_key(l) for l in PLAN_LAMBDAS]
                 and all(L[k]["lambdas"][e13.lam_key(l)]["lam"] == l for l in PLAN_LAMBDAS)
                 for k in AKEYS)
    check("1d lambda sweep == plan v2 (12 values)",
          lam_ok and tuple(L["_meta"]["lambdas"]) == PLAN_LAMBDAS and tuple(e13.LAMBDAS) == PLAN_LAMBDAS
          and tuple(SEL["sweep"][i]["lam"] for i in range(12)) == PLAN_LAMBDAS
          and tuple(H["_meta"]["lambdas"]) == PLAN_LAMBDAS,
          f"{L['_meta']['lambdas']}")
    check("1e forbidden knobs listed in meta", set(L["_meta"]["forbidden_knobs"]) == set(e12.PURE_OFF),
          f"{L['_meta']['forbidden_knobs']}")


# ---------------------------------------------------------------------------
# (2) COUNTERS
# ---------------------------------------------------------------------------
def check_counters():
    regr = e12.run_cr(UC, QIN, 0.5, 0.03, None, None, 1.0)
    ident = float(np.max(np.abs(regr["cum_cap"] - regr["cum_rel"] - regr["N_s"])))
    check("2a cum_cap - cum_rel == N_s (A=5 kc=0.5 kr=0.03 dt=1)", ident < 1e-8, f"max|.| = {ident:.3e}")
    C, R, tau = e13.window_counts(regr)
    check("2b window_counts C > R > 0", C > R > 0, f"C={C:.4f} R={R:.4f} tau={tau:.4f} s")
    p = e13.parts_of(regr, 5)
    ref = L["A5"]["ref"]
    mine = abs(np.log10((R + 1) / (ref["R_d"] + 1))) + abs(np.log10((C + 1) / (ref["C_d"] + 1)))
    check("2c eps_R of parts_of == formula from C_m, R_m, ref C_d, R_d", rd(p["eps_R"], mine) < 1e-9
          and rd(p["C_m"], C) < 1e-9 and rd(p["R_m"], R) < 1e-9,
          f"parts {p['eps_R']:.12f} vs mine {mine:.12f} (C_d={ref['C_d']}, R_d={ref['R_d']})")


# ---------------------------------------------------------------------------
# (3) REPRODUCTION of winners and classical rows (deterministic, 1e-9)
# ---------------------------------------------------------------------------
def check_reproduction():
    keys = ("W1", "W1_rel", "Ns_mae", "omega_err", "C_m", "R_m", "eps_R")
    worst, worst_e7, fails, rows = 0.0, 0.0, [], []
    for A in (1, 5, 10):
        k = f"A{A}"
        for lam in (0.0, LAM_S):
            w, blk = winner(k, lam)
            regr = e12.run_cr(UC, QIN, w["kappa_c"], w["kappa_r"], w["gamma"], w["ws_frac"], 0.5)
            m = my_parts(regr, A)
            J = m["W1_rel"] + lam * penalty(m)
            d = max(rd(m[x], w["parts_dt05"][x]) for x in keys)
            dJ = rd(J, w["J_dt05"])
            worst = max(worst, d, dJ)
            worst_e7 = max(worst_e7, rd(m["W1_e7"], w["parts_dt05"]["W1"]))
            if d > 1e-9 or dJ > 1e-9:
                fails.append((k, lam, d, dJ))
            rows.append(f"{k} lam={lam:g} {blk['winner']} J={J:.6f}/{w['J_dt05']:.6f} W1={m['W1']:.3f}")
    check("3a winner parts_dt05 + J_dt05 reproduced (A=1,5,10; lam=0, lam*)", not fails,
          f"max rel diff {worst:.2e}; " + "; ".join(rows))
    check("3b W1 via e7.load_rho_mean vs stored (loader agreement, tol 1e-6)", worst_e7 < 1e-6,
          f"max rel diff {worst_e7:.2e} (rho_mean loaders differ by ~2e-6 veh/km for A not in {{1,10}})")
    worst, fails = 0.0, []
    for A in (1, 5, 10):
        k = f"A{A}"
        regr = e12.run_classical(UC, QIN)
        m = my_parts(regr, A)
        cl = L[k]["classical"]["u15_q2500"]
        d = max(rd(m[x], cl[x]) for x in keys)
        d = max(d, rd(L[k]["W1_cl"], cl["W1"]), rd(m["W1"], L[k]["W1_cl"]))
        for lam in (0.0, LAM_S):
            d = max(d, rd(1.0 + lam * penalty(m), L[k]["lambdas"][e13.lam_key(lam)]["J_classical"]))
        worst = max(worst, d)
        if d > 1e-9:
            fails.append((k, d))
    check("3c classical parts, W1_cl and J_classical = 1 + lam P_cl reproduced (A=1,5,10)", not fails,
          f"max rel diff {worst:.2e}")


# ---------------------------------------------------------------------------
# (4) FLAGS and SWEEP
# ---------------------------------------------------------------------------
def my_flags(p, cl, J, Jcl):
    return dict(field_ok=p["W1_rel"] <= 1.0,
                collapsed=(p["Ns_mae"] <= cl["Ns_mae"] + 2.0 and p["W1_rel"] > 1.10),
                in_box=(p["Ns_mae"] <= 5.0 and abs(p["omega_err"]) <= 0.10),
                events_ok=p["eps_R"] <= 0.5, beats_classical_J=J < Jcl)


def my_sweep():
    rows = []
    for lam in PLAN_LAMBDAS:
        r = dict(lam=lam, sum_W1_rel=0.0, sum_penalty=0.0, sum_J=0.0, sum_J_classical=0.0,
                 n_field_ok=0, n_collapsed=0, n_in_box=0, n_events_ok=0, n_beats_classical=0)
        W1, Ns, om, eR, taus = [], [], [], [], []
        for k in AKEYS:
            w, blk = winner(k, lam)
            p, cl = w["parts_dt05"], L[k]["classical"]["u15_q2500"]
            fl = my_flags(p, cl, w["J_dt05"], blk["J_classical"])
            r["sum_W1_rel"] += p["W1_rel"]; r["sum_penalty"] += penalty(p); r["sum_J"] += w["J_dt05"]
            r["sum_J_classical"] += blk["J_classical"]
            for f in ("field_ok", "collapsed", "in_box", "events_ok", "beats_classical"):
                r["n_" + f] += int(fl[f if f != "beats_classical" else "beats_classical_J"])
            W1.append(p["W1"]); Ns.append(p["Ns_mae"]); om.append(abs(p["omega_err"]))
            eR.append(p["eps_R"]); taus.append(p["tau_model_s"])
        r.update(mean_W1=np.mean(W1), mean_Ns_mae=np.mean(Ns), mean_abs_omega=np.mean(om),
                 mean_eps_R=np.mean(eR))
        taus = [t for t in taus if t is not None]
        r["median_tau_s"] = float(np.median(taus)) if taus else None
        rows.append(r)
    return rows


def chord_knee(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xn = (x - x.min()) / (np.ptp(x) or 1.0); yn = (y - y.min()) / (np.ptp(y) or 1.0)
    p0, p1 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    d = p1 - p0; n = np.linalg.norm(d)
    dist = np.array([abs(np.cross(d, np.array([xn[i], yn[i]]) - p0)) / n for i in range(len(x))])
    inner = np.arange(1, len(x) - 1)
    return int(inner[np.argmax(dist[inner])]), dist


def check_flags_sweep():
    bad, badJ, badw = [], [], []
    for k in AKEYS:
        cl = L[k]["classical"]["u15_q2500"]
        for lam in PLAN_LAMBDAS:
            w, blk = winner(k, lam)
            fl = my_flags(w["parts_dt05"], cl, w["J_dt05"], blk["J_classical"])
            if any(bool(fl[f]) != bool(blk["flags"][f]) for f in fl):
                bad.append((k, lam, fl, blk["flags"]))
            for cname, fit in blk["fits"].items():
                for dt in ("dt1", "dt05"):
                    p = fit["parts_" + dt]
                    if rd(fit["J_" + dt], p["W1_rel"] + lam * penalty(p)) > 1e-9 \
                            or rd(p["W1_rel"], p["W1"] / L[k]["W1_cl"]) > 1e-9:
                        badJ.append((k, lam, cname, dt))
            if blk["winner"] != min(blk["fits"], key=lambda n: blk["fits"][n]["J_dt05"]):
                badw.append((k, lam))
    check("4a flags re-derived for all 120 (A, lam) blocks", not bad, f"{len(bad)} mismatches {bad[:2]}")
    check("4b J_dt1/J_dt05 == W1_rel + lam*penalty(parts) and W1_rel == W1/W1_cl (480 fits x 2 dt)",
          not badJ, f"{len(badJ)} mismatches {badJ[:3]}")
    check("4c winner == argmin J_dt05 over C1..C4", not badw, f"{len(badw)} mismatches {badw[:3]}")
    mine, stored = my_sweep(), SEL["sweep"]
    worst, badc = 0.0, []
    for a, b in zip(mine, stored):
        for key in a:
            if key.startswith("n_"):
                if a[key] != b[key]:
                    badc.append((a["lam"], key, a[key], b[key]))
            else:
                worst = max(worst, rd(a[key], b[key]))
    check("4d sweep rows (sums, means, median tau) recomputed from ladder13", worst < 1e-9 and not badc,
          f"max rel diff {worst:.2e}; count mismatches {badc}")
    nA = 10
    lam_max = next((r["lam"] for r in mine if r["n_collapsed"] >= 0.5 * nA), None)
    sub = [r for r in mine if lam_max is None or r["lam"] < lam_max]
    i_k, dist = e13.knee([r["sum_penalty"] for r in sub], [r["sum_W1_rel"] for r in sub])
    i_k2, dist2 = chord_knee([r["sum_penalty"] for r in sub], [r["sum_W1_rel"] for r in sub])
    lam_knee = sub[i_k]["lam"] if len(sub) >= 3 else None
    dom = [r["lam"] for r in mine if r["n_field_ok"] >= 0.8 * nA]
    lam_dom = max(dom) if dom else None
    lam_star = lam_dom if lam_dom is not None else (lam_knee if lam_knee is not None else 0.0)
    kd_stored = [r["knee_distance"] for r in stored]
    kd_ok = all(rd(dist[i], kd_stored[i]) < 1e-9 for i in range(len(sub))) and i_k == i_k2 \
        and np.allclose(dist, dist2, atol=1e-12)
    check("4e selection rules: lambda_max / dominance / restricted knee / lambda*",
          lam_max == SEL["lambda_max"] and lam_dom == SEL["lambda_dominance"]
          and lam_knee == SEL["lambda_knee_restricted"] and lam_star == SEL["lambda_star"]
          and SEL["rule"] == ("dominance" if lam_dom is not None else "knee") and kd_ok
          and SEL["lambdas_for_B"] == sorted({0.0, lam_star} | ({lam_knee} if lam_knee is not None else set())),
          f"lambda_max={lam_max} dominance={lam_dom} knee={lam_knee} (dist {dist[i_k]:.4f}) "
          f"lambda*={lam_star} rule={SEL['rule']}; n_field_ok by lam "
          f"{[r['n_field_ok'] for r in mine]}; n_collapsed {[r['n_collapsed'] for r in mine]}")
    r0 = mine[0]
    cf = {}
    for lm in (0.2, 0.5, 1.0):
        r1 = next(r for r in mine if r["lam"] == lm)
        dP = r0["sum_penalty"] - r1["sum_penalty"]
        cf[f"lam_max={lm:g}"] = (r1["sum_W1_rel"] - r0["sum_W1_rel"]) / dP if dP > 0 else None
    lm_ref = lam_max if lam_max is not None else 1.0
    pa = {}
    for k in AKEYS:
        w0, _ = winner(k, 0.0); w1, _ = winner(k, lm_ref)
        dP = penalty(w0["parts_dt05"]) - penalty(w1["parts_dt05"])
        pa[k] = (w1["parts_dt05"]["W1_rel"] - w0["parts_dt05"]["W1_rel"]) / dP if dP > 0 else None
    pv = [v for v in pa.values() if v and v > 0]
    spread = max(pv) / min(pv)
    d_cf = max(rd(cf[c], SEL["closed_form_knee"][c]) for c in cf)
    d_pa = max(rd(pa[k], SEL["per_A_exchange_rate"][k]) for k in AKEYS)
    check("4f closed-form knees, per-A exchange rates, spread, two-lambda flag",
          d_cf < 1e-9 and d_pa < 1e-9 and rd(spread, SEL["per_A_spread"]) < 1e-9
          and SEL["two_lambda_ladder"] == (spread > 3.0)
          and SEL["closed_form_agree_x1p5"] == (max(cf.values()) / min(cf.values()) <= 1.5),
          f"closed-form {({c: round(v, 5) for c, v in cf.items()})}; per-A "
          f"{[round(pa[k], 4) for k in AKEYS]}; spread {spread:.4f}; max rel diff {max(d_cf, d_pa):.2e}")


# ---------------------------------------------------------------------------
# (5) NESTED models
# ---------------------------------------------------------------------------
def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(np.asarray(x, float), np.asarray(y, float))
    return float(r.correlation), float(r.pvalue)


def quartet_consistency(q, tag):
    """totals == sums, nestedness, eta, gap, ridge residual, Spearman."""
    errs, worst = [], 0.0
    tot = q["totals_dt1"]
    for M in ("M_none", "M_c", "M_r", "M_both"):
        s = sum(q[M]["per_A_J_dt1"][k] for k in AKEYS)
        worst = max(worst, rd(tot[M], s), rd(q[M]["total_dt1"], s))
    if not (tot["M_none"] >= tot["M_c"] - 1e-9 and tot["M_none"] >= tot["M_r"] - 1e-9
            and tot["M_c"] >= tot["M_both"] - 1e-9 and tot["M_r"] >= tot["M_both"] - 1e-9):
        errs.append("nesting violated")
    if not all(q["nested"].values()):
        errs.append("stored nested flags not all True")
    den = tot["M_none"] - tot["M_both"]
    gap = tot["M_c"] - tot["M_r"]
    worst = max(worst, rd(q["gap_c_minus_r"], gap), rd(q["gap_rel_both"], gap / tot["M_both"]),
                rd(q["eta_c"], (tot["M_none"] - tot["M_c"]) / den),
                rd(q["eta_r"], (tot["M_none"] - tot["M_r"]) / den), rd(q["gain_none_to_both"], den))
    mc, mr = q["M_c"], q["M_r"]
    resid = max(abs(np.log10(mc["ladder"][k] / mc["shared"]["kr"]) - np.log10(mr["shared"]["kc"] / mr["ladder"][k]))
                for k in AKEYS)
    worst = max(worst, rd(q["ridge_residual_dex"], resid))
    A = list(range(1, 11))
    for M, lad in (("M_c", [mc["ladder"][k] for k in AKEYS]), ("M_r", [mr["ladder"][k] for k in AKEYS])):
        rho, p = spearman(A, lad)
        worst = max(worst, rd(q[M]["spearman"]["rho"], rho), rd(q[M]["spearman"]["p"], p))
        mono = all(np.diff(lad) <= 1e-12) if M == "M_c" else all(np.diff(lad) >= -1e-12)
        if q[M]["monotone_ok"] != mono:
            errs.append(f"{M} monotone flag")
    mb = q["M_both"]
    kcs, krs = [mb["ladder_kc"][k] for k in AKEYS], [mb["ladder_kr"][k] for k in AKEYS]
    for nm, v in (("spearman_kc", kcs), ("spearman_kr", krs), ("spearman_ratio", np.array(kcs) / np.array(krs))):
        rho, p = spearman(A, v)
        worst = max(worst, rd(mb[nm]["rho"], rho), rd(mb[nm]["p"], p))
    for k in AKEYS:                     # M_both per-A J == W1_rel + lam*penalty(parts)
        pp = mb["per_A_parts"][k]
        worst = max(worst, rd(mb["per_A_J_dt1"][k], pp["W1_rel"] + q["lam"] * penalty(pp, q["wN"], q["wO"], q["wR"])))
    if worst > 1e-9:
        errs.append(f"max rel diff {worst:.2e}")
    return errs, worst, gap, resid


def check_nested():
    bad, worst = [], 0.0
    for lk, q in H["per_lambda"].items():
        errs, w, gap, resid = quartet_consistency(q, lk)
        worst = max(worst, w)
        if errs:
            bad.append((lk, errs))
    check("5a per-lambda quartets: totals = sum per-A J, nesting, eta, gap, ridge residual, Spearman",
          not bad, f"12 lambdas, max rel diff {worst:.2e}; gaps "
          f"{[round(q['gap_c_minus_r'], 4) for q in H['per_lambda'].values()]}; {bad}")
    bad = []
    for tag, q in H["at_star"].items():
        errs, w, _, _ = quartet_consistency(q, tag)
        worst = max(worst, w)
        wts = dict(robust_ws=(1, 1, 1), queue_only=(1, 0, 0), overtaking_only=(0, 1, 0), events_only=(0, 0, 1))[tag]
        if (q["wN"], q["wO"], q["wR"]) != wts or q["lam"] != LAM_S or (tag == "robust_ws") != (q["ws_frac"] == 0.6):
            errs.append("weights/ws/lam wrong")
        if errs:
            bad.append((tag, errs))
    check("5b at_star ablation quartets consistent (weights, ws, totals, nesting)", not bad,
          f"{list(H['at_star'])}; {bad}")
    gv = H["gap_vs_lambda"]
    ok = len(gv) == 12 and all(rd(g["gap"], H["per_lambda"][e13.lam_key(g["lam"])]["gap_c_minus_r"]) < 1e-12 for g in gv)
    check("5c gap_vs_lambda list mirrors per_lambda", ok, f"{len(gv)} rows")
    q = H["per_lambda"][e13.lam_key(LAM_S)]
    ev = H["events"]["u15_q2500"]
    bad = []
    for which, lad in (("kappa_c", q["M_both"]["ladder_kc"]), ("kappa_r", q["M_both"]["ladder_kr"])):
        macro = np.array([lad[k] for k in AKEYS]); micro = np.array([max(ev[k][which][0], 1e-6) for k in AKEYS])
        dex = np.abs(np.log10(macro / micro)); rho, p = spearman(macro, micro)
        mm = q["micro_macro"][which]
        if rd(mm["median_abs_dex"], np.median(dex)) > 1e-9 or rd(mm["spearman_macro_vs_micro"]["rho"], rho) > 1e-9 \
                or mm["magnitude_agree"] != (np.median(dex) <= 0.5) or mm["trend_agree"] != (rho >= 0.6 and p < 0.05):
            bad.append(which)
    check("5d micro-macro agreement at lambda* recomputed", not bad,
          f"kc median {q['micro_macro']['kappa_c']['median_abs_dex']:.3f} dex, kr median "
          f"{q['micro_macro']['kappa_r']['median_abs_dex']:.3f} dex; {bad}")
    mc = q["M_c"]
    j3 = e13.j_one(3, mc["ladder"]["A3"], mc["shared"]["kr"], LAM_S)
    d3 = rd(j3, mc["per_A_J_dt1"]["A3"])
    mr = q["M_r"]
    j3r = e13.j_one(3, mr["shared"]["kc"], mr["ladder"]["A3"], LAM_S)
    d3r = rd(j3r, mr["per_A_J_dt1"]["A3"])
    mb = q["M_both"]
    j3b = e13.j_one(3, mb["ladder_kc"]["A3"], mb["ladder_kr"]["A3"], LAM_S)
    d3b = rd(j3b, mb["per_A_J_dt1"]["A3"])
    check("5e lambda* A=3: j_one at M_c / M_r / M_both kappas == stored per-A J (dt=1)",
          max(d3, d3r, d3b) < 1e-9, f"M_c {j3:.9f} vs {mc['per_A_J_dt1']['A3']:.9f}; M_r {j3r:.9f}; "
          f"M_both {j3b:.9f}; max rel diff {max(d3, d3r, d3b):.2e}")
    zs = np.linspace(-2.5, 2.0, 41)
    scan = np.array([e13.j_one(3, 10.0 ** z, mc["shared"]["kr"], LAM_S) for z in zs])
    check("5f lambda* A=3: stored M_c per-A J <= 1.01 x min of 41-pt log10 kappa_c scan at shared kappa_r",
          mc["per_A_J_dt1"]["A3"] <= 1.01 * scan.min(),
          f"stored {mc['per_A_J_dt1']['A3']:.6f}, scan min {scan.min():.6f} at log10 kc = {zs[np.argmin(scan)]:.3f} "
          f"(stored log10 kc = {np.log10(mc['ladder']['A3']):.3f})")


# ---------------------------------------------------------------------------
# (6) JACKKNIFE
# ---------------------------------------------------------------------------
def check_jackknife():
    bad, worst = [], 0.0
    for lk, jk in H["jackknife"].items():
        g = np.array([f["gap"] for f in jk["folds"]]); n = len(g)
        se = float(np.sqrt((n - 1) / n * np.sum((g - g.mean()) ** 2)))
        same = bool(np.all(np.sign(g) == np.sign(g[0])) and np.all(g != 0))
        worst = max(worst, rd(jk["gap_mean"], g.mean()), rd(jk["gap_se_jack"], se))
        fold_ok = all(rd(f["gap"], f["totals"]["M_c"] - f["totals"]["M_r"]) < 1e-9 and f["fold"] == i
                      for i, f in enumerate(jk["folds"]))
        if n != 5 or jk["same_sign"] != same or not fold_ok or jk["eta_c"] != [f["eta_c"] for f in jk["folds"]]:
            bad.append(lk)
    check("6a jackknife gap_mean / SE_jack / same_sign / fold gaps recomputed", worst < 1e-9 and not bad,
          f"lambdas {list(H['jackknife'])}, max rel diff {worst:.2e}; " + "; ".join(
              f"{lk}: mean {jk['gap_mean']:+.5f} se {jk['gap_se_jack']:.5f} |g|>2se {abs(jk['gap_mean']) > 2 * jk['gap_se_jack']}"
              for lk, jk in H["jackknife"].items()))
    tt, rm4, meas4, w1cl4, ref4 = e13._data(5, fold=0)
    tt, rm5, meas5, w1cl5, ref5 = e13._data(5)
    files = sorted(glob.glob(str(HERE / "out" / "e4" / "A5_u15_q2500_r*.npz")))
    w = (tt >= 260) & (tt <= 740)
    cap = np.array([np.load(f)["cap"] for f in files], float)
    C_d4 = float(np.mean(cap[1:][:, w].sum(axis=1)))
    check("6b _data(5, fold=0): 4 runs, rho_mean differs from 5-run mean, C_d over runs 1..4",
          ref4["n_runs"] == 4 and len(meas4["reps"]) == 4 and meas4["rho"].shape[0] == 4
          and np.max(np.abs(rm4 - rm5)) > 0 and rd(ref4["C_d"], C_d4) < 1e-9 and ref5["n_runs"] == 5,
          f"n_runs {ref4['n_runs']}, max|rho4 - rho5| = {np.max(np.abs(rm4 - rm5)):.3f} veh/km, "
          f"C_d fold0 {ref4['C_d']:.3f} (mine {C_d4:.3f}) vs all {ref5['C_d']:.3f}")


# ---------------------------------------------------------------------------
# (7) EXTRAS: ridge, events_field, dx sensitivity, turnover
# ---------------------------------------------------------------------------
def check_extras():
    ratios, mags = X["_meta"]["ratios"], X["_meta"]["magnitudes"]
    Wk = np.asarray(X["ridge"]["A5"]["W1"])
    rng = np.random.default_rng(0)
    picks = [(int(rng.integers(len(mags))), int(rng.integers(len(ratios)))) for _ in range(6)]
    worst, rows = 0.0, []
    for im, ir in picks:
        j = e13.j_one(5, ratios[ir] * mags[im], mags[im], LAM_S)
        worst = max(worst, rd(j, Wk[im, ir])); rows.append(f"({mags[im]:g},{ratios[ir]:g}) {j:.6f}/{Wk[im, ir]:.6f}")
    check("7a ridge A=5: 6 random entries recomputed with j_one (dt=1, seed 0)", worst < 1e-9,
          f"max rel diff {worst:.2e}; " + " ".join(rows))
    bad = []
    for k in AKEYS:
        r = X["ridge"][k]; W = np.asarray(r["W1"])
        im, ir = np.unravel_index(np.argmin(W), W.shape)
        line = W[:, ir]; ok = line <= 1.01 * line.min(); lm = np.log10(mags)
        basin = float(lm[ok].max() - lm[ok].min())
        w0, _ = winner(k, LAM_S)
        ident = bool(mags[im] < 0.1 and basin < 1.0 and w0["parts_dt05"]["W1_rel"] <= 1.0)
        if rd(r["min"], W.min()) > 1e-12 or r["argmin"] != dict(magnitude=mags[im], ratio=ratios[ir]) \
                or rd(r["basin_width_dex_along_best_ratio"], basin) > 1e-12 or r["magnitude_identified"] != ident \
                or any(rd(b["J"], W[m].min()) > 1e-12 or b["ratio"] != ratios[int(np.argmin(W[m]))]
                       for m, b in enumerate(r["best_ratio_per_magnitude"])):
            bad.append(k)
    check("7b ridge summaries (min, argmin, best ratio per magnitude, basin, identified) from stored matrix",
          not bad, f"{bad}; identified {[X['ridge'][k]['magnitude_identified'] for k in AKEYS]}")
    ef = X["events_field"]["A10"]; ev = H["events"]["u15_q2500"]["A10"]
    kc, kr = ev["kappa_c"][0], max(ev["kappa_r"][0], 1e-6)
    p = e13.parts_one(10, kc, kr, None, None, 0.5); J = p["W1_rel"] + LAM_S * penalty(p)
    pw = e13.parts_one(10, kc, kr, 0.6, None, 0.5); Jw = pw["W1_rel"] + LAM_S * penalty(pw)
    d = max(rd(J, ef["J_dt05"]), rd(Jw, ef["J_ws_dt05"]), rd(ef["kappa_c"], kc), rd(ef["kappa_r"], ev["kappa_r"][0]),
            max(rd(p[x], ef["parts"][x]) for x in ("W1", "Ns_mae", "omega_err", "eps_R")))
    check("7c events_field A=10: J_dt05 (and +w_s) from event kappas", d < 1e-9,
          f"J {J:.6f}/{ef['J_dt05']:.6f}, J_ws {Jw:.6f}/{ef['J_ws_dt05']:.6f}, max rel diff {d:.2e}")
    w, _ = winner("A5", LAM_S)
    regr, raw = e13.run_at(UC, QIN, w["kappa_c"], w["kappa_r"], w["gamma"], w["ws_frac"], 25.0, 0.5)
    p = e13.parts_of(regr, 5); st = X["dx_sensitivity"]["A5"]["winner_lamstar"]["dx25_dt0.5"]
    d = max(rd(p[x], st[x]) for x in ("W1", "Ns_mae", "omega_err", "eps_R", "tau_model_s"))
    d = max(d, rd(p["W1_rel"] + LAM_S * penalty(p), st["J"]))
    m = my_parts(regr, 5)
    d2 = max(rd(m[x], st[x]) for x in ("W1", "Ns_mae", "omega_err", "eps_R"))
    ch = X["dx_sensitivity"]["A5"]["winner_lamstar"]
    dch = max(rd(ch["change_50_to_25"][x], ch["dx25_dt0.5"][x] - ch["dx50_dt0.5"][x]) for x in ("W1", "Ns_mae", "omega_err"))
    check("7d dx_sensitivity A=5 winner_lamstar dx25_dt0.5 recomputed (run_at + parts_of, and first-principles)",
          max(d, d2, dch) < 1e-9, f"W1 {p['W1']:.4f}/{st['W1']:.4f} Ns {p['Ns_mae']:.4f} om {p['omega_err']:+.4f} "
          f"J {st['J']:.4f}; max rel diff {max(d, d2, dch):.2e}")
    files = sorted(glob.glob(str(HERE / "out" / "e4" / "A5_u15_q2500_r*.npz")))
    Z = [np.load(f) for f in files]; tt = Z[0]["tt"]; wm = (tt >= 260) & (tt <= 740)
    nc = np.array([z["n_caught"] for z in Z], float); rel = np.array([z["rel"] for z in Z], float)
    cap = np.array([z["cap"] for z in Z], float)
    tau = nc[:, wm].sum() * 10.0 / rel[:, wm].sum()
    C_d, R_d = cap[:, wm].sum(1).mean(), rel[:, wm].sum(1).mean(); n_max = nc[:, wm].max(1).mean()
    ref, to = L["A5"]["ref"], X["turnover"]["A5"]
    d = max(rd(tau, ref["tau_data_s"]), rd(tau, to["tau_data_s"]), rd(C_d, ref["C_d"]), rd(R_d, ref["R_d"]),
            rd(C_d, to["C_d"]), rd(max(5.0, n_max), ref["N_0"]), rd(to["C_m_lamstar"], w["parts_dt05"]["C_m"]),
            rd(to["tau_lamstar_s"], w["parts_dt05"]["tau_model_s"]))
    check("7e turnover A=5: tau_data = sum n_caught*10 / sum rel over [260,740] pooled; C_d, R_d, N_0 from npz",
          len(files) == 5 and d < 1e-9, f"tau {tau:.4f} s (stored {ref['tau_data_s']}), C_d {C_d:.2f} R_d {R_d:.2f} "
          f"N_0 {max(5.0, n_max):.2f}; max rel diff {d:.2e}")


# ---------------------------------------------------------------------------
# (8) PREHULL
# ---------------------------------------------------------------------------
def check_prehull():
    P = PRE["A4"]; pts = P["points"]
    rng = np.random.default_rng(1)
    idx = sorted(int(i) for i in rng.choice(len(pts), 5, replace=False))
    worst, rows = 0.0, []
    for i in idx:
        q = pts[i]
        p = e13.parts_one(4, q["kappa_c"], q["kappa_r"], q["ws_frac"], q["gamma"])
        d = max(rd(p[x], q[x]) for x in ("W1", "W1_rel", "Ns_mae", "omega_err", "eps_R", "tau_model_s"))
        worst = max(worst, d); rows.append(f"#{i} {q['slice']} kc={q['kappa_c']:.3g} kr={q['kappa_r']:.3g} W1 {p['W1']:.3f}")
    check("8a prehull A=4: 5 random points recomputed with parts_one (dt=1, seed 1)", worst < 1e-9,
          f"max rel diff {worst:.2e}; " + "; ".join(rows))
    inbox = [r for r in pts if r["Ns_mae"] <= 5.0 and abs(r["omega_err"]) <= 0.10]
    inbox_f = [r for r in inbox if r["W1_rel"] <= 1.0]; inbox_fr = [r for r in inbox_f if r["eps_R"] <= 0.5]
    bp = min(pts, key=lambda r: penalty(dict(r, N_0=L["A4"]["ref"]["N_0"])))   # stored points carry no N_0
    ok = (len(pts) == 260 == P["n_points"] and P["n_in_box"] == len(inbox) and P["n_in_box_field_ok"] == len(inbox_f)
          and P["n_in_box_field_ok_events_ok"] == len(inbox_fr) and P["mechanism_reachable"] == bool(inbox)
          and P["mechanism_reachable_field_ok"] == bool(inbox_f)
          and rd(P["min_Ns_mae"], min(r["Ns_mae"] for r in pts)) < 1e-12
          and rd(P["min_abs_omega"], min(abs(r["omega_err"]) for r in pts)) < 1e-12
          and rd(P["min_eps_R"], min(r["eps_R"] for r in pts)) < 1e-12
          and rd(P["best_penalty_point"]["W1"], bp["W1"]) < 1e-12)
    check("8b prehull A=4 counts (n_in_box, field_ok, events_ok), minima, best-penalty point", ok,
          f"n_in_box {len(inbox)}/{P['n_in_box']}, field_ok {len(inbox_f)}/{P['n_in_box_field_ok']}, "
          f"+events {len(inbox_fr)}/{P['n_in_box_field_ok_events_ok']}")


# ---------------------------------------------------------------------------
# (9) SUMMARY tables vs JSON at the printed rounding; figure twins
# ---------------------------------------------------------------------------
def _f(v, fmt=".2f"):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)


def table_rows(start, stop):
    body = SUMMARY.split(start, 1)[1].split(stop, 1)[0]
    rows = [ln for ln in body.splitlines() if ln.startswith("| ") and not ln.startswith("|---")]
    return [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in rows[1:]]   # drop header


def check_summary():
    rows = table_rows("## 1. Lambda sweep", "Per-A exchange rates")
    bad = []
    for cells, r in zip(rows, SEL["sweep"]):
        exp = [f"{r['lam']:g}", f"{r['sum_W1_rel']:.3f}", f"{r['sum_penalty']:.3f}", f"{r['sum_J']:.3f}",
               f"{r['sum_J_classical']:.3f}", str(r["n_field_ok"]), str(r["n_collapsed"]), str(r["n_in_box"]),
               str(r["n_events_ok"]), str(r["n_beats_classical"]), f"{r['mean_W1']:.1f}", f"{r['mean_Ns_mae']:.1f}",
               f"{100 * r['mean_abs_omega']:.0f} %", f"{r['mean_eps_R']:.2f}", _f(r["median_tau_s"], ".1f"),
               f"{r['knee_distance']:.3f}"]
        if cells != exp:
            bad.append((r["lam"], [(a, b) for a, b in zip(cells, exp) if a != b]))
    check("9a summary section 1 (sweep table, 12 rows x 16 cols) == lambda_star.json at printed rounding",
          len(rows) == 12 and not bad, f"{len(rows)} rows; mismatches {bad[:3]}")
    rows = table_rows("## 5. Nested shared-parameter models", "Jackknife (leave-one-run-out")
    bad = []
    for cells, (lk, q) in zip(rows, H["per_lambda"].items()):
        t = q["totals_dt1"]; mm = q["micro_macro"]["kappa_r"]
        exp = [f"{q['lam']:g}", f"{t['M_none']:.3f}", f"{t['M_c']:.3f}", f"{t['M_r']:.3f}", f"{t['M_both']:.3f}",
               f"{100 * q['gap_rel_both']:+.2f} %", _f(q["eta_c"]), _f(q["eta_r"]), f"{q['ridge_residual_dex']:.2f}",
               f"{q['M_c']['spearman']['rho']:+.2f} (p={q['M_c']['spearman']['p']:.3f})",
               f"{q['M_r']['spearman']['rho']:+.2f} (p={q['M_r']['spearman']['p']:.3f})",
               f"{q['M_both']['spearman_kc']['rho']:+.2f} / {q['M_both']['spearman_kr']['rho']:+.2f}",
               f"{mm['median_abs_dex']:.2f} dex, rho {_f(mm['spearman_macro_vs_micro']['rho'])}"]
        if cells != exp:
            bad.append((lk, [(a, b) for a, b in zip(cells, exp) if a != b]))
    check("9b summary section 5 (nested table, 12 rows x 13 cols) == hypothesis13.json at printed rounding",
          len(rows) == 12 and not bad, f"{len(rows)} rows; mismatches {bad[:3]}")
    rows = table_rows("Jackknife (leave-one-run-out", "At lambda* =")
    bad = []
    for cells, (lk, jk) in zip(rows, H["jackknife"].items()):
        exp = [f"{jk['lam']:g}", f"{jk['gap_mean']:+.4f}", f"{jk['gap_se_jack']:.4f}",
               str(abs(jk["gap_mean"]) > 2 * jk["gap_se_jack"]), str(jk["same_sign"]),
               ", ".join(_f(v) for v in jk["eta_c"]), ", ".join(_f(v) for v in jk["eta_r"])]
        if cells != exp:
            bad.append((lk, [(a, b) for a, b in zip(cells, exp) if a != b]))
    check("9c summary jackknife table == hypothesis13.json at printed rounding", len(rows) == 3 and not bad,
          f"{len(rows)} rows; mismatches {bad[:3]}")
    m = re.search(r"lambda\* = ([0-9.]+) \(rule: (\w+); dominance ([0-9.None]+), restricted knee ([0-9.None]+), "
                  r"lambda_max ([0-9.None]+)", SUMMARY)
    check("9d summary header lambda* / rule / dominance / knee / lambda_max == lambda_star.json",
          m is not None and float(m.group(1)) == LAM_S and m.group(2) == SEL["rule"]
          and m.group(3) == str(SEL["lambda_dominance"]) and m.group(4) == str(SEL["lambda_knee_restricted"])
          and m.group(5) == str(SEL["lambda_max"]), m.group(0) if m else "header not found")
    pngs = sorted(OUT.glob("fig_e13_*.png"))
    bad = [p.name for p in pngs if not p.with_suffix(".pdf").exists() or p.stat().st_size <= 20_000
           or p.with_suffix(".pdf").stat().st_size <= 20_000]
    check("9e every fig_e13_*.png has a .pdf twin and both are > 20 kB", len(pngs) >= 9 and not bad,
          f"{len(pngs)} png; sizes kB {[round(p.stat().st_size / 1e3) for p in pngs]}; bad {bad}")


def main():
    e13.preload(e13.SETTINGS)
    for fn in (check_purity, check_counters, check_reproduction, check_flags_sweep, check_nested,
               check_jackknife, check_extras, check_prehull, check_summary):
        try:
            fn()
        except Exception as exc:                       # a crash is a failed check, not a crash of the audit
            import traceback
            check(f"{fn.__name__} raised", False, f"{type(exc).__name__}: {exc} | {traceback.format_exc().splitlines()[-3]}")
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed; {n_fail} failed", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
