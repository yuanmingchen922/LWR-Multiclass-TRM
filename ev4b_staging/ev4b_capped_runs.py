"""E-V4b capped-vs-uncapped comparison, scratchpad-standalone edition.

The real ev4_compare.py, loader.py/fd.py and out/e1/*.npz are unreachable in
this session (macOS TCC denies ~/Desktop), so this driver reruns the SIM side
of the E-V4 grid with the patched solver (ev4b/solver.py, q_xi_max support)
and computes every metric that does not need the measured density fields:

  - cum_overtake_sim_veh   = trapz(omega, t) over the full run (the E-V4
                             convention: omega vanishes outside the slow
                             window, so this equals the windowed integral)
  - omega_cum_rel_err      = |sim - meas| / meas per rep; the measured
                             overtake counts are REAL data carried inside
                             out/ev4/metrics.json (mock-harness copy)
  - N_s(740), max N_s, linear-fit slope and R^2 of N_s on [300, 740]
                             (the A=1 queue-linearity question)

rho_RMSE / Ns_MAE / e_s need out/e1 measured fields -> not computable here.

Configs: forms lf/af x q_xi_max {None, 2000, 2440} veh/h, 8 scenarios,
kappa from out/ev3_kappa.json point estimates (kappa_cl_mod / kappa_cA_mod,
kappa_r_mod) -- ZERO refitting.  dt=0.5 s, save cadence 10 s, dx=50 m.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH / "ev4b"))

from solver import SimConfig, simulate  # noqa: E402

OUT = SCRATCH / "out"
p = json.loads((SCRATCH / "ev4b" / "out" / "params.json").read_text())
V_F = p["v_f_kmh"] / 3.6
W = p["w_kmh"] / 3.6
P = p["P_vehkm"] / 1000.0

KTAB = json.loads((OUT / "ev3_kappa.json").read_text())
MEAS = {tag: v["cum_overtake_meas_veh"]
        for tag, v in json.loads((OUT / "ev4" / "metrics.json").read_text()).items()
        if tag != "_meta"}

CORE = [(A, uc, qin) for A in (1, 10) for uc in (15, 20)
        for qin in (2000, 2500)]
KC_KEY = {"lf": "kappa_cl_mod", "af": "kappa_cA_mod"}


def run_one(tag: str, uc: float, qin: float, form: str,
            qxi_vehh: float | None):
    kc = KTAB[tag][KC_KEY[form]][0]
    kr = KTAB[tag]["kappa_r_mod"][0]
    cfg = SimConfig(v_f=V_F, w=W, P=P, q_in=qin / 3600.0, u_xi=uc,
                    kappa_c=kc, kappa_r=kr, capture_form=form,
                    dt=0.5, save_every=20,
                    q_xi_max=None if qxi_vehh is None else qxi_vehh / 3600.0)
    res = simulate(cfg)
    cum = float(np.trapz(res.omega, res.t))
    rel = [abs(cum - m) / m for m in MEAS[tag]]
    i740 = int(np.argmin(np.abs(res.t - 740.0)))
    fit = (res.t >= 300.0) & (res.t <= 740.0)
    tt, nn = res.t[fit], res.N_s[fit]
    A_ = np.vstack([tt, np.ones_like(tt)]).T
    (slope, icpt), sse = np.linalg.lstsq(A_, nn, rcond=None)[:2]
    ss_tot = float(np.sum((nn - nn.mean()) ** 2))
    r2 = 1.0 - (float(sse[0]) / ss_tot if sse.size and ss_tot > 0 else 0.0)
    return res, {
        "kappa_c": kc, "kappa_r": kr,
        "cum_overtake_sim_veh": cum,
        "cum_overtake_meas_veh": MEAS[tag],
        "omega_cum_rel_err": {"per_rep": rel,
                              "mean": float(np.mean(rel))},
        "Ns_740_veh": float(res.N_s[i740]),
        "Ns_max_veh": float(res.N_s.max()),
        "Ns_fit_300_740": {"slope_veh_per_s": float(slope),
                           "intercept_veh": float(icpt),
                           "R2": r2},
    }


def main():
    all_rows = {}          # (form, qxi) -> {tag: metrics}
    all_ns = {}            # (form, qxi) -> {tag: (t, N_s)}
    for form in ("lf", "af"):
        for qxi in (None, 2000.0, 2440.0):
            key = (form, qxi)
            all_rows[key] = {}
            all_ns[key] = {}
            for A, uc, qin in CORE:
                tag = f"A{A:g}_u{uc:g}_q{qin:g}"
                res, row = run_one(tag, uc, qin, form, qxi)
                all_rows[key][tag] = row
                all_ns[key][tag] = (res.t, res.N_s)
            suffix = "" if form == "lf" else "_af"
            name = (f"ev4b_uncapped{suffix}" if qxi is None
                    else f"ev4b_q{qxi:g}{suffix}")
            d = OUT / name
            d.mkdir(parents=True, exist_ok=True)
            meta = {
                "capture_form": form,
                "q_xi_max_vehh": qxi,
                "q_xi_max": None if qxi is None else qxi / 3600.0,
                "beta": 0.5,
                "kappa_c_key": KC_KEY[form],
                "kappa_r_key": "kappa_r_mod",
                "kappa_source": "out/ev3_kappa.json (E-V3 table, no refit)",
                "params_si": {"v_f_ms": V_F, "w_ms": W, "P_vehm": P},
                "note": ("sim-side rerun; rho_RMSE/Ns_MAE/e_s omitted -- "
                         "out/e1 measured fields unreachable this session"),
            }
            (d / "metrics.json").write_text(json.dumps(
                {"_meta": meta, **all_rows[key]}, indent=2))
            print(f"wrote {d / 'metrics.json'}")

    # N_s grid figure per capped dir: capped (solid) vs uncapped (dashed)
    for form in ("lf", "af"):
        suffix = "" if form == "lf" else "_af"
        for qxi in (2000.0, 2440.0):
            d = OUT / f"ev4b_q{qxi:g}{suffix}"
            fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
            for ax, (A, uc, qin) in zip(axes.ravel(), CORE):
                tag = f"A{A:g}_u{uc:g}_q{qin:g}"
                t0, n0 = all_ns[(form, None)][tag]
                t1, n1 = all_ns[(form, qxi)][tag]
                ax.plot(t0, n0, "k--", lw=1.2, label="uncapped")
                ax.plot(t1, n1, "-", color="tab:red", lw=1.6,
                        label=f"q_xi={qxi:g} veh/h")
                r2 = all_rows[(form, qxi)][tag]["Ns_fit_300_740"]["R2"]
                ax.set_title(f"{tag}  R2={r2:.4f}", fontsize=9)
                ax.axvspan(250, 750, color="0.92", zorder=0)
                if ax is axes.ravel()[0]:
                    ax.legend(fontsize=7)
            for ax in axes[-1]:
                ax.set_xlabel("t [s]")
            for ax in axes[:, 0]:
                ax.set_ylabel("N_s [veh]")
            fig.suptitle(f"E-V4b N_s(t), form={form}, cap={qxi:g} veh/h "
                         f"(dashed = uncapped baseline)")
            fig.tight_layout()
            fig.savefig(d / "fig_Ns_grid.png", dpi=110)
            plt.close(fig)
            print(f"wrote {d / 'fig_Ns_grid.png'}")

    # compact comparison table
    print("\n=== omega_cum_rel_err (mean over 5 reps) and Ns_740 ===")
    hdr = ("scenario      "
           "  base_lf  q2000lf  q2440lf  base_af  q2000af  q2440af"
           "  | Ns740: base_lf q2000lf q2440lf base_af q2000af q2440af")
    print(hdr)
    for A, uc, qin in CORE:
        tag = f"A{A:g}_u{uc:g}_q{qin:g}"
        oe = [all_rows[(f, q)][tag]["omega_cum_rel_err"]["mean"]
              for f in ("lf", "af") for q in (None, 2000.0, 2440.0)]
        ns = [all_rows[(f, q)][tag]["Ns_740_veh"]
              for f in ("lf", "af") for q in (None, 2000.0, 2440.0)]
        print(f"{tag:14s}" + "".join(f" {v:8.4f}" for v in oe)
              + "  |" + "".join(f" {v:7.1f}" for v in ns))
    print("\nmean omega_cum_rel_err over 8 scenarios:")
    for f in ("lf", "af"):
        for q in (None, 2000.0, 2440.0):
            m = np.mean([all_rows[(f, q)][t]["omega_cum_rel_err"]["mean"]
                         for t in all_rows[(f, q)]])
            print(f"  form={f} qxi={q}: {m:.4f}")
    print("\ncum_overtake_sim (veh) per config:")
    for f in ("lf", "af"):
        for q in (None, 2000.0, 2440.0):
            row = {t: round(all_rows[(f, q)][t]["cum_overtake_sim_veh"], 2)
                   for t in all_rows[(f, q)]}
            print(f"  form={f} qxi={q}: {row}")
    print("\nA=1 N_s linear-fit R2 / slope (veh/s) on [300,740]:")
    for f in ("lf", "af"):
        for q in (None, 2000.0, 2440.0):
            for tag in [f"A1_u{uc:g}_q{qin:g}" for uc in (15, 20)
                        for qin in (2000, 2500)]:
                ft = all_rows[(f, q)][tag]["Ns_fit_300_740"]
                print(f"  form={f} qxi={q} {tag}: R2={ft['R2']:.5f} "
                      f"slope={ft['slope_veh_per_s']:.4f}")


if __name__ == "__main__":
    main()
