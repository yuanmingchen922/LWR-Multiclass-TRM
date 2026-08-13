"""E-V5 closure: same-box spectral comparison of the capped model field vs the
SUMO field for A10_u15_q2500 (and A1 as reference).

The dispersion analysis (ev5_dispersion.py) proves the calibrated model has no
spontaneous modulational instability; the data analysis (ev5_waves.py) shows
the SUMO stripes are forward-convected fluctuations whose amplitude doubles
with A. Here we quantify what the deterministic model field contains in the
SAME analysis box: an amplitude/backward-fraction/PSD comparison.

Output: out/ev5/sim_vs_data_summary.json, out/ev5/fig_sim_vs_data_psd.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ev4_compare as ev4
import ev5_waves as w5
from solver import SimConfig, simulate

HERE = Path(__file__).parent
OUT = HERE / "out" / "ev5"

QXI_VEHH = 2000.0
CASES = [(1.0, 15.0, 2500.0), (10.0, 15.0, 2500.0)]


def sim_field(A, uc, qin):
    kap = json.loads((HERE / "out" / "ev3_kappa.json").read_text())
    tag = f"A{A:g}_u{uc:g}_q{qin:g}"
    cfg = SimConfig(v_f=ev4.V_F, w=ev4.W, P=ev4.P, q_in=qin / 3600.0, u_xi=uc,
                    kappa_c=float(kap[tag]["kappa_cl_mod"][0]),
                    kappa_r=float(kap[tag]["kappa_r_mod"][0]),
                    capture_form="lf", dt=0.5, save_every=20,
                    q_xi_max=QXI_VEHH / 3600.0)
    regr = ev4.regrid_sim(simulate(cfg))
    return regr["rho_tot"], regr["tt"]      # veh/km on the data grid


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharex=True, sharey=True)
    for col, (A, uc, qin) in enumerate(CASES):
        tag = f"A{A:g}_u{uc:g}_q{qin:g}"
        # data rep 0 field and its analysis box (same code path as ev5_waves)
        rho_d, t_d, x_d = w5._load_field(A, uc, qin, 0)
        tail = w5._queue_tail_km(A, uc, qin, 0, t_d)
        t_sl = (t_d >= 400.0) & (t_d <= 740.0)
        x_hi = float(np.nanmin(tail[t_sl])) - 0.5
        x_lo = 1.5 if x_hi - 1.5 >= 3.0 else 1.0
        x_sl = (x_d >= x_lo) & (x_d <= x_hi)
        ti = np.where(t_sl)[0]
        xi = np.where(x_sl)[0]
        box = dict(t_slice=slice(ti[0], ti[-1] + 1),
                   x_slice=slice(xi[0], xi[-1] + 1))

        rho_s, t_s = sim_field(A, uc, qin)
        assert np.allclose(t_s, t_d), "sim/data time grids must match"

        res_d = w5.analyze_field(rho_d, w5.DX_M, w5.DT_S, **box)
        res_s = w5.analyze_field(rho_s, w5.DX_M, w5.DT_S, **box)
        results[tag] = dict(
            box_x_km=[x_lo, round(x_hi, 2)],
            data=dict(amp=res_d["amp"], backward_frac=res_d["backward_frac"],
                      c_dom_kmh=res_d["c_dom_kmh"]),
            sim_capped_lf_q2000=dict(amp=res_s["amp"],
                                     backward_frac=res_s["backward_frac"],
                                     c_dom_kmh=res_s["c_dom_kmh"]),
            amp_ratio_sim_over_data=res_s["amp"] / max(res_d["amp"], 1e-12))
        print(f"{tag}: data amp={res_d['amp']:.2f} veh/km "
              f"(bwd {res_d['backward_frac']:.2f}, c {res_d['c_dom_kmh']:+.0f})"
              f" | sim amp={res_s['amp']:.3f} veh/km "
              f"(ratio {results[tag]['amp_ratio_sim_over_data']:.3f})")

        for row, (name, r) in enumerate([("SUMO rep0", res_d),
                                         ("model (lf, q_xi=2000)", res_s)]):
            ax = axes[row][col]
            psd = 10 * np.log10(np.maximum(r["psd"], 1e-12))
            ax.pcolormesh(r["k_axis"], r["f_axis"] * 1000, psd,
                          cmap="magma", shading="auto")
            for c in (10, 20, -10, -20):
                kk = np.linspace(-4, 4, 50)
                ax.plot(kk, -c / 3.6 * kk, "w--" if c > 0 else "c--",
                        lw=0.7, alpha=0.7)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-25, 25)
            ax.set_title(f"{tag} — {name}  (amp {r['amp']:.2f} veh/km)",
                         fontsize=9)
            if row == 1:
                ax.set_xlabel("k [cycles/km]")
            if col == 0:
                ax.set_ylabel("f [mHz]")
    fig.suptitle("E-V5 closure: PSD of upstream density fluctuations, "
                 "same analysis box (guide lines ±10, ±20 km/h)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_sim_vs_data_psd.png", dpi=160)
    (OUT / "sim_vs_data_summary.json").write_text(json.dumps(results, indent=2))
    print("wrote", OUT / "sim_vs_data_summary.json", "and figure")


if __name__ == "__main__":
    main()
