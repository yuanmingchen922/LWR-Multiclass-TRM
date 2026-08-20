"""E8 final configuration: figures and summary for the hybrid model
(capacity cap + stuck-class flux w_s + downstream-release + W1-fitted kappas).

Writes out/e8/final_config.json and the final comparison figures
fig_profiles_final_{tag}.png / fig_heat3_final_{tag}.png (data vs classical
LWR+MB vs the hybrid, with the s/f split).
"""

from __future__ import annotations

import json
from pathlib import Path

import ev4_compare as ev4
import e7_wasserstein as e7
import e8_ladder as L

HERE = Path(__file__).parent
OUT = HERE / "out" / "e8"

QXI = 2000.0 / 3600.0
W_S = 0.6 * ev4.W

FINAL = {
    1.0: dict(kappa_c=5.323e-02, kappa_r=1.654e-03),
    10.0: dict(kappa_c=1.664e-01, kappa_r=1.399e-02),
}


def main():
    summary = {"_meta": dict(
        structure="q_xi_max=2000 veh/h + w_s=0.6w + downstream_release + "
                  "W1-fitted (kappa_c, kappa_r); lf capture form",
        fit="u15_q2500 rep-mean field, 1-Wasserstein objective")}
    for A, kap in FINAL.items():
        tag = f"A{A:g}_u15_q2500"
        extra = {"downstream_release": True, "w_s": W_S, "q_xi_max": QXI}
        s, regr, meas = L.eval_rung(A, 15.0, 2500.0, kap["kappa_c"],
                                    kap["kappa_r"], extra)
        m1 = e7.run_sim(15.0, 2500.0, 0.0, 0.0, "lf", dt=e7.DT_PRODUCTION,
                        q_xi_max=QXI)
        L.fig_profiles_e8(tag, meas, m1, "M1: LWR + MB (DM-G cap)",
                          regr, "final hybrid (cap + w_s + DR)", OUT)
        L.fig_heat3_e8(tag, meas, m1, "M1: LWR + MB",
                       regr, "final hybrid", OUT)
        for p in OUT.glob(f"fig_*_e8_{tag}.png"):
            p.rename(p.with_name(p.name.replace("_e8_", "_final_")))
        summary[tag] = dict(**kap, summary=s)
        print(f"{tag}: W1={s['W1']:.1f} wake={s['wake']['wake_mean_vehkm']:.1f}"
              f" e_s={s['e_s']:+.1%} om={s['omega_err']:+.1%} "
              f"ds={s['ds_stuck']['frac']:.3f}")
    (OUT / "final_config.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print("wrote", OUT / "final_config.json", "and final figures")


if __name__ == "__main__":
    main()
