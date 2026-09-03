"""E10 final: the general model that replaces the downstream-release kludge.

Structure (all state/interface-based, no scenario information in any source
term):  DM-G capacity cap (Q_xi = 2000 veh/h) + stuck-class branch w_s = 0.6 w
(active only while class A is slow, u_s < v_f) + s-IMPERMEABLE moving-
bottleneck interface ('only free vehicles overtake') + catch & release with
W1-fitted (kappa_c, kappa_r).  The connectivity leader-loss term is NOT part
of the final model (E10 ladder: under W1 it degenerates the fit to zero
capture); after the bottleneck deactivates the residual s label is inert by
construction (Delta v = 0 and the w_s branch is gated off).

Writes out/e10/final_config.json and fig_profiles_final_e10_{tag}.png /
fig_heat3_final_e10_{tag}.png (data vs the E8 kludge vs the final model).
"""
import json
from pathlib import Path
import numpy as np
import e7_wasserstein as e7
import e8_ladder as L
import ev4_compare as ev4

OUT = Path("out/e10"); OUT.mkdir(exist_ok=True)
QXI = 2000.0 / 3600.0; W_S = 0.6 * ev4.W
FINAL = {"q_xi_max": QXI, "w_s": W_S, "s_impermeable": True}
KLUDGE = {"q_xi_max": QXI, "w_s": W_S, "downstream_release": True}
e8 = json.load(open("out/e8/final_config.json"))

def brief(s):
    return dict(W1=s["W1"], RMSE=s["RMSE"], e_s=s["e_s"], omega_err=s["omega_err"],
                wake=s["wake"]["wake_mean_vehkm"], ds_stuck=s["ds_stuck"]["frac"],
                ds_max_s=s["ds_max_s_vehkm"], rarefaction_m=s["rarefaction"]["width_growth_m"],
                s_layer_m=s["s_layer"]["max_m"])

res = {"_meta": dict(structure="cap 2000 + w_s=0.6w (gated u_s<v_f) + s_impermeable + W1-fit kappas; lf",
                     fit="u15_q2500 rep-mean field, W1, E7 protocol; q2000 = zero-refit transfer")}
for A in (1.0, 10.0):
    grids = {} if A == 1.0 else dict(kc_grid=np.logspace(-2.5, -0.5, 6))
    fit = e7.fit_field(A, 15.0, 2500.0, form="lf", metric="w1", extra_cfg=FINAL, **grids)
    kc, kr = fit["kappa_c"], fit["kappa_r"]
    kl = e8[f"A{A:g}_u15_q2500"]
    block = dict(kappa_c=kc, kappa_r=kr)
    for qin in (2500.0, 2000.0):
        s_fin, regr_fin, meas = L.eval_rung(A, 15.0, qin, kc, kr, FINAL)
        s_kl, regr_kl, _ = L.eval_rung(A, 15.0, qin, kl["kappa_c"], kl["kappa_r"], KLUDGE)
        block[f"q{qin:g}"] = dict(final=brief(s_fin), kludge_E8=brief(s_kl))
        tag = f"A{A:g}_u15_q{qin:g}"
        print(f"{tag} FINAL kc={kc:.3e} kr={kr:.3e} | W1 {s_fin['W1']:.1f} (kludge {s_kl['W1']:.1f}) "
              f"wake {s_fin['wake']['wake_mean_vehkm']:.1f} ({s_kl['wake']['wake_mean_vehkm']:.1f}) "
              f"e_s {s_fin['e_s']:+.1%} ({s_kl['e_s']:+.1%}) om {s_fin['omega_err']:+.1%} ({s_kl['omega_err']:+.1%}) "
              f"ds_slow-window-stuck {s_fin['ds_stuck']['frac']:.4f} rare {s_fin['rarefaction']['width_growth_m']:+.0f}")
        if qin == 2500.0:
            L.fig_profiles_e8(tag, meas, regr_kl, "E8 kludge (downstream_release)",
                              regr_fin, "final: cap + w_s + s-impermeable", OUT)
            L.fig_heat3_e8(tag, meas, regr_kl, "E8 kludge", regr_fin, "final model", OUT)
            for p in OUT.glob(f"fig_*_e8_{tag}.png"):
                p.rename(p.with_name(p.name.replace("_e8_", "_final_e10_")))
    res[f"A{A:g}"] = block
json.dump(res, open(OUT / "final_config.json", "w"), indent=2, default=float)
print("wrote", OUT / "final_config.json", "and figures")
