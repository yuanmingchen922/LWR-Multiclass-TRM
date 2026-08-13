"""E-V3: calibrate the transition coefficients (kappa_c, kappa_r) from the E1
classification via Poisson-exposure maximum likelihood.

Model (tex): per-vehicle rates
    sigma = kappa_c * a * Dv        (capture, per free vehicle)
    mu    = kappa_r * (P - rho) * Dv (release, per caught vehicle)
For an inhomogeneous Poisson process with rate kappa * X(t), the MLE is
    kappa_hat = (#events) / integral X dt   ("exposure").
Zero events give the rule-of-three upper bound 3 / exposure.

Exposures are accumulated from the classified per-cell fields over the slow
window, per scenario pooled over the 5 reps:
    E_r      = sum_t sum_j  s_cnt_j (P - rho_j) Dv_j  dt
    E_c^A    = sum_t sum_j fz_cnt_j  a_dens_j    Dv_j  dt   (strict tex: a f)
    E_c^l    = sum_t sum_j fz_cnt_j (a+s)_dens_j Dv_j  dt   (refined: l f)
with the CAV as a point mass of 1 veh smeared over 2 cells.

Dv variants:  model  Dv = [v(rho) - u_c]+ with the E-V2 FD;
              emp    Dv = [v_left - u_c]+ (mean left-lane speed; fallback model).

Also reported: per-capita time rates sigma_hat, mu_hat [1/h] and the sync/free
occupancy -> empirical chi, for consistency with tex
eq. (reaction-equilibrium).

Input : out/e1/*.npz, out/params.json
Output: out/ev3_kappa.json, out/ev3_kappa.png, out/ev3_rate_fit.png
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fd
from loader import CELL_LEN, T_SAMPLE

HERE = Path(__file__).parent
OUT = HERE / "out"
E1 = OUT / "e1"

PARAMS = json.loads((OUT / "params.json").read_text())
V_F, W, P = PARAMS["v_f_kmh"], PARAMS["w_kmh"], PARAMS["P_vehkm"]

DT_H = T_SAMPLE / 3600.0
DX_KM = CELL_LEN / 1000.0


def v_of_rho(rho):
    with np.errstate(divide="ignore"):
        vw = W * (P / np.maximum(rho, 1e-9) - 1.0)
    return np.where(rho > 1e-9, np.minimum(V_F, vw), V_F)


def rate_series(d, uc):
    """Per-time-bin exposure increments (before *dt) for one run."""
    n_t = len(d["tt"])
    rho = d["rho"]
    s_cnt = d["s_field"] * DX_KM          # [veh] per cell
    fz_cnt = d["fz_field"] * DX_KM
    a_dens = np.zeros_like(rho)           # [veh/km]
    jc = d["j_cav"]
    for i in range(n_t):
        if jc[i] >= 0:
            a_dens[i, jc[i]] += 0.5 / DX_KM
            a_dens[i, max(jc[i] - 1, 0)] += 0.5 / DX_KM

    dv_mod = np.maximum(v_of_rho(rho) - uc * fd.MS_TO_KMH, 0.0)
    v_left = d["v_left"] * fd.MS_TO_KMH
    dv_emp = np.where(np.isfinite(v_left),
                      np.maximum(v_left - uc * fd.MS_TO_KMH, 0.0), dv_mod)

    l_dens = a_dens + d["s_field"]        # [veh/km]
    out = {}
    for tag, dv in (("mod", dv_mod), ("emp", dv_emp)):
        out[f"Er_{tag}"] = (s_cnt * np.maximum(P - rho, 0.0) * dv).sum(axis=1)
        out[f"EcA_{tag}"] = (fz_cnt * a_dens * dv).sum(axis=1)
        out[f"Ecl_{tag}"] = (fz_cnt * l_dens * dv).sum(axis=1)
    return out


def mle(n_events: int, exposure: float):
    """(kappa_hat, lo95, hi95); zero events -> (0, 0, 3/exposure)."""
    if exposure <= 0:
        return np.nan, np.nan, np.nan
    if n_events == 0:
        return 0.0, 0.0, 3.0 / exposure
    k = n_events / exposure
    half = 1.96 * np.sqrt(n_events) / exposure
    return k, max(k - half, 0.0), k + half


def main():
    runs = defaultdict(list)
    for f in sorted(E1.glob("A*_r*.npz")):
        stem = f.stem                      # A{A}_u{uc}_q{qin}_r{r}
        scen = stem.rsplit("_r", 1)[0]
        runs[scen].append(f)

    results = {}
    series_store = {}
    for scen, files in sorted(runs.items()):
        A = float(scen.split("_")[0][1:])
        uc = float(scen.split("_")[1][1:])
        qin = float(scen.split("_")[2][1:])
        acc = defaultdict(float)
        n_cap = n_rel = 0
        t_caught = t_free = 0.0            # vehicle-hours in window
        ser = None
        for f in files:
            d = np.load(f)
            w = d["in_win"]
            rs = rate_series(d, uc)
            for k, v in rs.items():
                acc[k] += float(v[w].sum()) * DT_H
            n_cap += int(d["cap"][w].sum())
            n_rel += int(d["rel"][w].sum())
            t_caught += float(d["n_caught"][w].sum()) * DT_H
            t_free += float(d["n_free_zone"][w].sum()) * DT_H
            if ser is None:                # keep rep 0 series for the fit plot
                ser = dict(tt=d["tt"], w=w, cap=d["cap"], rel=d["rel"],
                           Er_emp=rs["Er_emp"], Ecl_emp=rs["Ecl_emp"])
        series_store[scen] = ser

        row = dict(A=A, uc=uc, qin=qin, n_cap=n_cap, n_rel=n_rel,
                   t_caught_vehh=t_caught, t_free_vehh=t_free,
                   sigma_time=n_cap / t_free if t_free > 0 else np.nan,
                   mu_time=n_rel / t_caught if t_caught > 0 else np.nan,
                   chi_occ=t_free / (t_free + t_caught)
                   if t_free + t_caught > 0 else np.nan)
        for tag in ("mod", "emp"):
            row[f"kappa_r_{tag}"] = mle(n_rel, acc[f"Er_{tag}"])
            row[f"kappa_cA_{tag}"] = mle(n_cap, acc[f"EcA_{tag}"])
            row[f"kappa_cl_{tag}"] = mle(n_cap, acc[f"Ecl_{tag}"])
            row[f"exp_r_{tag}"] = acc[f"Er_{tag}"]
            row[f"exp_cA_{tag}"] = acc[f"EcA_{tag}"]
            row[f"exp_cl_{tag}"] = acc[f"Ecl_{tag}"]
        results[scen] = row

        kr = row["kappa_r_emp"]
        kc = row["kappa_cl_emp"]
        print(f"{scen}: cap={n_cap:3d} rel={n_rel:3d} "
              f"chi_occ={row['chi_occ']:.2f} "
              f"kappa_c^l={kc[0]:.2e} [{kc[1]:.1e},{kc[2]:.1e}] "
              f"kappa_r={kr[0]:.2e} [{kr[1]:.1e},{kr[2]:.1e}] (emp dv) "
              f"| mu={row['mu_time']:.1f}/h sigma={row['sigma_time']:.1f}/h")

    (OUT / "ev3_kappa.json").write_text(json.dumps(results, indent=2))

    # ---- figure: kappa per scenario --------------------------------------
    scens = sorted(results, key=lambda s: (results[s]["A"], results[s]["uc"],
                                           results[s]["qin"]))
    labels = [s.replace("_", " ") for s in scens]
    xpos = np.arange(len(scens))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)

    ax = axes[0]
    for tag, color, name in (("emp", "tab:red", r"empirical $\Delta v$"),
                             ("mod", "tab:gray", r"model $\Delta v(\rho)$")):
        k = np.array([results[s][f"kappa_cl_{tag}"] for s in scens])
        ax.errorbar(xpos, k[:, 0], yerr=[k[:, 0] - k[:, 1], k[:, 2] - k[:, 0]],
                    fmt="o", color=color, capsize=3, label=name)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\hat\kappa_c$ [1/veh]  ($\ell f$ form)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title(r"E-V3: Poisson-exposure MLE of $\kappa_c$, $\kappa_r$ "
                 "(pooled 5 reps; bars = 95% CI; upper bounds where 0 events)")

    ax = axes[1]
    for tag, color, name in (("emp", "tab:red", r"empirical $\Delta v$"),
                             ("mod", "tab:gray", r"model $\Delta v(\rho)$")):
        k = np.array([results[s][f"kappa_r_{tag}"] for s in scens])
        zero = k[:, 0] == 0
        ax.errorbar(xpos[~zero], k[~zero, 0],
                    yerr=[k[~zero, 0] - k[~zero, 1], k[~zero, 2] - k[~zero, 0]],
                    fmt="s", color=color, capsize=3, label=name)
        if zero.any() and tag == "emp":
            ax.scatter(xpos[zero], k[zero, 2], marker="v", color=color,
                       label=r"upper bound (0 releases)")
        elif zero.any():
            ax.scatter(xpos[zero], k[zero, 2], marker="v", color=color)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\hat\kappa_r$ [1/veh]")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ev3_kappa.png", dpi=160)

    # ---- figure: rate fit for two representative scenarios ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, scen in zip(axes, ("A1_u15_q2500", "A10_u15_q2500")):
        ser = series_store[scen]
        row = results[scen]
        tt = ser["tt"]
        w = ser["w"]
        ax.bar(tt[w] / 60, ser["cap"][w] / DT_H, width=0.14, color="tab:blue",
               alpha=0.6, label="capture rate (events/h)")
        ax.bar(tt[w] / 60, -ser["rel"][w] / DT_H, width=0.14, color="tab:red",
               alpha=0.6, label="release rate (down)")
        kc = row["kappa_cl_emp"][0]
        kr = row["kappa_r_emp"][0]
        ax.plot(tt[w] / 60, kc * ser["Ecl_emp"][w], "b-", lw=1.6,
                label=r"$\hat\kappa_c \ell f \Delta v$")
        ax.plot(tt[w] / 60, -kr * ser["Er_emp"][w], "r-", lw=1.6,
                label=r"$-\hat\kappa_r (P-\rho) s \Delta v$")
        ax.set_title(f"{scen.replace('_', ' ')} (rep 0)")
        ax.set_xlabel("t [min]")
        ax.set_ylabel("rate [veh/h]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "ev3_rate_fit.png", dpi=160)
    print("wrote", OUT / "ev3_kappa.json", "and figures")


if __name__ == "__main__":
    main()
