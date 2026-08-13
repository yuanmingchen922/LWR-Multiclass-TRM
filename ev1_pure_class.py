"""E-V1: verify the pure-class limits of the transport layer (tex, Prop.
"Pure-class consistency") against the SUMO data, using the E-V2 parameters.

Checks
  1. Free-flow limit (pure B_f): before the CAV enters (t <= 100 s) every
     cell should follow q = v_f * rho.
  2. Shared congested branch (pure B_s-like): deep inside a fully developed
     queue (A=3, u_c <= 14 m/s, q_in = 2500) both lanes are dense, and the
     aggregate should follow q = w (P - rho); queue speed ~ u_c.
  3. Qualitative A-contrast (motivation figure): density heatmaps of A=1 vs
     A=10 at u_c = 15 m/s, q_in = 2500 with the CAV trajectory overlaid.

Outputs (analysis/out/): ev1_metrics.json, ev1_pure_class.png,
ev1_heatmaps_A1_vs_A10.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fd
from loader import (CELL_LEN, ctrl_position, list_scenarios, load_scenario)

HERE = Path(__file__).parent
OUT = HERE / "out"
SECOND = HERE.parent / "Second"

PARAMS = json.loads((OUT / "params.json").read_text())
V_F, W, P = PARAMS["v_f_kmh"], PARAMS["w_kmh"], PARAMS["P_vehkm"]


def free_flow_points():
    """(rho, q) from the pre-CAV window (t <= 100 s), across A in {1,3,10}."""
    rho_all, q_all = [], []
    for p, A, uc, qin, _ in list_scenarios(SECOND, flag=True):
        if A not in (1.0, 3.0, 10.0) or uc not in (15.0, 20.0) or qin != 2500:
            continue
        sc = load_scenario(p, fields=True, ctrl=False)
        n_pre = np.sum(sc.t_field <= 100.0)
        for r, rep in sorted(sc.reps.items()):
            if rep.rho is None:
                continue
            rho = rep.rho[:n_pre].ravel()
            q = rep.q[:n_pre].ravel()
            keep = rho > 0.5
            rho_all.append(rho[keep])
            q_all.append(q[keep])
    return np.concatenate(rho_all), np.concatenate(q_all)


def queue_interior_points(rho_thresh: float = 50.0):
    """(rho, q, uc) deep inside fully developed queues (A=3 strong cases)."""
    rho_all, q_all, uc_all = [], [], []
    for p, A, uc, qin, _ in list_scenarios(SECOND, flag=True):
        if A != 3.0 or qin != 2500 or uc > 14:
            continue
        sc = load_scenario(p, fields=True, ctrl=True)
        tt = sc.t_field
        sel_t = np.where((tt >= 350) & (tt <= 700))[0]
        for r, rep in sorted(sc.reps.items()):
            if rep.rho is None or rep.ctrl is None:
                continue
            x_cav = ctrl_position(sc, r, tt)
            for i in sel_t:
                row = rep.rho[i]
                jj = np.where(row > rho_thresh)[0]
                if not jj.size or not np.isfinite(x_cav[i]):
                    continue
                x_tail = (jj.min() + 1) * CELL_LEN
                lo, hi = x_tail + 300.0, x_cav[i] - 200.0
                if hi <= lo:
                    continue
                cells = np.where(((np.arange(len(row)) + 1) * CELL_LEN >= lo)
                                 & ((np.arange(len(row)) + 1) * CELL_LEN <= hi)
                                 & (row > rho_thresh))[0]
                rho_all.append(row[cells])
                q_all.append(rep.q[i][cells])
                uc_all.append(np.full(cells.size, uc))
    return (np.concatenate(rho_all), np.concatenate(q_all),
            np.concatenate(uc_all))


def heatmap_panel(ax, sc, r, vmax=90):
    rep = sc.reps[r]
    tt = sc.t_field
    im = ax.imshow(rep.rho, origin="lower", aspect="auto",
                   extent=[CELL_LEN / 1000, 30.0, tt[0] / 60, tt[-1] / 60],
                   cmap="turbo", vmin=0, vmax=vmax)
    c = rep.ctrl
    ax.plot(c.x / 1000, c.t / 60, "w-", lw=2)
    ax.set_xlabel("x [km]")
    return im


def main():
    OUT.mkdir(exist_ok=True)

    # ---- 1. free-flow limit ---------------------------------------------
    rho_f, q_f = free_flow_points()
    pred = V_F * rho_f
    rel = (q_f - pred) / np.maximum(pred, 1e-9)
    m_free = dict(n=int(len(rho_f)),
                  rel_bias=float(np.mean(rel)),
                  rel_rmse=float(np.sqrt(np.mean(rel ** 2))))
    print(f"free-flow: n={m_free['n']}, rel bias {m_free['rel_bias']:+.3f}, "
          f"rel RMSE {m_free['rel_rmse']:.3f}")

    # ---- 2. queue interior vs shared congested branch --------------------
    rho_q, q_q, uc_q = queue_interior_points()
    pred_q = fd.q_tri(rho_q, V_F, W, P)
    res_q = q_q - pred_q
    v_q = q_q / np.maximum(rho_q, 1e-9)             # [km/h]
    m_queue = dict(n=int(len(rho_q)),
                   q_rmse_vehh=float(np.sqrt(np.mean(res_q ** 2))),
                   q_bias_vehh=float(np.mean(res_q)),
                   speed_vs_uc_kmh=float(np.mean(v_q - uc_q * fd.MS_TO_KMH)))
    print(f"queue interior: n={m_queue['n']}, "
          f"q bias {m_queue['q_bias_vehh']:+.0f} veh/h, "
          f"q RMSE {m_queue['q_rmse_vehh']:.0f} veh/h, "
          f"mean(v - u_c) = {m_queue['speed_vs_uc_kmh']:+.1f} km/h")

    (OUT / "ev1_metrics.json").write_text(
        json.dumps(dict(free_flow=m_free, queue_interior=m_queue,
                        params=dict(v_f=V_F, w=W, P=P)), indent=2))

    # ---- figure: both scatter checks ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    rr = np.linspace(0, P, 400)

    ax = axes[0]
    ax.scatter(rho_f[::5], q_f[::5], s=3, c="tab:blue", alpha=0.3)
    ax.plot(rr, V_F * rr, "k-", lw=2, label=rf"$v_f\rho$, $v_f$={V_F:.0f} km/h")
    ax.set_xlim(0, 40); ax.set_ylim(0, 4000)
    ax.set_xlabel(r"$\rho$ [veh/km]"); ax.set_ylabel(r"$q$ [veh/h]")
    ax.set_title(f"Pure $B_f$ limit: pre-CAV window (t $\\leq$ 100 s)\n"
                 f"rel. RMSE = {m_free['rel_rmse']:.1%}")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    sc_pts = ax.scatter(rho_q, q_q, s=6, c=uc_q, cmap="viridis", alpha=0.5)
    fig.colorbar(sc_pts, ax=ax, label=r"$u_\xi$ [m/s]")
    ax.plot(rr, fd.q_tri(rr, V_F, W, P), "k-", lw=2,
            label=rf"$Q_0$ ($w$={W:.1f}, $P$={P:.0f})")
    ax.set_xlim(40, 110); ax.set_ylim(2500, 5500)
    ax.set_xlabel(r"$\rho$ [veh/km]"); ax.set_ylabel(r"$q$ [veh/h]")
    ax.set_title(f"Queue interior vs shared congested branch\n"
                 f"q RMSE = {m_queue['q_rmse_vehh']:.0f} veh/h")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ev1_pure_class.png", dpi=160)

    # ---- figure: A = 1 vs A = 10 heatmaps -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, A in zip(axes, (1.0, 10.0)):
        p = SECOND / f"data_{int(A)}_15_2500_True.mat"
        sc = load_scenario(p, fields=True, ctrl=True)
        im = heatmap_panel(ax, sc, r=0)
        ax.set_title(f"$A$ = {A:g},  $u_\\xi$ = 15 m/s,  $q_{{in}}$ = 2500 veh/h")
    axes[0].set_ylabel("t [min]")
    fig.colorbar(im, ax=axes, label=r"$\rho$ [veh/km]", shrink=0.9)
    fig.savefig(OUT / "ev1_heatmaps_A1_vs_A10.png", dpi=160,
                bbox_inches="tight")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
