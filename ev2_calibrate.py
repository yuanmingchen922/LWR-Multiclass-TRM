"""E-V2: calibrate the transport-layer parameters (v_f, w, P) of the
triangular fundamental diagram from the A=3 dense scenario set (True files).

Data sources:
  * full (rho, q) field cloud  -> free-flow branch (v_f)
  * steady-state upstream points (rho-, q-) from `updown` across the
    u_c sweep -> congested branch (w, P), the ECC22 trick: each bottleneck
    speed samples a different point of the congested branch.

Outputs (analysis/out/):
  params.json                  calibrated parameters (both P-free and P-fixed)
  ev2_fd_calibration.png       cloud + steady points + fitted FD
  ev2_residuals.png            congested-branch residuals vs rho
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fd
from loader import T_FAST, T_SLOW, T_SAMPLE, list_scenarios, load_scenario

HERE = Path(__file__).parent
OUT = HERE / "out"
SECOND = HERE.parent / "Second"

C_STEADY = 0.15          # ECC22 steady-state tolerance
SUBSAMPLE_CLOUD = 7      # keep every k-th field point for plotting/fitting


def gather():
    """Collect field cloud and steady updown points from all A=3 True files."""
    cloud_rho, cloud_q = [], []
    pts = []  # rows: uc[m/s], qin, rep, rho-, q-, rho+, q+
    i_slow, i_fast = int(T_SLOW / T_SAMPLE), int(T_FAST / T_SAMPLE)
    for p, A, uc, qin, _ in list_scenarios(SECOND, flag=True):
        if A != 3.0:
            continue
        sc = load_scenario(p, fields=True, ctrl=False)
        for r, rep in sorted(sc.reps.items()):
            if rep.rho is None or rep.q is None:
                continue
            rho = rep.rho.ravel()
            q = rep.q.ravel()
            keep = rho > 0.5
            cloud_rho.append(rho[keep][::SUBSAMPLE_CLOUD])
            cloud_q.append(q[keep][::SUBSAMPLE_CLOUD])
            if rep.updown is None:
                continue
            row = [uc, qin, r]
            for i in range(4):
                m, _ = fd.steady_state_mean(rep.updown[i], i_slow, i_fast, C_STEADY)
                row.append(m)
            pts.append(row)
    return (np.concatenate(cloud_rho), np.concatenate(cloud_q),
            np.array(pts, float))


def tail_shock_check(rho_thresh: float = 50.0):
    """Rankine-Hugoniot consistency of the measured aggregate states.

    During queue growth the upstream tail is a shock between the inflow state
    (rho_in, qin) and the queue state (rho-, q-); RH predicts its speed
    z = (q- - qin) / (rho- - rho_in). We measure the tail slope from the
    density field over t in [300, 650] s and compare.

    NOTE the queue-discharge wave (-w) is NOT directly observable for a
    moving bottleneck: the rho>thresh boundary after release is a material
    boundary drifting with the (moving) platoon, contaminated by finite-
    acceleration bunching. The tail shock during growth is clean.
    """
    i_slow, i_fast = int(T_SLOW / T_SAMPLE), int(T_FAST / T_SAMPLE)
    out = []
    for p, A, uc, qin, _ in list_scenarios(SECOND, flag=True):
        if A != 3.0 or qin < 2400 or uc > 14:
            continue
        sc = load_scenario(p, fields=True, ctrl=False)
        tt = sc.t_field
        sel = (tt >= 300) & (tt <= 650)
        cells_in = slice(19, 40)          # x in [2, 4] km: inflow reference
        for r, rep in sorted(sc.reps.items()):
            if rep.rho is None or rep.updown is None:
                continue
            xs, ts = [], []
            for i in np.where(sel)[0]:
                jj = np.where(rep.rho[i] > rho_thresh)[0]
                if jj.size:
                    xs.append((jj.min() + 1) * 100.0)
                    ts.append(tt[i])
            if len(xs) < 8:
                continue
            z_emp = np.polyfit(ts, xs, 1)[0] * fd.MS_TO_KMH  # [km/h]
            rho_in = float(np.mean(rep.rho[sel][:, cells_in]))
            q_in = float(np.mean(rep.q[sel][:, cells_in]))
            rho_m, _ = fd.steady_state_mean(rep.updown[0], i_slow, i_fast, C_STEADY)
            q_m, _ = fd.steady_state_mean(rep.updown[1], i_slow, i_fast, C_STEADY)
            if not (np.isfinite(rho_m) and rho_m > rho_in + 5):
                continue
            z_rh = (q_m - q_in) / (rho_m - rho_in)
            out.append((uc, qin, r, z_emp, z_rh))
    return np.array(out)


def main():
    OUT.mkdir(exist_ok=True)
    rho_c, q_c, pts = gather()
    print(f"cloud points: {len(rho_c)},  scenario-rep updown rows: {len(pts)}")

    # ---- free-flow branch ------------------------------------------------
    v_f, free_mask = fd.fit_free_speed(rho_c, q_c)
    print(f"v_f = {v_f:.1f} km/h = {v_f / 3.6:.2f} m/s "
          f"({int(free_mask.sum())} pts)")

    # ---- congested branch from steady upstream points --------------------
    uc_ms, qin, rep = pts[:, 0], pts[:, 1], pts[:, 2]
    rho_m, q_m = pts[:, 3], pts[:, 4]
    # NOTE: (rho-, q-) is a 2-lane aggregate right behind the CAV: right lane
    # queued at u_xi + left lane overtaking faster. Mid-uc points are convex
    # mixtures of two FD states and lie BELOW the concave FD (hence ECC22 fits
    # an upper concave envelope). Only strongly congested states (both lanes
    # dense) sample the congested branch itself -> select by density.
    cong = np.isfinite(rho_m) & np.isfinite(q_m) & (rho_m >= 60.0)
    print(f"congested steady points (rho- >= 60): {int(cong.sum())} / {len(pts)}")

    fit_free_P = fd.fit_congested(rho_m[cong], q_m[cong], P_fixed=None)
    fit_fix_P = fd.fit_congested(rho_m[cong], q_m[cong], P_fixed=fd.P_PRIOR)
    for tag, ft in [("P free ", fit_free_P), ("P fixed", fit_fix_P)]:
        print(f"[{tag}] w = {ft['w']:.1f} km/h, P = {ft['P']:.0f} veh/km, "
              f"rmse = {ft['rmse']:.0f} veh/h, n = {ft['n_used']}")

    # ---- Rankine-Hugoniot consistency of the tail shock ------------------
    ts_chk = tail_shock_check()
    if len(ts_chk):
        z_emp, z_rh = ts_chk[:, 3], ts_chk[:, 4]
        print(f"tail-shock RH check (n={len(ts_chk)}): "
              f"z_emp median {np.median(z_emp):.1f} km/h vs "
              f"z_RH median {np.median(z_rh):.1f} km/h, "
              f"median |diff| {np.median(np.abs(z_emp - z_rh)):.1f} km/h")
    else:
        print("tail-shock RH check: no usable runs")

    # ---- choose baseline: P fixed by bumper-to-bumper prior --------------
    # (data reach rho ~ 90 veh/km only; extrapolating to q=0 is ill-posed,
    #  so the P-free fit is reported as a sensitivity, not the baseline)
    w, P = fit_fix_P["w"], fit_fix_P["P"]
    params = dict(
        v_f_kmh=v_f, v_f_ms=v_f / 3.6,
        w_kmh=w, w_ms=w / 3.6, P_vehkm=P,
        crit_density=fd.crit_density(v_f, w, P),
        capacity=fd.capacity(v_f, w, P),
        alt_P_free=dict(w_kmh=fit_free_P["w"], P_vehkm=fit_free_P["P"],
                        rmse=fit_free_P["rmse"]),
        tail_shock_rh=dict(
            n=int(len(ts_chk)),
            z_emp_median_kmh=float(np.median(ts_chk[:, 3])) if len(ts_chk) else None,
            z_rh_median_kmh=float(np.median(ts_chk[:, 4])) if len(ts_chk) else None,
            median_abs_diff_kmh=(float(np.median(np.abs(ts_chk[:, 3] - ts_chk[:, 4])))
                                 if len(ts_chk) else None)),
        n_cloud=int(len(rho_c)), n_congested=int(cong.sum()),
        C_steady=C_STEADY,
        source="A=3 True files, steady updown upstream pts + field cloud",
    )
    (OUT / "params.json").write_text(json.dumps(params, indent=2))
    print("capacity =", round(params["capacity"]), "veh/h,  rho_crit =",
          round(params["crit_density"], 1), "veh/km")

    # ---- figures ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(rho_c, q_c, s=2, c="0.8", label="field cloud (A=3, all cells)")
    dn = np.isfinite(pts[:, 5])
    ax.scatter(pts[dn, 5], pts[dn, 6], s=28, c="tab:blue", marker="o",
               label=r"steady $(\rho_+,q_+)$ downstream")
    ax.scatter(rho_m[cong], q_m[cong], s=34, c="tab:red", marker="s",
               label=r"steady $(\rho_-,q_-)$ upstream (queue)")
    rr = np.linspace(0, P, 400)
    ax.plot(rr, fd.q_tri(rr, v_f, w, P), "k-", lw=2,
            label=(rf"$Q_0$: $v_f$={v_f:.0f} km/h, $w$={w:.1f} km/h, "
                   rf"$P$={P:.0f} veh/km"))
    ax.plot(rr, fd.q_tri(rr, v_f, fit_free_P["w"], fit_free_P["P"]), "k--",
            lw=1, alpha=0.6,
            label=rf"alt fit ($P$ free): $w$={fit_free_P['w']:.1f}, "
                  rf"$P$={fit_free_P['P']:.0f}")
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 5200)
    ax.set_xlabel(r"$\rho$ [veh/km]")
    ax.set_ylabel(r"$q$ [veh/h]")
    ax.set_title("E-V2: triangular FD calibration (A=3 dense set)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ev2_fd_calibration.png", dpi=160)

    fig, ax = plt.subplots(figsize=(7, 4))
    res = q_m[cong] - fd.q_tri(rho_m[cong], v_f, w, P)
    sc = ax.scatter(rho_m[cong], res, c=uc_ms[cong], cmap="viridis", s=30)
    fig.colorbar(sc, ax=ax, label=r"$u_\xi$ [m/s]")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel(r"$\rho_-$ [veh/km]")
    ax.set_ylabel(r"$q_- - Q_0(\rho_-)$ [veh/h]")
    ax.set_title("E-V2: congested-branch residuals")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ev2_residuals.png", dpi=160)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
