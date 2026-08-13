"""E1: per-vehicle caught/free classification and capture/release event
extraction for the core comparison set A in {1, 10} x u_c in {15, 20} m/s
x q_in in {2000, 2500} veh/h (True files, 5 reps).

Classification (hysteresis, per vehicle):
  * enter caught: behind the CAV inside the queue zone and v <= u_c + EPS_IN
    for 2 consecutive samples;
  * stay caught while still behind, within zone + slack, and v <= u_c + EPS_OUT;
  * exit otherwise (or when the trajectory ends / overtakes).
The queue zone is the connected rho > RHO_Q region anchored within
ANCHOR_CELLS upstream of the CAV cell (KDE dilutes density at the CAV itself),
with GAP_CELLS hole tolerance, TAIL_MARGIN extension, FALLBACK_ZONE minimum
and ZONE_CAP maximum.

Events (slow window [T_LO, T_HI] only -- after t = 750 s the model has
Delta v = 0 and queue dissolution is transport, not release):
  * capture: free -> caught transition;
  * release: caught -> free transition confirmed by overtaking the CAV within
    CONFIRM_HORIZON (unconfirmed exits are logged separately);
  * pass-through: overtake without having been caught in the last 60 s.

Stored per (scenario, rep) for the Poisson-exposure MLE in ev3:
  s_field, f_field      caught/free count-densities [veh/km] per cell
  fz_field              free count-density restricted to the queue zone
  n_caught, n_free_zone occupancy counts (-> empirical chi)
  cap, rel, ...         event series per 10 s bin
  rho, v_left, x_cav    aggregate density, mean left-lane speed [m/s], CAV pos

Output: out/e1/A{A}_u{uc}_q{qin}_r{r}.npz + out/e1/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from loader import CELL_LEN, N_CELL, T_SAMPLE, list_scenarios, load_scenario

HERE = Path(__file__).parent
OUT = HERE / "out" / "e1"
SECOND = HERE.parent / "Second"

EPS_IN = 2.0              # [m/s] enter-caught speed slack
EPS_OUT = 4.0             # [m/s] exit-caught speed slack (hysteresis)
RHO_Q = 50.0              # [veh/km] queue-region density threshold
ANCHOR_CELLS = 3          # search this many cells upstream of the CAV cell
GAP_CELLS = 2             # tolerated holes in the queue region
TAIL_MARGIN = 200.0       # [m]
FALLBACK_ZONE = 300.0     # [m]
ZONE_CAP = 4000.0         # [m]
ZONE_SLACK = 300.0        # [m] extra zone length tolerated while caught
T_LO, T_HI = 260.0, 740.0  # [s]
CONFIRM_HORIZON = 120.0   # [s]

CORE = [(A, uc, qin) for A in (1.0, 10.0) for uc in (15.0, 20.0)
        for qin in (2000.0, 2500.0)]


def queue_zone(rho_row: np.ndarray, j_cav: int) -> float:
    """Zone length [m]: connected rho > RHO_Q region anchored near j_cav."""
    if j_cav < 0:
        return 0.0
    j_hi = min(j_cav, N_CELL - 1)
    j_anchor = None
    for jj in range(j_hi, max(j_hi - ANCHOR_CELLS, 0) - 1, -1):
        if rho_row[jj] > RHO_Q:
            j_anchor = jj
            break
    if j_anchor is None:
        return 0.0
    gaps, j_tail = 0, j_anchor
    for jj in range(j_anchor, -1, -1):
        if rho_row[jj] > RHO_Q:
            j_tail, gaps = jj, 0
        else:
            gaps += 1
            if gaps > GAP_CELLS:
                break
    return min((j_hi - j_tail + 1) * CELL_LEN, ZONE_CAP)


def process_rep(sc, r):
    rep = sc.reps[r]
    tt = sc.t_field
    n_t = len(tt)
    uc = sc.uc

    x_cav = np.interp(tt, rep.ctrl.t, rep.ctrl.x, left=np.nan, right=np.nan)
    j_cav = np.full(n_t, -1, int)
    m = np.isfinite(x_cav)
    j_cav[m] = np.clip((x_cav[m] // CELL_LEN).astype(int), 0, N_CELL - 1)

    L_zone = np.zeros(n_t)
    for i in range(n_t):
        if j_cav[i] >= 0:
            lz = queue_zone(rep.rho[i], j_cav[i])
            L_zone[i] = max(lz + TAIL_MARGIN, FALLBACK_ZONE)

    # --- align trajectories on the field grid --------------------------------
    nv = len(rep.trajs)
    X = np.full((nv, n_t), np.nan)
    V = np.full((nv, n_t), np.nan)
    LN = np.full((nv, n_t), np.nan)
    for k, tr in enumerate(rep.trajs):
        i0 = int(round(tr.t[0] / T_SAMPLE)) - 1
        m_len = len(tr.t)
        lo, hi = max(i0, 0), min(i0 + m_len, n_t)
        s0 = lo - i0
        X[k, lo:hi] = tr.x[s0:s0 + hi - lo]
        V[k, lo:hi] = tr.v[s0:s0 + hi - lo]
        LN[k, lo:hi] = tr.lane[s0:s0 + hi - lo]

    # --- hysteresis state machine -------------------------------------------
    caught = np.zeros((nv, n_t), bool)
    inzone = np.zeros((nv, n_t), bool)
    for k in range(nv):
        st, pend = False, 0
        for i in range(n_t):
            x, v = X[k, i], V[k, i]
            if not np.isfinite(x) or not np.isfinite(x_cav[i]):
                st, pend = False, 0
                continue
            gap = x_cav[i] - x
            inz = 0.0 < gap <= L_zone[i]
            inzone[k, i] = inz
            if st:
                st = (0.0 < gap <= L_zone[i] + ZONE_SLACK) and v <= uc + EPS_OUT
            else:
                if inz and v <= uc + EPS_IN:
                    pend += 1
                    if pend >= 2:
                        st = True
                else:
                    pend = 0
            caught[k, i] = st

    in_win = (tt >= T_LO) & (tt <= T_HI)

    # --- events --------------------------------------------------------------
    cap = np.zeros(n_t, int)
    rel = np.zeros(n_t, int)
    rel_unconf = np.zeros(n_t, int)
    for k in range(nv):
        d = np.diff(caught[k].astype(int))
        for i in np.where(d == 1)[0] + 1:
            if in_win[i]:
                cap[i] += 1
        for i in np.where(d == -1)[0] + 1:
            if not in_win[i]:
                continue
            fut = np.arange(i, min(i + int(CONFIRM_HORIZON / T_SAMPLE) + 1, n_t))
            fin = fut[np.isfinite(X[k, fut]) & np.isfinite(x_cav[fut])]
            if fin.size and np.any(X[k, fin] > x_cav[fin] + 20.0):
                rel[i] += 1
            else:
                rel_unconf[i] += 1

    # --- overtakes and pass-throughs ----------------------------------------
    ot = np.zeros(n_t, int)
    passthru = np.zeros(n_t, int)
    crossed = (X[:, :-1] <= x_cav[:-1]) & (X[:, 1:] > x_cav[1:])
    for k in range(nv):
        for i in np.where(crossed[k])[0] + 1:
            if not in_win[i]:
                continue
            ot[i] += 1
            if not np.any(caught[k, max(i - 6, 0):i]):
                passthru[i] += 1

    # --- per-cell classified count-densities ---------------------------------
    s_field = np.zeros((n_t, N_CELL), np.float32)
    f_field = np.zeros((n_t, N_CELL), np.float32)
    fz_field = np.zeros((n_t, N_CELL), np.float32)
    v_left = np.full((n_t, N_CELL), np.nan, np.float32)
    for i in range(n_t):
        ok = np.isfinite(X[:, i])
        jj = np.clip((X[ok, i] // CELL_LEN).astype(int), 0, N_CELL - 1)
        cg = caught[ok, i]
        iz = inzone[ok, i]
        np.add.at(s_field[i], jj[cg], 1000.0 / CELL_LEN)
        np.add.at(f_field[i], jj[~cg], 1000.0 / CELL_LEN)
        np.add.at(fz_field[i], jj[~cg & iz], 1000.0 / CELL_LEN)
        left = ok.copy()
        left[ok] = LN[ok, i] == 1
        if left.any():
            jl = np.clip((X[left, i] // CELL_LEN).astype(int), 0, N_CELL - 1)
            sums = np.zeros(N_CELL)
            cnts = np.zeros(N_CELL)
            np.add.at(sums, jl, V[left, i])
            np.add.at(cnts, jl, 1.0)
            v_left[i, cnts > 0] = sums[cnts > 0] / cnts[cnts > 0]

    return dict(tt=tt, x_cav=x_cav, j_cav=j_cav, L_zone=L_zone,
                n_caught=caught.sum(axis=0),
                n_free_zone=(inzone & ~caught).sum(axis=0),
                cap=cap, rel=rel, rel_unconf=rel_unconf, overtakes=ot,
                passthru=passthru, s_field=s_field, f_field=f_field,
                fz_field=fz_field, v_left=v_left,
                rho=rep.rho.astype(np.float32),
                overtake_meas=rep.overtake, in_win=in_win)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for p, A, uc, qin, _ in list_scenarios(SECOND, flag=True):
        if (A, uc, qin) not in CORE:
            continue
        sc = load_scenario(p, fields=True, ctrl=True, trajs=True)
        for r in sorted(sc.reps):
            rep = sc.reps[r]
            if rep.rho is None or rep.ctrl is None or not rep.trajs:
                continue
            res = process_rep(sc, r)
            tag = f"A{A:g}_u{uc:g}_q{qin:g}_r{r}"
            np.savez_compressed(OUT / f"{tag}.npz", **res)
            w = res["in_win"]
            nc, nf = res["n_caught"][w], res["n_free_zone"][w]
            occ = nc.sum() / max(nc.sum() + nf.sum(), 1)
            row = dict(tag=tag, A=A, uc=uc, qin=qin, rep=r,
                       captures=int(res["cap"].sum()),
                       releases=int(res["rel"].sum()),
                       rel_unconfirmed=int(res["rel_unconf"].sum()),
                       overtakes=int(res["overtakes"].sum()),
                       passthrough=int(res["passthru"].sum()),
                       max_caught=int(res["n_caught"].max()),
                       sync_occupancy=float(occ),
                       max_zone_m=float(res["L_zone"].max()))
            summary.append(row)
            print(f"{tag}: cap={row['captures']:4d} rel={row['releases']:4d} "
                  f"(unconf {row['rel_unconfirmed']:3d}) "
                  f"ot={row['overtakes']:4d} pass={row['passthrough']:4d} "
                  f"maxN_s={row['max_caught']:3d} "
                  f"sync_occ={row['sync_occupancy']:.2f} "
                  f"maxL={row['max_zone_m']:5.0f} m")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{len(summary)} runs written to {OUT}")


if __name__ == "__main__":
    main()
