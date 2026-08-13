"""E-V4b cap-discretization variants, evaluated on the t8 metrics."""

import json
import sys
from pathlib import Path

import numpy as np

EV4B = Path(__file__).parent / "ev4b"
sys.path.insert(0, str(EV4B))

from solver import (SimConfig, cav_density, cav_position, u_s_of_t,  # noqa: E402
                    demand, supply)

p = json.loads((EV4B / "out" / "params.json").read_text())
V_F = p["v_f_kmh"] / 3.6
W = p["w_kmh"] / 3.6
P = p["P_vehkm"] / 1000.0

Q_XI, U_XI, Q_IN, BETA = 2000.0 / 3600.0, 15.0, 2500.0 / 3600.0, 0.5
sigma_xi = BETA * W * P / (V_F + W)
OM = max(Q_XI - U_XI * sigma_xi, 0.0)
rho_minus = (W * P - OM) / (W + U_XI)
q_minus = OM + U_XI * rho_minus
rho_in = Q_IN / V_F
z_rh = (q_minus - Q_IN) / (rho_minus - rho_in)


def run(variant: str):
    cfg = SimConfig(v_f=V_F, w=W, P=P, q_in=Q_IN, u_xi=U_XI,
                    kappa_c=0.0, kappa_r=0.0, dt=0.5, save_every=20)
    nx = int(round(cfg.L_road / cfg.dx))
    lam = cfg.dt / cfg.dx
    f = np.zeros(nx)
    s = np.zeros(nx)
    saves = {}
    tails = []
    n_steps = int(round(cfg.t_end / cfg.dt))
    for n in range(n_steps + 1):
        t = n * cfg.dt
        if n % cfg.save_every == 0:
            a = cav_density(cfg, t, nx)
            rho = a + f + s
            saves[round(t)] = (f.copy(), s.copy(), a, rho)
            if 350.0 <= t <= 650.0:
                ks = np.where(rho > 0.6 * rho_minus)[0]
                tails.append((t, (ks[0] + 0.5) * cfg.dx if ks.size else np.nan))
        if n == n_steps:
            break
        u_s = u_s_of_t(cfg, t)
        a = cav_density(cfg, t, nx)
        rho = a + f + s
        r_safe = np.maximum(rho, 1e-300)
        pi_f = np.where(rho > 0.0, f / r_safe, 0.0)
        D_f = demand(rho, cfg.v_f, cfg.w, cfg.P)
        S_f = supply(rho, cfg.v_f, cfg.w, cfg.P)
        F_f = np.empty(nx + 1)
        F_f[1:-1] = pi_f[:-1] * np.minimum(D_f[:-1], S_f[1:])
        F_f[0] = min(cfg.q_in, S_f[0])
        F_f[-1] = pi_f[-1] * D_f[-1]
        if cfg.t_slow <= t <= cfg.t_fast:
            x_cav = cav_position(cfg, t)
            if np.isfinite(x_cav):
                j = min(int(x_cav / cfg.dx), nx - 1)
                if variant == "A":            # rho_j incl. a (spec literal)
                    caps = {j + 1: OM + U_XI * rho[j]}
                elif variant == "B":          # traffic density in cell j
                    caps = {j + 1: OM + U_XI * (f[j] + s[j])}
                elif variant == "C":          # upstream-neighbor traffic
                    caps = {j + 1: OM + U_XI * (f[j - 1] + s[j - 1])}
                elif variant == "F":          # max(j-1, j) traffic
                    caps = {j + 1: OM + U_XI * max(f[j] + s[j],
                                                   f[j - 1] + s[j - 1])}
                elif variant == "E":          # two faces j+1/2, j+3/2
                    caps = {j + 1: OM + U_XI * (f[j] + s[j]),
                            j + 2: OM + U_XI * (f[j + 1] + s[j + 1])}
                elif variant == "D":          # a shares the capped budget
                    caps = {}
                    cap = OM + U_XI * rho[j]
                    if F_f[j + 1] > pi_f[j] * cap:
                        F_f[j + 1] = pi_f[j] * cap
                else:
                    raise ValueError(variant)
                if variant != "D":
                    for face, cap in caps.items():
                        if face <= nx and F_f[face] > cap:
                            F_f[face] = cap
        f = f - lam * np.diff(F_f)

    # metrics
    f7, s7, a7, rho7 = saves[700]
    xc = cav_position(cfg, 700.0)
    x = (np.arange(nx) + 0.5) * cfg.dx
    m = (x >= xc - 1000.0) & (x <= xc - 200.0)
    rho_sim = rho7[m].mean()
    # face flux at t=700 for this variant: recompute one step
    j = int(xc / cfg.dx)
    rho = a7 + f7 + s7
    r_safe = np.maximum(rho, 1e-300)
    pi = np.where(rho > 0, f7 / r_safe, 0.0)
    D = demand(rho, V_F, W, P)
    S = supply(rho, V_F, W, P)
    Fun = pi[j] * min(D[j], S[j + 1])
    if variant == "A":
        cap = OM + U_XI * rho[j]
    elif variant in ("B", "E"):
        cap = OM + U_XI * (f7[j] + s7[j])
    elif variant == "C":
        cap = OM + U_XI * (f7[j - 1] + s7[j - 1])
    elif variant == "F":
        cap = OM + U_XI * max(f7[j] + s7[j], f7[j - 1] + s7[j - 1])
    elif variant == "D":
        cap = pi[j] * (OM + U_XI * rho[j])
    q_face = min(Fun, cap)
    ts = np.array([tt for tt, _ in tails])
    xs = np.array([xx for _, xx in tails])
    ok = np.isfinite(xs)
    slope = np.polyfit(ts[ok], xs[ok], 1)[0] if ok.sum() > 2 else np.nan
    print(f"[{variant}] mean_rho {rho_sim:.6f} ({100*(rho_sim/rho_minus-1):+6.2f}%)"
          f"  q_face {q_face:.4f} ({100*(q_face/q_minus-1):+6.2f}%)"
          f"  tail_z {slope:7.4f} ({100*(slope/z_rh-1):+6.2f}%)"
          f"  max_rho_traffic {np.max(f7+s7)*1000:.1f}")


print(f"targets: rho_minus={rho_minus:.6f}, q_minus={q_minus:.6f}, "
      f"z_rh={z_rh:.4f}, crit={W*P/(V_F+W)*1000:.1f} veh/km")
for v in ("A", "B", "C", "F", "E", "D"):
    run(v)
