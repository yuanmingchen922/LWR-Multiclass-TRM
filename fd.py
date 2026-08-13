"""Triangular fundamental diagram (FD) and calibration helpers.

Model (Multi-class LWR Equations.tex, eq. intrinsic-fd):
    Q_c(r) = min{ c r, w (P - r) },   r in [0, P]
with c the free-flow speed, w the backward wave speed, P the jam density.
Units here: r [veh/km], q [veh/h], speeds [km/h].
"""

from __future__ import annotations

import numpy as np

MS_TO_KMH = 3.6

# Bumper-to-bumper prior: 2 lanes / (length 5 m + minGap 2.5 m)
P_PRIOR = 2 * 1000.0 / 7.5   # = 266.7 veh/km


def q_tri(rho, c, w, P):
    return np.minimum(c * np.asarray(rho), w * (P - np.asarray(rho)))


def demand(rho, c, w, P):
    cap = c * w * P / (c + w)
    return np.minimum(c * np.asarray(rho), cap)


def supply(rho, c, w, P):
    cap = c * w * P / (c + w)
    return np.minimum(cap, w * (P - np.asarray(rho)))


def crit_density(c, w, P):
    return w * P / (c + w)


def capacity(c, w, P):
    return c * w * P / (c + w)


def fit_free_speed(rho, q, n_iter: int = 3, keep_frac: float = 0.9):
    """Slope of the free-flow branch through the origin, robust to congested
    contamination: start from a high quantile of point speeds, then iteratively
    refit on points whose speed is within `keep_frac` of the current estimate.

    Returns (v_f [km/h], mask of points used).
    """
    rho = np.asarray(rho, float)
    q = np.asarray(q, float)
    ok = rho > 2.0
    speed = np.where(ok, q / np.maximum(rho, 1e-9), np.nan)
    v = np.nanpercentile(speed[ok], 90)
    mask = ok
    for _ in range(n_iter):
        mask = ok & (speed >= keep_frac * v)
        # least squares through origin on the retained points
        v = float(np.sum(q[mask] * rho[mask]) / np.sum(rho[mask] ** 2))
    return v, mask


def fit_congested(rho, q, P_fixed: float | None = None, n_iter: int = 5,
                  trim: float = 2.5):
    """Fit the congested branch q = w (P - rho).

    If P_fixed is given, only w is fitted (LSQ). Otherwise fit the affine law
    q = b0 + b1 rho with iterative sigma-trimming (drop points beyond
    `trim` * robust std of residuals), then w = -b1, P = -b0/b1.

    Returns dict(w, P, rmse, n_used, mask).
    """
    rho = np.asarray(rho, float)
    q = np.asarray(q, float)
    mask = np.isfinite(rho) & np.isfinite(q)
    if P_fixed is not None:
        for _ in range(n_iter):
            z = P_fixed - rho[mask]
            w = float(np.sum(q[mask] * z) / np.sum(z ** 2))
            res = q - w * (P_fixed - rho)
            sd = 1.4826 * np.median(np.abs(res[mask] - np.median(res[mask])))
            new = np.isfinite(res) & (np.abs(res - np.median(res[mask])) <= trim * max(sd, 1e-9))
            if new.sum() == mask.sum():
                break
            mask = new & np.isfinite(rho)
        res = q[mask] - w * (P_fixed - rho[mask])
        return dict(w=w, P=P_fixed, rmse=float(np.sqrt(np.mean(res ** 2))),
                    n_used=int(mask.sum()), mask=mask)

    for _ in range(n_iter):
        b1, b0 = np.polyfit(rho[mask], q[mask], 1)
        res = q - (b0 + b1 * rho)
        sd = 1.4826 * np.median(np.abs(res[mask] - np.median(res[mask])))
        new = np.isfinite(res) & (np.abs(res - np.median(res[mask])) <= trim * max(sd, 1e-9))
        if new.sum() == mask.sum():
            break
        mask = new & np.isfinite(rho)
    w = -float(b1)
    P = float(-b0 / b1)
    res = q[mask] - (b0 + b1 * rho[mask])
    return dict(w=w, P=P, rmse=float(np.sqrt(np.mean(res ** 2))),
                n_used=int(mask.sum()), mask=mask)


def steady_state_mean(series: np.ndarray, i_slow: int, i_fast: int,
                      C: float = 0.15, min_samples: int = 5):
    """ECC22 eq. (7): average of `series` over the steady-state window
    [t_ss, t_fast], where t_ss is the earliest time in [t_slow, t_fast] from
    which the relative deviation w.r.t. series[i_fast] stays within C.

    Returns (mean, n_window) or (nan, 0) if no valid window exists.
    """
    ref = series[i_fast]
    if not np.isfinite(ref) or abs(ref) < 1e-9:
        return np.nan, 0
    seg = series[i_slow:i_fast + 1]
    ok = np.abs(1.0 - seg / ref) <= C
    # last False before i_fast determines the window start
    bad = np.where(~ok)[0]
    start = 0 if bad.size == 0 else bad[-1] + 1
    if len(seg) - start < min_samples:
        return np.nan, int(len(seg) - start)
    return float(np.mean(seg[start:])), int(len(seg) - start)
