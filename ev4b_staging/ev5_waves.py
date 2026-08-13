"""E-V5 (data side): spectral quantification of wave stripes in SUMO density fields.

Method
------
For an analysis box rho[t, x] (veh/km) the field is two-way demeaned (grand mean,
per-time spatial mean, per-cell temporal mean removed), tapered with a 2D Hann
window and transformed with a zero-padded FFT2.  The power spectral density is
expressed over (k [cycles/km], f [Hz]).  For a field stored as rho[t, x] and
modes ~ exp(2*pi*i*(f*t + k*x)), a structure moving with speed c satisfies
f = -c*k, i.e. c = -f/k.  Backward (upstream, c < 0) power is therefore the
power in bins with sign(f) == sign(k).

A DC guard band (|k| < 0.25 cycles/km OR |f| < 1/480 Hz) is excluded from all
statistics.

Self-test (MANDATORY, gates everything): synthetic plane waves on the SUMO data
grid (dx=100 m, dt=10 s) with c = +20 and c = -18 km/h, wavelength 1.5 km, must
be recovered within 10 % with the correct sign, and backward_frac must exceed
0.8 (backward case) / stay below 0.2 (forward case).

Main analysis: scenarios A in {1,3,10} x u_xi in {15,20} x q_in=2500 plus
A in {1,10} x u15 x q2000, True files, 5 reps.  Box: t in [400, 740] s,
x in [1.5 km, min_t(queue tail) - 0.5 km] (tail = x_cav - L_zone from
out/e1/*.npz for A in {1,10}; x_cav(ctrl) - 2.5 km proxy for A=3).

Outputs: out/ev5/waves_summary.json, fig_waves_examples.png, fig_waves_psd.png,
fig_waves_trend.png.

Resolution caveat (measured on synthetic waves at the real box size, n_t≈34):
with the 340 s window, |c| estimates for lambda≈1.5 km waves are reliable for
|c| >~ 15 km/h; slower waves (period >~ 350 s) hit the window/guard-band floor
and |c_dom| is biased toward ≈19 km/h (sign and backward_frac stay correct).
Treat |c_dom| in that regime as a lower bound on the period, not a speed.

NOTE: the loader / e1-npz field names below are written against the documented
conventions and marked with "VERIFY:" comments; confirm them against loader.py
before the first real run (this file was authored in an environment where the
analysis directory was not readable).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# paths and constants
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent                  # .../SUMO/analysis
DATA_DIR = BASE.parent / "Second"                       # SUMO csv/data root
OUT_DIR = BASE / "out" / "ev5"
E1_DIR = BASE / "out" / "e1"

DX_M = 100.0            # SUMO grid spacing [m], upper-edge convention XS=(1:300)*100
DT_S = 10.0             # SUMO grid step [s], t = (1..100)*10 s
T_BOX = (400.0, 740.0)  # analysis window [s]
X_LO_KM = 1.5           # default lower edge [km]
X_LO_MIN_KM = 1.0       # widened lower edge if box would be too narrow
MIN_WIDTH_KM = 3.0      # required box width
TAIL_MARGIN_KM = 0.5    # stay this far downstream of the queue tail
A3_TAIL_PROXY_KM = 2.5  # conservative x_cav offset for A=3 (no e1 npz)

K_GUARD = 0.25          # DC guard: exclude |k| < 0.25 cycles/km
F_GUARD = 1.0 / 480.0   # DC guard: exclude |f| < 1/480 Hz

REPS = range(5)
# (A, u_xi [m/s], q_in [veh/h])
SCENARIOS = [(a, u, 2500) for a in (1, 3, 10) for u in (15, 20)] + \
            [(a, 15, 2000) for a in (1, 10)]

CAV_T_ENTER = 100.0     # CAV enters x=0 [s]
CAV_V_FREE = 27.78      # free speed before/after slow window [m/s]
CAV_T_SLOW = (250.0, 750.0)


# ----------------------------------------------------------------------------
# core spectral analysis
# ----------------------------------------------------------------------------
def analyze_field(rho, dx_m, dt_s, x_slice, t_slice,
                  pad_factor=4, refine_bins=None):
    """Spectrally quantify wave stripes in a density field.

    Parameters
    ----------
    rho : (n_t, n_x) array
        Density field [veh/km], time along axis 0, space along axis 1.
    dx_m, dt_s : float
        Grid spacings [m], [s].
    x_slice, t_slice : slice
        Select the analysis box: rho[t_slice, x_slice].
    pad_factor : int
        FFT zero-padding factor per axis (finer spectral sampling).
    refine_bins : int or None
        Half-width (in padded bins) of the PSD-weighted centroid used to
        refine the dominant peak location; defaults to ``pad_factor``.

    Returns
    -------
    dict with keys
        amp            : std of the detrended box [veh/km]
        backward_frac  : fraction of retained PSD with phase speed c < 0
        c_dom_kmh      : phase speed of the dominant PSD peak [km/h] (c=-f/k)
        c_top10_kmh    : energy-weighted mean speed over top-decile PSD bins
        k_dom_perkm    : dominant spatial frequency [cycles/km] (k>0 convention)
        f_dom_hz       : dominant temporal frequency [Hz] (sign matched to k>0)
        psd            : (n_f, n_k) fftshifted PSD (guard band NOT masked)
        k_axis, f_axis : shifted axes [cycles/km], [Hz]
    """
    box = np.asarray(rho, dtype=float)[t_slice, x_slice]
    n_t, n_x = box.shape
    if n_t < 8 or n_x < 8:
        raise ValueError(f"analysis box too small: {box.shape}")

    # --- detrend: 2-way demean (grand mean + row means + column means) -----
    grand = box.mean()
    row = box.mean(axis=1, keepdims=True)   # per-time spatial mean profile
    col = box.mean(axis=0, keepdims=True)   # per-cell temporal mean
    det = box - row - col + grand

    amp = float(det.std())

    # --- 2D Hann window + zero-padded FFT2 ---------------------------------
    w_t = np.hanning(n_t)[:, None]
    w_x = np.hanning(n_x)[None, :]
    win = det * w_t * w_x

    n_ft = pad_factor * n_t
    n_fx = pad_factor * n_x
    F = np.fft.fft2(win, s=(n_ft, n_fx))
    psd = np.abs(F) ** 2 / (np.sum((w_t * w_x) ** 2) * n_t * n_x)
    psd = np.fft.fftshift(psd)

    dx_km = dx_m / 1000.0
    f_axis = np.fft.fftshift(np.fft.fftfreq(n_ft, d=dt_s))    # [Hz]
    k_axis = np.fft.fftshift(np.fft.fftfreq(n_fx, d=dx_km))   # [cycles/km]
    K = k_axis[None, :]
    Fq = f_axis[:, None]

    # --- DC guard band ------------------------------------------------------
    guard = (np.abs(K) < K_GUARD) | (np.abs(Fq) < F_GUARD)
    retained = ~guard
    p_ret = psd[retained]
    p_tot = p_ret.sum()

    # --- backward fraction: c = -f/k < 0  <=>  sign(f) == sign(k) ----------
    backward = retained & ((K * Fq) > 0.0)
    backward_frac = float(psd[backward].sum() / p_tot)

    # --- dominant peak (with centroid refinement) --------------------------
    p_masked = np.where(retained, psd, 0.0)
    i_pk, j_pk = np.unravel_index(np.argmax(p_masked), p_masked.shape)

    r = pad_factor if refine_bins is None else refine_bins
    i0, i1 = max(i_pk - r, 0), min(i_pk + r + 1, n_ft)
    j0, j1 = max(j_pk - r, 0), min(j_pk + r + 1, n_fx)
    sub_p = p_masked[i0:i1, j0:j1]
    sub_f = f_axis[i0:i1][:, None]
    sub_k = k_axis[j0:j1][None, :]
    s = sub_p.sum()
    if s > 0:
        f_dom = float((sub_p * sub_f).sum() / s)
        k_dom = float((sub_p * sub_k).sum() / s)
    else:  # pragma: no cover - retained set empty only for degenerate input
        f_dom, k_dom = float(f_axis[i_pk]), float(k_axis[j_pk])

    if k_dom < 0:            # real field: (k,f) and (-k,-f) carry equal power
        k_dom, f_dom = -k_dom, -f_dom
    c_dom_kmh = float(-f_dom / k_dom * 3600.0)

    # --- energy-weighted mean speed over top-decile retained bins ----------
    thr = np.quantile(p_ret, 0.9)
    top = retained & (psd >= thr)
    c_bins = -Fq / np.where(K == 0.0, np.nan, K) * 3600.0
    w = psd[top]
    c_top10_kmh = float(np.nansum(w * c_bins[top]) / w.sum())

    return dict(amp=amp, backward_frac=backward_frac, c_dom_kmh=c_dom_kmh,
                c_top10_kmh=c_top10_kmh, k_dom_perkm=float(k_dom),
                f_dom_hz=float(f_dom), psd=psd, k_axis=k_axis, f_axis=f_axis)


# ----------------------------------------------------------------------------
# MANDATORY self-test — gates everything (sign conventions!)
# ----------------------------------------------------------------------------
def _synthetic_wave(c_kmh, lam_km=1.5, amp=5.0, sigma=0.5, seed=0,
                    n_t=100, n_x=300):
    """rho(t,x) = amp*sin(2 pi (x/L - t/T)) + noise on the SUMO data grid.

    Signed period T = L/c (hours -> s), so constant phase moves at x = c*t:
    c > 0 forward (downstream), c < 0 backward (upstream).
    """
    t = np.arange(1, n_t + 1) * DT_S                    # [s]
    x = np.arange(1, n_x + 1) * DX_M / 1000.0           # [km]
    T_s = lam_km / c_kmh * 3600.0                       # signed period [s]
    phase = 2.0 * np.pi * (x[None, :] / lam_km - t[:, None] / T_s)
    rng = np.random.default_rng(seed)
    return amp * np.sin(phase) + sigma * rng.standard_normal((n_t, n_x))


def self_test(verbose=True):
    """Assert correct speed magnitude, sign, and backward_frac on synthetic waves."""
    cases = [(+20.0, "forward"), (-18.0, "backward")]
    for c_true, name in cases:
        rho = _synthetic_wave(c_true)
        res = analyze_field(rho, DX_M, DT_S, slice(None), slice(None))
        c_est = res["c_dom_kmh"]
        rel = abs(c_est - c_true) / abs(c_true)
        if verbose:
            lam = 1.0 / res["k_dom_perkm"]
            per = 1.0 / abs(res["f_dom_hz"])
            print(f"[self-test] {name:8s} c_true={c_true:+7.2f} km/h  "
                  f"c_dom={c_est:+7.2f} ({100*rel:.1f}% err)  "
                  f"c_top10={res['c_top10_kmh']:+7.2f}  "
                  f"backward_frac={res['backward_frac']:.3f}  "
                  f"lambda={lam:.2f} km  T={per:.0f} s")
        assert np.sign(c_est) == np.sign(c_true), \
            f"{name}: sign wrong (c_dom={c_est:.2f} vs {c_true:.2f})"
        assert rel < 0.10, \
            f"{name}: |c| error {100*rel:.1f}% > 10% (c_dom={c_est:.2f})"
        if c_true < 0:
            assert res["backward_frac"] > 0.8, \
                f"backward case: backward_frac={res['backward_frac']:.3f} <= 0.8"
        else:
            assert res["backward_frac"] < 0.2, \
                f"forward case: backward_frac={res['backward_frac']:.3f} >= 0.2"
    if verbose:
        print("[self-test] PASSED (sign convention c = -f/k for rho[t, x] confirmed)")


# ----------------------------------------------------------------------------
# data access (VERIFY against loader.py / e1 npz before first real run)
# ----------------------------------------------------------------------------
def _load_field(A, u_xi, q_in, rep):
    """Return (rho[t,x] veh/km, t[s], x[km]) for a True-file scenario rep.

    VERIFY: exact loader.load_scenario signature/return.  Documented use:
    'load density fields via loader.load_scenario (fields only)' from
    analysis/../Second.  Adjust the call below if the real API differs.
    """
    sys.path.insert(0, str(BASE))
    import loader  # noqa: E402  (analysis-dir module)
    sc = loader.load_scenario(A=A, u=u_xi, q=q_in, rep=rep,
                              base=DATA_DIR, true=True)   # VERIFY kwargs
    rho = np.asarray(sc["rho"], dtype=float)              # VERIFY key (n_t, n_x)
    t = np.asarray(sc.get("t", np.arange(1, rho.shape[0] + 1) * DT_S))
    x = np.asarray(sc.get("x", np.arange(1, rho.shape[1] + 1) * DX_M)) / 1000.0
    return rho, t, x


def _load_ctrl_xcav(A, u_xi, q_in, rep, t_grid):
    """CAV position [km] on t_grid from the ctrl trajectory (A=3 proxy path).

    VERIFY: loader call with ctrl=True and returned trajectory keys.
    """
    sys.path.insert(0, str(BASE))
    import loader
    sc = loader.load_scenario(A=A, u=u_xi, q=q_in, rep=rep,
                              base=DATA_DIR, true=True, ctrl=True)  # VERIFY
    t_c = np.asarray(sc["t_cav"], dtype=float)            # VERIFY key
    x_c = np.asarray(sc["x_cav"], dtype=float) / 1000.0   # VERIFY key/unit [m]
    return np.interp(t_grid, t_c, x_c)


def _queue_tail_km(A, u_xi, q_in, rep, t_grid):
    """Queue-tail position [km] on t_grid.

    A in {1, 10}: x_cav - L_zone from out/e1/*.npz when available.
    A = 3 (no npz): x_cav(ctrl) - 2.5 km conservative proxy.
    Fallback (npz missing/unreadable): kinematic x_cav - 2.5 km proxy.
    """
    if A in (1, 10):
        # VERIFY: e1 npz naming pattern and keys (t, x_cav, L_zone).
        patterns = [f"A{A}_u{u_xi}_q{q_in}_rep{rep}.npz",
                    f"a{A}_u{u_xi}_q{q_in}_r{rep}.npz"]
        for pat in patterns:
            p = E1_DIR / pat
            if p.exists():
                with np.load(p) as z:
                    keys = set(z.files)
                    if {"t", "x_cav", "L_zone"} <= keys:
                        tail = (np.asarray(z["x_cav"], float)
                                - np.asarray(z["L_zone"], float)) / 1000.0
                        return np.interp(t_grid, np.asarray(z["t"], float), tail)
                    print(f"  [warn] {p.name}: unexpected keys {sorted(keys)}; "
                          f"falling back to kinematic proxy")
        print(f"  [warn] no e1 npz for A{A} u{u_xi} q{q_in} rep{rep}; "
              f"kinematic proxy used")
        return _kinematic_xcav_km(t_grid, u_xi) - A3_TAIL_PROXY_KM
    # A = 3: ctrl trajectory minus conservative offset
    try:
        return _load_ctrl_xcav(A, u_xi, q_in, rep, t_grid) - A3_TAIL_PROXY_KM
    except Exception as e:  # loader/ctrl unavailable -> kinematic fallback
        print(f"  [warn] ctrl trajectory unavailable ({e}); kinematic proxy")
        return _kinematic_xcav_km(t_grid, u_xi) - A3_TAIL_PROXY_KM


def _kinematic_xcav_km(t_grid, u_xi):
    """Nominal CAV position [km]: enter x=0 at t=100 s at 27.78 m/s, u_xi in [250,750]."""
    t = np.asarray(t_grid, dtype=float)
    t0, (t1, t2) = CAV_T_ENTER, CAV_T_SLOW
    x = np.where(
        t < t1,
        CAV_V_FREE * np.clip(t - t0, 0.0, None),
        CAV_V_FREE * (t1 - t0) + u_xi * np.clip(t - t1, 0.0, None)
        + (CAV_V_FREE - u_xi) * np.clip(t - t2, 0.0, None),
    )
    return x / 1000.0


def _analysis_box(t, x, tail_km):
    """Index slices for the analysis box; returns (t_slice, x_slice, info)."""
    t_sl = slice(int(np.searchsorted(t, T_BOX[0])),
                 int(np.searchsorted(t, T_BOX[1], side="right")))
    x_hi = float(np.min(tail_km) - TAIL_MARGIN_KM)
    x_lo, widened = X_LO_KM, False
    if x_hi - x_lo < MIN_WIDTH_KM:
        x_lo, widened = X_LO_MIN_KM, True
    width = x_hi - x_lo
    x_sl = slice(int(np.searchsorted(x, x_lo)),
                 int(np.searchsorted(x, x_hi, side="right")))
    return t_sl, x_sl, dict(x_lo_km=x_lo, x_hi_km=x_hi, width_km=width,
                            widened=widened)


# ----------------------------------------------------------------------------
# main analysis
# ----------------------------------------------------------------------------
def run_analysis():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, examples = {}, {}
    for (A, u_xi, q_in) in SCENARIOS:
        key = f"A{A}_u{u_xi}_q{q_in}"
        per_rep = []
        for rep in REPS:
            rho, t, x = _load_field(A, u_xi, q_in, rep)
            tail = _queue_tail_km(A, u_xi, q_in, rep,
                                  t[(t >= T_BOX[0]) & (t <= T_BOX[1])])
            t_sl, x_sl, box = _analysis_box(t, x, tail)
            if box["width_km"] < MIN_WIDTH_KM:
                print(f"  [warn] {key} rep{rep}: box width "
                      f"{box['width_km']:.2f} km < {MIN_WIDTH_KM} km even after "
                      f"widening to x_lo={box['x_lo_km']} km")
            elif box["widened"]:
                print(f"  [note] {key} rep{rep}: lower edge widened to "
                      f"{box['x_lo_km']} km (width {box['width_km']:.2f} km)")
            res = analyze_field(rho, DX_M, DT_S, x_sl, t_sl)
            res["box"] = box
            per_rep.append(res)
            if rep == 0 and u_xi == 15 and q_in == 2500:
                det = _detrended_box(rho, t_sl, x_sl)
                examples[A] = dict(det=det, res=res, box=box,
                                   t=t[t_sl], x=x[x_sl])

        def stat(name):
            v = np.array([r[name] for r in per_rep], dtype=float)
            return dict(mean=float(v.mean()), min=float(v.min()),
                        max=float(v.max()))

        lam = np.array([1.0 / r["k_dom_perkm"] for r in per_rep])
        per = np.array([1.0 / abs(r["f_dom_hz"]) for r in per_rep])
        summary[key] = dict(
            A=A, u_xi=u_xi, q_in=q_in, n_reps=len(per_rep),
            amp=stat("amp"), backward_frac=stat("backward_frac"),
            c_dom_kmh=stat("c_dom_kmh"), c_top10_kmh=stat("c_top10_kmh"),
            wavelength_km=dict(mean=float(lam.mean()), min=float(lam.min()),
                               max=float(lam.max())),
            period_s=dict(mean=float(per.mean()), min=float(per.min()),
                          max=float(per.max())),
            boxes=[r["box"] for r in per_rep],
        )

    with open(OUT_DIR / "waves_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    _fig_examples(examples)
    _fig_psd(examples)
    _fig_trend(summary)
    _print_table(summary)
    return summary


def _detrended_box(rho, t_sl, x_sl):
    box = np.asarray(rho, dtype=float)[t_sl, x_sl]
    return box - box.mean(axis=1, keepdims=True) \
               - box.mean(axis=0, keepdims=True) + box.mean()


def _fig_examples(examples):
    if not examples:
        return
    vmax = max(np.abs(e["det"]).max() for e in examples.values())
    fig, axes = plt.subplots(1, len(examples), figsize=(4.2 * len(examples), 4),
                             sharey=True, constrained_layout=True)
    for ax, A in zip(np.atleast_1d(axes), sorted(examples)):
        e = examples[A]
        im = ax.pcolormesh(e["x"], e["t"], e["det"], cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_title(f"A={A} (u15 q2500 rep0)")
        ax.set_xlabel("x [km]")
    np.atleast_1d(axes)[0].set_ylabel("t [s]")
    fig.colorbar(im, ax=axes, label=r"$\rho'$ [veh/km]", shrink=0.85)
    fig.suptitle("Detrended density: wave stripes")
    fig.savefig(OUT_DIR / "fig_waves_examples.png", dpi=160)
    plt.close(fig)


def _fig_psd(examples):
    if not examples:
        return
    fig, axes = plt.subplots(1, len(examples), figsize=(4.6 * len(examples), 4),
                             sharey=True, constrained_layout=True)
    for ax, A in zip(np.atleast_1d(axes), sorted(examples)):
        r = examples[A]["res"]
        p = np.log10(r["psd"] + 1e-12)
        ax.pcolormesh(r["k_axis"], r["f_axis"] * 1000.0, p, cmap="magma",
                      shading="auto")
        kk = np.array([r["k_axis"].min(), r["k_axis"].max()])
        for c in (10.0, 20.0, -10.0, -20.0):     # guide lines f = -c*k
            ax.plot(kk, -c / 3600.0 * kk * 1000.0, lw=0.8,
                    ls="--" if c > 0 else "-", color="w", alpha=0.7)
            ax.annotate(f"{c:+.0f}", (kk[1], -c / 3600.0 * kk[1] * 1000.0),
                        color="w", fontsize=7, ha="right")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-15, 15)
        ax.set_title(f"A={A}: c_dom={r['c_dom_kmh']:+.1f} km/h")
        ax.set_xlabel("k [cycles/km]")
    np.atleast_1d(axes)[0].set_ylabel("f [mHz]")
    fig.suptitle("2D PSD (log10), guide lines c = ±10, ±20 km/h")
    fig.savefig(OUT_DIR / "fig_waves_psd.png", dpi=160)
    plt.close(fig)


def _fig_trend(summary):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    panels = [("amp", r"amp [veh/km]"), ("backward_frac", "backward frac"),
              ("c_dom_kmh", r"$c_{dom}$ [km/h]")]
    marks = {(15, 2500): ("o", "u15 q2500"), (20, 2500): ("s", "u20 q2500"),
             (15, 2000): ("^", "u15 q2000")}
    for ax, (name, lab) in zip(axes, panels):
        for (u, q), (m, leg) in marks.items():
            rows = sorted((s["A"], s[name]) for s in summary.values()
                          if s["u_xi"] == u and s["q_in"] == q)
            if not rows:
                continue
            As = [r[0] for r in rows]
            mean = [r[1]["mean"] for r in rows]
            lo = [r[1]["mean"] - r[1]["min"] for r in rows]
            hi = [r[1]["max"] - r[1]["mean"] for r in rows]
            ax.errorbar(As, mean, yerr=[lo, hi], marker=m, capsize=3,
                        ls="-", label=leg)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10], ["1", "3", "10"])
        ax.set_xlabel("A")
        ax.set_ylabel(lab)
        if name == "c_dom_kmh":
            ax.axhline(0.0, color="k", lw=0.6)
    axes[0].legend(fontsize=8)
    fig.suptitle("Wave metrics vs assertiveness A (mean, min–max over reps)")
    fig.savefig(OUT_DIR / "fig_waves_trend.png", dpi=160)
    plt.close(fig)


def _print_table(summary):
    hdr = (f"{'scenario':16s} {'amp':>7s} {'bwd_frac':>9s} {'c_dom':>8s} "
           f"{'c_top10':>8s} {'lambda':>7s} {'T':>6s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for key, s in summary.items():
        print(f"{key:16s} {s['amp']['mean']:7.2f} "
              f"{s['backward_frac']['mean']:9.3f} "
              f"{s['c_dom_kmh']['mean']:+8.2f} {s['c_top10_kmh']['mean']:+8.2f} "
              f"{s['wavelength_km']['mean']:7.2f} {s['period_s']['mean']:6.0f}")
    print("(amp veh/km; c km/h; lambda km; T s; means over reps)")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    self_test()                       # gates everything
    if "--selftest" not in sys.argv:
        run_analysis()
