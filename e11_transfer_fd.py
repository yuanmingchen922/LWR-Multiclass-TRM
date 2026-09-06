"""E11 (paper): speed-transfer curve and fundamental-diagram figure.

Two paper figures in the shared style (paperfig.py) plus their numbers:

fig_transfer   Zero-refit speed transfer of the catch & release model in
               the paper's final (no-cap) structure, A = 3 anchor:
               stuck-class branch w_s = 0.6 w + s-impermeable bottleneck
               interface + catch & release (lf form, ell = a + s), NO DM-G
               capacity cap.  (kappa_c, kappa_r) are fitted ONCE on the
               A=3 / u_xi=15 m/s (54 km/h) / q_in=2500 veh/h rep-mean
               density field by 1-Wasserstein (e7_wasserstein.fit_field,
               loader-direct path: A=3 has no out/e1 npz), then u_xi is
               swept over {10, 12, 14, 15, 16, 18, 20, 22, 24} m/s at
               q_in = 2500 with the kappas frozen.  The classical baseline
               M1 (kappa = 0, DM-G cap 2000 veh/h, no w_s, no impermeable
               face -- no fitted parameter at all) is scored identically.
               Panels: W1 (top) and e_s (bottom) vs u_xi [km/h].
               The previous capped-structure sweep (E10 structure: cap
               2000 veh/h + w_s = 0.6 w + s_impermeable; A = 1 / A = 10
               u20 triangles) is kept in transfer.json['capped_reference']
               and no longer plotted.

fig_fd         The calibrated triangular FD (out/params.json) over the A=3
               field cloud and the ECC22-style steady upstream / downstream
               states (ev2_calibrate.gather), critical density and the
               classical bottleneck capacity Q_xi marked -- the ECC22
               Fig. 2 analog.

Metrics (all vs the rep-mean data of ../Second *_True.mat):
  W1        mean 1-Wasserstein distance of cumulative densities [veh km],
            t in [100, 1000] s, x <= 20 km (e7_wasserstein.w1_mean);
  rho-RMSE  pointwise density RMSE [veh/km], same window;
  e_s       ECC22 eq. (6) analogue at X_q = 15 km over [576 s, nominal CAV
            arrival at X_q] (ev4_compare.flux_sim_xq / t_reach; the flows
            field at data cell 149): per rep -> mean, min, max; also vs the
            rep-mean flow.
A=3 data straight from the .mat via loader.load_scenario (fields only,
rep-stacked); A in {1, 10} via ev4_compare.load_measured (kept for the
archived capped-reference block).  All evaluation sims run at production
resolution dt = 0.5 s / save_every 20.

Outputs (out/paper/): fig_transfer.{png,pdf}, fig_fd.{png,pdf},
transfer.json (fit, sweep, headline, FD numbers, capped_reference).
Captions: out/paper/captions.md (hand-written from the numbers).

CLI:  python3 e11_transfer_fd.py [--refit] [--skip-transfer] [--skip-fd]
      (the A=3 anchor fit is cached inside transfer.json and reused unless
      --refit is given or the cached fit used a different structure)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.ticker import PercentFormatter

import e7_wasserstein as e7w
import ev4_compare as ev4
import fd
import paperfig as pf
from loader import load_scenario

HERE = Path(__file__).parent
OUT = HERE / "out" / "paper"
SECOND = HERE.parent / "Second"
PARAMS_JSON = HERE / "out" / "params.json"
TRANSFER_JSON = OUT / "transfer.json"

# ---- anchor structure (final, no cap) and the classical baseline ----------
QXI_VEHH = 2000.0                                   # classical DM-G cap [veh/h]
ANCHOR = dict(w_s=0.6 * ev4.W, s_impermeable=True)  # no q_xi_max -> no cap
CLASSICAL = dict(q_xi_max=QXI_VEHH / 3600.0)        # kappa = 0, nothing else
FORM = "lf"

A_SWEEP, U_FIT, Q_FIT = 3.0, 15.0, 2500.0
U_SWEEP = (10.0, 12.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0)
ES_BAND = 0.10                                      # +-10 % reference band
WAKE_T0 = 650.0                                     # [s] wake-flow window start
KMH = 3.6


def _cfg_json(cfg):
    return {k: (v if isinstance(v, bool) else float(v)) for k, v in cfg.items()}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_data(A, uc, qin):
    """dict(tt, rho_mean [veh/km], q_xq (n_rep, n_t) [veh/h], n_reps).

    A in {1, 10}: ev4_compare.load_measured (out/e1 npz + .mat flows);
    other A (no E1 npz): densities and flows straight from the .mat."""
    if float(A) in (1.0, 10.0):
        m = ev4.load_measured(float(A), float(uc), float(qin))
        return dict(tt=np.asarray(m["tt"], float),
                    rho_mean=np.mean(m["rho"], axis=0),
                    q_xq=np.asarray(m["q_xq"], float), n_reps=len(m["reps"]),
                    source="ev4_compare.load_measured (out/e1 npz)")
    a_tag = (f"{A:g}" if float(A).is_integer()
             else f"{int(round(float(A) * 100))}")
    path = SECOND / f"data_{a_tag}_{uc:g}_{qin:g}_True.mat"
    sc = load_scenario(path, fields=True, ctrl=False, trajs=False)
    reps = sorted(r for r in sc.reps
                  if sc.reps[r].rho is not None and sc.reps[r].q is not None)
    if not reps:
        raise FileNotFoundError(f"no per-rep density+flow fields in {path}")
    rho = np.asarray([sc.reps[r].rho for r in reps], float)
    q_xq = np.asarray([sc.reps[r].q[:, ev4.J_XQ] for r in reps], float)
    return dict(tt=sc.t_field, rho_mean=rho.mean(axis=0), q_xq=q_xq,
                n_reps=len(reps), source="loader.load_scenario (.mat)")


# ---------------------------------------------------------------------------
# simulation + scoring
# ---------------------------------------------------------------------------

def run_model(uc, qin, kc, kr, extra_cfg):
    """Production-resolution run -> ev4_compare.regrid_sim dict."""
    return e7w.run_sim(uc, qin, kc, kr, FORM, dt=e7w.DT_PRODUCTION,
                       **extra_cfg)


def evaluate(regr, uc, data):
    """W1 / rho-RMSE vs the rep-mean field, e_s at X_q per rep (mean, min,
    max) and vs the rep-mean flow; windows as in e7_wasserstein / ev4."""
    tt = np.asarray(regr["tt"], float)
    assert np.allclose(tt, data["tt"]), "sim/data time grids differ"
    w1 = e7w.w1_mean(regr["rho_tot"], data["rho_mean"], tt)
    rmse = e7w.rmse_mean(regr["rho_tot"], data["rho_mean"], tt)
    t1 = ev4.t_reach(ev4.X_Q, uc)
    we = (tt >= ev4.T0_ES) & (tt <= t1)
    num = float(ev4.flux_sim_xq(regr, uc)[we].sum())
    per = []
    for q in data["q_xq"]:
        den = float(np.asarray(q, float)[we].sum())
        per.append(num / den - 1.0 if den > 0 else np.nan)
    per = np.asarray(per, float)
    den_mean = float(np.mean(data["q_xq"], axis=0)[we].sum())
    # through-flow past the bottleneck: mean flow at X_q while the CAV is
    # upstream of X_q and slow (wake state), t in [650 s, min(t1, 750) - 10]
    hi = min(t1, ev4.T_FAST) - 10.0
    ww = (tt >= WAKE_T0) & (tt <= hi)
    q_sim = ev4.flux_sim_xq(regr, uc)
    return dict(w1_vehkm=float(w1), rho_rmse_vehkm=float(rmse),
                e_s=float(np.nanmean(per)), e_s_min=float(np.nanmin(per)),
                e_s_max=float(np.nanmax(per)),
                e_s_per_rep=[float(v) for v in per],
                e_s_vs_repmean=(num / den_mean - 1.0 if den_mean > 0
                                else float("nan")),
                es_window_s=[float(ev4.T0_ES), float(t1)],
                wake_flow_Xq_sim_vehh=float(q_sim[ww].mean()),
                wake_flow_Xq_data_vehh=float(
                    np.mean(data["q_xq"], axis=0)[ww].mean()),
                wake_window_s=[float(WAKE_T0), float(hi)],
                peak_omega_vehh=float(np.max(regr["omega"])),
                n_reps=int(data["n_reps"]))


def score_pair(A, uc, qin, kc, kr, model_cfg=ANCHOR):
    """(model, classical) evaluate dicts for one scenario."""
    data = load_data(A, uc, qin)
    model = evaluate(run_model(uc, qin, kc, kr, model_cfg), uc, data)
    classical = evaluate(run_model(uc, qin, 0.0, 0.0, CLASSICAL), uc, data)
    return model, classical, data["source"]


# ---------------------------------------------------------------------------
# the A=3 u15 q2500 anchor fit (final no-cap structure)
# ---------------------------------------------------------------------------

def get_fit(payload, refit=False):
    """A=3 u15 q2500 lf/W1 fit with the anchor structure fixed via
    extra_cfg; cached in transfer.json['fit'] and reused unless refit=True
    or the cached fit was made with a different structure."""
    fit = payload.get("fit")
    if fit and not refit and fit.get("extra_cfg") == _cfg_json(ANCHOR):
        print(f"[fit] reusing cached A=3 anchor: kappa_c={fit['kappa_c']:.3e} "
              f"kappa_r={fit['kappa_r']:.3e} (W1 {fit['objective']:.2f} "
              f"at dt_fit; {fit['at_production']['w1']:.2f} at dt=0.5)")
        return fit
    print(f"[fit] A={A_SWEEP:g} u{U_FIT:g} q{Q_FIT:g} {FORM}/W1 field fit, "
          f"anchor structure {ANCHOR} (no capacity cap) ...")
    return e7w.fit_field(A_SWEEP, U_FIT, Q_FIT, form=FORM, metric="w1",
                         extra_cfg=ANCHOR)


# ---------------------------------------------------------------------------
# headline
# ---------------------------------------------------------------------------

def headline(sweep):
    us = sorted(sweep)
    r = {u: sweep[u]["model"]["w1_vehkm"] / sweep[u]["classical"]["w1_vehkm"]
         for u in us}
    g = {u: sweep[u]["model"]["w1_vehkm"] / sweep[U_FIT]["model"]["w1_vehkm"]
         for u in us}
    es_m = {u: sweep[u]["model"]["e_s"] for u in us}
    es_c = {u: sweep[u]["classical"]["e_s"] for u in us}
    in_band_m = [u for u in us if abs(es_m[u]) <= ES_BAND]
    in_band_c = [u for u in us if abs(es_c[u]) <= ES_BAND]
    worst = max(r, key=lambda u: r[u])
    if max(r.values()) <= 1.0:
        verdict = "model below classical W1 at every tested speed"
    elif max(r.values()) <= 1.15:
        verdict = "model within 15% of classical W1 at every tested speed"
    else:
        verdict = f"model exceeds classical W1 by {r[worst] - 1:.0%} at u={worst:g}"
    wake = {f"u{u:g}": dict(data=round(sweep[u]["model"]["wake_flow_Xq_data_vehh"], 1),
                           model=round(sweep[u]["model"]["wake_flow_Xq_sim_vehh"], 1),
                           classical=round(sweep[u]["classical"]["wake_flow_Xq_sim_vehh"], 1))
            for u in us}
    sigma_xi = 0.5 * ev4.W * ev4.P / (ev4.V_F + ev4.W) * 1000.0   # [veh/km]
    om_max = {f"u{u:g}": round(max(QXI_VEHH - u * KMH * sigma_xi, 0.0), 1)
              for u in us}
    d_m = np.array([sweep[u]["model"]["wake_flow_Xq_data_vehh"] for u in us])
    q_m = np.array([sweep[u]["model"]["wake_flow_Xq_sim_vehh"] for u in us])
    q_c = np.array([sweep[u]["classical"]["wake_flow_Xq_sim_vehh"] for u in us])
    u20 = 20.0 in sweep
    return dict(
        verdict=verdict,
        note=("the classical model's through-flow past the bottleneck is set "
              "by the DM-G cap omega_max = (Q_xi - u_xi sigma_xi)_+ with "
              f"sigma_xi = beta w P/(v_f + w) = {sigma_xi:.2f} veh/km "
              "(beta = 0.5), which decays linearly in u_xi "
              "(omega_max_classical_vehh below), whereas the SUMO "
              f"through-flow at X_q stays at {d_m.min():.0f}-{d_m.max():.0f} "
              "veh/h at every speed; catch & release without the cap has "
              "no such bound -- its through-flow is set by the "
              "s-impermeable interface plus capture "
              f"({q_m.min():.0f}-{q_m.max():.0f} veh/h over the sweep vs "
              f"{q_c.min():.0f}-{q_c.max():.0f} veh/h for the classical "
              "model)"),
        wake_flow_Xq_vehh=wake,
        omega_max_classical_vehh=om_max,
        w1_ratio_model_over_classical={f"u{u:g}": round(r[u], 3) for u in us},
        w1_growth_vs_fit_speed={f"u{u:g}": round(g[u], 3) for u in us},
        w1_ratio_max_u=float(worst),
        w1_ratio_mean=float(np.mean(list(r.values()))),
        w1_ratio_min=float(min(r.values())),
        w1_ratio_max=float(max(r.values())),
        e_s_model={f"u{u:g}": round(es_m[u], 4) for u in us},
        e_s_classical={f"u{u:g}": round(es_c[u], 4) for u in us},
        e_s_max_abs_model=float(max(abs(v) for v in es_m.values())),
        e_s_max_abs_classical=float(max(abs(v) for v in es_c.values())),
        e_s_within_10pct_model=[float(u) for u in in_band_m],
        e_s_within_10pct_classical=[float(u) for u in in_band_c],
        u20_dip=(None if not u20 else dict(
            e_s_model=round(es_m[20.0], 4), e_s_classical=round(es_c[20.0], 4),
            model_within_10pct=bool(abs(es_m[20.0]) <= ES_BAND),
            classical_within_10pct=bool(abs(es_c[20.0]) <= ES_BAND))))


# ---------------------------------------------------------------------------
# fig_transfer
# ---------------------------------------------------------------------------

def fig_transfer(sweep):
    us = sorted(sweep)
    uk = np.array(us) * KMH                        # [km/h]
    M = {k: np.array([sweep[u]["model"][k] for u in us])
         for k in ("w1_vehkm", "e_s", "e_s_min", "e_s_max")}
    C = {k: np.array([sweep[u]["classical"][k] for u in us])
         for k in ("w1_vehkm", "e_s", "e_s_min", "e_s_max")}

    fig, (ax1, ax2) = pf.figure("single", height=3.9, nrows=2, ncols=1,
                                sharex=True)
    kw_m = dict(color=pf.COL["model"], ls=pf.LS["model"], marker="o", ms=3.2)
    kw_c = dict(color=pf.COL["classical"], ls=pf.LS["classical"], marker="s",
                ms=3.0, mfc="white")

    # -- W1 -------------------------------------------------------------------
    ax1.plot(uk, C["w1_vehkm"], label="classical LWR+MB", zorder=2, **kw_c)
    ax1.plot(uk, M["w1_vehkm"], label="catch & release", zorder=3, **kw_m)
    ax1.set_ylabel(r"$W_1$ [veh km]")
    w1_lo = min(M["w1_vehkm"].min(), C["w1_vehkm"].min())
    w1_hi = max(M["w1_vehkm"].max(), C["w1_vehkm"].max())
    ax1.set_ylim(10.0 * np.floor(0.9 * w1_lo / 10.0),
                 10.0 * np.ceil(1.2 * w1_hi / 10.0))     # legend headroom

    # -- e_s ------------------------------------------------------------------
    ax2.axhspan(-ES_BAND, ES_BAND, color="0.5", alpha=0.15, lw=0,
                label=r"$\pm10\,\%$", zorder=0)
    ax2.axhline(0.0, color="k", lw=0.6, zorder=1)
    ax2.errorbar(uk, C["e_s"], yerr=[C["e_s"] - C["e_s_min"],
                                      C["e_s_max"] - C["e_s"]],
                 elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=2, **kw_c)
    ax2.errorbar(uk, M["e_s"], yerr=[M["e_s"] - M["e_s_min"],
                                      M["e_s_max"] - M["e_s"]],
                 elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=3, **kw_m)
    ax2.set_ylabel(r"$e_s$ at $X_q$ = 15 km")
    ax2.set_xlabel(r"bottleneck speed $u_\xi$ [km/h]")
    lo = min(M["e_s_min"].min(), C["e_s_min"].min(), -ES_BAND)
    hi = max(M["e_s_max"].max(), C["e_s_max"].max(), ES_BAND)
    pad = 0.08 * (hi - lo)
    ax2.set_ylim(lo - pad, hi + 2.0 * pad)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    # -- calibration speed ----------------------------------------------------
    for ax in (ax1, ax2):
        ax.axvline(U_FIT * KMH, color="0.4", ls=":", lw=0.9, zorder=1)
    ax1.text(U_FIT * KMH + 1.0, 0.04, r"calibration ($u_\xi$ = 54 km/h)",
             fontsize=7, color="0.3", ha="left", va="bottom",
             transform=ax1.get_xaxis_transform())

    ax1.legend(loc="upper right", ncol=1, handlelength=2.2,
               handletextpad=0.5, borderaxespad=0.4)
    ax2.legend(loc="upper left", ncol=1, handlelength=2.2)
    ax1.set_xticks([40, 50, 60, 70, 80])
    ax1.set_xlim(uk[0] - 3.0, uk[-1] + 3.0)
    return pf.save(fig, "fig_transfer")


# ---------------------------------------------------------------------------
# fig_fd
# ---------------------------------------------------------------------------

def fig_fd(cloud_sub=6, rho_cong=60.0):
    """Calibrated triangular FD over the A=3 field cloud + steady states."""
    import ev2_calibrate as ev2
    prm = json.loads(PARAMS_JSON.read_text())
    v_f, w, P = prm["v_f_kmh"], prm["w_kmh"], prm["P_vehkm"]
    rho_c, cap = fd.crit_density(v_f, w, P), fd.capacity(v_f, w, P)

    rho_cl, q_cl, pts = ev2.gather()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(rho_cl), size=len(rho_cl) // cloud_sub, replace=False)
    up = np.isfinite(pts[:, 3]) & np.isfinite(pts[:, 4])
    dn = np.isfinite(pts[:, 5]) & np.isfinite(pts[:, 6])
    cong = up & (pts[:, 3] >= rho_cong)

    fig, ax = pf.figure("single", height=2.5)
    ax.scatter(rho_cl[idx], q_cl[idx], s=1.2, c="0.78", lw=0, rasterized=True,
               label="field cells (A = 3)", zorder=1)
    ax.scatter(pts[up & ~cong, 3], pts[up & ~cong, 4], s=12, marker="s",
               facecolor="white", edgecolor=pf.COL["stuck"], lw=0.7,
               label=r"upstream $(\rho_-, q_-)$, mixed", zorder=3)
    ax.scatter(pts[cong, 3], pts[cong, 4], s=12, marker="s",
               facecolor=pf.COL["stuck"], edgecolor="none", alpha=0.9,
               label="upstream, queued", zorder=4)
    ax.scatter(pts[dn, 5], pts[dn, 6], s=11, marker="o",
               facecolor=pf.COL["free"], edgecolor="none", alpha=0.9,
               label=r"downstream $(\rho_+, q_+)$", zorder=5)
    rr = np.linspace(0.0, P, 400)
    ax.plot(rr, fd.q_tri(rr, v_f, w, P), color=pf.COL["data"], lw=1.4,
            label=r"$Q(\rho)$", zorder=6)
    ax.axhline(QXI_VEHH, color="0.35", ls="-.", lw=0.8, zorder=2)
    ax.text(0.985 * P, QXI_VEHH + 0.02 * cap, r"$Q_\xi$",
            fontsize=7, color="0.25", ha="right", va="bottom")
    ax.axvline(rho_c, color="0.35", ls=":", lw=0.8, zorder=2)
    ax.text(rho_c + 3.0, 0.03, rf"$\rho_c$ = {rho_c:.1f} veh/km", rotation=90,
            fontsize=7, color="0.25", ha="left", va="bottom",
            transform=ax.get_xaxis_transform())
    ax.set_xlim(0.0, P)
    ax.set_ylim(0.0, 1.12 * cap)
    ax.set_xlabel(r"density $\rho$ [veh/km]")
    ax.set_ylabel(r"flow $q$ [veh/h]")
    # legend: FD line first so the bottom row (a marker) sits clear of the
    # congested branch that runs under the legend's lower-left corner
    hs, ls = ax.get_legend_handles_labels()
    order = [ls.index(l) for l in (r"$Q(\rho)$", "field cells (A = 3)",
                                   r"upstream $(\rho_-, q_-)$, mixed",
                                   "upstream, queued",
                                   r"downstream $(\rho_+, q_+)$")]
    ax.legend([hs[i] for i in order], [ls[i] for i in order],
              loc="upper right", fontsize=6.8, handlelength=1.6,
              borderaxespad=0.3, labelspacing=0.3, markerscale=1.4)
    path = pf.save(fig, "fig_fd")
    return path, dict(
        v_f_kmh=v_f, w_kmh=w, P_vehkm=P, rho_crit_vehkm=float(rho_c),
        capacity_vehh=float(cap), Q_xi_vehh=QXI_VEHH,
        n_cloud_total=int(len(rho_cl)), n_cloud_plotted=int(len(idx)),
        n_steady_upstream=int(up.sum()), n_steady_upstream_queued=int(cong.sum()),
        n_steady_downstream=int(dn.sum()), rho_queued_threshold=rho_cong,
        cloud_source="ev2_calibrate.gather (A=3 True files, rho > 0.5, "
                     "every 7th cell; a further random 1/%d plotted)" % cloud_sub,
        rho_minus_range=[float(np.nanmin(pts[up, 3])), float(np.nanmax(pts[up, 3]))],
        rho_plus_range=[float(np.nanmin(pts[dn, 5])), float(np.nanmax(pts[dn, 5]))])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def archive_capped(payload):
    """Move a capped-structure (E10) result set, if present at the top
    level, to payload['capped_reference'] (once)."""
    fit = payload.get("fit") or {}
    if "capped_reference" in payload or "q_xi_max" not in fit.get("extra_cfg", {}):
        return payload
    ref = dict(note=("previous structure (E10): DM-G cap Q_xi = 2000 veh/h + "
                     "w_s = 0.6 w + s_impermeable + catch & release (lf); "
                     "A = 3 anchor fit and u-sweep, plus the A = 1 / A = 10 "
                     "u20 transfer points with the E10 kappas; kept for "
                     "reference, not plotted"),
               _meta=payload.get("_meta"))
    for k in ("fit", "A3_sweep_q2500", "A1_A10_transfer", "headline"):
        if k in payload:
            ref[k] = payload.pop(k)
    payload["capped_reference"] = ref
    print("[archive] moved the capped-structure results to "
          "transfer.json['capped_reference']")
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--skip-transfer", action="store_true")
    ap.add_argument("--skip-fd", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    pf.setup()

    payload = (json.loads(TRANSFER_JSON.read_text()) if TRANSFER_JSON.exists()
               else {})
    payload = archive_capped(payload)
    payload["_meta"] = dict(
        script="e11_transfer_fd.py",
        structure=("A = 3 anchor, final (no-cap) structure: stuck-class "
                   "branch w_s=0.6w + s_impermeable + catch & release (lf, "
                   "ell = a + s); NO DM-G capacity cap; kappas W1-fitted at "
                   "u15 q2500 (54 km/h, 2500 veh/h) and frozen for the sweep"),
        classical="M1: kappa_c=kappa_r=0, DM-G cap 2000 veh/h, no w_s, no "
                  "s_impermeable (no fitted parameter)",
        anchor_cfg=_cfg_json(ANCHOR),
        classical_cfg=_cfg_json(CLASSICAL),
        resolution=dict(dt_s=e7w.DT_PRODUCTION, save_every=20, dx_m=50.0),
        windows=dict(w1_rmse_t_s=list(e7w.T_WIN), w1_rmse_x_km=[0.0, 20.0],
                     es_t0_s=ev4.T0_ES, es_Xq_m=ev4.X_Q, es_cell=ev4.J_XQ,
                     wake_t0_s=WAKE_T0),
        fd_params_si=dict(v_f_ms=ev4.V_F, w_ms=ev4.W, P_vehm=ev4.P),
        u_sweep_ms=list(U_SWEEP), u_sweep_kmh=[u * KMH for u in U_SWEEP])

    if not args.skip_transfer:
        fit = get_fit(payload, refit=args.refit)
        kc, kr = fit["kappa_c"], fit["kappa_r"]
        payload["fit"] = fit

        print(f"\nA={A_SWEEP:g} q={Q_FIT:g} zero-refit sweep "
              f"(model kc={kc:.3e} kr={kr:.3e}, no cap | classical kappa=0, "
              f"cap {QXI_VEHH:.0f})")
        print(f"{'u m/s':>6s} {'km/h':>5s} {'W1_mod':>8s} {'W1_cl':>8s} "
              f"{'RMSE_mod':>9s} {'RMSE_cl':>8s} {'es_mod':>8s} {'es_cl':>8s} "
              f"{'Xq SUMO':>8s} {'Xq mod':>7s} {'Xq cl':>7s}")
        sweep = {}
        for u in U_SWEEP:
            m, c, src = score_pair(A_SWEEP, u, Q_FIT, kc, kr)
            sweep[u] = dict(model=m, classical=c, data_source=src,
                            role="fit anchor" if u == U_FIT else "zero-refit")
            print(f"{u:6g} {u * KMH:5.1f} {m['w1_vehkm']:8.2f} {c['w1_vehkm']:8.2f} "
                  f"{m['rho_rmse_vehkm']:9.2f} {c['rho_rmse_vehkm']:8.2f} "
                  f"{m['e_s']:+8.1%} {c['e_s']:+8.1%} "
                  f"{m['wake_flow_Xq_data_vehh']:8.0f} "
                  f"{m['wake_flow_Xq_sim_vehh']:7.0f} "
                  f"{c['wake_flow_Xq_sim_vehh']:7.0f}")
        payload["A3_sweep_q2500"] = {f"u{u:g}": sweep[u] for u in U_SWEEP}
        payload["headline"] = headline(sweep)
        print("\nheadline:", payload["headline"]["verdict"])
        d = payload["headline"]["u20_dip"]
        if d:
            print(f"u_xi = 20 m/s (72 km/h): e_s model {d['e_s_model']:+.1%} "
                  f"(within 10 %: {d['model_within_10pct']}), classical "
                  f"{d['e_s_classical']:+.1%} (within 10 %: "
                  f"{d['classical_within_10pct']})")
        print("wrote", fig_transfer(sweep))

    if not args.skip_fd:
        path, fdinfo = fig_fd()
        payload["fd_figure"] = fdinfo
        print("wrote", path)

    TRANSFER_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print("wrote", TRANSFER_JSON)


if __name__ == "__main__":
    main()
