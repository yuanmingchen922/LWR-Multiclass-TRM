# E8 Results — Downstream Release, the Plug Discovery, and the Final Hybrid

Date: 2026-08-19 · Responds to Mladen's third review (downstream stuck vehicles /
free-flow waviness, analytical / A=10 wrong + wedge / one-at-a-time ablation).
3-agent workflow + main-loop follow-up; solver suite 16/16, independent audit PASS.

## 1. The fix he asked for — and what it revealed

**Downstream-release constraint (DR)**: a vehicle cannot be caught by a
bottleneck that is behind it. Implemented as a zero-parameter definitional
constraint (`SimConfig.downstream_release`): every cell strictly downstream of
the CAV converts s→f after each reaction substep. Conservation exact;
bit-identical when off (t14); downstream s ≡ 0.0 exactly when on (t15).

**One-at-a-time ladder** (his request; A=1, u15 q2500, each rung refit):

| rung | W1 | wake [veh/km] | downstream stuck | rarefaction | ω err |
|---|---|---|---|---|---|
| L0 RMSE (E6) | 174.7 | 47.1 sub | 3.8% | +500 m | −32% |
| L1 +W1 metric | 162.6 | 53.9 super | 3.1% | +200 m | −23% |
| L2 +w_s=0.6w | 149.9 | — | 0.7% | −700 m | — |
| L3 +DR (refit) | 147.0 | 40.6 sub | **0 exactly** | −500 m | −7.5% |
| L4 DR, no w_s | 154.3 | — | 0 | — | — |

**The plug discovery**: evaluating the E7 winner's κ *with* DR (no refit)
collapsed everything — W1 147→228, wake 55.5→34.5, ω −9%→**+42%**. The stuck
vehicles that leaked downstream had been acting as a **plug** that throttled
the bottleneck: the uncapped model can only restrict passing flow by parking
mass at/past the CAV, which is exactly the unphysical behavior Mladen flagged.
Conclusion: **downstream cleanliness and bottleneck throttling cannot both come
from catch & release alone** — the throttling must come from the explicit
capacity term. This resolves the E6 "native model" question precisely:

> capacity cap = how much gets past · catch & release = who and how fast ·
> downstream release = consistency.

## 2. Final hybrid configuration (cap 2000 veh/h + w_s = 0.6w + DR, W1-fit κ)

| | κ_c | κ_r | W1 | wake (data) | ds-stuck | waviness | rarefaction | e_s | ω err |
|---|---|---|---|---|---|---|---|---|---|
| A=1 | 5.3e-2 | 1.7e-3 | 152.2 | **53.8 super** (58.6) | **0** | 1.37 | +300 m | +4.7% | **−2.5%** |
| A=10 | 1.66e-1 | 1.40e-2 | 167.4 | 47.7 (55.4) | **0** | 1.53 | −300 m | **+0.2%** | −18% |

- A=1: every complaint fixed simultaneously — supercritical wake hugging the
  data plateau (t=700), clean downstream, no waviness, post-release rarefaction
  fan tracking the data (t=850), best-ever ω. W1 only 3.5% above the leaky
  optimum (the price of physicality).
- A=10: κ_r = 1.40e-2 lands on the **event-measured value** (E-V3: 1.56e-2) —
  the field fit and the microscopic calibration now agree. e_s essentially
  exact. Remaining gaps, honestly: wake 47.7 vs 55.4 and s-layer 2.2 km vs the
  thin layer Mladen sketched (the geometry-vs-turnover limit from E7 stands).
- w_s ablation under the cap: still needed (A=1 W1 152→165 and wake 53.8→41.8
  without it) — the cap throttles, w_s densifies; different jobs.
- Zero-refit q2000 transfers: A=1 W1=146.7, e_s +1.8%; A=10 W1=113.2,
  e_s −2.2%; downstream clean in both.

## 3. The analytical piece he asked for (free-flow waviness)

From the E-V5 dispersion relation (symbolically verified): in free flow a mixed
(f, s) state has the exact, k-independent source growth rate

  λ = Δv · (κ_c ρ − κ_r (P − ρ))   (lf form)

**No-waviness condition: κ_c ρ ≤ κ_r (P − ρ).** The E7 winner violated it
downstream (predicted e-fold 180 s in the frozen-f form vs 199 s measured in
the simulation — ratio 1.10, verified in `out/e8/ladder.json`). Under DR the
downstream state has s ≡ 0, so the source vanishes **identically**: free flow
is exactly neutral scalar LWR and waviness is impossible there, while emergent
waves remain possible inside the congestion (where they would be interesting) —
precisely the separation he asked for. Verified at production resolution:
max |s| downstream = 0.0.

## 4. The A=10 wedge artifact — explained

The intermediate-density wedge (x≈13–15 km) was downstream s travelling at
u_s = 15 m/s inside a v_f = 27.9 m/s free stream — a two-speed mixture band,
not a concave-flux effect (his LWR intuition was the right instinct: the
s-class *was* dynamically a second branch down there). DR removes it; the
wedge indicator is zero in the final config.

## 5. Files

Solver: `downstream_release` flag (tests t14–t16; audit_e8.py PASS, incl.
git-reconstructed bit-identity and CAV-cell boundary semantics).
`e8_ladder.py` → `out/e8/ladder.json` (full one-at-a-time table + analytic
checks), `e8_final.py` → `out/e8/final_config.json`,
`fig_profiles_final_*.png`, `fig_heat3_final_*.png`;
candidate evaluations in `out/e8/final_candidates.json`, `hybrid_fit.json`.
