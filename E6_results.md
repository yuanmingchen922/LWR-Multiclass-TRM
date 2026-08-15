# E6 Results — Catch & Release as a *Native* Moving-Bottleneck Model

Date: 2026-08-15 · Responds to Mladen's feedback points 3–6.
Focus: u_ξ = 15 m/s (per point 2); stop-and-go line on ice (point 1).

## Question

Can the catch & release dynamics alone — no Delle Monache–Goatin flux
constraint — reproduce the moving-bottleneck state evolution, when
(κ_c, κ_r) is calibrated against the observed density field instead of the
microscopic events? If yes, catch & release is a **new way to model moving
bottlenecks**, and it can natively represent the low-assertiveness split into
a stuck stream and a free stream (a state whose aggregate (ρ, q) lies inside
the flux function — inexpressible in scalar LWR).

## Setup (all models get the same information: inflow, empty road, CAV schedule)

| Model | Definition |
|---|---|
| **M1 — LWR + MB** | κ = 0, DM-G capacity cap (Q_ξ = 2000 veh/h). The canonical scalar-LWR moving-bottleneck baseline (ECC22-equivalent prediction). |
| **M2 — CR native** | Catch & release only, **no cap**. (κ_c, κ_r) fitted per assertiveness by matching the rep-mean density field of `u15_q2500` (coarse log-grid + Nelder–Mead), then **transferred unchanged** to `u15_q2000`. |
| **M3 — CR capped** | E-V4b reference: event-MLE κ + cap. |

Fitted values (lf capture form wins for both A; the strict a·f form is
pathological under field fitting — at A=1 it runs away to κ_c ≈ 8·10³ because
the point-mass covariate is nearly degenerate):

| | κ_c | κ_r | fit ρ-RMSE |
|---|---|---|---|
| A=1 (lf) | 0.504 | 3.06×10⁻² | 5.50 veh/km |
| A=10 (lf) | 0.164 | 1.29×10⁻² | 5.60 veh/km |

## Results (u15; q2000 columns are zero-refit transfer)

| scenario | model | ρ-RMSE | N_s-MAE | ω err | e_s |
|---|---|---|---|---|---|
| A1 u15 q2500 | M1 LWR+MB | 7.12 | (40.2) | −2.5% | +5.8% |
| | **M2 CR native** | **5.83** | 15.3 | −33.3% | **−0.5%** |
| | M3 CR capped | 7.20 | 34.2 | −3.0% | +5.6% |
| A1 u15 q2000 ⁽ᵗ⁾ | M1 | 4.52 | (15.9) | −6.2% | +3.5% |
| | **M2** | **4.19** | 11.5 | −28.0% | −4.3% |
| | M3 | 4.57 | 11.0 | −6.2% | +3.2% |
| A10 u15 q2500 | M1 | 7.18 | 0.60 | −18.3% | −5.6% |
| | **M2** | **6.33** | 23.5 | −29.1% | −4.8% |
| | M3 | 7.25 | 0.62 | −18.3% | −6.7% |
| A10 u15 q2000 ⁽ᵗ⁾ | M1 | 5.08 | 0.02 | −20.3% | −7.4% |
| | **M2** | **3.32** | 8.4 | −12.6% | +3.1% |
| | M3 | 5.09 | 0.07 | −20.3% | −7.5% |

⁽ᵗ⁾ transfer scenario (κ fitted at q2500 only). N_s-MAE in parentheses is
meaningless for M1 (it has no s class).

## What this says

1. **The native model beats the classical baseline on the state evolution.**
   Density-field RMSE is lower than LWR+MB in **all four** u15 scenarios,
   including both zero-refit transfers, and |e_s| ≤ 4.8% everywhere (M1:
   up to 7.4%). Mladen's point-4 claim is supported: catch & release,
   properly calibrated, *is* a moving-bottleneck model on its own.
2. **The mechanism is the two-stream state (point 3).** In the M2 queue the
   caught stream (s ≈ 30 veh/km at u_s) and the free stream (f ≈ 14 veh/km at
   v(ρ)) coexist; the aggregate lies inside the flux function — exactly the
   low-assertiveness regime scalar LWR cannot express. See
   `fig_profiles_A1_u15_q2500.png`: SUMO's queue plateau (~58 veh/km) sits
   between M1's over-dense narrow wedge (~69) and M2's wider mixed wedge
   (~44 total).
3. **Honest costs.** (a) M2 underpredicts the overtaking flow (−13…−33%):
   capture removes vehicles instead of letting a capacity gate meter them.
   (b) At A=10 the field-fitted s is a *modeling device* (a slow stream near
   the bottleneck), no longer the literal caught population (micro count ≈ 0,
   N_s-MAE 8–24). The event-based κ of E-V3 remains the interpretable
   calibration; the field-based κ is the predictive one. Both belong in the
   paper with this distinction stated.
4. **Refinement candidates** (post-paper or reviewer-response): queue-weighted
   fitting objective (the plain field RMSE lets M2 trade plateau density for
   extent); arrival-flux capture law (three independent diagnostics from
   E1/E-V4/E-V4b still point to it); joint (κ_c, κ_r, ω) objective if the
   overtaking-flow gap matters for control applications.

## Files

`e6_native_mb.py` · `out/e6/fits.json` · `out/e6/metrics_e6.json` (incl. u20 for
completeness) · `out/e6/fig_heat3_*.png` (data vs M1 vs M2 heatmaps) ·
`out/e6/fig_profiles_*.png` (profiles at 300/500/700 s with the f/s split)
