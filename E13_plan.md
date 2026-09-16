# E13 Plan — Joint calibration (density field + queue size + overtaking flow) on the pure E7 model

Date: 2026-09-15 · Follows E12 (`E12_results.md` §6). Same model, same
parameters, same A = 1…10; only the **scoring rule** changes.

## 1. Why

E12 showed that with the density-field objective W1 alone the pure E7 model
(i) beats classical LWR+MB everywhere but (ii) at A ≥ 4 does so with the
wrong mechanism (a 3 km dilute stuck band, 25–50 phantom stuck vehicles, 40 %
overtaking deficit), and (iii) cannot decide whether A acts through κ_c or
κ_r, because for A ≥ 2 the fit sits on the fast-equilibrium ridge where only
κ_c/κ_r is identified. The queue size N_s(t) and the overtaking flow ω(t) are
measured per run and are exactly the observables that distinguish "many
stuck, slow turnover" from "few stuck, fast turnover" at the same density
field. E13 puts them into the objective.

**Nothing in the model changes**: equations, states, parameters (κ_c, κ_r;
optional γ, w_s) and the purity guard are those of E12. N_s and ω are read
off the solution (they already are in the E12 tables); E13 only lets them
enter the score.

## 2. Objective

For one A on the fit scenario (u_ξ = 15 m/s, q_in = 2500 veh/h):

    J_λ(θ) = W1(θ) / W1_cl(A)  +  λ · [ MAE_N(θ) / N_0  +  |ε_ω(θ)| ]

- W1: as in E7/E12 (cumulative-density Wasserstein vs the 5-run mean field,
  t ∈ [100, 1000] s, x ≤ 20 km); normalised by the classical model's W1 for
  the same A so that the field term is O(1) and comparable across A.
- MAE_N: mean over the 5 runs of the mean absolute error of the queue size
  N_s(t) vs the classified stuck count of that run, slow window [260, 740] s
  (`ev4_compare.metrics` → `Ns_mae`). N_0 = 10 veh (the data's own
  run-to-run spread of the queue is 2–6 veh at A ≥ 3, ~10 at A = 1).
- ε_ω: relative error of the cumulative overtaking count over the slow window
  (`omega_cum_rel_err`, mean over runs).
- λ ≥ 0: a **weight, not a model parameter**. λ = 0 reproduces E12.
  Sweep λ ∈ {0, 0.02, 0.05, 0.1, 0.2, 0.5, 1}; the two penalty terms share
  λ (both are O(1) relative errors once scaled). Ablations at the adopted λ:
  queue-only (drop ε_ω) and overtaking-only (drop MAE_N) to see which
  observable breaks the degeneracy.

Choice of λ* (reported, not hidden): the knee of the summed-over-A trade-off
curve (Σ W1/W1_cl vs Σ penalty), i.e. the λ with the largest distance from
the chord joining the λ = 0 and λ = max end points after normalising both
axes to [0, 1]; alongside, the threshold view "smallest λ for which at least
8 of 10 A have MAE_N ≤ 10 veh and |ε_ω| ≤ 25 %". If the two disagree the
knee is adopted and the disagreement is stated.

## 3. Experiments

### E13a — per-A joint fits over the λ sweep

For each A, each λ, configurations C1 (κ_c, κ_r), C2 (+γ), C3 (+w_s),
C4 (+γ, +w_s) — γ is kept although it never won in E12, because γ → 0
localises capture at the CAV and is precisely a queue-shrinking knob that a
queue penalty may now reward. Fit = 7×7 log grid (κ_c ∈ [10^-2.5, 10^2],
κ_r ∈ [10^-4, 10^1], wider than E7 because the E12 optima left the E7 grid)
+ Nelder–Mead (40 evaluations) at dt = 1 s, re-evaluated at dt = 0.5 s; all
production metrics (W1, RMSE, MAE_N, ε_ω, e_s, wake, s-layer, turnover) at
q2500 and, zero-refit, at q2000 (full) and u20 (W1 only). Winner per (A, λ)
= minimum J_λ at dt = 0.5. Consistency check: the λ = 0 C1 winner must
coincide with the E12 C1 fit (same objective up to a positive scale).

### E13b — nested shared-parameter test as a function of λ

For every λ: M_none, M_c (κ_c(A) free, κ_r shared), M_r (κ_r(A) free, κ_c
shared), M_both (per-A 2-D polish), objective Σ_A J_λ, with the E12
machinery (parent-optimum fallback ⇒ exact nesting). Report Σ J and the gap
M_c − M_r vs λ: at λ = 0 it is 0.02 % (E12); the question is whether and how
fast it opens. At λ*: robustness with w_s = 0.6 w fixed; the queue-only and
overtaking-only ablations; monotonicity and Spearman of the M_c and M_r
ladders; comparison with the event-based ladder (κ_c flat, κ_r rising).

### E13e — identifiability at λ*

(ratio, magnitude) scan of J_λ* per A (as E12 stage E) — does the
fast-equilibrium ridge become a basin? — and the event-based κ evaluated
under J_λ*.

### E13c — figures (paper style, `out/e13/`)

- `fig_e13_tradeoff`: Σ W1/W1_cl vs Σ penalty over the sweep with the knee;
  per-A small multiples of W1, MAE_N, ε_ω vs λ.
- `fig_e13_gap_vs_lambda`: Σ J of the four nested models vs λ and the
  M_c − M_r gap; the λ* ladders κ_c(A) under M_c, κ_r(A) under M_r, both under
  M_both, event MLE.
- `fig_e13_kappa_vs_A`, `fig_e13_metrics_vs_A` (classical / E12 W1-only /
  E13 joint: W1, MAE_N, ε_ω, wake per A), heatmap grids for every A at λ*
  (q2500 and q2000), `fig_e13_profiles` (A = 1, 5, 10), `fig_e13_ridge`.

## 4. Files

`e13_joint.py` (--stage a|b|e|f, --run, --figures, --summary, --smoke),
`e13_figures.py`, `test_e13.py`, `audit_e13.py`, `out/e13/`,
`E13_results.md`, README step 14.

---

## v2 (2026-09-15, after two independent methodology reviews and the v1 run)

**What v1 found (`out/e13_v1/`)**: with λ·[MAE_N/N_0 + |ε_ω|] the M_c − M_r
gap stayed below 0.4 % at every λ ∈ {0…1}. Both reviewers had predicted it
and one verified it numerically: in the fast-equilibrium regime (turnover
< 10 s, where every E12 optimum for A ≥ 2 sits) the queue size and the
overtaking flow are functions of the ratio κ_c/κ_r only, exactly like the
density field. They constrain the ratio better, not the magnitude. v1 also
showed the λ ≥ 0.2 winners collapsing into a one-cell "plug" (γ → 0,
κ_c = 10²–10³, N_s ≈ 2 veh, ε_ω = −30…−59 %, W1 above classical) and that
the chord-distance knee is fixed by the sweep end points.

**Changes adopted in v2** (reviewer findings 1–6, all MAJOR):

1. **Magnitude-sensitive term**: the gross event counts. The solver now
   returns the exact capture and release integrals per step
   (`solver.reaction_exact(return_gross=True)`, `SimResult.cum_cap/cum_rel`,
   bit-identical update; E12 audit 34/34 unchanged). Term
   ε_R = |log₁₀((R_m+1)/(R_d+1))| + |log₁₀((C_m+1)/(C_d+1))| in dex, data
   counts C_d, R_d from the per-run `cap`, `rel` series (E1/E4), slow window.
   The E12 optima give 10³–10⁵ captures per window against 14–80 in SUMO.
2. **Objective** J = W1/W1_cl + λ[MAE_N/N_0(A) + |ε_ω|/ε_0 + ε_R],
   N_0(A) = max(5 veh, rep-mean peak queue), ε_0 = 0.2; ablations
   queue-only / overtaking-only / events-only at λ*.
3. **Pre-registered structural check (stage p)**: (W1, MAE_N, ε_ω, ε_R) on
   the (ratio, magnitude) grid with γ ∈ {none, 0, 0.3} and w_s = 0.6 w slices
   for every A; mechanism box MAE_N ≤ 5 veh and |ε_ω| ≤ 10 %. Rule: if no
   point of the reachable set at A ≥ 4 is in the box, the pure model cannot
   reproduce the observed mechanism at any κ and E13's deliverable is the
   trade-off frontier, not a corrected mechanism.
4. **Fitting in ridge coordinates** (r, m) = (log₁₀ κ_c/κ_r, log₁₀ κ_r),
   grid 13 × 11, Nelder–Mead (maxfev 100, fatol 1e-6) with continuation in
   λ and multi-start (grid, previous λ, E12 optimum, event MLE), dt = 0.5
   polish before the winner is chosen; per-A path check (W1 non-decreasing,
   penalty non-increasing in λ). Stage B: 0.1-dex pre-scan + bounded Brent
   with the parent fallback.
5. **λ selection**: dominance rule primary (largest λ with ≥ 80 % of A still
   beating the classical field), restricted chord knee secondary (λ < λ_max,
   λ_max = smallest λ with ≥ 50 % of A collapsed), closed-form knee for
   several end points and per-A exchange rates reported; classical row
   J_cl = 1 + λ P_cl and flags (field_ok, collapsed, in_box, events_ok)
   everywhere; sweep {0, .01, .02, .03, .05, .07, .1, .15, .2, .3, .5, 1}.
6. **Significance and interpretation**: leave-one-run-out jackknife of the
   gap (5 refits per λ, SE_jack); η_c, η_r explained-variation fractions;
   ridge residual max_A |log₁₀(κ_c(A)/κ_r^sh) − log₁₀(κ_c^sh/κ_r(A))|
   (< 0.1 dex ⇒ the two ladders are one ratio ladder); magnitude counted as
   identified only if the J_λ* minimum along the best-ratio line has κ_r < 0.1
   with a basin < 1 dex and W1 ≤ W1_cl; micro–macro agreement iff median
   |log₁₀(κ^macro/κ^micro)| ≤ 0.5 dex and Spearman(macro, micro) ≥ 0.6;
   turnover τ = ∫N_s dt / R reported next to the data's Little's-law value.
7. **Resolution check**: winners at λ = 0 and λ* and the classical model at
   dx = 50/25/12.5 m, dt = 0.5/0.25 (the CAV is a density 1/dx, so this is
   a sensitivity report, not a convergence study).
