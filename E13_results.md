# E13 Results — Joint calibration of the pure E7 model: field + queue + overtaking + event counts

Date: 2026-09-15 · Follows E12 (`E12_results.md` §6); plan `E13_plan.md`
(v1 and the v2 section written after two independent methodology reviews).
Driver `e13_joint.py` (+ `e13_figures.py`), tests `test_e13.py` (11),
solver counters `solver.reaction_exact(return_gross=True)`, audit
`audit_e13.py`. Model, parameters and purity guard unchanged from E12; only
the scoring rule changes. All numbers at u_ξ = 54 km/h, fit on q_in = 2500.

## 0. Summary in five lines

1. **Queue size and overtaking flow alone do not separate κ_c from κ_r**
   (v1, `out/e13_v1/`): the M_c − M_r gap stayed < 0.4 % at every λ. In the
   fast-equilibrium regime they are functions of the ratio κ_c/κ_r only,
   like the density field — both reviewers predicted this, one verified it.
2. **The gross event counts do** (v2): with ε_R in the objective the ridge
   residual jumps from 0 to 0.85–1.0 dex for λ ≥ 0.03, the fitted rates
   move from κ_r ≈ 0.4–4 (turnover 0.1 s) to κ_r ≈ 1e-3 (turnover
   300–650 s), and the J_λ* landscape has a sharp valley along constant
   κ_c ≈ 0.032 (the E12 ridge ran along constant κ_c/κ_r); along that
   valley κ_r is pinned to 0–1 dex at the 1 % level (6/10 A within the
   plan's 1-dex rule; A = 3–5 sit exactly at 1 dex).
3. **κ_r(A) is the better single carrier once the events count**, and the
   advantage grows with λ (gap +0.3 % at λ = 0.05, jackknife-significant in
   5/5 folds; +6 % at λ = 1; η_r 0.70–0.75 vs η_c 0.44–0.62) — but at the
   adopted λ* = 0.07 the two are tied within optimizer/jackknife noise, and
   both ladders are nearly flat for A ≥ 2. The strong A-signal is A = 1
   (no release) vs A ≥ 2.
4. **The macroscopic κ_c agrees with the microscopic one** (median 0.07 dex:
   0.034–0.041 vs 0.031–0.052 veh⁻¹); the macroscopic κ_r does not (≈ 1e-3
   flat vs 3e-3 → 1.6e-2 rising, 0.77 dex). Mladen's "κ_c falls with A" is
   present only as a 20 % drift (0.041 → 0.034, Spearman −0.84 under M_both,
   −0.67 under M_c), far inside the ±0.5 dex band; the hypothesis is not
   supported as a mechanism.
5. **Structural limit confirmed by the pre-registered check**: for A ≥ 3
   the pure model can reach the mechanism box (MAE_N ≤ 5 veh, |ε_ω| ≤ 10 %)
   only by giving up the density field (every in-box point has W1 19–37 %
   above classical); only A = 2 reaches both. At λ* the model keeps the
   field (9/10 A better than classical), fixes ω (|ε_ω| ≤ 15 %, mean 5 %)
   and the event counts (7/10 A within 0.5 dex), but the queue stays 3–10×
   the data at A ≥ 3 and the turnover 3–30× too long.

## 1. What was run

- Objective (v2): J_λ = W1/W1_cl(A) + λ [MAE_N/N_0(A) + |ε_ω|/0.2 + ε_R],
  ε_R = |log₁₀((R_m+1)/(R_d+1))| + |log₁₀((C_m+1)/(C_d+1))| in dex, model
  counts from exact solver integrals (bit-identical update; E12 audit 34/34
  re-run), data counts from the per-run `cap`/`rel` series: C_d = 80 (A=1)
  → 14 (A=10), R_d = 0 (A=1), 24 → 13 (A=2…10).
- Stage p: reachable set on a 13 × 5 (ratio, magnitude) grid × 4 slices
  (γ ∈ {–, 0, 0.3}, w_s = 0.6 w) per A = 260 points.
- Stage a: per (A, λ), λ ∈ {0, .01, .02, .03, .05, .07, .1, .15, .2, .3, .5, 1},
  C1–C4 in ridge coordinates (13 × 11 grid, Nelder–Mead 100 with
  continuation in λ and multi-start, dt = 0.5 polish); winner = min J at
  dt = 0.5; flags, classical J, transfers. 15 min per A, 10 in parallel.
- Stage f: λ* by the dominance rule (largest λ with ≥ 8/10 A still beating
  the classical field) = **0.07**; restricted knee 0.05; closed-form knees
  0.029–0.033 (consistent); per-A exchange rates 0.013–0.044 (spread 3.4 →
  two-λ ladder reported: 0.05 and 0.07); λ_max undefined (no λ collapses
  ≥ 5 A under the stricter v2 definition).
- Stage b: M_none / M_c / M_r / M_both at every λ (0.1-dex pre-scan +
  Brent, parent fallback ⇒ exact nesting at every λ), leave-one-run-out
  jackknife at λ = 0, 0.05, 0.07, ablations at λ*, η fractions, ridge
  residual, micro–macro criterion. 20 min.
- Stage e: ridge scan under J_λ* (13 × 11), event-MLE κ under J_λ*,
  turnover, dx/dt sensitivity.
- Optimizer sanity: the winner path in λ shows 1–4 monotonicity violations
  per A (W1 should not decrease, penalty should not increase with λ) —
  Nelder–Mead noise of order 1–3 % in W1; the conclusions below are all
  larger than that.

## 2. Reachable set (stage p, `fig_e13_hull`)

| A | in box | in box & W1 ≤ W1_cl | + ε_R ≤ 0.5 | min MAE_N | min ε_R | best in-box point |
|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 2.8 | 1.41 | – |
| 2 | 4 | **4** | 2 | 2.4 | 0.17 | κ_c 0.032, κ_r 1e-3, W1/W1_cl 0.91 |
| 3–10 | 8–12 | **0** | 0 | 0.6–1.9 | 0.02–0.17 | γ = 0, κ_c 0.32, κ_r 0.01, W1/W1_cl 1.16–1.34 |

The in-box points at A ≥ 3 are "nearly transparent CAV" solutions (γ = 0,
few captures, a 44 m stuck sliver): queue, overtaking and event counts are
right, the wedge is gone. The falsification rule of plan v2 §3 therefore
fires for A ≥ 3: **the pure uncapped model cannot reproduce SUMO's
mechanism (few stuck vehicles, 2000 veh/h passing) and SUMO's density field
(a 55 veh/km wedge) at the same time, at any κ, γ, w_s**. This is E8's plug
diagnosis reached from the other side: in this model the wedge *is* parked
stuck mass, and nothing else holds a congested free-class wedge at the CAV.

## 3. The λ sweep (`fig_e13_tradeoff`)

| λ | Σ W1/W1_cl | Σ penalty | A better than classical (field) | in box | ε_R ≤ 0.5 | mean W1 | mean MAE_N | mean |ε_ω| | mean ε_R | median τ [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 (E12) | 7.75 | 117.1 | 10 | 0 | 0 | 138.7 | 29.6 | 31 % | 6.08 | 0.2 |
| 0.03 | 8.55 | 42.9 | 10 | 1 | 1 | 153.4 | 16.6 | 12 % | 1.58 | 83 |
| 0.05 (knee) | 9.01 | 26.3 | 10 | 1 | 4 | 161.8 | 14.4 | 6 % | 0.49 | 464 |
| **0.07 (λ*)** | 9.10 | 24.1 | 9 | 1 | 7 | 163.5 | 14.3 | 5 % | 0.37 | 549 |
| 0.1 | 9.78 | 16.8 | 7 | 1 | 8 | 176.2 | 11.5 | 3 % | 0.29 | 372 |
| 0.2 | 10.95 | 7.9 | 4 | 5 | 10 | 197.5 | 6.7 | 1 % | 0.15 | 220 |
| 1 | 11.39 | 6.2 | 2 | 6 | 10 | 204.9 | 5.2 | 0 % | 0.10 | 195 |
| classical | 10.00 | 38.4 | – | 0 | 0 | 178.9 | 6.6 | 19 % | 2.6 | – |

Between λ = 0 and 0.07 the field costs 18 % (mean W1 139 → 164, still 9 %
below classical) and buys: overtaking error 31 % → 5 % (classical 19 %),
event-count error 6.1 → 0.37 dex (classical 2.6), queue 30 → 14 veh
(classical 7, data noise 0.5–2). Beyond λ ≈ 0.1 the model pays the field
(W1 above classical for most A) for the last of the queue. The classical
model has a *lower* J than the pure model only for λ ∈ [0.03, 0.1] at 1–2
of the 10 A (`n_beats_classical` 8/10 there, 10/10 elsewhere).

## 4. Per-A winners at λ* = 0.07 (`fig_e13_metrics_vs_A`, profiles, heatmaps)

| A | cfg | κ_c | κ_r | γ | W1 cl / λ=0 / λ* | MAE_N cl / λ=0 / λ* | ε_ω cl / λ=0 / λ* | ε_R λ=0 / λ* | τ λ* (data) | C_m/C_d | R_m/R_d |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | C3 | 3.9e-2 | 1e-5 | – | 174 / 143 / 149 | 40 / 11 / 15 | −2.5 / −13 / −0.3 % | 2.2 / 0.12 | ∞ (∞) | 81/80 | 0/0 |
| 2 | C4 | 7.0e-2 | 1.2e-3 | 0.46 | 170 / 125 / 127 | 10 / 6 / 5 | −22 / −14 / −0.8 % | 1.1 / 0.01 | 298 (210) | 66/66 | 24/24 |
| 3 | C4 | 1.0e-1 | 1.1e-3 | 0.28 | 167 / 130 / 139 | 5 / 15 / 12 | −22 / −19 / −5.6 % | 4.4 / 0.32 | 339 (106) | 69/34 | 24/23 |
| 4 | C4 | 6.0e-2 | 1.0e-3 | 0.58 | 173 / 136 / 164 | 5 / 28 / 13 | −21 / −38 / −3.4 % | 6.4 / 0.33 | 367 (98) | 74/34 | 23/23 |
| 5 | C4 | 4.8e-1 | 1.3e-3 | 0.00 | 174 / 135 / 157 | 4 / 34 / 12 | −20 / −38 / −15 % | 7.6 / 0.18 | 352 (88) | 45/32 | 22/20 |
| 6 | C4 | 1.6e-1 | 7.3e-4 | 0.14 | 181 / 142 / 170 | 2 / 38 / 17 | −20 / −38 / −8.8 % | 7.7 / 0.39 | 527 (61) | 66/27 | 18/18 |
| 7 | C4 | 5.9e-2 | 5.9e-4 | 0.52 | 186 / 143 / 178 | 1 / 39 / 16 | −19 / −38 / −0.9 % | 7.7 / 0.55 | 629 (42) | 65/18 | 13/13 |
| 8 | C4 | 7.3e-2 | 5.9e-4 | 0.41 | 188 / 143 / 181 | 1 / 40 / 18 | −19 / −37 / −3.6 % | 8.1 / 0.64 | 636 (28) | 69/15 | 14/14 |
| 9 | C4 | 1.5e-1 | 6.8e-4 | 0.15 | 188 / 144 / 176 | 1 / 41 / 20 | −18 / −36 / −9.0 % | 7.9 / 0.48 | 570 (26) | 70/24 | 18/17 |
| 10 | C3 | 3.7e-2 | 5.8e-4 | – | 190 / 146 / **193** | 1 / 45 / 17 | −18 / −38 / −0.3 % | 7.8 / 0.69 | 649 (22) | 72/14 | 13/13 |

(w_s = 0.50 w at the search bound for every A.) Reading: the joint fit
matches the *release* count exactly at every A and the overtaking flow
within 15 %, but still captures 2–5× too many vehicles at A ≥ 3 and holds
them 3–30× too long — the residual queue is the price of the wedge. The
profiles (`fig_e13_profiles`) show the λ* solution as a moderate wedge
(52–57 veh/km, SUMO 55–60) over a thin stuck layer, instead of E12's 3 km
dilute band; q2000 transfers stay below classical for 9/10 A.

## 5. Is the degeneracy broken, and which observable breaks it? (`fig_e13_gap_vs_lambda`)

| λ | gap (M_c − M_r)/M_both | η_c | η_r | ridge residual [dex] | jackknife (5 folds) |
|---|---|---|---|---|---|
| 0 | +0.05 % | 0.76 | 0.77 | 0.00 | −0.0001 ± 0.0085, signs mixed |
| 0.02 | +0.01 % | 0.84 | 0.84 | 0.02 | – |
| 0.03 | +0.21 % | 0.56 | 0.71 | 0.88 | – |
| 0.05 | +0.31 % | 0.53 | 0.69 | 0.86 | **+0.035 ± 0.007, same sign 5/5** |
| 0.07 (λ*) | +0.07 % | 0.62 | 0.64 | 0.85 | −0.003 ± 0.040, signs mixed |
| 0.1 | +0.46 % | 0.59 | 0.70 | 0.86 | – |
| 0.2 | +1.5 % | 0.49 | 0.74 | 0.99 | – |
| 0.5 | +4.0 % | 0.44 | 0.74 | 1.02 | – |
| 1 | +6.2 % | 0.44 | 0.74 | 1.04 | – |

Ablations at λ* (which term carries it): events-only gap +0.91 %, ridge
residual 0.84 dex; queue-only +0.47 % but residual 0.10 dex (still one
ratio ladder); overtaking-only +0.00 %, residual 0.00 (fully degenerate);
w_s = 0.6 w: +0.35 %, 0.82 dex. **The event counts are the observable that
separates κ_c from κ_r; the queue helps little; the overtaking flow not at
all** — exactly the reviewers' prediction. Nestedness holds exactly at every
λ and in every fold.

Reading the verdict honestly: κ_r(A) is the better single carrier for every
λ ≥ 0.03 and the advantage grows with λ, but the pre-registered rule
"κ_r is the carrier iff η_r ≥ 0.8 and η_c ≤ 0.5" is met only for
λ ≥ 0.15–0.2 in the η_c half and never in the η_r half (0.70–0.75); at λ*
itself the gap is inside the jackknife noise. The two single-carrier
ladders at λ*: M_c gives κ_c(A) = 0.048 → 0.034 (Spearman −0.67, p = 0.03);
M_r gives κ_r(A) = 1e-4 (A = 1), 1.0–1.6e-3 (A ≥ 2, Spearman +0.65,
p = 0.04). Under M_both: κ_c = 0.034–0.041 (−0.84), κ_r = 3e-7 (A = 1),
0.9–1.4e-3 (A ≥ 2, flat). So the only large A-effect the field + events
can support is "A = 1 never releases, A ≥ 2 release at ≈ 1e-3 veh⁻¹"; the
remaining A-dependence is a 20 % drift in κ_c and none in κ_r.

## 6. Micro–macro

Criterion: median |log₁₀(κ^macro/κ^micro)| ≤ 0.5 dex and Spearman(macro,
micro) ≥ 0.6. At λ*: κ_c median 0.07 dex → magnitude **agrees** (the joint
calibration lands on the microscopic 0.03–0.05 veh⁻¹ without being told);
trend does not (ρ = −0.12). κ_r median 0.77 dex, ρ = −0.26 → neither: the
macroscopic release rate is 3–16× below the microscopic one and flat where
the microscopic one rises 5× from A = 2 to 10. The event-MLE κ evaluated
under J_λ* are worse than the fitted ones at every A ≥ 3 (J 1.36–1.55 vs
1.11–1.44) because they produce the right counts with the wrong field.
Interpretation: to keep the wedge the macroscopic model needs each captured
vehicle to stay 300–650 s (data 22–210 s), so it under-releases per capita
and over-captures — the same structural conflict as §2 seen in the rates.

## 7. Identifiability and resolution (stage e)

- Ridge scan under J_λ*: the argmin sits at κ_r = 3e-4–3e-3. The audit
  pointed out that the basin measured along a fixed-ratio column is 0 dex
  by construction, because the J_λ* valley no longer runs along constant
  ratio but along **constant κ_c ≈ 0.032** (best κ_c at every magnitude
  κ_r ≤ 1e-3 is 0.032 for A ≥ 2). Along that valley the 1 % basin in κ_r is
  0.0 dex (A = 1, 2, 6), 0.5 dex (A = 7–10) and 1.0 dex (A = 3–5); at 2 %
  it is 0.5–1.5 dex. So the joint objective identifies κ_c sharply and κ_r
  to within half a decade to a decade — a real but partial identification,
  consistent with the micro–macro result (κ_c agrees, κ_r does not). By the
  plan's rule (argmin κ_r < 0.1, valley basin < 1 dex, W1 ≤ W1_cl) 6/10 A
  count as identified. The E12 ridge (flat in magnitude above κ_r = 0.1) is
  gone.
- Turnover: λ = 0 winners 0.1–2 s (A ≥ 3), λ* winners 300–650 s, data
  22–210 s (Little's law), i.e. the joint fit over-corrects the kinetics.
- Resolution: the CAV is a density 1/dx, so dx belongs to the model
  definition (50 m = KDE scale since E-V4). At dx = 12.5 m the CAV cell
  alone is 80 veh/km and *every* model, classical included, blocks
  overtaking completely (ε_ω = −100 %, peak cell density 105 veh/km); at
  dx = 25 m the λ = 0 winners change ε_ω by −0.4…−0.5 and the λ* winners by
  −0.2…−0.4. This is a property of the E-V4 discretisation, not of E13,
  and a fixed-length CAV footprint would be needed before any convergence
  claim.

## 8. Conclusions for the next step

1. The E12 question "does A act through κ_c or κ_r" is answered as far as
   this model and these data allow: with event counts in the calibration the
   two rates are separately identified; κ_r carries more of the (small)
   A-dependence than κ_c; the macroscopic κ_c matches the microscopic
   value; the macroscopic κ_r does not, and Mladen's κ_c hypothesis is not
   supported.
2. The dominant finding is structural, and it is now pre-registered and
   quantified: for A ≥ 3 the pure model cannot have SUMO's wedge and SUMO's
   mechanism at once. The wedge in SUMO at high A is made of vehicles that
   slow down briefly and pass; in the model the wedge can only be stuck
   mass. A bottleneck resistance that is *not* stuck mass — the DM-G
   capacity term of E-V4b/E8, or a lane-resolved free class — is the missing
   ingredient, and E8's hybrid (cap + catch & release) is where the
   κ_c/κ_r ladder should be re-asked next.
3. Optimizer noise along λ (1–4 monotonicity violations per A) and the
   λ*-sensitivity of the jackknife verdict argue for reporting the whole
   sweep rather than one λ, which is what the figures do.

Figures: `out/e13/fig_e13_hull`, `fig_e13_tradeoff`, `fig_e13_gap_vs_lambda`,
`fig_e13_kappa_vs_A`, `fig_e13_metrics_vs_A`, `fig_e13_heat_q2500_A1-5/_A6-10`,
`fig_e13_heat_q2000_*`, `fig_e13_profiles`, `fig_e13_ridge`; tables
`out/e13/summary13.md`; v1 outputs in `out/e13_v1/`.

## Audit block

`audit_e13.py` (independent re-computation, 2026-09-16): PASS_WITH_ISSUES,
exit 0 — every numerical check passes: purity of all 480 stored
configurations; counters exact (3e-12); bit-level reproduction (relative
error 0) of every winner's W1, MAE_N, ε_ω, C_m, R_m, ε_R and J at A = 1, 5,
10 for λ = 0 and λ*; classical J; all 120 flag sets; sweep sums and the
three selection rules re-derived; nested totals, exact nesting, η, gap,
ridge residual and Spearman values at all 12 λ; ablation quartets;
jackknife SE; ridge, prehull and resolution entries; summary tables vs
JSON; figure files. Findings, all addressed or noted: (MINOR) the
identifiability basin had been measured along a fixed-ratio column, which
cuts across the J_λ* valley — stage e now also reports the basin along the
valley and the "identified" count is restated (6/10, §7); (NOTE) the
restricted knee is a near tie (0.333 at λ = 0.05 vs 0.330 at 0.07) and does
not affect λ*, which comes from the dominance rule; (NOTE) the jackknife
gap at λ* is not distinguishable from zero while it is at λ = 0.05 (stated
in §5); (NOTE) the E13 5-run mean field is computed from the classified
npz arrays and differs from `e7_wasserstein.load_rho_mean` by ≤ 2e-6 veh/km
(W1 relative difference 4e-9); (NOTE) the data event window is (250, 740]
and the model window (260, 740] — the bin ending at 260 s holds no events
in any run, so no number changes.
