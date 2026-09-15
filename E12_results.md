# E12 Results — Pure E7 model on the integer assertiveness ladder A = 1…10

Date: 2026-09-15 · Responds to Mladen's fifth review (roll back to E7; no
kludge, no imposed constraint; test "low A → high κ_c, high A → low κ_c";
restrict A to the ten integers; look at every A's heatmap).
Plan: `E12_plan.md`. Driver: `e12_assertiveness.py` (+ `e12_figures.py`),
tests `test_e12.py` (17), independent audit `audit_e12.py`.
Everything below is at u_ξ = 15 m/s (54 km/h); fit on q_in = 2500 veh/h,
every other number is a zero-refit transfer.

## 0. What was run

- **Model**: exactly the E7 structure — uncapped two-class catch & release,
  J_c = κ_c (a + γ s) f Δv, J_r = κ_r (P − ρ) s Δv, optional γ and w_s ≤ w.
  The four post-E7 knobs (capacity cap, downstream release, leader-loss,
  impermeable interface) are **forbidden**: the driver asserts they are off on
  every catch & release run, a test pins the solver defaults, and the audit
  rebuilds every stored configuration and re-checks. The classical baseline
  (κ = 0 + Delle Monache–Goatin cap 2000 veh/h) is the only capped run.
- **A = 1, 2, …, 10** (1.5 / 1.75 / 2.25 / 2.5 dropped). Event metrics
  (e_s, ω, N_s) for A ∉ {1, 10} come from the E4 classification (`out/e4`).
- **Stage A** per A: C1 (κ_c, κ_r), C2 (+γ), C3 (+w_s), C4 (+γ, +w_s), each
  fitted with the E7 W1 machinery (6×6 log grid + Nelder–Mead 40, dt = 1 s;
  re-evaluated at dt = 0.5 s), winner by the E7 rule, C1 W1 landscape stored.
- **Stage B**: nested shared-parameter models over all ten A — M_none (one
  (κ_c, κ_r) for all A), M_c (κ_c(A) free, κ_r shared = Mladen's
  hypothesis), M_r (κ_r(A) free, κ_c shared = E-V3 event picture), M_both
  (per-A optimum). Nestedness is enforced (parent optimum is a fallback), so
  Σ W1 can only go down along M_none → M_c/M_r → M_both.
- **Stage E**: the event-based (E-V3/E4) κ evaluated in the field with no
  fitting; a (κ_c/κ_r, κ_r) identifiability scan per A.
- Runtime: stage A 21 s per A (10 in parallel), stage B 56 s, stage E 4 s.

## 1. Every A: catch & release vs classical (W1, veh km)

| A | winner | κ_c | κ_r | γ | w_s/w | q2500 (fit) cl / CR | q2000 cl / CR | u20 q2000 cl / CR | u20 q2500 cl / CR |
|---|---|---|---|---|---|---|---|---|---|
| 1 | C4 | 5.64e-2 | 4.36e-5 | 0.657 | 0.609 | 173.7 / **148.9** | 146.5 / 146.6 | 175.2 / **139.1** | 184.0 / **157.5** |
| 2 | C3 | 4.54e-2 | 1.75e-3 | – | 0.615 | 169.7 / **127.0** | 152.7 / **127.0** | 162.3 / **120.4** | 174.5 / **133.7** |
| 3 | C3 | 3.12 | 0.367 | – | 0.554 | 167.3 / **140.6** | 127.2 / **92.7** | 131.3 / **92.1** | 174.6 / **147.4** |
| 4 | C3 | 4.19 | 0.391 | – | 0.634 | 172.7 / **141.8** | 119.1 / **92.8** | 124.8 / **89.4** | 186.1 / **166.0** |
| 5 | C3 | 4.05 | 0.368 | – | 0.647 | 173.9 / **143.3** | 119.5 / **95.5** | 124.9 / **90.9** | 188.5 / **168.4** |
| 6 | C1 | 39.3 | 3.30 | – | – | 181.2 / **146.5** | 119.4 / **101.3** | 126.4 / **94.5** | 196.4 / **151.5** |
| 7 | C3 | 3.95 | 0.354 | – | 0.637 | 185.9 / **152.1** | 122.3 / **100.4** | 127.4 / **95.2** | 201.6 / **178.7** |
| 8 | C3 | 0.761 | 0.0766 | – | 0.500 | 188.3 / **167.9** | 120.0 / **94.2** | 128.5 / **98.0** | 205.0 / **191.6** |
| 9 | C1 | 31.4 | 2.53 | – | – | 187.6 / **151.7** | 121.0 / **106.8** | 129.8 / **99.5** | 206.3 / **154.4** |
| 10 | C1 | 4.59 | 0.378 | – | – | 190.4 / **157.2** | 122.0 / **107.3** | 130.4 / **100.2** | 208.6 / **180.8** |

**On the field metric the pure E7 model beats the classical model at every A
and in all 40 scenario cells** (the single exception, A = 1 at q2000, is a tie:
146.6 vs 146.5; the E7 winner C3 would transfer to 142.8 — C4 wins the fit
scenario only by the wake tie-break, 56.1 vs 55.5 veh/km). Gains are
14–25 % at q2500 and 12–27 % at q2000 for A ≥ 3. So "at A = 10 the model is
worse than classical" is **not** a property of the E7 model; it came from
the E10/E11 impermeable-interface + cap configuration (A = 10 q2000:
145.6 vs classical 122.0). The pure model transfers A = 10 to q2000 at 107.3.

Figures: `out/e12/fig_e12_heat_q2500_A1-5`, `_A6-10` (fit) and
`fig_e12_heat_q2000_A1-5`, `_A6-10` (transfer); `fig_e12_data_heat` (SUMO
only, all A); `fig_e12_profiles` (A = 1, 5, 10 with the s/f split);
`fig_e12_kappa_vs_A` panel (d) is the W1-vs-A summary.

## 2. Where the pure model IS worse than classical — the physics at A ≥ 4

The heatmaps show it: for A ≥ 3 the catch & release wake is not a thin dense
wedge but a **2.5–3.2 km dilute band at ~33 veh/km** (SUMO: ~55 veh/km,
0.5–1 km). The per-run event metrics quantify the price of the W1 gain:

| A | ω error cl / CR | N_s MAE [veh] cl / CR | wake cl / CR (SUMO) | e_s cl / CR |
|---|---|---|---|---|
| 1 | −2.5 % / −9.9 % | 40.2 / 13.3 | 68.4 / 56.1 (58.6) | +5.8 / +3.0 % |
| 2 | −21.7 / −9.4 % | 10.3 / 5.8 | 68.4 / 42.1 (62.9) | −9.5 / +4.1 % |
| 3 | −21.7 / −22.2 % | 5.0 / 10.7 | 68.4 / 48.4 (58.8) | −8.2 / +9.7 % |
| 4–7 | −19…−21 / **−38…−40 %** | 1–5 / **24–43** | 68.4 / 38–46 (55–57) | −6…−7 / +4…+5 % |
| 8 | −18.7 / −14.5 % | 0.8 / 21.5 | 68.4 / 49.0 (55.3) | −4.7 / +8.4 % |
| 9–10 | −18 / **−40 %** | 0.6–0.9 / **44–50** | 68.4 / 37–39 (54–55) | −5…−6 / +3…+4 % |

At A ≥ 4 the model passes 40 % too few vehicles (classical: 20 % too few),
its queue holds 25–50 vehicles that SUMO does not have (SUMO's queue at
A ≥ 4 is < 5 veh — the classical model's empty queue is right by
construction), and its wake is 15–18 veh/km too dilute. This is the E7
"geometry vs turnover" limit — turnover 0.9 s in the slow window for A = 5 and
10 — and it is now shown to be generic for A ≥ 3, not an A = 10 curiosity.
e_s stays within +3…+10 % everywhere (classical −5…−10 %). **The A-high
problem is real, but it is a problem of the wrong mechanism at equal or
better W1, not of a worse density field.**

## 3. The hypothesis "low A → high κ_c, high A → low κ_c"

### 3.1 The field cannot separate κ_c from κ_r for A ≥ 2

Nested comparison (Σ over the ten A of W1 at q2500, dt = 1; `fig_e12_hypothesis`):

| model | free per A | shared | params | Σ W1 |
|---|---|---|---|---|
| M_none | – | κ_c = 45.1, κ_r = 3.94 | 2 | 1605.4 |
| M_c (hypothesis) | κ_c(A) | κ_r = 3.16 | 11 | 1558.2 |
| M_r (event picture) | κ_r(A) | κ_c = 71.5 | 11 | 1558.4 |
| M_both | κ_c(A), κ_r(A) | – | 20 | 1544.2 |
| same three with w_s = 0.6 w | | | | 1548.7 / 1492.7 / 1492.7 |

- **M_c and M_r are indistinguishable** (0.3 veh km apart, 0.02 % of the
  total; identical per-A W1 to 0.1). Letting κ_c vary with A and letting κ_r
  vary with A explain the fields equally well, because for A ≥ 2 the fit sits
  on the **fast-equilibrium ridge**: W1 depends on the ratio κ_c/κ_r and is
  flat in the magnitude once κ_r ≳ 0.1 (identifiability scan
  `fig_e12_ridge`, summary §9: for every A ≥ 2 the best ratio is 10 at
  κ_r = 0.1, 1 and 10 with W1 within 1 %). Halving κ_c is the same as doubling
  κ_r. The ladders under M_c and M_r are flat and **not monotone** (κ_c(A)
  from 28 to 40 with Spearman +0.91 — increasing, not decreasing; κ_r(A) from
  8.2 to 5.7).
- **What the field does identify**: (i) the ratio, κ_c/κ_r ≈ 9.4–12.5 for
  A = 2…10, 15 for A = 1 — nearly A-independent; (ii) slow kinetics at A = 1
  only (κ_c ≈ 0.14, κ_r ≈ 0.009: the scan's minimum is at κ_r = 0.001, and the
  fast limit is 9 % worse), i.e. A = 1 is the only level where the catch &
  release *rates* matter; from A = 2 up the model behaves as an equilibrium
  partition s/f fixed by the ratio.
- **A-dependence in the pure C1 model is weak**: shared parameters cost only
  61 veh km over ten A (3.8 %), and 41 of those are A = 1 and 2 (173→162,
  164→134). For A ≥ 3 one parameter pair serves all A within 0–6 veh km. The
  differences between the A ≥ 3 heatmaps come from the data, not from κ.
- The per-A C1 magnitudes (κ_c from 0.14 at A = 1 to 12–48 at A = 3–6, back to
  0.95 at A = 8) are ridge positions, not physics — they left the fit grid
  (log κ_c up to 1.7 vs grid max 0.5) and their Spearman correlation with A is
  +0.21 (p = 0.56).

So the hypothesis is neither supported nor refuted by the density fields:
the pure E7 model, calibrated on W1, has no way of telling whether A acts on
capture or on release.

### 3.2 The microscopic (event-based) view does separate them — and points to κ_r

The E-V3/E4 Poisson-exposure MLE identifies κ_c and κ_r separately from the
vehicle events (integer A only, q2500): κ_c is flat at 0.031–0.052
(Spearman +0.14, p = 0.70) while **κ_r rises from 0 (A = 1) to 1.6e-2
(A = 10)** (Spearman +0.99, p < 1e-7; same at q2000). Microscopically,
assertiveness acts through *release*, not capture — the opposite carrier from
the hypothesis, but with the same macroscopic effect (fewer stuck vehicles).

However, the event-based κ do **not** work in the field (summary §8): plugged
into the C1 model without fitting they give W1 = 141 at A = 2 (better than
classical) but 197–262 for A ≥ 3 (classical 167–190) and 202 at A = 1
(κ_r = 0: nothing is ever released). The macroscopic effective rates at
A ≥ 3 are two orders of magnitude above the microscopic ones, and their
ratio (10) is above the microscopic ratio (2–7). The two calibrations
measure different things (E6's "two tracks"), and only the microscopic track
carries the A-signal in κ.

## 4. Structural knobs across A

- **w_s (C3)** wins at A = 2, 3, 4, 5, 7, 8 (w_s = 0.50–0.65 w) and gives the
  best q2000 transfers everywhere (92–101 vs C1's 93–107); at A = 6, 9, 10 C1
  wins the fit scenario by ≤ 15 veh km but C3 still transfers better.
- **γ (C2)** never wins and is worse than classical at A ≥ 4 (173–202): with
  κ_r → 5e-5 the fit freezes release. **C4** (γ + w_s) wins only at A = 1
  (γ = 0.66, w_s = 0.61 w, κ_r = 4e-5 — again no release) and transfers worse
  than C3. Conclusion: in the pure model γ is not a useful A-carrier; w_s is
  a robust improvement of the wake geometry at every A.

## 5. Identifiability, audit, files

- 6×6 grid near-optimal sets (≤ 3 %): 1–4 points; the ridge is diagonal in
  (log κ_c, log κ_r), so the coarse grid under-reports it — the dedicated
  (ratio, magnitude) scan of stage E is the authoritative statement.
- Independent audit `audit_e12.py`: purity of every stored configuration,
  bit-level reproduction of classical and winner W1 at A = 1, 5, 10, grid
  entries, winner rule, hypothesis totals and nestedness, Spearman values,
  M_c per-A optimum vs a 41-point scan, A = 5 event metrics, summary tables
  vs JSON, figure files — see the audit block at the end of this file.
- `out/e12/`: `ladder.json` (+ per-A), `hypothesis.json`, `extras.json`,
  `fields_A*.npz`, `summary.md`, figures `fig_e12_*.png/.pdf`.

## 6. What this means for the next step

1. The "A = 10 worse than classical" verdict is withdrawn for the pure model
   on the field metric, but the **mechanism** at A ≥ 4 is wrong (40 % ω
   deficit, 25–50 phantom stuck vehicles, dilute 3 km band). W1 alone cannot
   see this, because it only looks at the density field.
2. The field cannot decide between κ_c(A) and κ_r(A); the microscopic data
   say κ_r(A). To make the hypothesis testable macroscopically the calibration
   must include an observable that breaks the ratio degeneracy — the queue
   size N_s(t) and/or the overtaking flow ω(t), both measured per run.
   Proposal E13: joint objective W1 + λ_N · N_s-MAE + λ_ω · |ω error| per A,
   same pure structure, then repeat the M_none / M_c / M_r / M_both ladder.
   If the microscopic picture is right, κ_r(A) should then come out
   monotone and κ_c flat, with the queue and ω of the A ≥ 4 fields fixed at a
   modest W1 cost.
3. Whatever the outcome, the A-dependence has to be carried by the source
   terms as functions of state — no scenario information enters the model.

## Audit block

`audit_e12.py` (independent re-computation, 2026-09-15): **34/34 checks
pass**, exit 0. Purity of all 40 stored configurations; bit-identical
reproduction (relative error 0) of classical and winner W1 at A = 1, 5, 10 on
both inflows; grid entries and grid minima; winner rule; hypothesis totals at
both resolutions; nestedness holds exactly without tolerance; all Spearman
values; A = 5 event metrics; 80 summary numbers vs JSON; 9 figures with PDF
twins. Notes from the auditor, all addressed: the stage-B log had been
written by an earlier optimizer version (whole program re-run into a fresh
log); the verdict string now names the ridge branch when |M_c − M_r| is below
1 % of M_both; the A = 1 q2000 margin is printed with two decimals
(+0.03 veh km); `fig_e12_ridge` added to the plan. The auditor's own fine
scan at A = 3 confirms the ridge: along κ_c/κ_r = 9.42 the objective is flat
to 0.01 veh km for κ_c ∈ [10, 60], while at fixed κ_r the valley in κ_c is
only ~0.1 dex wide — the ratio is identified, the magnitude is not.
