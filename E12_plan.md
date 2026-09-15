# E12 Plan — Back to the pure E7 model, integer assertiveness ladder A = 1…10

Date: 2026-09-15 · Responds to Mladen's fifth review (relayed by Mingchen):

> Roll back to E7 and iterate from there. The model does well at low A but at
> high A (A = 10) it is even worse than the classical model. From now on: no
> kludge and no imposed physical constraint — after the CAV speeds up the
> downstream flow is still affected (the captured vehicles do NOT all become
> free spontaneously). Working hypothesis: low A → high κ_c (polite drivers
> stay stuck behind), high A → low κ_c (aggressive drivers pass the CAV /
> obstacle). First restrict A to the ten integers 1…10, then look at the
> heatmaps of every A level.

## 1. What "pure E7" means (and what is now forbidden)

Model = two-class catch & release, uncapped, exactly the E7 structure:

- states a (CAV), f (free), s (stuck); shared triangular FD (v_f, w, P from
  E-V2, frozen); u(ρ) = min(u_s, v(ρ)); Δv = [v(ρ) − u_s]₊;
- capture J_c = κ_c · (a + γ s) · f · Δv, release J_r = κ_r · (P − ρ) · s · Δv;
- optional state-only structure knobs (E7): γ ∈ [0, 1] (capture agent), w_s ≤ w
  (stuck-class congested branch, P_s = P). Both are functions of the state
  only; they are model structure, not kludges.

Forbidden from now on (the E12 driver refuses to run if any is set, and a
unit test pins the solver defaults to the pure path):

| knob (`SimConfig`) | introduced | why it is out |
|---|---|---|
| `q_xi_max` (DM-G capacity cap) | E-V4b | imposed bottleneck capacity — not part of catch & release |
| `downstream_release` | E8 | the kludge (uses x_cav, i.e. scenario information) |
| `eta_la` / `ll_mode` / `tau_ll` (leader-loss) | E9/E10 | failed; a release mechanism bolted on |
| `s_impermeable` | E10 | imposed interface constraint; Mladen: stuck vehicles do not all turn free after the CAV speeds up |

The classical baseline keeps its cap (scalar LWR + Delle Monache–Goatin
moving bottleneck, Q_ξ = 2000 veh/h, κ = 0): that *is* the classical model.

## 2. Data restriction

A ∈ {1, 2, …, 10} only (drop 1.5, 1.75, 2.25, 2.5). Focus u_ξ = 15 m/s
(54 km/h). Fit scenario q_in = 2500 veh/h; zero-refit transfer to q_in = 2000.
Density fields: 5-run mean (`e7_wasserstein.load_rho_mean`). Per-run event
metrics (e_s, ω, N_s) for A ∉ {1, 10} come from the E4 classification
(`out/e4/A{A}_u15_q{q}_r{r}.npz`, same keys as `out/e1`).

## 3. Experiments

### E12a — per-A calibration ladder (E7 configs, W1 objective)

For each A, on u15/q2500, `e7_wasserstein.fit_field` (6×6 log grid +
Nelder–Mead 40, dt = 1 s; re-evaluated at dt = 0.5 s):

| config | fitted | note |
|---|---|---|
| C1 | κ_c, κ_r | pure E7 |
| C2 | + γ ∈ [0, 1] | capture localisation |
| C3 | + w_s/w ∈ [0.5, 1] | stuck-class branch |
| C4 | + γ, + w_s/w | joint (new) |

Winner per A by the E7 rule (min W1 at dt 0.5; ties within 5 % broken by wake
density then rarefaction). Every config is scored at q2500 and, zero-refit, at
q2000 (W1, RMSE, e_s, ω error, wake, rarefaction, s-layer, N_s MAE) and W1-only
at u20 (q2000, q2500). Classical baseline scored identically.

### E12b — hypothesis test: does A act through κ_c?

1. **Per-A trends**: κ_c(A), κ_r(A), κ_c/κ_r from C1; Spearman rank
   correlation with A; side by side with the event-based MLE values
   (`out/e4/kappa_vs_A.json`, integer A only, with 95 % bounds).
2. **Identifiability**: per A, the W1 landscape on the 6×6 grid; the
   near-optimal set (≤ 3 % above the minimum) and its extent in log κ_c,
   log κ_r and along the ratio (E7 found a fast-equilibrium ridge at A = 3).
   (Stage E adds a dedicated (κ_c/κ_r, κ_r) scan and a zero-fit evaluation of
   the event-based κ in the field.)
3. **Nested shared-parameter models** (all 10 A fitted jointly at q2500,
   objective = Σ_A W1):
   - M_none: one (κ_c, κ_r) for all A (2 params);
   - M_c: κ_c free per A, κ_r shared (11 params) — Mladen's hypothesis;
   - M_r: κ_r free per A, κ_c shared (11 params) — the E-V3 event picture;
   - M_both: the per-A C1 fits (20 params).
   If Σ W1(M_c) ≈ Σ W1(M_both) ≪ Σ W1(M_r): A acts through κ_c. If M_r wins:
   through κ_r. If M_c ≈ M_r ≈ M_both: the field cannot tell (ridge) — then
   only the ratio is a meaningful A-dependence. Monotonicity of κ_c(A) under
   M_c is checked explicitly. Robustness: repeat M_c / M_r with w_s = 0.6 w.

### E12c — heatmaps of every A

Paper style (`paperfig`), `out/e12/`:

- `fig_e12_heat_q2500_A1-5`, `_A6-10`: rows A, columns SUMO 5-run mean |
  classical LWR+MB | catch & release (per-A winner); W1 in the model titles;
  CAV trajectory overlaid. Same for q2000 (zero-refit).
- `fig_e12_data_heat`: SUMO 5-run mean only, all 10 A × {q2500, q2000}.
- `fig_e12_kappa_vs_A`: κ_c, κ_r, ratio vs A (field fit vs event-based) and
  W1 vs A for classical vs catch & release at q2500 and q2000 — this panel
  shows exactly where the model loses to the classical one.
- `fig_e12_hypothesis`: Σ W1 of M_none / M_c / M_r / M_both and the
  per-A parameter ladders.
- `fig_e12_profiles`: A ∈ {1, 5, 10}, t = 500 / 700 / 850 s, s/f split.
- `fig_e12_ridge`: the identifiability scan of E12b.2 — W1 vs κ_c/κ_r per A,
  one curve per κ_r magnitude (added during the run: the 6×6 grid is too
  coarse for the diagonal ridge).

## 4. Files

`e12_assertiveness.py` (--smoke | --run | --figures | --summary),
`test_e12.py` (purity guard, A set, data loader, mechanics),
`audit_e12.py` (independent re-computation), `out/e12/*.json`, figures,
`E12_results.md`, README step 13.
