# E7 Results — Wasserstein Calibration, Flux-Function Ablation, Speed Transfer

Date: 2026-08-17 · Responds to Mladen's second review (W1 metric / wake density /
rarefaction / dynamic equilibrium / speed transfer). 5-agent workflow, audited;
solver suite 13/13 green, all new knobs bit-compatible when off.

## Setup

- **W1 objective** (his suggestion): W1(t) = ∫|N_sim − N_data|dx between
  *cumulative* density curves, averaged over t∈[100,1000] s; unnormalized, so
  mass errors are charged too. Synthetic gate: shifted equal-mass blobs recover
  W1 = mass × shift exactly.
- **Two structural knobs** (default off, bit-identical when off; +1 param each):
  `gamma` (capture agent ℓ = a + γs, interpolates lf↔af) and `w_s` (stuck-class
  congested branch; w_s ≤ w required by the invariant-domain proof — *lowering*
  w_s lowers the s-critical density r*_s = w_s·P/(c_s+w_s) so the s-supply binds
  inside the wake and densifies it; this supersedes the plan's "steeper branch"
  sketch, and is the correct direction under the tex scheme).
- Ablation per A (u15, q2500, lf): C0 = RMSE refit, C1 = W1, C2 = W1+γ,
  C3 = W1+w_s. Fits on the rep-mean field (coarse log-grid + Nelder–Mead at
  dt=1, re-evaluated at dt=0.5).

## Ablation results

**A=1** (data: wake 58.6 veh/km supercritical; post-release spreading +1200 m):

| config | κ_c | κ_r | extra | W1 | wake ρ | rarefaction | e_s | ω err |
|---|---|---|---|---|---|---|---|---|
| C0 RMSE | 0.178 | 9.5e-3 | — | 174.7 | 47.1 (sub) | +500 m | −2.1% | −32% |
| C1 W1 | 0.141 | 9.3e-3 | — | 162.6 | 53.9 (**super**) | +200 m | +10.1% | −20% |
| C2 W1+γ | 0.065 | 4.5e-4 | γ=0.64 | 169.4 | 60.1 (super) | +600 m | +3.2% | −22% |
| **C3 W1+w_s** | **0.046** | **5.7e-4** | **w_s=0.60w** | **147.1** | **55.5 (super)** | **+500 m** | **+3.0%** | **−9%** |

**The one-bug-two-symptoms hypothesis is confirmed**: switching RMSE→W1 alone
pushes the wake supercritical; adding the stuck-class flux function (Mladen's
own suggestion) wins outright — wake 55.5 vs data 58.6, post-release
rarefaction restored (visible spreading fan at t=850, `fig_profiles_e7_A1…`),
e_s +3.0%, and even the overtaking-flow error collapses from −32% to −9%.
Zero-refit transfer to q2000: W1=142.8, e_s=−2.4% (data wake there is itself
subcritical, and the model correctly produces no spreading — matching data).

**A=10** (data: wake 55.4): winner is **C1 (pure W1)**: W1=157.2, e_s=+2.8%.
The optimizer prefers a broad *subcritical* ~33 veh/km band over a dense narrow
wedge. **Dynamic-equilibrium verdict** (Mladen's "ideal outcome"): the
*slow-but-not-stuck band exists* — mean speed 22.9 m/s (strictly between
u_ξ+2 and v_f−3) at density 1.3× the inflow, a genuine inside-the-flux-function
aggregate that scalar LWR cannot express. But its composition is off his
picture: the s-layer spans 3.2 km (not a thin near-CAV layer) and the
capture/release turnover is ~1 s (fast churn), vs the 20–50 s measured
microscopically. γ-localization (C2) produces the thin dense layer he sketched
(835 m, wake 56.9) but only by freezing release (turnover 8455 s) at the worst
W1. **Partial structural limit, honestly stated**: with instantaneous
Δv-driven rates, the model can have either the right geometry or the right
turnover, not both. (A relaxation-time source is the natural fix — same
extension the stop-and-go analysis pointed to.)

## Speed transfer (his robustness question)

Fit A=3 at u=15/q2500 only (κ_c=12.4, κ_r=1.32 — see identifiability note),
then predict u ∈ {10, 12, 18, 20, 24} with κ frozen:

- **W1 stays flat: 146.8–157.7 veh·km across the whole sweep** (≤5.9% above
  the fit-speed value), and is ≤ the parameter-free classical M1 baseline at
  every speed except u=18 (+0.6%). ρ-RMSE: CR beats M1 at all u except u=10.
- A=1/A=10 with the E6 u15-κ applied to u=20: CR beats M1 at 3 of 4 points
  (A1 both inflows, A10 q2000); A10 q2500 is 3% worse — the single losing point.
- The non-flat part is e_s: it grows from +4% (u10) to +31% (u24) — the
  robustness claim rests on the field metrics (W1/RMSE); M1's e_s is itself
  −29% at u20.
- **Identifiability note**: on the A=3 field the fit sits on a fast-equilibrium
  ridge — only the *ratio* κ_c/κ_r ≈ 9.4 is identified (W1 changes <2% under a
  20× joint rescaling). Quote the ridge, not the point values.

## Audit

PASS_WITH_ISSUES → all addressed: γ ≥ 0 guard added (negative γ voided reaction
positivity); w_s-direction flip vs the plan documented above; the untested
cap+w_s combination noted as out of scope (E7 runs are uncapped). Bit-identity
of all default-off paths verified against the pre-change solver from git.

## Files

`e7_wasserstein.py` (W1 metric + fit machinery, `--selftest`) ·
`e7_ablation.py` → `out/e7/ablation.json`, `fig_profiles_e7_*.png` (incl. the
t=850 post-release panel), `fig_heat3_e7_*.png` ·
`e7_transfer.py` → `out/e7/transfer.json`, `fig_transfer_curve.png` ·
`audit_e7.py` · solver knobs in `solver.py` (tests t10–t13).
