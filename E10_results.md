# E9 / E10 Results — Replacing the Downstream-Release Kludge by General Model Structure

Date: 2026-09-03 · Responds to Mladen's fourth review ("hardcoding the conversion of
stuck vehicles to free when they overtake the bottleneck is not the solution … the
model should be usable in a general setting without kludgy fixes").
Solver suite 24/24; independent audits (E9 PASS, E10: 195 checks, one stale
provenance probe); all new fields default off = bit-identical legacy.

## 0. Verdict

The kludge (`downstream_release`, a label conversion keyed to the CAV position) is
replaced by a **general interface constraint**: *only free vehicles overtake the
bottleneck*, i.e. the synchronized class has zero flux relative to the moving
bottleneck (`s_impermeable`). It lives at the same interface and under the same
activity condition as the classical Delle Monache–Goatin capacity constraint, has no
parameters, references no label conversion, and applies to any moving bottleneck.
On the calibration scenario it reproduces the kludge's results to within noise
(A=1: W1 152.0 vs 152.2, wake 55.1 vs 53.8 veh/km, e_s +3.9% vs +4.7%, ω −2.5%
both, downstream exactly clean while the bottleneck is active).

## 1. E9 — the first general attempt (ratio-form leader-loss) and why it failed

Idea: a stuck vehicle whose look-ahead window contains less slow traffic than its
own cell has lost its leader; release rate ∝ (1 − s_ahead/s_j), front receding at the
start-up wave speed w (prefactor c_wave/ℓ_eff makes the front speed exactly w on any
grid — verified 0.04% off, dx = 100…12.5 m). Outcome (out/e9/ladder.json):
95–97% of its release happened **inside the queue** (fitted queues have s decreasing
toward the head, so the ratio coverage is < 1 there) — it became a κ_r substitute,
not a downstream constraint — and the downstream s it was meant to clean is mostly
**numerical front diffusion** of the s-class across the CAV cell (first-order upwind,
Courant 0.15 → D_num ≈ 320 m²/s → a co-moving plume 21/12/7/4/2 veh/km over the five
cells ahead of the CAV) that no finite rate out-cleans. Lesson: the leak is at the
interface; the fix must be at the interface.

## 2. E10 — one-at-a-time ladder (u15 q2500, lf, W1 refit per rung, cap 2000, w_s = 0.6w)

| rung | A=1: W1 / wake / e_s / ω / ds-stuck(slow) | A=10: W1 / wake / e_s / ω / ds-stuck(slow) | admissible |
|---|---|---|---|
| R0 cap + w_s (leaky reference) | 150.4 / 54.7 / +6.6% / −2.6% / 0.0042 | 172.8 / 55.7 / +4.3% / −18.3% / 0.0016 | no (plume 8.2 / 3.6 veh/km) |
| **R1 + s-impermeable interface** | **152.0 / 55.1 / +3.9% / −2.5% / 0.0000** | **180.1 / 60.7 / −5.9% / −18.3% / 0.0000** | yes (active window) |
| R2 R1 + connectivity leader-loss | 173.7, fit degenerates: κ_c → 5·10⁻¹⁵, N_s ≡ 0 | 190.4, same degeneracy | — |
| R3 leader-loss only (no interface) | 160.2 / 43.2 / +10.9% / −9.0% / 0.0003, plume 2.8 | 180.4 / 45.9 / −2.3% / −18.5% / 0.0003, plume 2.6 | no |
| R4 R2 without w_s | 169.6 / 69.5 / +0.3% / −8.4% | 190.4 (degenerate) | — |
| RDR the E8 kludge (reference) | 152.2 / 53.8 / +4.7% / −2.5% / 0.0000 | 167.4 / 47.7 / +0.2% / −18.3% / 0.0000 | (kludge) |

Attribution: (i) the interface constraint **alone** removes the plume and the plug
while the bottleneck is active — exactly, by construction — at the best W1 of the
ladder; (ii) the connectivity leader-loss term (general, state-only: coverage =
fixed point of "connected through slow traffic to a bottleneck vehicle within η")
makes s so costly under W1 that the optimizer abandons catch & release altogether;
it is therefore **not** part of the final model (kept in the code, documented);
(iii) leader-loss without the interface cannot clean a numerical leak; (iv) w_s is
still needed. Data-side reference: wake 58.6 (A=1) / 55.4 (A=10) veh/km.

## 3. The gating experiment (tried, measured, reverted)

Hypothesis: the stuck-class branch w_s should exist only while class A is slow
(u_s < v_f), so that a residual s label after the bottleneck deactivates is inert.
Result: W1 worsened for **both** models (final 152 → 167, kludge 152 → 170). A
platoon released when the bottleneck deactivates keeps a slower congested branch
until it disperses — that is the start-up-wave physics the data show (the receding
front at t = 850 s) — so the ungated branch is the physical choice; reverted and
documented in `solver.py`. Consequence: the only thing left of the "downstream s"
question is a label on an already-released platoon behind a free-driving vehicle,
with no bottleneck active; we report admissibility on the active-bottleneck window.

## 4. Final general model and numbers

Structure: DM-G capacity cap (Q_ξ = 2000 veh/h) + stuck-class congested branch
w_s = 0.6w + s-impermeable bottleneck interface + catch & release, (κ_c, κ_r)
fitted by 1-Wasserstein on u15/q2500 only; q2000 = zero-refit transfer.

| scenario | κ_c / κ_r | W1 (kludge) | wake (kludge; data) | e_s (kludge) | ω err (kludge) |
|---|---|---|---|---|---|
| A=1 q2500 | 4.46e-2 / 1.46e-3 | **152.0** (152.2) | 55.1 (53.8; 58.6) | +3.9% (+4.7%) | −2.5% (−2.5%) |
| A=1 q2000 ⁽ᵗ⁾ | same | 161.1 (146.7) | 46.9 (40.8; 42.3) | −6.0% (+1.8%) | −13.2% (−6.2%) |
| A=10 q2500 | 3.16e-2 / 1.28e-5 | 180.1 (167.4) | 60.7 (47.7; 55.4) | −5.9% (+0.2%) | −18.3% (−18.3%) |
| A=10 q2000 ⁽ᵗ⁾ | same | 133.6 (113.2) | 43.0 (32.3; 25.9) | −12.1% (−2.2%) | −21.3% (−20.3%) |

Honest reading: on the calibration case the general model equals the kludge; at
A=10 and in the q2000 transfers it is 8–18% worse in W1 and 4–10 points worse in
e_s. The kludge could keep a larger κ_r "for free" because it deleted whatever
leaked; the general model has to hold the queue physically, and its W1 fit drives
κ_r toward zero (A=10: 1.3·10⁻⁵ — i.e. essentially no overtaking release, which
is where the ω error comes from). Post-release rarefaction: +300 m (A=1, kludge
+300, data +1200); +100 m (A=10, data +100).

## 5. Audit trail

E9 audit PASS (285 checks). E10 audit: the working-tree solver was compared bit
for bit against `git HEAD:solver.py` with all new fields off; the audit found one
**major** defect — with the impermeable face the DM-G violation test still
subtracted u_cav·s_j although s no longer crosses, disengaging the cap whenever
the cap cell held s — fixed (`rho_pass = f_j` under impermeability) and the ladder
re-run; two minor items fixed (contiguous CAV footprint; the coverage fixed point
is unique, not merely least); one noted (500-sweep cap on the coverage iteration,
unreachable below 25 km queues). Final audit: 195 checks ok, one stale provenance
probe on a regenerated reference file. Three w_s-bearing bit-identity references
were regenerated from the flag-off paths (proven identical to the older solvers).

## 6. Files

`solver.py` (fields: `eta_la`, `ll_mode`, `tau_ll`, `s_impermeable`; tests t17–t24),
`e9_ladder.py` → `out/e9/`, `e10_ladder.py` → `out/e10/ladder.json` + rung figures,
`e10_final.py` → `out/e10/final_config.json`, `fig_profiles_final_e10_*.png`,
`fig_heat3_final_e10_*.png`; `audit_e9.py`, `audit_e10.py`.
