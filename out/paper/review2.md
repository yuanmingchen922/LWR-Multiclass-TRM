# Paper figure set — re-review after the fix pass (tag `review2`)

Scope: the seven PNG/PDF pairs in `out/paper/` (every PNG opened at full
resolution, plus zoomed crops of each legend flagged in `review.md`),
`captions.md`, `table_metrics.md`, checked against `tuned_config.json`,
`transfer.json` and `out/e4/kappa_vs_A.json`. Producer scripts
(`e11_tune.py`, `e4_sweep.py`, `e11_transfer_fd.py`, `paperfig.py`) were
read, not modified (mtimes 23:45–23:48, Sep 5, unchanged by this review).
`captions.md` received six wording corrections (§4); a copy of the file as
found is in the session scratchpad (`captions_before_review2.md`).

Context change since `review.md`: the tuning round's five-parameter capped
sets (A = 1: kappa_c 1.557e-1, kappa_r 6.84e-3, gamma 0.510, w_s 0.584 w,
Q_xi 2512 veh/h; A = 10: 3.795e-1, 8.70e-5, 0, 0.40 w, 2290 veh/h) are now
`tuned_config.json[A]['capped_reference']`. The ADOPTED final model
(`[A]['final']['adopted'] = 'nocap'`, 4 parameters, `q_xi_max = None`) is
A = 1: kappa_c 1.695e-1, kappa_r 7.129e-3 veh^-1, gamma 0.471,
w_s 0.579 w; A = 10: 3.875e-1, 1.534e-4, 0.000, 0.400 w. Fit-scenario W1
138.85 vs 138.97 (capped) and 165.39 vs 165.28 veh km. Figures, table and
captions were all checked against the adopted sets.

## 0. Summary

| Fig | file | verdict | note |
|---|---|---|---|
| 1 | `fig_fd` | **ready** | D1.1, D1.2 done |
| 2 | `fig_assertiveness` | **ready** | D2.1–D2.4, G5 done |
| 3 | `fig_heatmaps_q2500` | **ready** | G1, G3, G4, D3.4 done; D3.5 (colormap) left optional |
| S1 | `fig_heatmaps_q2000` | **ready** | as Fig 3; D3.3 caption fixed |
| 4 | `fig_es` | **ready** | D4.1–D4.5 done |
| 5 | `fig_profiles` | **ready** | D5.1–D5.6 done |
| 6 | `fig_transfer` | **ready** | G2 closed (triangles dropped); D6.2–D6.6 done; remarks R1, R5 |
| – | `table_metrics.md` | **ready** | 232 numbers reproduce `tuned_config.json`; §2.8 cosmetics done |
| – | `captions.md` | **ready** after §4 | every quoted number re-derived from its JSON |

PDFs: all seven exist (`fig_assertiveness.pdf` 35 kB, `fig_es.pdf` 24 kB,
`fig_fd.pdf` 38 kB, `fig_heatmaps_q2000.pdf` 86 kB, `fig_heatmaps_q2500.pdf`
92 kB, `fig_profiles.pdf` 63 kB, `fig_transfer.pdf` 27 kB), Matplotlib 3.10
pdf backend, fonts embedded as `FontFile2` (TrueType: DejaVuSerif,
DejaVuSans, DejaVuSans-Oblique), zero `/Type3` objects. MediaBoxes: 239 x 174
(fd), 249 x 174 (es), 238 x 278 (transfer) pt = single column 3.3–3.5 in;
499 x 164 (assertiveness), 512 x 318 (heatmaps), 503 x 391 (profiles) pt =
double column 6.9–7.1 in.

PNGs: 300 dpi everywhere (fd 996 x 727, assertiveness 2077 x 681, heatmaps
2134 x 1324, es 1038 x 721, profiles 2096 x 1631, transfer 994 x 1160 px).

## 1. Global defects

- **G1 (`pf.setup()`) — closed.** `e11_tune.py:869` (`make_outputs`),
  `e4_sweep.py:187`, `e11_transfer_fd.py:417`. All four E11 figures are now
  serif, 300 dpi, TrueType, unframed legends, no top/right spines, faint grid
  (heatmaps have the grid switched off, correctly). The 7 pt two-line tick
  labels of Fig 4 and the 7.5 pt legends of Figs 4 and 5 are legible at
  column width.
- **G2 (two "the model"s) — closed by option (b).** The A = 1 / A = 10
  triangles are gone from Fig 6; `transfer.json` has no top-level
  `A1_A10_transfer` (the E10 block is archived under
  `capped_reference`, note "kept for reference, not plotted"). Figs 3–5 and
  Table 1 use `tuned_config[A]['final']` (no cap); Fig 6 uses the same
  structure (`extra_cfg = {w_s: 3.7276 m/s = 0.6 w, s_impermeable: True}`,
  no `q_xi_max`) at A = 3 with (kappa_c, kappa_r) fitted and gamma = 1,
  w_s = 0.6 w fixed. The legend/title string is "catch & release" in every
  figure, and no scenario appears anywhere with two different parameter
  sets. The only residual difference is the fitting protocol (4 vs 2 free
  parameters), stated in the caption header and the Fig 6 caption — R1.
- **G3 (units) — closed.** Fig 4: tick labels 54 / 72 with x-label
  "u_xi [km/h] / q_in [veh/h]"; Fig 6: "bottleneck speed u_xi [km/h]", ticks
  40–80, annotation "calibration (u_xi = 54 km/h)"; heatmaps: "t [s]",
  ticks 0, 250, 500, 750, 1000. Captions: km/h and s throughout; the only
  "m/s" strings are the conversions in the conventions paragraph and the
  table header ("54 km/h (15 m/s)"), which are intended.
- **G4 (vocabulary) — closed.** "SUMO (5-run mean)" (Figs 3, S1, 5),
  "classical LWR+MB" (3, S1, 4, 5, 6), "catch & release" (3, S1, 4, 6;
  "catch & release rho" in Fig 5 to distinguish the total from the s / f
  fills). Sources: `MODEL_LABEL`/`DATA_LABEL` at `e11_tune.py:98–99`,
  literals at `e11_transfer_fd.py:277–278`.
- **G5 (red/green) — closed.** kappa_r-hat is `tab:purple`
  (`KR_COL`, `e4_sweep.py:173`); the legend's censored-point handle uses the
  same colour.

## 2. Per-figure

### 2.1 Fig 1 — `fig_fd` — READY

D1.1 legend entry now "Q(rho)"; D1.2 on-plot text now "Q_xi" (value in the
caption). Zoom of the legend region: the congested branch passes to the left
of the legend's marker column and the densest queued squares (rho_- up to
about 90 veh/km) end roughly 1 mm before the legend — clear, but keep the
legend where it is (R6). Caption numbers = `transfer.json[fd_figure]`:
v_f 100.57 km/h, w 22.37 km/h, P 266.7 veh/km, rho_c 48.52 veh/km,
capacity 4879 veh/h, 67 944 of 407 667 cells, 144 upstream / 53 queued /
146 downstream states, queued threshold 60 veh/km, Q_xi 2000 veh/h.

### 2.2 Fig 2 — `fig_assertiveness` — READY

- D2.1: legend shortened to "kappa_c-hat / kappa_r-hat / upper bound",
  anchored lower right in panel (b). Zoom: the legend's top edge sits at
  about 1.5e-4 veh^-1, the lowest CI whisker (q2000 kappa_c-hat at A = 7)
  ends near 1e-3; the A = 1 censored triangle (1.7e-5) is far to the left.
  No overlap.
- D2.2: q_in = 2000 veh/h drawn as open grey squares in (a) (`mfc="white"`).
- D2.3: legend triangle purple = plotted triangle.
- D2.4: caption restricts the kappa_c-hat range to q2500, says "almost three
  orders of magnitude", "within 2.5 %".
- Numbers vs `kappa_vs_A.json`: omega_0 926.1 / 1157.6 veh/h; peaks 886.0
  (A = 2.25, q2000) and 887.5 (A = 2.5, q2500); A = 10: 865.5 / 844.9;
  largest inflow disagreement for A >= 2: 2.4 % (A = 10); q2000 peak /
  omega_0 = 0.957; kappa_r-hat q2500: 1.7e-5 upper bound at A = 1 (0
  releases) -> 1.56e-2 at A = 10, 3.12e-5 (A = 1.5) -> 4.78e-3 (A = 2.25);
  kappa_c-hat q2500 2.64e-2 – 5.24e-2; chi_s 0.610 -> 0.018 (q2500) and
  0.530 -> 0.003 (q2000). All agree with the caption; "865" corrected to
  "866" (865.51 veh/h), and the "black / grey" colour statement restricted
  to panel (a) (§4 a, b).

### 2.3 Fig 3 / Fig S1 — `fig_heatmaps_q2500` / `_q2000` — READY

D3.1 (G1), D3.2 (t in s), D3.3 (q2000 caption), D3.4 (letters (a)–(f), G4
titles) all done. D3.5 remains optional: `turbo` maps 0–5 veh/km to
near-black, so the un-entered region and the empty road ahead of the CAV
look alike — acceptable. D3.6 handled by numbering q2000 as Fig S1. W1 in
the captions vs `tuned_config[A]['eval']`: 173.74 / 138.85 -> 174 / 139,
190.35 / 165.39 -> 190 / 165 (q2500); 146.54 / 131.57 -> 147 / 132,
122.02 / 145.60 -> 122 / 146 (q2000). The white trajectory kinks at
t = 250 s and 750 s as the caption states; the colourbar reads
"rho [veh/km]", scale 0–90.

### 2.4 Fig 4 — `fig_es` — READY

- D4.1 (G1) done. D4.2: classical series orange dashed with open squares
  (`style["classical"]`, `e11_tune.py:592`). D4.3: the figure now plots
  `ES_ORDER` = 54/2000, 54/2500, 72/2000, 72/2500 in both groups (tick
  labels read so) and the caption lists the same order; the eight values
  per model, re-read from `tuned_config[A]['eval']` in that order, match:
  catch & release +4.9, +7.4, +11.9, +19.4 % (A = 1), −12.8, +4.8, −5.3,
  +8.9 % (A = 10); classical +3.5, +5.8, −24.8, −24.0 % and −7.4, −5.6,
  −28.1, −25.9 %. D4.4: km/h tick labels; e_s in percent here and in Fig 6
  (`PercentFormatter` in both). D4.5: legend above the axes, "catch &
  release".
- Zoom: the group headers "A = 1" / "A = 10" sit at +27 %, above the
  highest min–max shading (+21.8 %); nothing is covered.
- Caption: the sentence "stays within ±10 % at A = 10; the only transfer
  outside the band at A = 10 is …" contradicted itself; reworded (§4 c).

### 2.5 Fig 5 — `fig_profiles` — READY

D5.1 (G1); D5.2 figure-level legend above the panels — the A = 1, t = 700 s
panel (blue peak 63, orange plateau 68 veh/km, rho_c line) is fully
visible; D5.3 "rho_c"; D5.4 legend texts "stuck class s / free class f /
catch & release rho"; D5.6 row-2 ylim 0–80 (the classical plateau no longer
touches the frame). Wake numbers vs `tuned_config`: 58.58 / 68.40 / 54.96 ->
58.6 / 68.4 / 55.0 veh/km (A = 1); 55.37 / 68.40 / 48.68 -> 55.4 / 68.4 /
48.7 (A = 10); window [600, 740] s = `e7_ablation.D1_T`, consistent between
caption and table header. Observation R2 (not a figure defect).

### 2.6 Fig 6 — `fig_transfer` — READY

- D6.1 (G2) closed. D6.2 "classical LWR+MB". D6.3 "calibration
  (u_xi = 54 km/h)". D6.4 y-label "e_s at X_q = 15 km" without "[–]",
  percent ticks. D6.5 top ylim 120–260. D6.6 ticks 40, 50, …, 80; the
  calibration speed is marked by the dotted line only.
- Legends: top panel upper right (above the 209 veh km maximum), bottom
  panel upper left (data there: −16 % and +4 %); nothing covered.
- Numbers vs `transfer.json`: `fit` kappa_c 6.428e-2, kappa_r 4.701e-3
  veh^-1, W1 142.42 veh km at dt = 0.5 s; `A3_sweep_q2500` W1 model 208.9,
  172.0, 146.7, 142.4, 149.9, 157.7, 156.2, 151.9, 148.1; classical 157.9,
  164.0, 167.3, 167.3, 165.0, 156.4, 174.6, 150.1, 150.6; ratios 1.323,
  1.049, 0.876, 0.851, 0.908, 1.008, 0.895, 1.012, 0.984, mean 0.9895; e_s
  model −15.5, −8.1, −0.6, +3.3, +11.3, +16.3, +19.0, +29.4, +31.6 %
  (monotone), classical +3.9, −0.2, −5.2, −8.2, −4.2, −6.3, −28.7, −1.7,
  −5.5 %; in-band speeds {43, 50, 54} km/h (model) and all but 72 km/h
  (classical); flow through X_q: model 2077–2326, classical 845–1791, SUMO
  1924–2085 veh/h; omega_max 1127 -> 254 -> 0 veh/h at 36 / 72 / 86 km/h,
  sigma_xi 24.26 veh/km. Every plotted point reads correctly against the
  axes and the error bars equal the per-run min–max.
- Caption: the sentence attached the classical through-flow numbers
  (1.8e3 -> 0.85–1.3e3 veh/h) to the DM-G bound, whose values are 1.1e3 ->
  0; rewritten with both quantities named (§4 d). Short form: the W1
  statement now follows the ratio table (§4 f). Remarks R1, R5.

### 2.7 `table_metrics.md` — READY

Block-aware programmatic check: 152 numbers of the main and spread tables
(2 A x 4 scenarios x 2 models x (5 + 4) plus the 8 bracketed SUMO wake
values) and 80 numbers of the ablation tables reproduce `tuned_config.json`
to the printed precision; the parameter header lines equal
`final.theta_public` / `capped_reference.theta_public`; "(fit)" is on 54
km/h, 2500 veh/h in both blocks. `review.md` §2.8: `ds_stuck` dropped,
RMSE-vs-per-run convention stated, sign convention stated, units in the
headers, "omega error [%]" — all done. Ablation deltas (no cap − capped):
|dW1| <= 0.94 veh km (A = 10, 72 km/h, 2500 veh/h: 187.44 vs 188.38),
|de_s| <= 0.28 pp, |d omega error| <= 0.66 pp — the Table 1 caption's
"at most 0.9 veh km" corrected to 0.94 (§4 e); "0.3 pp" and "0.7 pp" hold.

## 3. Cross-document consistency

- Caption header parameters = `final.theta_public`: 1.695e-1 -> 1.70e-1,
  7.129e-3 -> 7.13e-3, 0.471 -> 0.47, 0.579 -> 0.58 w; 3.875e-1 -> 3.87e-1,
  1.534e-4 -> 1.53e-4, 0, 0.40 w. Ablation W1 138.97 / 138.85 and 165.28 /
  165.39 veh km (−0.09 %, +0.07 %). Fig 6 "gamma = 1, w_s = 0.6 w" =
  `transfer.json._meta` (ell = a + s; w_s 3.7276 m/s = 0.6 x 6.2126 m/s).
  Classical Q_xi = 2000 veh/h in every file.
- Window constants quoted in captions and table header: W1 / RMSE
  t in [100, 1000] s, x <= 20 km (`e7_wasserstein.T_WIN`, `_meta.objective`);
  e_s from 576 s at X_q = 15 km (`ev4.T0_ES`, `X_Q`); wake [600, 740] s
  (`e7_ablation.D1_T`); Fig 2 window 260–740 s (`kappa_vs_A._meta`);
  analysis grid 100 m x 10 s. All consistent.
- Stale vocabulary in `captions.md` ("tuned", "rep mean", "cap only",
  "E10", "t [min]", bare "m/s" axes): none.
- Model identity: `tuned_config._meta.final_model[A]` and
  `transfer.json._meta.structure` both describe the no-cap structure
  (stuck-class branch w_s + s-impermeable interface + lf capture with
  ell = a + gamma s); Figs 3–6 and Table 1 all show this structure under the
  single name "catch & release".

## 4. Edits made to `captions.md` (wording only)

- (a) Fig 2: "decays slowly to 865 and 845 veh/h" -> "866 and 845" (865.51).
- (b) Fig 2: "q_in = 2500 veh/h (black, filled circles) and 2000 veh/h
  (grey, open squares)" was stated for all panels; panels (b), (c) use the
  coefficient colours / red. Now: filled circles vs open squares in every
  panel, black and grey in (a), coefficient colours in (b), red in (c).
- (c) Fig 4: "stays within ±10 % at A = 10; the only transfer outside the
  band at A = 10 is 54 km/h, 2000 veh/h (−12.8 %)" -> "stays within ±10 %
  at A = 10 except for the 54 km/h, 2000 veh/h transfer (−12.8 %)".
- (d) Fig 6: the DM-G bound now carries its own values (1.1e3 veh/h at
  36 km/h, 0.25e3 at 72, 0 at 86; sigma_xi = 24.3 veh/km) and the classical
  flow through X_q (1.8e3 -> 0.85–1.3e3 veh/h) is named as such; the e_s
  clause is no longer phrased as a consequence of the bound.
- (e) Table 1: "W1 by at most 0.9 veh km" -> "0.94 veh km".
- (f) Fig 6 short form: "W1: model better at 50–72 km/h except 65, worse at
  36–43 km/h" -> "better at 50–58 and 72 km/h, within 2 % of the classical
  value at 65–86 km/h, worse at 36–43 km/h" (ratios 1.008, 1.012, 0.984).

## 5. Residual remarks (non-blocking; for the text, not the figures)

- **R1 (Fig 6 protocol).** Same structure and name as Figs 3–5, but 2 fitted
  parameters at A = 3 (gamma = 1, w_s = 0.6 w fixed) versus 4 per A in
  Table 1. Stated in both captions; the text should say why the A = 3
  anchor keeps the E10 fixed values (a referee will ask whether the
  36–43 km/h excess of W1 (ratios 1.32, 1.05) would shrink with gamma and
  w_s fitted).
- **R2 (Fig 5, t = 850 s).** A narrow blue spike (about 32 veh/km) at the
  nominal CAV position (x about 14.4 km) in both columns: the s-impermeable
  interface still acts 100 s after release. Model behaviour, plainly
  visible; either mention it in the text or note that the interface is
  not switched off after the slow phase.
- **R3 (Figs 3, S1).** `turbo` colormap, see D3.5 — optional.
- **R4 (Fig 2b).** At q_in = 2000 veh/h kappa_c-hat drops to 6–8e-3 veh^-1
  for A >= 4 with CIs reaching about 1e-3; the caption's 2.6–5.2e-2 range is
  (correctly) q2500-only, and the text should not cite a single range.
- **R5 (Fig 6 windows).** e_s uses [576 s, t_xi(X_q)]; the through-flow at
  X_q uses [650 s, min(t_xi(X_q), 750) − 10 s] (`e11_transfer_fd.evaluate`).
  At 79 km/h these are [576, 741] and [650, 731] s, yet the classical e_s
  is −1.7 % while its through-flow is 1283 vs 1924 veh/h (−33 %). The
  caption states both numbers correctly, but the text's "decaying bound"
  explanation should be checked against the e_s integrand before it is
  relied on (the 72 km/h dip of −29 % may be specific to that speed).
- **R6 (Fig 1).** Legend clearance to the queued squares and the congested
  branch is about 1 mm; do not enlarge the legend font or move it left.

## 6. Verdict

All G1–G5 and every per-figure item of `review.md` are closed except the
explicitly optional D3.5. Figures 1–6 and S1, `table_metrics.md` and (after
§4) `captions.md` are ready for the manuscript.
