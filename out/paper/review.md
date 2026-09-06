# Paper figure set — quality review (tag `review`)

Scope: the seven PNG/PDF pairs in `out/paper/`, the three caption files
(`captions_sweep.md`, `captions_tune.md`, `captions_transfer.md`),
`table_metrics.md`, and the numbers in `tuned_config.json`,
`out/e4/kappa_vs_A.json`, `transfer.json`, `out/params.json`,
`out/e10/final_config.json`. Producer scripts (`e4_sweep.py`, `e11_tune.py`,
`e11_transfer_fd.py`, `paperfig.py`) were read, not modified. Merged,
renumbered captions: `captions.md` (this directory).

## 0. Summary

| Fig | file | verdict | blocking defects |
|---|---|---|---|
| 1 | fig_fd | **ready** | none (two optional polish items, D1.x) |
| 2 | fig_assertiveness | **needs-fix** | legend in (b) drawn over data; (a) marker fill contradicts caption |
| 3 | fig_heatmaps_q2500 | **needs-fix** | 100 dpi / Type 3 fonts / sans-serif (no `pf.setup()`); t axis in min |
| S1 | fig_heatmaps_q2000 | **needs-fix** | same; caption sentence contradicts the W1 at A = 10 |
| 4 | fig_es | **needs-fix** | same style bug; classical drawn solid but caption says dashed; caption scenario order wrong; km/h vs m/s |
| 5 | fig_profiles | **needs-fix** | same style bug; legend covers the A = 1, t = 700 s queue |
| 6 | fig_transfer | **needs-fix** | different parameter set from Figs 3–5 for the same scenarios (G2); legend labels |
| – | table_metrics.md | numbers OK | cosmetics only (§2.8) |

Numbers: every value in `table_metrics.md` reproduces `tuned_config.json`
(24 metric rows x 7 columns + 24 spread rows, checked programmatically to the
printed precision); the "E10 final" rows equal `out/e10/final_config.json`
and the A1/A10 u15 q2500 rows equal `transfer.json[A1_A10_transfer]`. Every
number quoted in the three caption files was re-derived from its JSON and
agrees. Reading the plotted points against the axes, no plotted value
disagrees with its JSON (details per figure in §2).

## 1. Global defects (fix once, several files affected)

**G1 — `e11_tune.py` never calls `pf.setup()`** (grep: only
`e4_sweep.py:186` and `e11_transfer_fd.py:394` do). Consequences for
`fig_heatmaps_q2500`, `fig_heatmaps_q2000`, `fig_es`, `fig_profiles`:

- PNGs written at 100 dpi (fig_es 341x240 px, heatmaps 711x441 px,
  profiles 690x530 px) instead of 300 dpi (the other three are 2077x681,
  996x727, 1028x1148) — unusable for submission/review PDFs;
- DejaVu Sans instead of the serif family used by Figs 1, 2, 6;
- PDFs embed **Type 3** fonts (`fig_es.pdf`, `fig_heatmaps_*.pdf`,
  `fig_profiles.pdf`) instead of TrueType (`pdf.fonttype = 42`) — Type 3 is
  rejected by many production systems;
- legends framed, top/right spines on, grid off, legend/tick sizes at
  matplotlib defaults.

Fix: add `pf.setup()` at the top of `make_outputs()` in `e11_tune.py` (or in
`main()` before `run()` / `figures_only()`), then
`python3 e11_tune.py --figures`. Re-check afterwards that the 6.5 pt two-line
tick labels of fig_es and the 6.5 pt legend of fig_profiles are still legible
(8.5/7.5 pt everywhere else).

**G2 — Two different "the model"s in the set.** Figs 3–5 show the E11
*tuned* instantiation (five parameters per A; A = 1: kappa_c 1.56e-1,
kappa_r 6.84e-3, gamma 0.51, w_s 0.58 w, Q_xi 2512 veh/h; A = 10: kappa_c
3.80e-1, kappa_r 8.70e-5, gamma 0, w_s 0.40 w, Q_xi 2290 veh/h). Fig 6 shows
the E10 fixed structure (Q_xi 2000 veh/h, w_s 0.6 w, gamma 1, two fitted
kappas) — for the A = 3 sweep *and* for the A = 1 / A = 10 triangles
(`transfer.json[A1_A10_transfer]` takes the kappas from
`out/e10/final_config.json`). For the same four scenarios the reader is
shown contradictory numbers:

| scenario | Fig 4 (tuned) e_s | Fig 6 (E10) e_s | Fig 4 W1 | Fig 6 W1 |
|---|---|---|---|---|
| A1 u20 q2500 | +19.5 % | −24.1 % | 154.8 | 184.7 |
| A1 u20 q2000 | +12.1 % | −24.9 % | 139.4 | 177.2 |
| A10 u20 q2500 | +8.7 % | −26.0 % | 187.4 | 207.6 |
| A10 u20 q2000 | −5.3 % | −28.2 % | 109.4 | 130.4 |

Options: (a) recompute the triangles in `e11_transfer_fd.py` with the tuned
sets (`tuned_config.json[A]['tuned']['theta_public']` gives kappa_c, kappa_r,
gamma, w_s_ms, Q_xi_vehh; pass `q_xi_max=Q_xi/3600, w_s=w_s_ms, gamma=gamma,
s_impermeable=True` as `extra_cfg`) and say in the caption that the A = 3
curve uses the fixed structure; (b) drop the triangles and keep Fig 6 as the
A = 3 fixed-structure transfer only; (c) keep everything and state the
parameter set in both captions (done in `captions.md`), accepting that a
referee will ask why "the model" gives +19.5 % and −24 % for one scenario.
Recommendation: (b), or (a) if the tuned set is the paper's headline model.
Related: the tuned kappa_c (0.16, 0.38 veh^-1) are 5–12x the event-measured
kappa_c-hat of Fig 2(b) (2.6–5.2e-2 veh^-1 at q_in = 2500); Fig 2's caption
must not imply that the plotted kappa-hats are the coefficients used in
Figs 3–5 (the old caption did not, but the "micro-macro link" framing invites
that reading — say "measured" in the caption and discuss the gap in the text).

**G3 — Units.** Bottleneck speed: Fig 6 axis and the Fig 2 caption in m/s;
Fig 4 tick labels and all `captions_tune.md` captions in km/h (54/72); data
files, scripts and README use u15/u20. Time: the heatmaps' axis is in minutes
(0–16.7), every other figure and every caption is in seconds (slow phase
250–750 s). Choose m/s and s everywhere (data grid and Fig 6 are already in
m/s; one "15 m/s (54 km/h)" in the text suffices). Code changes: `fig_es`
tick labels `f"{uc:g}\n{qin:.0f}"` and x-label
`$u_\xi$ [m/s] / $q_{in}$ [veh/h]`; `fig_heatmaps` extent/plot with `tt`
instead of `tt/60` and `t [s]` (ticks 0, 250, 500, 750, 1000).

**G4 — Legend vocabulary** for the same three series differs between
figures: "SUMO (rep mean)" (Figs 3, 5); "classical LWR+MB" (3, 4, 5) vs
"classical (cap only)" (6); "tuned model" (3), "tuned" (4), "tuned: total
rho" (5), "catch & release" (6). Use one set everywhere: **"SUMO (5-run
mean)", "classical LWR+MB", "catch & release"** (`MODEL_LABEL`,
`e11_tune.py:79`, the literals in `fig_profiles`, and the `label=` strings in
`e11_transfer_fd.py::fig_transfer`).

**G5 (minor) — red/green pairs.** Fig 2(b) (kappa_c-hat / kappa_r-hat
lines), Fig 1 (upstream / downstream markers), Fig 5 (stuck / free fills).
Figs 1 and 5 remain legible in greyscale (marker shape, stacking order), but
a deuteranope cannot separate the two *lines* in Fig 2(b). Optional: draw
kappa_r-hat in blue or purple.

## 2. Per-figure review

### 2.1 Fig 1 — `fig_fd` (`e11_transfer_fd.py::fig_fd`) — READY

Checks passed: axis labels "density rho [veh/km]" / "flow q [veh/h]";
legend inside, five entries covering every series; rho_c dotted with label,
Q_xi dash-dotted with label; 3.4 x 2.5 in at 300 dpi, serif, TrueType fonts.
Values: v_f 100.6 km/h, w 22.4 km/h, P 266.7 -> 267 veh/km, rho_c 48.5
veh/km, C 4879 veh/h = `params.json`; caption counts 6.8e4 / 4.1e5 cells
(67 944 / 407 667), 144 upstream / 53 queued / 146 downstream =
`transfer.json[fd_figure]`. The congested branch (zorder 6) passes about 1 mm
left of the legend's "downstream" marker — clear, but do not shrink the
legend further.

Optional polish: D1.1 legend entry "Q(rho) calibrated" -> "Q(rho)" (the
caption says calibrated); D1.2 the on-plot "Q_xi = 2000 veh/h" text could be
just "Q_xi" with the value in the caption; D1.3 the FD line is drawn in the
"data" black — fine (it is the road, not a model), but keep it out of the
model-blue.

### 2.2 Fig 2 — `fig_assertiveness` (`e4_sweep.py`) — NEEDS-FIX

Checks passed: 7.0 x 2.35 in, 300 dpi, serif, TrueType; log x-axis with
ticks 1, 2, 3, 5, 10 and "assertiveness A [-]" on all panels; y-labels with
units; panel letters; legends inside. Values vs `kappa_vs_A.json`: omega_0
926 / 1158 veh/h; peaks 886.0 (A = 2.25, q2000) and 887.5 (A = 2.5, q2500);
A = 10: 865.5 / 844.9; A = 1 q2500 kappa_r upper bound 1.71e-5, A = 10
1.56e-2; A = 1.5 -> 2.25: 3.12e-5 -> 4.78e-3; kappa_c range at q2500
2.64e-2 – 5.24e-2; chi_s 0.610 -> 0.018 and 0.530 -> 0.003. Plotted points
read correctly against the table.

- **D2.1 (blocking)** Panel (b) legend (`loc="lower right"`, three long
  entries) extends left to A ~ 1.05 and is drawn over data: the q2000
  kappa_r-hat points at A = 1 and 1.5 with their CI whiskers, the segment
  between them, and the A = 1 censored triangle; the legend's own "v" handle
  sits exactly where the real censored point is (A = 1, 1.7e-5). Fix:
  shorten to "kappa_c-hat", "kappa_r-hat", "upper bound" and anchor the
  legend in the empty region A in [3, 10], kappa in [1e-5, 1e-4]
  (`loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=6.5`); confirm
  it does not touch the lower CI whiskers at A = 2–2.5 (q2000, ~7.8e-3).
- **D2.2** Panel (a) draws q_in = 2000 veh/h as *filled* grey squares (no
  `mfc="white"`), panels (b), (c) as open squares; the caption says "open
  squares". Add `mfc="white"` in (a) (line + shading stay grey).
- **D2.3** The legend's censored-point handle is dark grey (`color="0.3"`)
  while the plotted censored triangles are in the coefficient colour
  (green). Use the same colour in the legend.
- **D2.4 (caption)** "kappa_c-hat stays within [2.6e-2, 5.2e-2] over the
  whole sweep" holds for q_in = 2500 only (at q2000 it falls to 6.3e-3 for
  A >= 4, with wide CIs); "more than 911-fold" -> "by almost three orders of
  magnitude"; "to within 2 %" is 2.4 % (max over A >= 2) -> "2.5 %". All
  three fixed in `captions.md`.
- D2.5 (minor) The dashed omega_0 reference lines share the dashed style
  reserved for the classical model; acceptable (ECC22 does the same, no model
  is shown here) as long as the caption says "dashed" — it does.

### 2.3 Fig 3 / Fig S1 — `fig_heatmaps_q2500`, `fig_heatmaps_q2000` (`e11_tune.py::fig_heatmaps`) — NEEDS-FIX

Checks passed: 2 x 3 panels, shared colour scale 0–90 veh/km with labelled
colourbar; "x [km]" on the bottom row and "t [min]" on the left column;
titles one line; white CAV trajectory in every panel; caption W1 values =
table (173.7 / 139.0, 190.4 / 165.3; 146.5 / 130.9, 122.0 / 146.3).

- **D3.1 (blocking)** G1: 711 x 441 px at 100 dpi, sans-serif, Type 3 PDF.
- **D3.2** Time axis in minutes while the caption (slow phase 250–750 s) and
  all other figures are in seconds -> G3.
- **D3.3 (caption, q2000)** "at A = 10 it produces the broad 'slow but not
  stuck' band that the classical model cannot express" is boilerplate copied
  from the q2500 caption; at q2000, A = 10 the tuned model is *worse* than
  the classical one (W1 146 vs 122 veh km) and its band is visibly broader
  than SUMO's. Reworded in `captions.md`.
- **D3.4** Titles "SUMO (rep mean) / classical LWR+MB / tuned model" -> G4
  vocabulary; add panel letters (a)–(f) so the caption can point at panels.
- D3.5 (minor) `turbo` maps 0–5 veh/km to near-black, so the empty road
  ahead of the CAV and the un-entered region look identical; `viridis` or a
  small negative vmin separates them. Optional.
- D3.6 (minor) Two main-text figures with identical layout (q2500, q2000)
  are unlikely to survive review; either make the q2000 one supplementary
  (numbered S1 in `captions.md`) or build one 4 x 3 composite.

### 2.4 Fig 4 — `fig_es` (`e11_tune.py::fig_es`) — NEEDS-FIX

Checks passed: all 16 means and 16 min–max bands agree with the table when
read against the grid (+0.07 / +0.05 / +0.12 / +0.20 and −0.06 / −0.07 /
−0.28 / −0.26, etc.); ±10 % band and zero line; "A = 1 / A = 10" group
headers; y-label with unit; 3.4 x 2.5 in.

- **D4.1 (blocking)** G1: 341 x 240 px at 100 dpi, Type 3 PDF.
- **D4.2 (blocking)** The classical series is drawn *solid* with filled
  circles (`ls="-"` for both models, `e11_tune.py` fig_es loop) while the
  colour convention and the caption say orange dashed. Use `ls=pf.LS[m]` and
  the open-square marker used in Fig 6 for the classical series.
- **D4.3 (blocking, caption)** The caption lists the within-group order as
  q_in = 2000, 2500, 2000, 2500; the figure (`SCENARIOS`) is 2500, 2000,
  2000, 2500. The eight quoted e_s values are in figure order, only the
  sentence is wrong. Fixed in `captions.md`.
- **D4.4** Tick labels 54 / 72 km/h vs Fig 6's m/s axis -> G3. Also e_s is a
  fraction here (0.2) and a percentage in Fig 6 (20 %); use one format.
- **D4.5** Legend label "tuned" -> "catch & release" (G4). The legend
  occupies the lower 20 % of the axes (ymin −0.45 for data reaching −0.30);
  with `frameon=False` after G1, `ymin = lo_all − 0.08` still fits `ncol=3`.
- D4.6 (minor) Lines connect categorical scenarios; ECC22 Fig 4 does the
  same, so acceptable — but markers with min–max bars (as in Fig 6) would
  give the same quantity the same encoding in both figures.

### 2.5 Fig 5 — `fig_profiles` (`e11_tune.py::fig_profiles`) — NEEDS-FIX

Checks passed: 3 x 2 panels, shared x, y per row; "rho [veh/km]" and
"x [km]"; colours exactly per convention (data black solid, classical orange
dashed, model blue solid, stuck red fill 0.3, free green fill 0.2, stacked);
rho_crit dashed, CAV dotted; one-line panel texts; caption wake numbers =
table (58.6 / 68.4 / 55.3; 55.4 / 68.4 / 48.4 veh/km); 7.0 x 5.4 in.

- **D5.1 (blocking)** G1.
- **D5.2 (blocking)** The single legend (`axes[1,0]`, upper left, bbox
  (0, 0.88), framed, two columns) spans x ~ 0.3–9.8 km, rho ~ 37–63 veh/km
  in the A = 1, t = 700 s panel and covers the top of the modelled queue
  (blue peak 63 veh/km at x ~ 7.1 km, orange plateau at 68 veh/km) and the
  rho_c line; the semi-opaque frame partly hides them. Move it out of the
  data: `fig.legend(handles, labels, loc="upper center", ncol=5,
  frameon=False, bbox_to_anchor=(0.5, 1.0))` with
  `fig.tight_layout(rect=[0, 0, 1, 0.95])`.
- **D5.3** "rho_crit" label -> "rho_c" (Fig 1 uses rho_c).
- **D5.4** Legend texts "tuned: stuck s / tuned: free f / tuned: total rho"
  -> "stuck class s / free class f / catch & release rho" (G4).
- D5.5 (minor) The dotted CAV line is the tuned model's `x_cav`; it is the
  prescribed trajectory, so it also holds for SUMO — say "nominal position"
  in the caption (done).
- D5.6 (minor) Row 2: the classical plateau (68 veh/km) touches the top of
  the axes (ylim ~ 70); set the row-2 ylim to 0–75.

### 2.6 Fig 6 — `fig_transfer` (`e11_transfer_fd.py::fig_transfer`) — NEEDS-FIX

Checks passed: 3.4 x 3.9 in, 300 dpi, serif, TrueType; "W_1 [veh km]", "e_s
at X_q = 15 km" with percent ticks; "bottleneck speed u_xi [m/s]" with ticks
at the nine sweep speeds; model blue solid filled circles, classical orange
dashed open squares; ±10 % band, zero line, dotted fit-speed line; legends
inside. Values vs `transfer.json[A3_sweep_q2500]`: W1 model 148.9, 159.5,
161.9, 165.0, 162.8, 156.3, 174.5, 149.9, 150.6; classical 157.9, 164.0,
167.3, 167.3, 165.0, 156.4, 174.6, 150.1, 150.6; e_s model −2.9, −1.5, −4.9,
−8.4, −4.4, −7.4, −28.7, −1.9, −5.5 %; classical +3.9, −0.2, −5.2, −8.2,
−4.2, −6.3, −28.7, −1.7, −5.5 %; triangles A1 184.7 (filled) / 177.2 (open),
A10 207.6 / 130.4, e_s −24.1 / −24.9, −26.0 / −28.2 %; caption ratios
0.94–1.00, mean 0.98 (0.9837), "vanishes above 18 m/s" (0.999 at 18); fit
kappas 2.55e-2 / 1.85e-5. All agree.

- **D6.1 (blocking)** G2: the triangles and the caption's "−24 to −28 %"
  describe a different parameter set from Figs 3–5 for the same scenarios.
- **D6.2** Legend "classical (cap only)" -> "classical LWR+MB"; keep "catch
  & release" (G4).
- **D6.3** Annotation "fit (u_xi = 15)" lacks the unit -> "calibration
  (u_xi = 15 m/s)".
- D6.4 (minor) y-label "e_s ... [–]" together with percent ticks: drop the
  "[–]" or the PercentFormatter, and do the same in Fig 4.
- D6.5 (minor) Top panel ylim starts at 0, so the curves use the upper third
  of the panel; ylim (120, 220) would make the 0.94–1.00 ratio visible.
  Either is defensible.
- D6.6 (minor) Ticks 14 / 15 / 16 nearly touch at 3.4 in; mark 15 by the
  dotted line only and use ticks 10, 12, ..., 24.
- D6.7 (notes, not caption) `captions_transfer.md` "Notes" claim the u_xi =
  20 m/s emitted wake state 891 veh/h is "exactly the simulated value";
  `transfer.json[headline]` gives 961 (model) / 963 (classical) veh/h — 8 %
  off. Write "to within 10 %" or recompute sigma_xi.

### 2.7 `captions_*.md` — merged into `captions.md`

Numbering as requested: Fig 1 FD, Fig 2 assertiveness, Fig 3 heatmaps
(q2500; q2000 as Fig S1), Fig 4 e_s, Fig 5 profiles, Fig 6 transfer.
Captions describe the figures *after* the fixes above (dashed classical in
Fig 4, m/s tick labels, unified labels, s on the heatmap axis); everything
else is unchanged in substance. Corrections made relative to the source
captions: Fig 4 scenario order (D4.3); Fig S1 A = 10 claim (D3.3); Fig 2
q2500-only range, "911-fold", "2 %" (D2.4); Fig 6 "decays ... to 0.96e3 at
20 m/s" (the through-flow is not monotone: 1279 veh/h at 22, 844 at 24 m/s —
now stated as "0.84–0.96e3 veh/h for u_xi >= 20 m/s"); parameter set named
in every model caption (G2). "ECC22" replaced by a citation placeholder
[ECC22].

### 2.8 `table_metrics.md` — numbers OK

All 48 rows and both parameter headers reproduce `tuned_config.json`;
"E10 final" rows = `out/e10/final_config.json`; cap-activity notes =
`tuned_config.json[A]['tuned']['cap_activity']` (1202 vs 1158 veh/h, W1
139.0; 980 vs 1158, 0.50 veh/km, 165.4). For the paper table: drop
`ds_stuck` (0.0000 in all 24 rows); say that "RMSE" is against the rep-mean
field while "rho_rmse min/max" are per-rep values (they differ: 6.86 vs
6.35–7.62); give the sign convention of e_s and omega err (+ = model passes
more flow than SUMO); put units in the headers (e_s [%]); rename "omega err"
-> "omega error [%]".

## 3. Fix list in execution order

1. `e11_tune.py`: `pf.setup()` in `make_outputs()`; `ls=pf.LS[m]` + open
   square for classical in `fig_es`; m/s tick labels + x-label; `MODEL_LABEL`
   and profile legend texts per G4; profile legend as `fig.legend` above the
   panels; rho_c label; heatmap axis in s; panel letters (a)–(f) in the
   heatmaps; then `python3 e11_tune.py --figures`.
2. `e4_sweep.py`: shorten panel-(b) legend labels and re-anchor; `mfc="white"`
   for q2000 in panel (a); legend triangle in the coefficient colour; rerun
   (`python3 e4_sweep.py`, classification is cached).
3. `e11_transfer_fd.py`: decide G2 (drop or recompute the triangles);
   relabel "classical (cap only)"; "calibration (u_xi = 15 m/s)"; rerun
   (`--skip-fd`, anchor fit is cached).
4. Replace the three caption files by `captions.md`; update the paper table
   per §2.8.
