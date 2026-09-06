# Figure captions (paper set; tag `final`)

Every number below is re-derived from `tuned_config.json` (Figs 3–5,
Table 1), `transfer.json` (Figs 1, 6) and `out/e4/kappa_vs_A.json` (Fig 2)
as regenerated after the no-cap ablation. Figure files: `fig_*.png`
(300 dpi) and `fig_*.pdf` (TrueType fonts) in this directory.

**Conventions** (all figures). Bottleneck speed $u_\xi$ in km/h (54 km/h =
15 m/s, 72 km/h = 20 m/s), time in s, densities in veh/km (two-lane
aggregate), flows in veh/h, $W_1$ in veh km. Series labels and styles:
"SUMO (5-run mean)" (black, solid), "classical LWR+MB" (orange, dashed, open
squares), "catch & release" (blue, solid, filled circles). Classical LWR+MB
= scalar LWR with the Delle Monache–Goatin moving-bottleneck constraint,
capacity $Q_\xi = 2000$ veh/h, no fitted parameter.

**The catch & release model (final form).** Two-class LWR with free ($f$)
and stuck ($s$) vehicles behind the controlled vehicle ($a$): capture
$J_c = \kappa_c (a + \gamma s) f\,\Delta v$, release
$J_r = \kappa_r (P - \rho)\, s\,\Delta v$, a stuck-class congested branch
with wave speed $w_s$, and an $s$-impermeable bottleneck interface (only free
vehicles overtake). **No capacity cap** is imposed at the bottleneck. Figs
3–5 use, per assertiveness, the four parameters $(\kappa_c, \kappa_r, \gamma,
w_s)$ fitted by the 1-Wasserstein distance $W_1$ on the $u_\xi = 54$ km/h,
$q_\mathrm{in} = 2500$ veh/h scenario (Table 1): A = 1: $\kappa_c =
1.70\times10^{-1}$, $\kappa_r = 7.13\times10^{-3}$ veh$^{-1}$, $\gamma =
0.47$, $w_s = 0.58\,w$; A = 10: $\kappa_c = 3.87\times10^{-1}$, $\kappa_r =
1.53\times10^{-4}$ veh$^{-1}$, $\gamma = 0$, $w_s = 0.40\,w$. Fig 6 uses the
A = 3 anchor form of the same structure with $\gamma = 1$ and $w_s = 0.6\,w$
fixed and only $(\kappa_c, \kappa_r)$ fitted. (Adding the DM-G capacity cap
$Q_\xi$ as a fifth fitted parameter changes the fit-scenario $W_1$ by less
than 0.1 % for A = 1 and A = 10 — 138.97 vs 138.85 and 165.28 vs 165.39
veh km — so the cap was dropped; the capped sets are kept in
`tuned_config.json[A]['capped_reference']` and in Table 1's ablation block.)
[ECC22] = Krook, Cicic & Johansson, ECC 2022.

---

**Fig. 1. Calibrated fundamental diagram** (`fig_fd`). Triangular flux
function $Q(\rho) = \min\{v_f\rho,\, w(P-\rho)\}$ with $v_f = 100.6$ km/h,
$w = 22.4$ km/h and $P = 267$ veh/km (bumper-to-bumper prior, two lanes),
critical density $\rho_c = 48.5$ veh/km (dotted) and capacity 4879 veh/h,
over the $(\rho, q)$ cloud of all $A = 3$ SUMO runs (grey; $6.8\times10^4$
of the $4.1\times10^5$ cells of 100 m $\times$ 10 s with $\rho > 0.5$
veh/km). Squares: steady-state averages immediately upstream of the
controlled vehicle, $(\rho_-, q_-)$, over the $u_\xi$ and $q_\mathrm{in}$
sweep (144 run–scenario pairs); circles: the corresponding downstream states
$(\rho_+, q_+)$ (146 pairs). Only the queued upstream states (filled,
$\rho_- \ge 60$ veh/km, $n = 53$) sample the congested branch and were used
to fit $w$; the open squares are two-lane mixtures of a queued right lane
and an overtaking left lane and lie inside the triangle, as in [ECC22]. The
free-flow branch is a least-squares slope through the cloud. Dash-dotted:
$Q_\xi = 2000$ veh/h, the capacity of the moving-bottleneck constraint of
the classical LWR+MB model.

**Fig. 2. Micro–macro link: lane-change assertiveness and the measured
catch & release quantities** (`fig_assertiveness`). All panels: $u_\xi = 54$
km/h, analysis window 260–740 s inside the slow phase, five SUMO runs per
point, $A$ (SUMO `lcAssertive`) on a logarithmic axis; $q_\mathrm{in} =
2500$ veh/h (black, filled circles) and 2000 veh/h (grey, open squares).
(a) Overtaking flow $\omega_\xi$ past the controlled vehicle (mean over runs,
shading min–max), the analogue of Fig. 1 in [ECC22]; dashed: free-flow
reference $\omega_0 = q_\mathrm{in}(1 - u_\xi/v_f) = 926$ and 1158 veh/h.
$\omega_\xi$ rises steeply from $A = 1$ to 2, peaks at 886 veh/h
($A = 2.25$, $q_\mathrm{in} = 2000$) and 888 veh/h ($A = 2.5$, 2500), and
decays slowly to 865 and 845 veh/h at $A = 10$; for $A \ge 2$ the two inflows
agree to within 2.5 %, i.e. the bottleneck is capacity- rather than
demand-limited, while at $q_\mathrm{in} = 2000$ veh/h the peak reaches 96 %
of $\omega_0$. (b) Poisson-exposure maximum-likelihood estimates of the
measured capture coefficient $\hat\kappa_c$ ($\ell f$ form, red) and release
coefficient $\hat\kappa_r$ (purple), pooled over the five runs with the model
speed difference $\Delta v(\rho)$; bars: 95 % confidence intervals;
triangles: rule-of-three upper bounds where no release was observed. At
$q_\mathrm{in} = 2500$ veh/h, $\hat\kappa_r$ rises from an upper bound of
$1.7\times10^{-5}$ veh$^{-1}$ at $A = 1$ (no release in five runs) to
$1.6\times10^{-2}$ veh$^{-1}$ at $A = 10$ — almost three orders of magnitude,
most steeply between $A = 1.5$ and 2.25 ($3.1\times10^{-5}$ to
$4.8\times10^{-3}$ veh$^{-1}$) — whereas $\hat\kappa_c$ stays within
$2.6$–$5.2\times10^{-2}$ veh$^{-1}$. (c) Synchronized occupancy $\chi_s$, the
fraction of vehicle-hours in the queue zone spent in the caught state,
falls from 0.61 ($A = 1$) to 0.02 ($A = 10$) at $q_\mathrm{in} = 2500$ veh/h
and from 0.53 to 0.00 at 2000 veh/h, mirroring the $\hat\kappa_r$ transition
in (b). (The measured $\hat\kappa_c$ are event counts normalised by the
model exposure; they are not the coefficients used in Figs 3–6, which are
field-calibrated — see the text.)

**Fig. 3. Density fields, calibration scenario** (`fig_heatmaps_q2500`).
Total density $\rho(x,t)$ on the measurement grid (100 m $\times$ 10 s) for
$u_\xi = 54$ km/h, $q_\mathrm{in} = 2500$ veh/h at $A = 1$ (a–c) and $A = 10$
(d–f): SUMO 5-run mean (a, d), classical LWR+MB (b, e), catch & release
with the parameters of Table 1 (c, f); white line: controlled-vehicle
trajectory (slow phase 250–750 s). Common colour scale 0–90 veh/km. $W_1$ =
mean 1-Wasserstein distance between simulated and measured cumulative
density profiles ($t \in [100, 1000]$ s, $x \le 20$ km), classical / catch &
release: 174 / 139 veh km ($A = 1$), 190 / 165 veh km ($A = 10$). At $A = 1$
the model reproduces the dense, slowly growing queue and its release fan; at
$A = 10$ it produces the broad "slow but not stuck" band that a single-class
model cannot represent.

**Fig. S1. Density fields, zero-refit transfer to $q_\mathrm{in} = 2000$
veh/h** (`fig_heatmaps_q2000`). As Fig. 3 with the same parameters,
$u_\xi = 54$ km/h. $W_1$ classical / catch & release: 147 / 132 veh km
($A = 1$), 122 / 146 veh km ($A = 10$). The transfer improves on the
classical model at $A = 1$; at $A = 10$ the calibrated capture rate makes the
model band broader than the SUMO one and $W_1$ exceeds the classical value.

**Fig. 4. Relative cumulative-flow error** (`fig_es`). $e_s$, the relative
error of the cumulative flow through $X_q = 15$ km over the window
$[576\ \mathrm{s},\, t_\xi(X_q)]$ (eq. (6) of [ECC22]; $e_s > 0$: the model
passes more flow than SUMO), for the eight core scenarios grouped by
assertiveness; within each group $u_\xi/q_\mathrm{in} = 54/2000$,
$54/2500$ (calibration), $72/2000$, $72/2500$ (km/h, veh/h). Lines: mean
over the five SUMO runs; shading: min–max over runs; orange dashed, open
squares: classical LWR+MB; blue, filled circles: catch & release (Table 1);
grey band: $\pm10$ %. Catch & release: $+4.9$, $+7.4$, $+11.9$, $+19.4$ %
($A = 1$) and $-12.8$, $+4.8$, $-5.3$, $+8.9$ % ($A = 10$); classical:
$+3.5$, $+5.8$, $-24.8$, $-24.0$ % and $-7.4$, $-5.6$, $-28.1$, $-25.9$ %.
At 72 km/h the classical constraint under-passes by a quarter in every
scenario, whereas catch & release over-passes by 12–19 % at $A = 1$ and
stays within $\pm10$ % at $A = 10$; the only transfer outside the band
at $A = 10$ is 54 km/h, 2000 veh/h ($-12.8$ %).

**Fig. 5. Density profiles** (`fig_profiles`), $u_\xi = 54$ km/h,
$q_\mathrm{in} = 2500$ veh/h, at $t = 500$ and 700 s (slow phase) and 850 s
(100 s after release), $A = 1$ (left) and $A = 10$ (right): SUMO 5-run mean
(black), classical LWR+MB (orange dashed) and catch & release total density
(blue) with its stuck ($s$, red) and free ($f$, green) classes stacked; grey
dashed: $\rho_c = 48.5$ veh/km; dotted vertical: nominal position of the
controlled vehicle. Mean wake density 0.2–1 km behind the vehicle,
$t \in [600, 740]$ s, SUMO / classical / catch & release: 58.6 / 68.4 / 55.0
veh/km ($A = 1$), 55.4 / 68.4 / 48.7 veh/km ($A = 10$). The classical model
queues at the congested-branch density and drains as a kinematic wave; the
two-class model keeps the aggregate state inside the flux triangle (stuck
and free streams coexist) and releases the platoon as a dispersing front.

**Fig. 6. Zero-refit transfer across bottleneck speed** (`fig_transfer`),
$A = 3$, $q_\mathrm{in} = 2500$ veh/h. Top: $W_1$ (as in Fig. 3); bottom:
$e_s$ at $X_q = 15$ km (as in Fig. 4; mean over five runs, bars min–max, grey
band $\pm10$ %). Catch & release in the A = 3 anchor form (no capacity cap,
$w_s = 0.6\,w$, $\gamma = 1$) is calibrated once at $u_\xi = 54$ km/h
(dotted line; $\kappa_c = 6.43\times10^{-2}$, $\kappa_r = 4.70\times10^{-3}$
veh$^{-1}$, $W_1$ fit, 142 veh km) and evaluated at eight other speeds
(36–86 km/h) with the rate constants frozen; classical LWR+MB with
$Q_\xi = 2000$ veh/h has no fitted parameter. In $W_1$ the transferred model
is below the classical one at 50–58 km/h (ratio 0.85–0.91) and at 72 km/h
(0.90), within $\pm2$ % of it at 65, 79 and 86 km/h, and above it at 36 km/h
(1.32) and 43 km/h (1.05); mean ratio 0.99. The two models fail $e_s$ in
opposite directions: the classical through-flow past the vehicle is bounded
by $(Q_\xi - u_\xi\sigma_\xi)_+$, which decays with $u_\xi$ (from
$1.8\times10^{3}$ veh/h at 36 km/h to $0.85$–$1.3\times10^{3}$ veh/h for
$u_\xi \ge 72$ km/h), giving $e_s$ within $\pm10$ % everywhere except
72 km/h ($-29$ %); catch & release without a cap passes
$2.1$–$2.3\times10^{3}$ veh/h, rising with $u_\xi$, so $e_s$ increases
monotonically from $-15$ % at 36 km/h to $+32$ % at 86 km/h ($+19$ % at
72 km/h) and is within $\pm10$ % only for 43–54 km/h. SUMO passes
$1.9$–$2.1\times10^{3}$ veh/h at every speed; neither the decaying
constraint nor the uncapped interface reproduces that speed-independent
through-flow.

**Table 1. Calibration and zero-refit transfer, A = 1 and A = 10**
(`table_metrics.md`). Classical LWR+MB versus catch & release (four fitted
parameters per A, listed in the section headers) on the calibration
scenario (54 km/h, 2500 veh/h) and the three zero-refit transfers: $W_1$ and
density RMSE against the SUMO 5-run-mean field ($t \in [100, 1000]$ s,
$x \le 20$ km), $e_s$ and the relative cumulative overtaking-flow error over
the slow window (means of the per-run values; positive = the model passes
more flow than SUMO), and the mean wake density 0.2–1 km behind the vehicle
($t \in [600, 740]$ s; SUMO value in brackets). The spread tables give the
per-run min–max of $e_s$ and of the per-run density RMSE (which is why the
latter differ from the RMSE column). The ablation block compares the
adopted four-parameter sets with the five-parameter capped sets: the
fit-scenario $W_1$ differs by $-0.09$ % ($A = 1$) and $+0.07$ % ($A = 10$),
and, over the eight scenarios, $W_1$ by at most 0.9 veh km, $e_s$ by at
most 0.3 pp and the overtaking-flow error by at most 0.7 pp.

---

Short forms (for a two-line caption budget):

**Fig. 1.** Calibrated triangular fundamental diagram ($v_f = 100.6$ km/h,
$w = 22.4$ km/h, $P = 267$ veh/km, $\rho_c = 48.5$ veh/km) over the $A = 3$
SUMO cell cloud (grey), with steady upstream $(\rho_-, q_-)$ (squares; filled
= queued states used to fit $w$) and downstream $(\rho_+, q_+)$ (circles)
states of all bottleneck speeds. Dash-dotted: bottleneck capacity $Q_\xi$
of the classical model.

**Fig. 4.** $e_s$ at $X_q = 15$ km (mean and min–max over five runs,
$\pm10$ % band) for the eight core scenarios; classical LWR+MB (dashed) and
catch & release (solid, Table 1 parameters, calibrated on 54 km/h /
2500 veh/h only). The classical constraint under-passes by $\approx25$ % at
72 km/h; catch & release stays within $+5$ to $+19$ % ($A = 1$) and $-13$ to
$+9$ % ($A = 10$).

**Fig. 6.** $W_1$ (top) and $e_s$ at $X_q = 15$ km (bottom; $\pm10$ % band,
bars min–max over five runs) versus bottleneck speed for $A = 3$,
$q_\mathrm{in} = 2500$ veh/h. Catch & release (no cap, solid) calibrated at
54 km/h only and transferred with frozen $\kappa$; classical LWR+MB
(dashed) has no fitted parameter. $W_1$: model better at 50–72 km/h except
65, worse at 36–43 km/h. $e_s$: the classical bound under-passes at 72 km/h
($-29$ %), the uncapped model over-passes increasingly with speed
($+11$ % at 58 to $+32$ % at 86 km/h); SUMO's through-flow is
speed-independent ($\approx 2.0\times10^{3}$ veh/h).
