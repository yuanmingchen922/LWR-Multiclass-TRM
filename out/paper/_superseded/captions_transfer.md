# Captions — transfer tag (`e11_transfer_fd.py`)

Figures: `out/paper/fig_transfer.{png,pdf}`, `out/paper/fig_fd.{png,pdf}`.
Numbers: `out/paper/transfer.json`.

## fig_transfer

**Fig. X. Zero-refit speed transfer.** Mean 1-Wasserstein distance $W_1$
between simulated and measured cumulative density profiles (top; $t \in
[100, 1000]$ s, $x \le 20$ km) and relative cumulative-outflow error $e_s$ at
$X_q = 15$ km (bottom; ECC22 eq. (6) window $[576\,\mathrm{s},\, t_\xi(X_q)]$,
mean over the five SUMO runs, bars = min–max) as functions of the bottleneck
speed $u_\xi$, for assertiveness $A = 3$ and inflow $q_\mathrm{in} = 2500$
veh/h. The catch & release model (blue, solid) is calibrated once, at
$u_\xi = 15$ m/s (dotted line; $\kappa_c = 2.55\cdot10^{-2}$, $\kappa_r =
1.85\cdot10^{-5}$ veh$^{-1}$, $W_1$ fit), and evaluated at all other speeds
with the rate constants frozen; the classical single-class LWR model with the
same Delle Monache–Goatin capacity constraint $Q_\xi = 2000$ veh/h (orange,
dashed) has no fitted parameter. Triangles: the $A = 1$ and $A = 10$ scenarios
at $u_\xi = 20$ m/s predicted with their own $u_\xi = 15$ m/s calibrations
(filled: $q_\mathrm{in} = 2500$ veh/h, open: 2000 veh/h). The transferred
model is never worse than the classical baseline in $W_1$ (ratio 0.94–1.00,
mean 0.98); its gain is confined to slow bottlenecks ($u_\xi \le 15$ m/s),
where queues form, and vanishes above 18 m/s. $e_s$ lies within $\pm 10\,\%$
at every speed except $u_\xi = 20$ m/s ($-29\,\%$ for both models, and
$-24$ to $-28\,\%$ for $A = 1$, $10$): SUMO passes $\approx 2.0\cdot10^{3}$
veh/h past the vehicle at all speeds, whereas the capacity constraint
$\omega \le (Q_\xi - u_\xi\sigma_\xi)_+$ lets the through-flow of both
models decay with $u_\xi$ (from $1.8\cdot10^{3}$ veh/h at 10 m/s to
$0.96\cdot10^{3}$ veh/h at 20 m/s); the outflow error is set by the
constraint, not by catch & release.

Short form: **Fig. X.** $W_1$ (top) and $e_s$ at $X_q = 15$ km (bottom;
$\pm10\,\%$ band, bars = min–max over 5 runs) versus bottleneck speed for
$A = 3$, $q_\mathrm{in} = 2500$ veh/h. Catch & release (solid) calibrated at
$u_\xi = 15$ m/s only and transferred with frozen $\kappa$; classical LWR with
the same capacity constraint (dashed) has no fitted parameter. Triangles:
$A = 1$ / $A = 10$ at $u_\xi = 20$ m/s with their $u_\xi = 15$ m/s
calibrations (filled $q_\mathrm{in} = 2500$, open 2000 veh/h). The dip at
$u_\xi = 20$ m/s is shared by both models and comes from the
speed-dependent capacity constraint $(Q_\xi - u_\xi\sigma_\xi)_+$, which
under-predicts the SUMO through-flow there.

## fig_fd

**Fig. Y. Calibrated fundamental diagram.** Triangular flux function
$Q(\rho) = \min\{v_f\rho,\, w(P - \rho)\}$ with $v_f = 100.6$ km/h, $w = 22.4$
km/h and $P = 267$ veh/km (bumper-to-bumper prior, two lanes), giving the
critical density $\rho_c = 48.5$ veh/km (dotted) and capacity $C = 4879$
veh/h, over the two-lane aggregate $(\rho, q)$ cloud of all $A = 3$
SUMO runs (grey; $6.8\cdot10^{4}$ of the $4.1\cdot10^{5}$ sampled
100 m $\times$ 10 s cells with $\rho > 0.5$ veh/km). Squares: ECC22-style
steady-state averages of the states immediately upstream of the vehicle,
$(\rho_-, q_-)$, over the $u_\xi$ and $q_\mathrm{in}$ sweep (144
run–scenario pairs); circles: the corresponding downstream states
$(\rho_+, q_+)$ (146 pairs). Only the strongly congested
upstream states (filled, $\rho_- \ge 60$ veh/km, $n = 53$) sample the
congested branch and were used to fit $w$; the remaining upstream states
(open) are two-lane mixtures of a queued right lane and an overtaking left
lane and lie inside the triangle, as in [ECC22]. The free-flow branch is a
least-squares slope through the cloud. The dash-dotted line is the
bottleneck capacity $Q_\xi = 2000$ veh/h of the moving-bottleneck
constraint.

Short form: **Fig. Y.** Calibrated triangular fundamental diagram
($v_f = 100.6$ km/h, $w = 22.4$ km/h, $P = 267$ veh/km; $\rho_c = 48.5$
veh/km) over the $A = 3$ SUMO cell cloud (grey), with the steady upstream
$(\rho_-, q_-)$ (squares; filled = queued states $\rho_- \ge 60$ veh/km used
to fit $w$, open = two-lane mixtures) and downstream $(\rho_+, q_+)$
(circles) states of all bottleneck speeds. Dash-dotted: bottleneck capacity
$Q_\xi$.

## Notes for the text (not captions)

- A=3 anchor fit (final structure: cap 2000 + $w_s = 0.6w$ +
  s-impermeable, lf, W1 on the rep-mean field): $\kappa_c = 2.550\cdot10^{-2}$,
  $\kappa_r = 1.854\cdot10^{-5}$; $W_1 = 165.0$ veh km, $\rho$-RMSE 5.83
  veh/km at production resolution (classical: 167.4 / 5.93). Same magnitude
  regime as the $A = 1$ / $A = 10$ fits (4.46e-2 / 3.16e-2).
- $W_1$ model / classical over the sweep: u10 0.943, u12 0.973, u14 0.968,
  u15 0.986, u16 0.987, u18 0.999, u20 0.999, u22 0.999, u24 1.000.
- $e_s$ model (classical): u10 −2.9 % (+3.9 %), u12 −1.5 % (−0.2 %), u14
  −4.9 % (−5.2 %), u15 −8.4 % (−8.2 %), u16 −4.4 % (−4.2 %), u18 −7.4 %
  (−6.3 %), u20 −28.7 % (−28.7 %), u22 −1.9 % (−1.7 %), u24 −5.5 % (−5.5 %).
- Through-flow past the vehicle at $X_q$ (mean flow over $[650\,\mathrm{s},
  \min(t_\xi(X_q), 750) - 10]$), SUMO vs model: u10 2085 vs 1787, u15 2084 vs
  1591, u18 2019 vs 1521, u20 2016 vs 961, u22 1924 vs 1279, u24 2084 vs 844
  veh/h. Model = classical to within 30 veh/h at every speed: the through-flow
  is fixed by the constraint $\omega_{\max} = (Q_\xi - u_\xi\sigma_\xi)_+$,
  $\sigma_\xi = \tfrac12\rho_c = 24.3$ veh/km (solver docstring), whose wake
  state $\hat q = \omega_{\max}(1 + u_\xi/(v_f - u_\xi))$ is 891 veh/h at
  $u_\xi = 20$ m/s — exactly the simulated value. SUMO's through-flow is
  speed-independent here; the constraint's linear decay in $u_\xi$ is the
  structural source of the $u_\xi = 20$ m/s outflow error (and of the
  $-24$…$-28\,\%$ at $A = 1$, $10$, $u_\xi = 20$).
- $A = 1$ / $A = 10$ transfer, model (classical) $W_1$ / $e_s$:
  A1 u20 q2500 184.7 (184.0) / −24.1 % (−24.0 %); A1 u20 q2000 177.2 (175.2)
  / −24.9 % (−24.8 %); A10 u20 q2500 207.6 (208.6) / −26.0 % (−25.9 %);
  A10 u20 q2000 130.4 (130.4) / −28.2 % (−28.1 %). Anchors reproduce
  `out/e10/final_config.json` exactly (A1 u15 q2500 151.96 / +3.9 %; A10
  180.10 / −5.9 %).
