# Paper table: classical LWR+MB vs catch & release (adopted final set)

Zero-refit evaluation at dt = 0.5 s. The catch & release parameters are fitted per A on u_xi = 54 km/h (15 m/s), q_in = 2500 veh/h only; every other scenario is a zero-refit transfer. Classical LWR+MB: scalar LWR + Delle Monache-Goatin moving bottleneck, Q_xi = 2000 veh/h, no fitted parameter.

Columns: W1 = mean 1-Wasserstein distance between simulated and measured cumulative density profiles [veh km] and RMSE = density RMSE [veh/km], both against the SUMO 5-run-mean field (t in [100, 1000] s, x <= 20 km); e_s [%] = relative cumulative-flow error at X_q = 15 km over the ECC22 eq. (6) window, mean over the 5 runs of the per-run values; omega error [%] = relative cumulative overtaking-flow error over the slow window, mean over runs; wake = mean density 0.2-1 km behind the controlled vehicle, t in [600, 740] s (SUMO value in brackets). Sign convention: e_s > 0 and omega error > 0 mean that the model passes MORE flow than SUMO. RMSE convention: the RMSE column is against the 5-run-mean field, whereas the rho_rmse min/max columns of the spread tables are per-run values (RMSE against each single run), which is why the two differ.

## A = 1 -- catch & release, 4 fitted parameters (log_kc, log_kr, gamma, ws_frac): kappa_c = 1.695e-01, kappa_r = 7.129e-03 veh^-1, gamma = 0.471, w_s = 0.579 w, no capacity cap

| scenario | model | W1 [veh km] | RMSE [veh/km] | e_s [%] | omega error [%] | wake [veh/km] |
|---|---|---|---|---|---|---|
| 54 km/h, 2500 veh/h (fit) | classical LWR+MB | 173.7 | 6.86 | +5.8 | -2.5 | 68.4 (58.6) |
|  | catch & release | 138.8 | 4.47 | +7.4 | +10.4 | 55.0 (58.6) |
| 54 km/h, 2000 veh/h | classical LWR+MB | 146.5 | 4.12 | +3.5 | -6.2 | 36.4 (42.3) |
|  | catch & release | 131.6 | 2.76 | +4.9 | +5.1 | 35.0 (42.3) |
| 72 km/h, 2000 veh/h | classical LWR+MB | 175.2 | 4.17 | -24.8 | -45.6 | 53.5 (36.5) |
|  | catch & release | 139.6 | 3.22 | +11.9 | +8.2 | 24.5 (36.5) |
| 72 km/h, 2500 veh/h | classical LWR+MB | 184.0 | 5.16 | -24.0 | -44.9 | 60.5 (49.9) |
|  | catch & release | 154.5 | 4.77 | +19.4 | +25.7 | 38.4 (49.9) |

Per-run spread (min .. max over the 5 SUMO runs):

| scenario | model | e_s min [%] | e_s max [%] | rho_rmse min [veh/km] | rho_rmse max [veh/km] |
|---|---|---|---|---|---|
| 54 km/h, 2500 veh/h | classical LWR+MB | +4.6 | +7.9 | 6.35 | 7.62 |
|  | catch & release | +6.2 | +9.5 | 4.01 | 5.59 |
| 54 km/h, 2000 veh/h | classical LWR+MB | +1.2 | +6.0 | 4.19 | 5.32 |
|  | catch & release | +2.6 | +7.4 | 3.09 | 3.86 |
| 72 km/h, 2000 veh/h | classical LWR+MB | -27.5 | -21.8 | 4.22 | 4.92 |
|  | catch & release | +8.0 | +16.4 | 3.50 | 3.99 |
| 72 km/h, 2500 veh/h | classical LWR+MB | -25.3 | -22.4 | 4.79 | 5.70 |
|  | catch & release | +17.3 | +21.8 | 4.55 | 5.37 |

## A = 10 -- catch & release, 4 fitted parameters (log_kc, log_kr, gamma, ws_frac): kappa_c = 3.875e-01, kappa_r = 1.534e-04 veh^-1, gamma = 0.000, w_s = 0.400 w, no capacity cap

| scenario | model | W1 [veh km] | RMSE [veh/km] | e_s [%] | omega error [%] | wake [veh/km] |
|---|---|---|---|---|---|---|
| 54 km/h, 2500 veh/h (fit) | classical LWR+MB | 190.4 | 6.55 | -5.6 | -18.3 | 68.4 (55.4) |
|  | catch & release | 165.4 | 4.96 | +4.8 | +6.6 | 48.7 (55.4) |
| 54 km/h, 2000 veh/h | classical LWR+MB | 122.0 | 4.75 | -7.4 | -20.3 | 36.4 (25.9) |
|  | catch & release | 145.6 | 4.29 | -12.8 | -18.3 | 44.9 (25.9) |
| 72 km/h, 2000 veh/h | classical LWR+MB | 130.4 | 6.17 | -28.1 | -50.0 | 53.5 (25.7) |
|  | catch & release | 109.5 | 2.83 | -5.3 | -15.9 | 34.2 (25.7) |
| 72 km/h, 2500 veh/h | classical LWR+MB | 208.6 | 7.44 | -25.9 | -49.1 | 60.5 (38.4) |
|  | catch & release | 188.4 | 3.97 | +8.9 | +6.8 | 37.2 (38.4) |

Per-run spread (min .. max over the 5 SUMO runs):

| scenario | model | e_s min [%] | e_s max [%] | rho_rmse min [veh/km] | rho_rmse max [veh/km] |
|---|---|---|---|---|---|
| 54 km/h, 2500 veh/h | classical LWR+MB | -8.4 | -3.5 | 6.92 | 7.63 |
|  | catch & release | +1.7 | +7.2 | 5.62 | 6.19 |
| 54 km/h, 2000 veh/h | classical LWR+MB | -8.2 | -6.6 | 4.87 | 5.29 |
|  | catch & release | -13.6 | -12.1 | 4.50 | 4.76 |
| 72 km/h, 2000 veh/h | classical LWR+MB | -29.9 | -26.6 | 6.21 | 6.63 |
|  | catch & release | -7.6 | -3.2 | 3.13 | 3.61 |
| 72 km/h, 2500 veh/h | classical LWR+MB | -29.0 | -22.9 | 7.88 | 8.17 |
|  | catch & release | +4.4 | +13.4 | 4.82 | 5.34 |

## Capacity-cap ablation (same protocol, 5 vs 4 parameters)

Capped reference = the 5-parameter set with the DM-G cap Q_xi as a fitted parameter; no cap = the 4-parameter set with q_xi_max = None, refitted from the capped point. Both scored at dt = 0.5 s. Rule: adopt the 4-parameter no-cap set if (W1_nocap - W1_capped)/W1_capped <= 0.015 at the fit scenario (dt = 0.5 s).

### A = 1

- capped reference (5): kappa_c = 1.557e-01, kappa_r = 6.841e-03 veh^-1, gamma = 0.510, w_s = 0.584 w, Q_xi = 2512 veh/h
- no cap (4): kappa_c = 1.695e-01, kappa_r = 7.129e-03 veh^-1, gamma = 0.471, w_s = 0.579 w, no capacity cap
- fit-scenario W1: capped 138.97 vs no cap 138.85 veh km (-0.09 %, tolerance 1.5 %) -> adopted: no cap (4 parameters)

| scenario | set | W1 [veh km] | RMSE [veh/km] | e_s [%] | omega error [%] | peak omega [veh/h] |
|---|---|---|---|---|---|---|
| 54 km/h, 2500 veh/h (fit) | capped (5) | 139.0 | 4.52 | +7.1 | +9.7 | 1158 |
|  | no cap (4) | 138.8 | 4.47 | +7.4 | +10.4 | 1158 |
| 54 km/h, 2000 veh/h | capped (5) | 130.9 | 2.84 | +5.0 | +5.0 | 926 |
|  | no cap (4) | 131.6 | 2.76 | +4.9 | +5.1 | 926 |
| 72 km/h, 2000 veh/h | capped (5) | 139.4 | 3.24 | +12.1 | +8.3 | 568 |
|  | no cap (4) | 139.6 | 3.22 | +11.9 | +8.2 | 568 |
| 72 km/h, 2500 veh/h | capped (5) | 154.8 | 4.80 | +19.5 | +25.7 | 710 |
|  | no cap (4) | 154.5 | 4.77 | +19.4 | +25.7 | 710 |

### A = 10

- capped reference (5): kappa_c = 3.795e-01, kappa_r = 8.698e-05 veh^-1, gamma = 0.000, w_s = 0.400 w, Q_xi = 2290 veh/h
- no cap (4): kappa_c = 3.875e-01, kappa_r = 1.534e-04 veh^-1, gamma = 0.000, w_s = 0.400 w, no capacity cap
- fit-scenario W1: capped 165.28 vs no cap 165.39 veh km (+0.07 %, tolerance 1.5 %) -> adopted: no cap (4 parameters)

| scenario | set | W1 [veh km] | RMSE [veh/km] | e_s [%] | omega error [%] | peak omega [veh/h] |
|---|---|---|---|---|---|---|
| 54 km/h, 2500 veh/h (fit) | capped (5) | 165.3 | 4.98 | +4.7 | +6.3 | 980 |
|  | no cap (4) | 165.4 | 4.96 | +4.8 | +6.6 | 1158 |
| 54 km/h, 2000 veh/h | capped (5) | 146.3 | 4.32 | -13.1 | -18.8 | 926 |
|  | no cap (4) | 145.6 | 4.29 | -12.8 | -18.3 | 926 |
| 72 km/h, 2000 veh/h | capped (5) | 109.4 | 2.83 | -5.3 | -16.0 | 543 |
|  | no cap (4) | 109.5 | 2.83 | -5.3 | -15.9 | 568 |
| 72 km/h, 2500 veh/h | capped (5) | 187.4 | 3.97 | +8.7 | +6.8 | 543 |
|  | no cap (4) | 188.4 | 3.97 | +8.9 | +6.8 | 710 |
