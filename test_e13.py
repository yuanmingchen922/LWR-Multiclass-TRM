"""Tests for the E13 v2 joint objective (fast, no full fits)."""

import numpy as np
import pytest

import e12_assertiveness as e12
import e13_joint as e13


def test_objective_reduces_to_normalised_w1_at_lambda_zero():
    p = e13.parts_one(5, 0.05, 1e-3)
    tt, rho_mean, meas, w1_cl, ref = e13._data(5)
    assert np.isclose(p["W1_rel"], p["W1"] / w1_cl)
    assert np.isclose(e13.J_of(p, 0.0), p["W1_rel"])
    assert p["N_0"] == ref["N_0"] >= e13.N_MIN


def test_penalty_terms_and_ablation_weights():
    p = dict(W1=100.0, W1_rel=0.6, Ns_mae=20.0, omega_err=-0.4, eps_R=1.5, N_0=10.0)
    assert np.isclose(e13.penalty(p), 20.0 / 10.0 + 0.4 / e13.EPS_0 + 1.5)
    assert np.isclose(e13.J_of(p, 0.1), 0.6 + 0.1 * e13.penalty(p))
    assert np.isclose(e13.J_of(p, 0.1, wN=1, wO=0, wR=0), 0.6 + 0.1 * 2.0)
    assert np.isclose(e13.J_of(p, 0.1, wN=0, wO=1, wR=0), 0.6 + 0.1 * 2.0)
    assert np.isclose(e13.J_of(p, 0.1, wN=0, wO=0, wR=1), 0.6 + 0.1 * 1.5)
    assert e13.J_of(p, 0.5) > e13.J_of(p, 0.1) > e13.J_of(p, 0.0)


def test_event_counts_are_exact_and_eps_R_zero_at_the_data_counts():
    regr = e12.run_cr(15.0, 2500.0, 0.5, 0.03, None, None, 1.0)
    C, R, tau = e13.window_counts(regr)
    net = regr["cum_cap"] - regr["cum_rel"]
    assert np.max(np.abs(net - regr["N_s"])) < 1e-8          # counters consistent with N_s
    assert C > R > 0 and tau > 0
    tt, rho_mean, meas, w1_cl, ref = e13._data(5)
    # eps_R vanishes iff the model counts equal the data counts
    fake = dict(regr)
    fake["cum_cap"] = np.zeros_like(regr["cum_cap"]); fake["cum_rel"] = np.zeros_like(regr["cum_rel"])
    i1 = int(np.argmin(np.abs(regr["tt"] - e13.SLOW[1])))
    fake["cum_cap"][i1:] = ref["C_d"]; fake["cum_rel"][i1:] = ref["R_d"]
    p = e13.parts_of(fake, 5)
    assert abs(p["eps_R"]) < 1e-9


def test_parts_match_e12_evaluate_at_production_dt():
    summ, _ = e12.evaluate(5, 15.0, 2500.0, "cr", 0.05, 1e-3, dt=0.5)
    p = e13.parts_one(5, 0.05, 1e-3, dt=0.5)
    assert np.isclose(p["W1"], summ["W1"]) and np.isclose(p["Ns_mae"], summ["Ns_mae"])
    assert np.isclose(p["omega_err"], summ["omega_err"])


def test_fold_data_leaves_one_run_out():
    tt, rho5, meas5, w15, ref5 = e13._data(5)
    tt, rho4, meas4, w14, ref4 = e13._data(5, fold=2)
    assert meas5["rho"].shape[0] == 5 and meas4["rho"].shape[0] == 4
    assert 2 not in [meas5["reps"].index(r) for r in meas4["reps"]]
    assert ref4["n_runs"] == 4 and ref5["n_runs"] == 5
    assert not np.allclose(rho4, rho5)


def test_all_cr_runs_go_through_the_purity_guard(monkeypatch):
    seen = []
    orig = e12.cr_config

    def spy(*a, **k):
        cfg = orig(*a, **k)
        seen.append(cfg)
        return cfg

    monkeypatch.setattr(e12, "cr_config", spy)
    e13.j_one(5, 0.05, 1e-3, 0.1, ws=0.6, gamma=0.5)
    assert seen and all(e12.assert_pure(c) for c in seen)
    assert seen[0].gamma == 0.5 and seen[0].w_s is not None


def test_ridge_coordinates_roundtrip():
    kc, kr = e13.rm_to_kappa(*e13.kappa_to_rm(3.7, 0.021))
    assert np.isclose(kc, 3.7) and np.isclose(kr, 0.021)


def test_knee_on_synthetic_l_curve():
    x = np.array([10.0, 4.0, 2.0, 1.5, 1.3, 1.2, 1.15])
    y = np.array([1.00, 1.02, 1.05, 1.12, 1.25, 1.45, 1.70])
    i, dist = e13.knee(x, y)
    assert 1 <= i <= len(x) - 2 and dist[0] == 0.0 and np.isclose(dist[-1], 0.0)


def test_regrid_any_matches_regrid_sim_at_dx50():
    regr, raw = e13.run_at(15.0, 2500.0, 0.05, 1e-3, None, None, 50.0, 0.5)
    ref = e12.run_cr(15.0, 2500.0, 0.05, 1e-3, None, None, 0.5)
    for k in ("rho_tot", "s", "N_s", "omega", "cum_cap", "cum_rel"):
        assert np.array_equal(regr[k], ref[k]), k


def test_settings():
    assert e13.lam_key(0.0) == "lam0" and e13.lam_key(0.05) == "lam0.05"
    assert e13.SETTINGS["lambdas"][0] == 0.0 and e13.SETTINGS["A_levels"] == tuple(range(1, 11))


def test_fit_joint_mechanics_tiny():
    S = e13.smoke_settings()
    A = 5.0
    grid = [e13._grid_task((A, *e13.rm_to_kappa(r, m), None, None)) for r in S["r_grid"] for m in S["m_grid"]]
    fit = e13.fit_joint(A, 0.1, "C1", None, grid, S, {})
    assert fit["J_dt1"] <= fit["grid_J_min"] + 1e-12
    assert fit["kappa_c"] > 0 and fit["kappa_r"] > 0
    assert set(fit["parts_dt05"]) >= {"W1", "W1_rel", "Ns_mae", "omega_err", "eps_R", "tau_model_s"}
