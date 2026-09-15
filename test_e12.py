"""Tests for the E12 driver: purity guard, A set, data routing, solver
bridge equivalence.  Fast (no full fits)."""

import numpy as np
import pytest

import e12_assertiveness as e12
import e7_wasserstein as e7
import ev4_compare as ev4
from solver import SimConfig


def _base(**kw):
    d = dict(v_f=ev4.V_F, w=ev4.W, P=ev4.P, q_in=2500 / 3600.0, u_xi=15.0,
             kappa_c=0.05, kappa_r=1e-3, dt=1.0, save_every=10)
    d.update(kw)
    return d


def test_solver_defaults_are_pure():
    assert e12._solver_defaults_are_pure()
    assert e12.assert_pure(SimConfig(**_base()))


@pytest.mark.parametrize("knob,val", [("q_xi_max", 2000 / 3600.0),
                                      ("downstream_release", True),
                                      ("eta_la", 200.0),
                                      ("s_impermeable", True)])
def test_assert_pure_rejects_each_forbidden_knob(knob, val):
    with pytest.raises(ValueError, match=knob):
        e12.assert_pure(SimConfig(**_base(**{knob: val})))
    with pytest.raises(ValueError, match=knob):
        e12.assert_pure({knob: val})


def test_assert_pure_accepts_e7_knobs():
    assert e12.assert_pure(SimConfig(**_base(gamma=0.5, w_s=0.6 * ev4.W)))
    assert e12.assert_pure(dict(gamma=0.0, w_s=0.5 * ev4.W, P_s=None))


def test_cr_config_has_no_forbidden_knob():
    cfg = e12.cr_config(15.0, 2500.0, 0.05, 1e-3, gamma=0.3, ws_frac=0.7)
    for k, v in e12.PURE_OFF.items():
        assert getattr(cfg, k) == v
    assert cfg.gamma == 0.3
    assert np.isclose(cfg.w_s, 0.7 * ev4.W) and cfg.P_s is None


def test_a_levels_are_the_ten_integers():
    assert e12.A_LEVELS == tuple(range(1, 11))
    assert e12.SETTINGS["A_levels"] == tuple(range(1, 11))
    assert all(isinstance(a, int) for a in e12.A_LEVELS)


def test_load_measured_any_routes_a5_to_e4_and_restores():
    e1_before = ev4.E1
    m = e12.load_measured_any(5, 15.0, 2500.0)
    assert ev4.E1 == e1_before
    assert m["rho"].shape == (5, 100, 300)
    assert len(m["tt"]) == 100 and len(m["reps"]) == 5
    assert m["q_xq"].shape[0] == 5


def test_load_measured_any_a1_uses_e1():
    m1 = e12.load_measured_any(1, 15.0, 2500.0)
    m2 = ev4.load_measured(1.0, 15.0, 2500.0)
    assert np.array_equal(m1["rho"], m2["rho"])


def test_load_measured_any_rejects_u20_for_intermediate_a():
    with pytest.raises(ValueError):
        e12.load_measured_any(5, 20.0, 2500.0)


def test_e1_dir_restored_on_exception():
    before = ev4.E1
    with pytest.raises(RuntimeError):
        with e12.e1_dir(e12.OUT_E4):
            assert ev4.E1 == e12.OUT_E4
            raise RuntimeError("boom")
    assert ev4.E1 == before


def test_run_cr_bit_identical_to_e7_run_sim():
    a = e12.run_cr(15.0, 2500.0, 0.05, 1e-3, gamma=0.3, ws_frac=0.7, dt=1.0)
    b = e7.run_sim(15.0, 2500.0, 0.05, 1e-3, "lf", gamma=0.3, ws_frac=0.7,
                   dt=1.0)
    for k in ("rho_tot", "s", "f", "x_cav", "omega", "N_s"):
        assert np.array_equal(a[k], b[k], equal_nan=True), k


def test_zero_kappa_has_no_stuck_class():
    r = e12.run_cr(15.0, 2500.0, 0.0, 0.0, dt=1.0)
    assert np.all(r["s"] == 0.0) and np.all(r["N_s"] == 0.0)
    assert r["rho_tot"].shape == (100, 300)


def test_classical_is_the_only_capped_run():
    r = e12.run_classical(15.0, 2500.0, dt=1.0)
    assert np.all(r["s"] == 0.0)
    assert r["rho_tot"].shape == (100, 300)


def test_w1_one_matches_direct_evaluation():
    tt, rho_mean = e7.load_rho_mean(5, 15.0, 2500.0)
    regr = e7.run_sim(15.0, 2500.0, 0.05, 1e-3, "lf", dt=1.0)
    direct = e7.w1_mean(regr["rho_tot"], rho_mean, tt)
    assert np.isclose(e12.w1_one(5, 0.05, 1e-3), direct)


def test_scen_key():
    assert e12.scen_key(15.0, 2500.0) == "u15_q2500"
    assert e12.scen_key(20.0, 2000.0) == "u20_q2000"
