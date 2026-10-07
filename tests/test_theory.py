import numpy as np
import pytest

from src import theory as th


def test_closed_forms_at_known_point():
    N, T = 30, 60
    assert th.predicted_variance_ratio_mean(N, T) == pytest.approx(30 / 59)
    assert th.predicted_variance_dof(N, T) == 30
    assert th.true_variance_ratio_mean(N, T) == pytest.approx(58 / 29)  # = 1 + 29/29
    assert th.true_variance_ratio_mean(N, T) == pytest.approx(1 + (N - 1) / (T - N - 1))
    assert th.variance_ratio_of_expectations(N, T) == pytest.approx(59 * 58 / (30 * 29))
    assert th.variance_ratio_mean_per_run(N, T) == pytest.approx((58 / 29) * (59 / 28))


def test_moment_existence_guards():
    with pytest.raises(ValueError):
        th.predicted_variance_ratio_mean(30, 30)
    with pytest.raises(ValueError):
        th.true_variance_ratio_mean(30, 31)       # T-N-1 = 0
    with pytest.raises(ValueError):
        th.variance_ratio_mean_per_run(30, 32)    # T-N-2 = 0
    assert th.variance_ratio_mean_per_run(30, 33) > 0


def test_n_equals_one_has_no_inflation():
    assert th.true_variance_ratio_mean(1, 20) == pytest.approx(1.0)  # 1 + 0/(T-2)


def test_c4_known_values():
    assert th.c4(2) == pytest.approx(np.sqrt(2 / np.pi))
    assert th.c4(60) == pytest.approx(1 - 1 / (4 * 59), abs=2e-4)
    assert th.c4(10) < th.c4(100) < 1


def test_exact_representation_matches_closed_forms_and_independence():
    N, T, n = 6, 40, 400_000
    Q, R = th.sample_exact_QR(N, T, n, np.random.default_rng(0))
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    assert abs(Q.mean() - th.predicted_variance_ratio_mean(N, T)) < 4 * se(Q)
    assert abs(R.mean() - th.true_variance_ratio_mean(N, T)) < 4 * se(R)
    r = R / Q
    assert abs(r.mean() - th.variance_ratio_mean_per_run(N, T)) < 4 * se(r)
    assert abs(np.corrcoef(Q, R)[0, 1]) < 0.01


def test_expected_std_ratio_deterministic_and_sane():
    a = th.expected_std_ratio(10, 60, 60, 50_000, seed=1)
    assert a == th.expected_std_ratio(10, 60, 60, 50_000, seed=1)
    assert a[0] > 1.0 and a[1] > 0  # estimation error inflates realised vs predicted risk
    # sqrt of the ratio of expectations is only a rough proxy (Jensen), same order of magnitude
    ref = np.sqrt(th.variance_ratio_of_expectations(10, 60))
    assert 0.7 * ref < a[0] < 1.3 * ref


def test_theory_table_columns():
    t = th.theory_table(30, windows=[60, 120], H=60)
    assert list(t.window) == [60, 120] and (t.th_true_var_over_V > 1).all()
    assert (t.th_pred_var_over_V < 1).all()


def test_recalled_form_differs_from_derived_by_offset_only():
    N, T = 30, 60
    assert th.true_variance_ratio_mean_recalled(N, T) == pytest.approx(1 + 29 / 28)
    assert th.true_variance_ratio_mean_recalled(N, T) > th.true_variance_ratio_mean(N, T)
    with pytest.raises(ValueError):
        th.true_variance_ratio_mean_recalled(30, 32)
