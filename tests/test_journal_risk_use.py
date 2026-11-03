import numpy as np

from src.journal import risk_use


def test_kupiec_accepts_nominal_rate_and_rejects_far_rate():
    assert risk_use.kupiec_lr(50, 1000, 0.05) > 0.9
    assert risk_use.kupiec_lr(120, 1000, 0.05) < 1e-6


def test_christoffersen_detects_clustering():
    rng = np.random.default_rng(0)
    iid = (rng.random(4000) < 0.05)
    clustered = np.zeros(4000, bool)
    for s in range(0, 4000, 80):
        clustered[s:s + 4] = True                 # same overall rate, strongly clustered
    assert risk_use.christoffersen_ind(iid) > 0.01
    assert risk_use.christoffersen_ind(clustered) < 1e-6


def test_origin_records_use_only_rows_before_origin_and_c_scales_forecast():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 8)) * 0.01
    a = risk_use.origin_records(X, 250, 120, 20)
    X2 = X.copy()
    X2[250:] += 5.0                                     # change the test block only
    b = risk_use.origin_records(X2, 250, 120, 20)
    assert a["plugin_raw"][1] == b["plugin_raw"][1]       # forecast unaffected by the future
    assert a["c_qlike"][1] > a["c"][1] > a["plugin_raw"][1]
