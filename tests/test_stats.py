import numpy as np

from recall.bench.common import bootstrap_ci, paired_bootstrap_diff


def test_bootstrap_contains_mean():
    x = np.r_[np.ones(60), np.zeros(40)]
    m, lo, hi = bootstrap_ci(x)
    assert m == 0.6 and lo < 0.6 < hi and hi - lo < 0.25


def test_cluster_bootstrap_wider():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(10), 20)
    x = (rng.random(10)[g] > 0.5).astype(float)
    _, lo1, hi1 = bootstrap_ci(x)
    _, lo2, hi2 = bootstrap_ci(x, groups=g)
    assert hi2 - lo2 > hi1 - lo1


def test_paired_diff():
    a = np.ones(50)
    b = np.zeros(50)
    m, lo, hi = paired_bootstrap_diff(a, b)
    assert m == 1.0 and lo == 1.0


def test_calibration_features_do_not_reveal_absent_queries():
    from recall.qa.calibrate import feat_matrix

    feats = {"feats": {"score": 0.3}}
    a = feat_matrix([{"kind": "absent", **feats}])
    w = feat_matrix([{"kind": "where", **feats}])
    assert (a == w).all()


def test_mobileclip2_preprocess_is_identity_normalised():
    """The phone feeds RGB/255 with mean 0 / std 1; the offline LiteRT path must use the same transform."""
    import pytest

    open_clip = pytest.importorskip("open_clip")
    try:
        _, _, pp = open_clip.create_model_and_transforms("MobileCLIP2-S2", pretrained="dfndr2b")
    except Exception:
        pytest.skip("weights not cached")
    norm = next(t for t in pp.transforms if type(t).__name__ == "Normalize")
    assert tuple(norm.mean) == (0.0, 0.0, 0.0) and tuple(norm.std) == (1.0, 1.0, 1.0)
