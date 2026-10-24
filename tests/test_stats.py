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
