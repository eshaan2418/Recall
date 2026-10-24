import numpy as np

from recall.trigger.policies import burst, oracle, threshold_for_budget, threshold_policy, uniform


def test_uniform():
    m = uniform(10, 3)
    assert m.tolist() == [True, False, False, True, False, False, True, False, False, True]


def test_burst_extends():
    f = np.zeros(20, bool)
    f[[2, 4]] = True
    m = burst(f, 3)
    assert np.flatnonzero(m).tolist() == [2, 3, 4, 5, 6]


def test_oracle():
    t = np.arange(20) / 2
    m = oracle(t, np.array([[3.0, 4.0]]), pad_s=0.5)
    assert np.flatnonzero(m).tolist() == [5, 6, 7, 8, 9]


def test_budget():
    rng = np.random.default_rng(0)
    s = rng.random(5000)
    thr = threshold_for_budget(s, 0.2, length=4)
    frac = threshold_policy(s, thr, 4).mean()
    assert abs(frac - 0.2) < 0.03
