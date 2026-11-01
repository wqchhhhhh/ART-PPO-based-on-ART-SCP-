from dataclasses import replace

import numpy as np

from cruise.config import Mission, Numerics, Settings
from cruise.dynamics import Discretization
from cruise.validation import validate


def test_independent_replay_detects_manipulated_terminal_state():
    settings = Settings(numerics=Numerics(intervals=2))
    initial = (30000., 1.8, .35, 2110., 0., 1.1, 20000.)
    mission = Mission(initial, initial[:5], 4.)
    u = np.tile([.06, 0., .6], (2, 1))
    actual = Discretization(settings, mission).propagate(initial, u)
    wrong_target = actual[-1, :5].copy()
    wrong_target[1] += .001
    mission = replace(mission, target=tuple(wrong_target))
    planned = actual.copy()
    planned[-1, :5] = wrong_target
    check = validate(settings, mission, planned, u)
    assert not check["passed"]
    assert "integration_consistency" in check["failures"]
    assert "terminal_position" in check["failures"]
    assert check["terminal_miss_m"] > 5000.
