from dataclasses import replace

import numpy as np
import torch

from cruise.config import CONTROL_SCALE, Mission, Numerics, Settings
from cruise.dynamics import Discretization
from cruise.inference import rollout
from cruise.model import ART, ModelConfig
from cruise.optimization import solve_scp


def short_problem():
    settings = Settings(numerics=Numerics(intervals=2, scp_iterations=12))
    x0 = (30000., 1.8, .35, 2110., 0., 1.1, 20000.)
    m = Mission(x0, x0[:5], 4.)
    u = np.tile([.06, 0., .6], (2, 1))
    x = Discretization(settings, m).propagate(x0, u)
    return settings, replace(m, target=tuple(x[-1, :5])), x, u


def test_scp_solution_survives_independent_integration():
    settings, mission, x, u = short_problem()
    result = solve_scp(settings, mission, (x, u))
    assert result.success, result.summary()
    assert result.validation["passed"]
    assert result.validation["terminal_miss_m"] < .01


def test_dynamics_rollout_uses_the_training_return_scale():
    settings, mission, x, u = short_problem()
    model = ART(ModelConfig(layers=1, heads=2, embedding=16, context_length=2, max_steps=2, dropout=0.))
    with torch.no_grad():
        model.action_head.weight.zero_()
        model.action_head.bias.copy_(torch.tensor(u[0]/CONTROL_SCALE, dtype=torch.float32))
    initial_rtg = x[-1, 6]-x[0, 6]
    result = rollout(model, settings, mission, initial_rtg)
    assert abs(result.validation["final_rtg_kg"]) < 1e-5
    assert result.validation["passed"]
    np.testing.assert_allclose(result.states, x, atol=1e-4)
