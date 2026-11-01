import numpy as np
import torch

from cruise.config import Mission, Vehicle, Zone
from cruise.dataset import labels, split_groups
from cruise.model import ART, ModelConfig


def test_negative_rtg_and_once_per_timestep_ctg():
    mission = Mission((30000., 0., 0., 2100., 0., 0., 20000.), (30000., .1, 0., 2100., 0.), 30.,
                      (Zone(0., 0., 10.), Zone(0., 0., 5.)))
    states = np.tile(mission.initial, (4, 1))
    states[:, 6] = [20000., 19997., 19995., 19994.]
    states[:, 1] = [0., .01, 0., .1]
    r, c = labels(states, mission, Vehicle())
    np.testing.assert_array_equal(r, [-6, -3, -1, 0])
    np.testing.assert_array_equal(c, [2, 1, 1, 0])
    np.testing.assert_array_equal(r[1:], r[:-1]+states[:-1, 6]-states[1:, 6])


def test_scenario_splits_do_not_leak_paired_variants():
    ids = np.repeat(np.arange(30), 2)
    split = split_groups(ids)
    assert not split["train"] & split["test"]
    assert not split["train"] & split["val"]
    assert not split["val"] & split["test"]
    assert set.union(*split.values()) == set(range(30))
    assert split == split_groups(ids)


def test_causal_action_and_state_heads():
    torch.manual_seed(3)
    model = ART(ModelConfig(layers=2, heads=2, embedding=32, context_length=4, max_steps=4, dropout=0.)).eval()
    batch = {"states": torch.randn(1, 4, 7), "actions": torch.randn(1, 4, 3),
             "rtgs": torch.randn(1, 4, 1), "ctgs": torch.randn(1, 4, 1),
             "timesteps": torch.arange(4)[None], "context": torch.randn(1, 18)}
    with torch.no_grad():
        action0, state0 = model(**batch)
        changed = {k: v.clone() for k, v in batch.items()}
        changed["actions"][:, 1:] += 100.
        changed["states"][:, 2:] += 100.
        changed["rtgs"][:, 2:] += 100.
        action1, state1 = model(**changed)
    torch.testing.assert_close(action0[:, :2], action1[:, :2])
    torch.testing.assert_close(state0[:, :1], state1[:, :1])
    assert not torch.allclose(state0[:, 1], state1[:, 1])
