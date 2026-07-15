import numpy as np
import pytest

from openpi.policies import yam_policy


def _example(*, state_dim: int = 42) -> dict:
    return {
        "images": {
            "cam_high": np.zeros((3, 8, 10), dtype=np.float32),
            "cam_left_wrist": np.zeros((3, 8, 10), dtype=np.float32),
            "cam_right_wrist": np.zeros((3, 8, 10), dtype=np.float32),
        },
        "state": np.arange(state_dim, dtype=np.float32),
        "actions": np.linspace(0.05, 0.95, 28, dtype=np.float32).reshape(2, 14),
        "prompt": "insert the plug",
    }


def test_yam_inputs_extracts_position_blocks_from_42d_state():
    result = yam_policy.YamInputs(adapt_to_pi=False)(_example())

    np.testing.assert_array_equal(result["state"], np.r_[0:7, 21:28])
    assert result["actions"].shape == (2, 14)
    assert result["image"]["base_0_rgb"].shape == (8, 10, 3)
    assert result["prompt"] == "insert the plug"


def test_yam_action_adaptation_round_trip():
    original = _example()["actions"]
    model_inputs = yam_policy.YamInputs(adapt_to_pi=True)(_example())
    decoded = yam_policy.YamOutputs(adapt_to_pi=True)({"actions": model_inputs["actions"]})

    np.testing.assert_allclose(decoded["actions"], original, atol=1e-6)


def test_yam_inputs_rejects_unknown_state_layout():
    with pytest.raises(ValueError, match="14D position state or 42D"):
        yam_policy.YamInputs(adapt_to_pi=False)(_example(state_dim=21))
