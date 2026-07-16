import pathlib

import numpy as np
import pytest

from openpi import transforms
from openpi.policies import yam_policy
from openpi.training import config as training_config


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


def test_live_14d_and_training_42d_states_are_equivalent():
    training = _example()
    live = _example(state_dim=14)
    live["state"] = training["state"][np.r_[0:7, 21:28]]

    training_result = yam_policy.YamInputs(adapt_to_pi=True)(training)
    live_result = yam_policy.YamInputs(adapt_to_pi=True)(live)

    np.testing.assert_allclose(live_result["state"], training_result["state"])


def test_hwc_uint8_and_chw_float_images_are_equivalent():
    rng = np.random.default_rng(0)
    hwc = rng.integers(0, 256, size=(8, 10, 3), dtype=np.uint8)
    training = _example()
    live = _example(state_dim=14)
    for key in training["images"]:
        training["images"][key] = np.moveaxis(hwc.astype(np.float32) / 255.0, -1, 0)
        live["images"][key] = hwc

    training_result = yam_policy.YamInputs(adapt_to_pi=True)(training)
    live_result = yam_policy.YamInputs(adapt_to_pi=True)(live)

    for key in training_result["image"]:
        np.testing.assert_array_equal(live_result["image"][key], training_result["image"][key])
    assert live_result["image_mask"] == training_result["image_mask"]


def test_yam_action_adaptation_round_trip():
    original = _example()["actions"]
    model_inputs = yam_policy.YamInputs(adapt_to_pi=True)(_example())
    decoded = yam_policy.YamOutputs(adapt_to_pi=True)({"actions": model_inputs["actions"]})

    np.testing.assert_allclose(decoded["actions"], original, atol=1e-6)


@pytest.mark.parametrize("gripper", [0.0, 0.5, 1.0])
def test_yam_action_gripper_conversions_round_trip(gripper):
    actions = np.zeros((3, 14), dtype=np.float32)
    actions[:, [6, 13]] = gripper
    example = _example()
    example["actions"] = actions

    encoded = yam_policy.YamInputs(adapt_to_pi=True)(example)["actions"]
    decoded = yam_policy.YamOutputs(adapt_to_pi=True)({"actions": encoded})["actions"]

    np.testing.assert_allclose(decoded[:, [6, 13]], gripper, atol=1e-6)


def test_yam_outputs_return_finite_robot_chunk():
    model_actions = np.linspace(-0.5, 0.5, 5 * 32, dtype=np.float32).reshape(5, 32)
    result = yam_policy.YamOutputs(adapt_to_pi=True)({"actions": model_actions})

    assert result["actions"].shape == (5, 14)
    assert np.all(np.isfinite(result["actions"]))


def test_wire_config_output_transforms_return_finite_robot_chunk():
    cfg = training_config.get_config("pi05_wire_insertion_success")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    model_state = yam_policy.YamInputs(adapt_to_pi=True)(_example(state_dim=14))["state"]
    outputs = transforms.compose(data_cfg.data_transforms.outputs)(
        {
            "state": model_state,
            "actions": np.linspace(-0.5, 0.5, 5 * cfg.model.action_dim, dtype=np.float32).reshape(
                5, cfg.model.action_dim
            ),
        }
    )

    assert outputs["actions"].shape == (5, 14)
    assert np.all(np.isfinite(outputs["actions"]))


def test_yam_inputs_rejects_unknown_state_layout():
    with pytest.raises(ValueError, match="14D position state or 42D"):
        yam_policy.YamInputs(adapt_to_pi=False)(_example(state_dim=21))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda item: item["images"].pop("cam_left_wrist"), "Missing required YAM cameras"),
        (lambda item: item.update(state=np.full(14, np.nan)), "finite numeric"),
        (lambda item: item.update(state=np.zeros(13)), "14D position state or 42D"),
        (lambda item: item["images"].update(cam_high=np.full((8, 10, 3), np.inf)), "finite numeric"),
    ],
)
def test_yam_inputs_fail_with_useful_errors(mutate, message):
    item = _example(state_dim=14)
    mutate(item)
    with pytest.raises(ValueError, match=message):
        yam_policy.YamInputs(adapt_to_pi=True)(item)


@pytest.mark.manual
def test_local_dataset_frame_matches_live_preprocessing():
    torch = pytest.importorskip("torch")
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    cfg = training_config.get_config("pi05_wire_insertion_success")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    root = pathlib.Path(data_cfg.local_files_path) if data_cfg.local_files_path else None
    if root is None or not root.exists():
        pytest.skip(f"Local wire-insertion dataset is unavailable at {root}")

    dataset = lerobot_dataset.LeRobotDataset("wire_insertion", root=root, video_backend="pyav")
    sample = dataset[0]
    training_sample = {key: value.numpy() if isinstance(value, torch.Tensor) else value for key, value in sample.items()}
    training_sample["prompt"] = training_sample["task"]
    repacked = transforms.compose(data_cfg.repack_transforms.inputs)(training_sample)
    # Action horizon construction is irrelevant to observation parity; omit the
    # scalar frame action so the delta-action transform is a no-op here.
    repacked.pop("actions")
    training_pre_norm = transforms.compose(data_cfg.data_transforms.inputs)(repacked)

    state_42d = training_sample["observation.state"]
    live_input = {
        "state": state_42d[np.r_[0:7, 21:28]],
        "images": {
            "cam_high": np.moveaxis(
                np.rint(training_sample["observation.images.agentview"] * 255).astype(np.uint8), 0, -1
            ),
            "cam_left_wrist": np.moveaxis(
                np.rint(training_sample["observation.images.wrist_left"] * 255).astype(np.uint8), 0, -1
            ),
            "cam_right_wrist": np.moveaxis(
                np.rint(training_sample["observation.images.wrist_right"] * 255).astype(np.uint8), 0, -1
            ),
        },
        "prompt": training_sample["task"],
    }
    live_pre_norm = yam_policy.YamInputs(adapt_to_pi=True)(live_input)

    np.testing.assert_allclose(live_pre_norm["state"], training_pre_norm["state"])
    assert live_pre_norm["image_mask"] == training_pre_norm["image_mask"]
    assert live_pre_norm["prompt"] == training_pre_norm["prompt"]
    for key in live_pre_norm["image"]:
        np.testing.assert_array_equal(live_pre_norm["image"][key], training_pre_norm["image"][key])
