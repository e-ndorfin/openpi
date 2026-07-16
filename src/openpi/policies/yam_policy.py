import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms

_JOINT_FLIP_MASK = np.array([1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1])
_POSITION_INDICES_42D = np.r_[0:7, 21:28]


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)
    value = np.arcsin(np.clip((0.022**2 + value**2 - 0.036**2) / (2 * 0.022 * value), -1.0, 1.0))
    return _normalize(value, min_val=0.4, max_val=1.5)


def _gripper_from_angular(value):
    value = _unnormalize(value, min_val=0.4, max_val=1.5)
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return _normalize(value, min_val=0.4, max_val=1.5)


def _position_state(state: np.ndarray) -> np.ndarray:
    """Select left/right position blocks from the dataset's 42D state."""
    if state.shape[-1] == 42:
        return state[..., _POSITION_INDICES_42D]
    if state.shape[-1] == 14:
        return state
    raise ValueError(f"Expected a 14D position state or 42D position/velocity/effort state, got {state.shape}")


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool) -> np.ndarray:
    state = _position_state(np.asarray(state)).copy()
    if not np.issubdtype(state.dtype, np.number) or not np.all(np.isfinite(state)):
        raise ValueError("YAM state must contain only finite numeric values")
    if adapt_to_pi:
        state *= _JOINT_FLIP_MASK
        state[..., [6, 13]] = _gripper_to_angular(state[..., [6, 13]])
    return state


def _convert_image(image) -> np.ndarray:
    image = np.asarray(image)
    if not np.issubdtype(image.dtype, np.number) or not np.all(np.isfinite(image)):
        raise ValueError("YAM images must contain only finite numeric values")
    if np.issubdtype(image.dtype, np.floating):
        if np.any(image < 0) or np.any(image > 1):
            raise ValueError("Floating-point YAM images must have values in [0, 1]")
        image = np.rint(255 * image).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got {image.shape}")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image with 3 channels, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 or float YAM image, got {image.dtype}")
    return image


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    """Converts bimanual YAM observations into OpenPI model inputs."""

    adapt_to_pi: bool = True
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        if "images" not in data:
            raise ValueError("Missing required YAM input 'images'")
        if "state" not in data:
            raise ValueError("Missing required YAM input 'state'")
        images = data["images"]
        unknown_cameras = set(images) - set(self.EXPECTED_CAMERAS)
        if unknown_cameras:
            raise ValueError(f"Unexpected YAM cameras: {sorted(unknown_cameras)}")
        missing_cameras = set(self.EXPECTED_CAMERAS) - set(images)
        if missing_cameras:
            raise ValueError(f"Missing required YAM cameras: {sorted(missing_cameras)}")

        decoded_images = {key: _convert_image(value) for key, value in images.items()}
        base_image = decoded_images["cam_high"]
        model_images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}
        for destination, source in {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }.items():
            if source in decoded_images:
                model_images[destination] = decoded_images[source]
                image_masks[destination] = np.True_
            else:
                model_images[destination] = np.zeros_like(base_image)
                image_masks[destination] = np.False_

        result = {
            "image": model_images,
            "image_mask": image_masks,
            "state": _decode_state(np.asarray(data["state"]), adapt_to_pi=self.adapt_to_pi),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"]).copy()
            if actions.shape[-1] != 14:
                raise ValueError(f"Expected 14D YAM actions, got {actions.shape}")
            if not np.issubdtype(actions.dtype, np.number) or not np.all(np.isfinite(actions)):
                raise ValueError("YAM actions must contain only finite numeric values")
            if self.adapt_to_pi:
                actions *= _JOINT_FLIP_MASK
                actions[..., [6, 13]] = _gripper_from_angular_inv(actions[..., [6, 13]])
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    """Converts OpenPI actions back to the bimanual YAM action space."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim < 1 or actions.shape[-1] < 14:
            raise ValueError(f"Expected model actions with at least 14 dimensions, got {actions.shape}")
        if not np.issubdtype(actions.dtype, np.number) or not np.all(np.isfinite(actions)):
            raise ValueError("Model actions must contain only finite numeric values")
        actions = actions[..., :14].copy()
        if self.adapt_to_pi:
            actions *= _JOINT_FLIP_MASK
            actions[..., [6, 13]] = _gripper_from_angular(actions[..., [6, 13]])
        return {"actions": actions}
