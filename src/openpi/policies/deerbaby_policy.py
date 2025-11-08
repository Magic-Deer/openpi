import dataclasses

import einops
import numpy as np

from openpi import transforms


LEADER_GRIPPER_JOINT_OPEN = -0.7194 #-0.8
LEADER_GRIPPER_JOINT_CLOSE = -1.6153 #-1.65

FOLLOWER_GRIPPER_JOINT_OPEN = 1.4910
FOLLOWER_GRIPPER_JOINT_CLOSE = 0.0


def make_deerbaby_example() -> dict:
    """Creates a random input example for the Deerbaby policy."""
    return {
        "state": np.ones((7,)),
        "images": {
            "camera_front": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "camera_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class DeerbabyInputs(transforms.DataTransformFn):
    """Inputs for the Deerbaby policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width].
    - state: [7]
    - actions: [action_horizon, 7]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # The base image fed to the model.
    base_view: str = 'camera_front'
    # The third image fed to the model.
    third_view: str | None = None

    def __call__(self, data: dict) -> dict:
        data = _decode_deerbaby(data, adapt_to_pi=self.adapt_to_pi)

        # Get the state. We are padding from 7 to the model action dim.
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        wrist_image = data["images"]["camera_wrist"]
        base_image = data["images"][f"{self.base_view}"]
        third_image = data["images"][self.third_view] if self.third_view \
            else np.zeros_like(wrist_image)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": third_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.third_view else np.False_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _decode_actions(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DeerbabyOutputs(transforms.DataTransformFn):
    """Outputs for the Deerbaby policy."""
    # The action dimension of the output.
    action_dim: int = 7

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 9 dims.
        actions = np.asarray(data["actions"][:, :self.action_dim])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_state_to_pi0(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    # Deerbaby already collects radian value (normalized), linear_to_radian call is not needed
    value = _unnormalize(value, min_val=FOLLOWER_GRIPPER_JOINT_CLOSE, max_val=FOLLOWER_GRIPPER_JOINT_OPEN)

    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # There are 4096 total encoder counts and aloha uses a zero of 2048.
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_action_from_pi0(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # We do not scale the output since the trossen model predictions are already in radians.
    # See the comment in _gripper_to_pi0 for a derivation of the constant
    value = value + 0.5476

    # These values are coming from the Deerbaby code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=FOLLOWER_GRIPPER_JOINT_CLOSE, max_val=FOLLOWER_GRIPPER_JOINT_OPEN)


def _gripper_action_to_pi0(value):
    # Directly inverts the gripper_action_from_pi0 function.
    value = _unnormalize(value, min_val=FOLLOWER_GRIPPER_JOINT_CLOSE, max_val=FOLLOWER_GRIPPER_JOINT_OPEN)
    return value - 0.5476


def _decode_deerbaby(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [arm_joint_angles, arm_gripper]
    # dim sizes: [6, 1]
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask()[:7] * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[6] = _gripper_state_to_pi0(state[6])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        dim = np.shape(actions)[1]
        actions = _joint_flip_mask()[:dim] * actions
        actions[:, 6] = _gripper_action_from_pi0(actions[:, 6])
    return actions


def _decode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        dim = np.shape(actions)[1]
        actions = _joint_flip_mask()[:dim] * actions
        actions[:, 6] = _gripper_action_to_pi0(actions[:, 6])
    return actions
