"""What the VLM's guidance function gets as input (main.closed_loop_ee_input): EE positions, EE poses, or gripper points."""
import torch

EE_INPUTS = ("position", "pose", "gripper_points")

# Fingertip body origins (gripper0_finger_joint{1,2}_tip) in the EE frame with the fingers open, metres.
# Measured in LIBERO sim, 2026-11-23 (closed: x = +-0.0097). EE x = finger line, EE z = approach direction.
FINGERTIP_A_EE_M = (-0.0475, 0.0, 0.0114)
FINGERTIP_B_EE_M = (0.0475, 0.0, 0.0114)
WRIST_BACK_M = 0.10  # wrist point: this far behind the EE origin along the approach direction


def gripper_points_from_poses(ee_poses):
    """
    Args:
        ee_poses: (..., 4, 4) world-frame EE poses.
    Returns:
        (..., 4, 3) world-frame points: 0 = midway between the fingertips, 1 = fingertip A, 2 = fingertip B, 3 = wrist.
    """
    assert ee_poses.shape[-2:] == (4, 4), f"expected (..., 4, 4) poses, got {tuple(ee_poses.shape)}"
    rot = ee_poses[..., :3, :3]  # (..., 3, 3) columns = EE axes in world frame
    pos = ee_poses[..., :3, 3]   # (..., 3)
    offsets_ee = torch.tensor([
        [0.0, 0.0, FINGERTIP_A_EE_M[2]],
        FINGERTIP_A_EE_M,
        FINGERTIP_B_EE_M,
        [0.0, 0.0, -WRIST_BACK_M],
    ], device=ee_poses.device, dtype=ee_poses.dtype)  # (4, 3) in EE frame
    offsets_world = torch.einsum('...ij,pj->...pi', rot, offsets_ee)  # (..., 4, 3)
    return pos.unsqueeze(-2) + offsets_world


def wrap_for_poses(fn, ee_input):
    """Turn a VLM function written for ee_input into one PI05PolicySteer calls with (B, T+1, 4, 4) poses."""
    assert ee_input in ("pose", "gripper_points"), f"wrap_for_poses needs pose or gripper_points, got {ee_input!r}"

    def guidance(keypoints, ee_poses):
        if ee_input == "pose":
            return fn(keypoints, ee_poses)
        return fn(keypoints, gripper_points_from_poses(ee_poses))

    guidance.takes_poses = True  # pi05_steer.py:442 then passes poses instead of positions
    return guidance


def check_on_dummy_poses(fn):
    """Raises ValueError if fn crashes or does not return a scalar tensor with a gradient (like load_functions_from_txt's check)."""
    rotvec = torch.randn(1, 11, 3) * 0.3
    from pytorch_kinematics.transforms import axis_angle_to_matrix
    poses = torch.eye(4).repeat(1, 11, 1, 1)  # (1, 11, 4, 4): batch 1, chunk of 10 + current pose
    poses[..., :3, :3] = axis_angle_to_matrix(rotvec)
    poses[..., :3, 3] = torch.randn(1, 11, 3) * 0.2
    poses.requires_grad_(True)
    keypoints = torch.randn(30, 3)
    try:
        result = fn(keypoints, poses)
    except Exception as e:
        raise ValueError(f"guidance function crashed on dummy poses: {e}") from e
    if not isinstance(result, torch.Tensor) or result.dim() != 0:
        raise ValueError(f"guidance function should return a scalar tensor, got {type(result)} "
                         f"{getattr(result, 'shape', '')}")
    if not result.requires_grad:
        raise ValueError("guidance function output has no gradient with respect to the poses")
