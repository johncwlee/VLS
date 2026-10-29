"""
Pose costs for grasp steering, shared by VLS (envs/vls) and the GraspGenX server (envs/graspgenx).

Pure torch with no other imports, so scripts/graspgenx_server.py can load this file by path
without pulling in the rest of VLS.

Conventions: poses are 4x4 homogeneous matrices; a grasp's approach axis is its local +Z and its
jaws close along local +X (GraspGenX convention), so a grasp and the same grasp rotated 180 deg
about Z are equivalent.
"""

import math

import torch
from torch import Tensor


def angle_to_orientation_cost(deg: float) -> float:
    """orientation_cost of a rotation error of deg degrees: (1 - cos theta) / 2."""
    return (1.0 - math.cos(math.radians(deg))) / 2.0


def orientation_cost_to_deg(cost: Tensor) -> Tensor:
    """Inverse of angle_to_orientation_cost: rotation error in degrees of an orientation_cost."""
    return torch.rad2deg(torch.arccos((1 - 2 * cost).clamp(-1, 1)))


def rotation_weight(rot_deg_equiv: float) -> float:
    """Weight that makes a rot_deg_equiv rotation error cost 1, the same as a sigma_trans translation error."""
    return 1.0 / angle_to_orientation_cost(rot_deg_equiv)


def orientation_cost(R: Tensor, R_goal: Tensor) -> Tensor:
    """Jaw-symmetric rotation cost (3 - tr(R^T R_goal)) / 4, min over R_goal and R_goal * Rz(pi).

    Args:
        R: (..., 3, 3) rotations.
        R_goal: (K, 3, 3) goal rotations.
    Returns:
        (..., K) cost in [0, 1]; (1 - cos theta) / 2 for a rotation error theta.
    """
    M = torch.einsum('...ji,kjl->...kil', R, R_goal)  # R^T R_goal, (..., K, 3, 3)
    tr = M[..., 0, 0] + M[..., 1, 1] + M[..., 2, 2]
    tr_flipped = -M[..., 0, 0] - M[..., 1, 1] + M[..., 2, 2]  # R_goal * Rz(pi) flips local x and y
    return (3.0 - torch.maximum(tr, tr_flipped)) / 4.0


def segment_dist_sq(p: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Squared distance from points to segments.

    Args:
        p: (..., 3) points.
        a, b: (K, 3) segment endpoints.
    Returns:
        (..., K) squared distances.
    """
    ab = b - a  # (K, 3)
    ap = p.unsqueeze(-2) - a  # (..., K, 3)
    t = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0.0, 1.0)
    closest = a + t.unsqueeze(-1) * ab
    return ((p.unsqueeze(-2) - closest) ** 2).sum(-1)


def grasp_costs(
    poses: Tensor,
    grasps: Tensor,
    pregrasp_dist: float,
    sigma_trans: float,
    rot_weight: float,
    gate_dist: float,
    gate_tau: float,
    rot_tol_deg: float = 0.0,
) -> Tensor:
    """Cost of each predicted TCP trajectory against each grasp target.

    corridor: mean over future waypoints of the squared distance to the pre-grasp -> grasp segment
    progress: squared distance from the final waypoint to the grasp position
    orientation: jaw-symmetric rotation cost of the final waypoint, gated on by proximity to the grasp;
        free within rot_tol_deg, so a roughly aligned gripper is not pushed toward the exact grasp angle

    Args:
        poses: (B, T+1, 4, 4) TCP trajectories; index 0 is the current pose (no gradient, skipped).
        grasps: (K, 4, 4) TCP-frame grasp targets.
    Returns:
        (B, K) costs in units where sigma_trans of translation error costs 1.
    """
    g_pos = grasps[:, :3, 3]
    pre_pos = g_pos - pregrasp_dist * grasps[:, :3, 2]
    p = poses[:, 1:, :3, 3]  # (B, T, 3)
    p_end = poses[:, -1, :3, 3]  # (B, 3)
    inv_var = 1.0 / sigma_trans ** 2

    corridor = segment_dist_sq(p, pre_pos, g_pos).mean(dim=1) * inv_var  # (B, K)
    end_dist_sq = ((p_end.unsqueeze(1) - g_pos) ** 2).sum(-1)  # (B, K)
    progress = end_dist_sq * inv_var
    gate = torch.sigmoid((gate_dist - end_dist_sq.clamp_min(1e-12).sqrt()) / gate_tau)
    rot_err = orientation_cost(poses[:, -1, :3, :3], grasps[:, :3, :3]) - angle_to_orientation_cost(rot_tol_deg)
    orientation = rot_weight * gate * rot_err.clamp_min(0.0)
    return corridor + progress + orientation


def pose_distance_cost(poses: Tensor, targets: Tensor, sigma_trans: float, rot_weight: float) -> Tensor:
    """Ungated pose-to-pose cost, used to pull sampled grasps toward target poses.

    Args:
        poses: (N, 4, 4); targets: (M, 4, 4).
    Returns:
        (N, M) cost: squared translation error / sigma_trans^2 + rot_weight * jaw-symmetric rotation cost.
    """
    trans = ((poses[:, None, :3, 3] - targets[None, :, :3, 3]) ** 2).sum(-1) / sigma_trans ** 2
    return trans + rot_weight * orientation_cost(poses[:, :3, :3], targets[:, :3, :3])


def soft_min_reward(costs: Tensor, log_scores: Tensor, beta: float) -> Tensor:
    """Quality-weighted soft-min over grasps as a reward: logsumexp_k(log s_k - beta * c_k) / beta.

    Args:
        costs: (B, K); log_scores: (K,).
    Returns:
        (B,) reward (higher is better).
    """
    return torch.logsumexp(log_scores - beta * costs, dim=-1) / beta


def soft_min_weights(costs: Tensor, log_scores: Tensor, beta: float) -> Tensor:
    """(B, K) soft-min weights matching soft_min_reward."""
    return torch.softmax(log_scores - beta * costs, dim=-1)
