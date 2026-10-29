"""Unit tests for grasp steering (P1): reward geometry, grasp selection, commit logic, FKD fix, VLM parsing.

Run from modules/VLS with vls.env loaded (core/__init__ imports the LIBERO adapter):
    set -a; . ../../.env; . ../../vls.env; set +a; ../../envs/vls/bin/python -m pytest tests -q
"""

import math

import pytest
import types

import numpy as np
import torch
from pytorch_kinematics.transforms import axis_angle_to_quaternion, quaternion_to_matrix

from core.fkd_class import FKD
from core.grasp_guidance import GraspReward, grasp_to_tcp, select_grasps
from core.pi05_steer import PI05PolicySteer
from utils.grasp_costs import grasp_costs, orientation_cost, rotation_weight
from vlm_query.vlm_agent import VLMAgent

CFG = dict(
    mode='one_way', top_k=16, approach_max_deg=30, keypoint_sigma=0.03,
    grasp_to_tcp={'translation': [0.0, 0.0, 0.097], 'quaternion_xyzw': [0.0, 0.0, 0.0, 1.0]},
    pregrasp_dist=0.08, sigma_trans=0.015, rot_deg_equiv=15, gate_dist=0.06, gate_tau=0.01, rot_tol_deg=0,
    beta=0.5, commit_weight=0.8, commit_dist=0.03, commit_rot_deg=180,
)
DOWN = torch.tensor([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])  # approach (local +Z) pointing down


def pose(pos, R=DOWN):
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = torch.as_tensor(pos, dtype=torch.float32)
    return T


def rot(axis_angle):
    return quaternion_to_matrix(axis_angle_to_quaternion(torch.as_tensor(axis_angle, dtype=torch.float32)))


def trajectory(start, deltas, R=DOWN):
    """(1, T+1, 4, 4) poses from a start position and per-step position deltas (differentiable in deltas)."""
    positions = torch.cat([start[None], start + torch.cumsum(deltas, dim=0)])
    poses = torch.eye(4).repeat(len(positions), 1, 1)
    poses[:, :3, :3] = R
    poses[:, :3, 3] = positions
    return poses[None]


def reward_for(grasps, scores=None, cfg=CFG):
    scores = np.ones(len(grasps)) if scores is None else np.asarray(scores)
    return GraspReward(torch.stack(grasps).numpy(), np.log(scores), cfg)


def test_gradient_points_toward_grasp():
    reward = reward_for([pose([0.5, 0.0, 0.1])])
    start = torch.tensor([0.6, 0.1, 0.3])
    deltas = torch.zeros(10, 3, requires_grad=True)
    reward(None, trajectory(start, deltas)).backward()
    ascent = deltas.grad.sum(0)  # moving every waypoint along the reward gradient
    to_grasp = torch.tensor([0.5, 0.0, 0.1]) - start
    assert torch.cosine_similarity(ascent, to_grasp, dim=0) > 0.9


def test_orientation_cost_is_jaw_symmetric():
    Rz = lambda a: rot([0.0, 0.0, a])
    goal = DOWN[None]
    assert orientation_cost(DOWN, goal).item() < 1e-6
    assert orientation_cost(DOWN @ Rz(math.pi), goal).item() < 1e-6  # jaws swapped: same grasp
    assert abs(orientation_cost(DOWN @ Rz(math.pi / 2), goal).item() - 0.5) < 1e-5


def test_rotation_weight_matches_translation_scale():
    w = rotation_weight(CFG['rot_deg_equiv'])
    tilted = DOWN @ rot([math.radians(CFG['rot_deg_equiv']), 0.0, 0.0])
    assert abs(w * orientation_cost(tilted, DOWN[None]).item() - 1.0) < 1e-4


def test_orientation_gate_only_near_object():
    grasp = pose([0.5, 0.0, 0.1])

    def orientation_grad(distance):
        axis_angle = torch.tensor([0.3, 0.0, 0.0], requires_grad=True)
        poses = torch.stack([pose([0.5, 0.0, 0.1 + distance]), pose([0.5, 0.0, 0.1 + distance])])
        poses[-1, :3, :3] = DOWN @ rot(axis_angle)
        grasp_costs(poses[None], grasp[None], CFG['pregrasp_dist'], CFG['sigma_trans'],
                    rotation_weight(CFG['rot_deg_equiv']), CFG['gate_dist'], CFG['gate_tau']).sum().backward()
        return axis_angle.grad.norm().item()

    assert orientation_grad(0.20) < 1e-3
    assert orientation_grad(0.01) > 1.0


def test_cost_is_zero_on_the_approach_corridor():
    grasp = pose([0.5, 0.0, 0.1])
    path = torch.stack([pose([0.5, 0.0, z]) for z in torch.linspace(0.18, 0.1, 6)])
    cost = grasp_costs(path[None], grasp[None], CFG['pregrasp_dist'], CFG['sigma_trans'],
                       rotation_weight(CFG['rot_deg_equiv']), CFG['gate_dist'], CFG['gate_tau'])
    assert cost.item() < 1e-6
    # Reward is 0 at the goal and negative elsewhere (VLS reward normalization relies on it)
    reward = reward_for([grasp, pose([0.5, 0.2, 0.1])], scores=[0.9, 0.5])
    assert abs(reward(None, path[None]).item()) < 0.05
    assert reward(None, trajectory(torch.tensor([0.5, 0.0, 0.3]), torch.zeros(3, 3))).item() < -10


def test_softmin_prefers_closer_then_better_grasp():
    end = trajectory(torch.tensor([0.5, 0.0, 0.12]), torch.zeros(1, 3))
    near_far = reward_for([pose([0.5, 0.0, 0.1]), pose([0.5, 0.2, 0.1])])
    assert near_far.weights(end)[0, 0] > 0.9
    equal_distance = reward_for([pose([0.52, 0.0, 0.1]), pose([0.48, 0.0, 0.1])], scores=[0.2, 0.9])
    w = equal_distance.weights(end)[0]
    assert w[1] > w[0]


def test_commit_locks_dominant_grasp():
    reward = reward_for([pose([0.5, 0.0, 0.1]), pose([0.5, 0.3, 0.1])])
    reward.update_commit(trajectory(torch.tensor([0.5, 0.0, 0.2]), torch.zeros(1, 3)))
    assert reward.committed == 0 and reward.commit_reason == 'weight'
    # Once committed, the far grasp no longer competes even when the TCP heads toward it
    far_end = trajectory(torch.tensor([0.5, 0.3, 0.12]), torch.zeros(1, 3))
    assert reward.weights(far_end).shape == (1, 1)


def test_select_grasps_filters_and_ranks():
    down, side = pose([0.5, 0.0, 0.1]).numpy(), pose([0.5, 0.0, 0.1], rot([0.0, math.pi / 2, 0.0])).numpy()
    side_far = side.copy()
    side_far[:3, 3] = [0.5, 0.1, 0.1]
    grasps_G = np.stack([down, side, side_far])
    conf = np.array([0.5, 0.9, 0.8])
    ee = np.array([0.5, 0.1, 0.3])

    tcp, _ = select_grasps(grasps_G, conf, CFG, ee, approach='top_down')
    assert len(tcp) == 1 and np.allclose(tcp[0], down @ grasp_to_tcp(CFG))
    tcp, _ = select_grasps(grasps_G, conf, CFG, ee, approach='side')
    assert len(tcp) == 2
    side_far_tcp = (side_far @ grasp_to_tcp(CFG))[:3, 3]
    tcp, _ = select_grasps(grasps_G, conf, CFG, ee, part_kp=side_far_tcp)
    assert np.allclose(tcp[0][:3, 3], side_far_tcp)  # the grasp at the part keypoint outranks higher scores
    tcp, _ = select_grasps(grasps_G, conf, {**CFG, 'mode': 'nearest'}, ee)
    assert len(tcp) == 1


def test_fkd_resamples_with_pi05_step_indices():
    """Regression: FKD never ran for pi0.5 because float flow times were cast to integers."""
    calls = []

    def guidance_fn(keypoints, traj):
        calls.append(1)
        return -1e3 * traj[..., 0].mean()  # large-magnitude reward: weights must not all underflow

    steer = types.SimpleNamespace(
        _action_chunk_horizon=10,
        _sample_to_trajectory_3d=lambda x: x[:, :, :3],
    )
    num_steps, start_time, bsize = 10, 0.8, 8
    fkd = PI05PolicySteer._init_fkd(steer, {'potential_type': 'max', 'lmbda': 10.0, 'adaptive_resampling': True,
                                            'resample_frequency': 5}, bsize, num_steps, start_time,
                                    torch.zeros(1, 3), [guidance_fn], 'cpu')
    x = torch.zeros(bsize, 10, 7)
    x[:, :, 0] = torch.arange(bsize, dtype=torch.float32)[:, None] / bsize  # particle reward decreases with index
    for step_idx in range(num_steps):
        if 1.0 - step_idx / num_steps <= start_time + 1e-6:
            x, _ = fkd.resample(sampling_idx=step_idx, latents=x, x0_preds=x)
    assert calls, "FKD reward was never evaluated"
    assert fkd.num_resamples >= 1
    assert torch.all(x[:, 0, 0] == 0.0), "resampling should keep the best particle (index 0 had reward 0)"


def test_parse_grasp_fields():
    output = '\n'.join([
        'num_stages = 2',
        'grasp_stage = 1   # pick',
        'grasp_object = "akita_black_bowl_1"',
        'grasp_keypoint = None',
        'grasp_approach = "top_down"',
    ])
    assert VLMAgent._parse_grasp_fields(output) == {
        'grasp_stage': 1, 'grasp_object': 'akita_black_bowl_1', 'grasp_keypoint': None, 'grasp_approach': 'top_down'}
    assert VLMAgent._parse_grasp_fields('num_stages = 1')['grasp_stage'] is None
    # Locals inside a stage function and non-integer indices must not leak into the grasp fields
    tricky = VLMAgent._parse_grasp_fields('\n'.join([
        'grasp_stage = 1', 'grasp_keypoint = -1',
        'def stage1_guidance(keypoints, action_sequence):', '    grasp_keypoint = keypoints[3]',
    ]))
    assert tricky['grasp_stage'] == 1 and tricky['grasp_keypoint'] is None


def test_orientation_acts_before_vls_eases_guidance_off():
    """VLS scales guidance by sigmoid(25 * (0.75 - n)), n = 1 - r / r_init, and asks the VLM to end the stage
    at n >= 0.8. Orientation steering must still be on (gate open, strength high) when the gate opens, and a
    misaligned gripper close to the grasp must not count as done."""
    cfg = {**CFG, 'gate_dist': 0.10}
    reward = reward_for([pose([0.5, 0.0, 0.1])], cfg=cfg)
    at = lambda d, R=DOWN: trajectory(torch.tensor([0.5, 0.0, 0.1 + d]), torch.zeros(1, 3), R)
    r0 = reward(None, at(0.25)).item()
    progress = lambda d, R=DOWN: 1 - reward(None, at(d, R)).item() / r0
    strength = lambda n: 1 / (1 + math.exp(25 * (n - 0.75)))

    assert strength(progress(cfg['gate_dist'])) > 0.8  # still steering ~90% strength when orientation switches on
    tilted = DOWN @ rot([math.radians(30), 0.0, 0.0])
    assert progress(0.03, tilted) < progress(0.03)  # orientation error delays the hand-over...
    assert progress(0.01) > 0.8  # ...and aligned + close lets VLS hand over to the policy
    # Known limit: progress is relative to the start, so 30 deg at 3 cm (~4 cm-equivalent) from a 25 cm start
    # already counts as ~85% done; chunks.jsonl end_rot_err_deg shows whether this matters in practice.


def test_rotation_tolerance_lets_vls_hand_over_when_roughly_aligned():
    """Within rot_tol_deg the gripper counts as aligned: no orientation pull, and VLS sees the stage as done
    once close (it then eases off and the policy makes the final approach). Beyond it, orientation still steers."""
    cfg = {**CFG, 'gate_dist': 0.30, 'rot_tol_deg': 30}
    reward = reward_for([pose([0.5, 0.0, 0.1])], cfg=cfg)
    at = lambda d, R=DOWN: trajectory(torch.tensor([0.5, 0.0, 0.1 + d]), torch.zeros(1, 3), R)
    r0 = reward(None, at(0.25)).item()
    progress = lambda d, R=DOWN: 1 - reward(None, at(d, R)).item() / r0

    def orientation_grad(deg):
        axis_angle = torch.tensor([math.radians(deg), 0.0, 0.0], requires_grad=True)
        reward(None, at(0.01, DOWN @ rot(axis_angle))).backward()
        return axis_angle.grad.norm().item()

    assert progress(0.01, DOWN @ rot([math.radians(20), 0.0, 0.0])) > 0.8
    assert orientation_grad(20) == 0.0
    assert orientation_grad(60) > 0.1
    assert progress(0.01, DOWN @ rot([math.radians(90), 0.0, 0.0])) < 0.8


@pytest.mark.parametrize('rule', ['distance', 'weight'])
def test_commit_requires_rough_alignment(rule):
    """A TCP that ends at a grasp's position but far from its orientation (e.g. pointing down at a side grasp)
    must not lock onto it, by either rule; the soft-min keeps choosing among grasps."""
    side = DOWN @ rot([math.pi / 2, 0.0, 0.0])
    grasps = [pose([0.5, 0.0, 0.1], side), pose([0.5, 0.3, 0.1])]  # near side grasp, far top-down grasp
    end = trajectory(torch.tensor([0.5, 0.0, 0.11]), torch.zeros(1, 3))
    cfg = {**CFG, 'commit_weight': 0.8 if rule == 'weight' else 1.01, 'commit_dist': 0.03 if rule == 'distance' else 0}
    reward = reward_for(grasps, cfg={**cfg, 'commit_rot_deg': 30})
    reward.update_commit(end)
    assert reward.committed is None
    reward = reward_for(grasps, cfg=cfg)  # commit_rot_deg=180: any orientation (previous behaviour)
    reward.update_commit(end)
    assert reward.committed == 0 and reward.commit_reason == rule


def test_guidance_value_ignores_softmin_bonus_but_keeps_its_gradient():
    """Many equally good grasps must not make the reward look near 0 (stage 'done') from far away."""
    many = reward_for([pose([0.5, 0.005 * k, 0.1]) for k in range(16)], cfg={**CFG, 'beta': 0.05})
    one = reward_for([pose([0.5, 0.0, 0.1])], cfg={**CFG, 'beta': 0.05})
    far = trajectory(torch.tensor([0.5, 0.0, 0.3]), torch.zeros(3, 3))
    assert many(None, far).item() == pytest.approx(one(None, far).item(), rel=0.05)

    deltas = torch.zeros(3, 3, requires_grad=True)
    many(None, trajectory(torch.tensor([0.5, 0.2, 0.3]), deltas)).backward()
    target_mean = torch.tensor([0.5, 0.0375, 0.1])  # soft-min pulls toward the weighted grasps, not one corner
    assert torch.cosine_similarity(deltas.grad.sum(0), target_mean - torch.tensor([0.5, 0.2, 0.3]), dim=0) > 0.9


@pytest.mark.parametrize('model', ['constant', 'lag'])
def test_batched_pose_prediction_matches_per_particle(model):
    from core.env_adapters.libero_adapter import predict_ee_poses
    cfg = {'ee_prediction': model, 'action_scale_pos': 0.01, 'action_scale_rot': 0.1,
           'ee_lag_model': {'pos_gain': [0.005, 0.006, 0.0058], 'pos_lag': [0.62, 0.57, 0.58],
                            'rot_gain': [0.047, 0.04, 0.05], 'rot_lag': [0.6, 0.59, 0.54]}}
    last = {'pos': np.array([0.002, -0.001, 0.0005]), 'rot': np.array([0.01, 0.0, -0.02])}
    actions = torch.randn(5, 10, 7, requires_grad=True)
    start_pos, start_rot = torch.tensor([0.4, 0.1, 0.3]), rot([0.1, -0.2, 0.3])
    batched = predict_ee_poses(actions, start_pos, start_rot, cfg, last)
    single = torch.stack([predict_ee_poses(a, start_pos, start_rot, cfg, last) for a in actions])
    assert batched.shape == (5, 11, 4, 4)
    assert torch.allclose(batched, single, atol=1e-6)
    batched[..., :3, 3].sum().backward()  # differentiable through the batched path
    assert actions.grad is not None
