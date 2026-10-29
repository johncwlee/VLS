"""
GraspGen-X grasp steering for VLS (see plans/graspgen-x_plan.md in the outer repo).

During the VLM-flagged grasp stage, the VLM's keypoint reward is replaced by GraspReward: a
quality-weighted soft-min over GraspGen-X grasps of a corridor + progress + orientation cost on the
predicted TCP trajectory. GraspGen-X runs as a separate process (scripts/graspgenx_server.py in
envs/graspgenx); this module only talks ZMQ to it and never imports graspgenx.

Modes (config main.grasp.mode):
    one_way        steer toward the GraspGen-X grasps
    coupled_cheap  also reweight grasps by how close unsteered VLA particles already get (D.1)
    coupled_full   regenerate grasps with GraspGen-X's sampler pulled toward the unsteered VLA poses (D.2)
    nearest        AnyGrasp-style baseline: one grasp, the nearest to the current TCP, no soft-min

Frames: everything is in the adapter's world frame. A GraspGen-X grasp G (+Z approach, +X closing,
origin at the gripper base) maps to the adapter's TCP by the robot's grasp_to_tcp transform
(backend.<name>.grasp_robot). Robot-specific pieces live in the adapter (delta_actions_to_ee_poses,
steerable_action_dims) and that config block, so this module has no robot or simulator assumptions.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import msgpack
import msgpack_numpy
import numpy as np
import torch
import zmq
from torch import Tensor

from core.env_adapters.base_adapter import BaseEnvAdapter
from utils.grasp_costs import (angle_to_orientation_cost, grasp_costs, orientation_cost, orientation_cost_to_deg,
                               rotation_weight, soft_min_reward, soft_min_weights)
from utils.logging_utils import SteerLogger

msgpack_numpy.patch()
log = SteerLogger("GraspSteering")

MODES = ('one_way', 'coupled_cheap', 'coupled_full', 'nearest')
SCENE_CROP_RADIUS = 0.3  # m; scene points farther than this from the object cannot touch a grasp
MAX_SCENE_POINTS = 8192  # bounds GraspGenX's collision-filter cdist memory (~1 GB at this size)
MIN_OBJECT_POINTS = 50


class GraspGenXClient:
    """REQ/REP msgpack client for scripts/graspgenx_server.py (mirrors graspgenx.serving.zmq_client)."""

    def __init__(self, endpoint: str, timeout_ms: int = 60000):
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self._sock = None

    def _connect(self):
        self._sock = zmq.Context.instance().socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self.endpoint)

    def request(self, payload: dict) -> dict:
        if self._sock is None:
            self._connect()
        try:
            self._sock.send(msgpack.packb(payload, use_bin_type=True))
            response = msgpack.unpackb(self._sock.recv(), raw=False)
        except zmq.error.Again as exc:
            # A REQ socket is unusable after a timeout; reconnect on the next call
            self._sock.close(linger=0)
            self._sock = None
            raise TimeoutError(f"GraspGenX server at {self.endpoint} did not answer within {self.timeout_ms} ms "
                               f"(start it with ./run_graspgenx_server.sh)") from exc
        if isinstance(response, dict) and 'error' in response:
            raise RuntimeError(f"GraspGenX server error: {response['error']}")
        return response


def grasp_to_tcp(cfg: dict) -> np.ndarray:
    """4x4 GraspGen-X grasp frame -> robot TCP, from grasp_robot.grasp_to_tcp {translation, quaternion_xyzw}."""
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(cfg['grasp_to_tcp']['quaternion_xyzw']).as_matrix()
    T[:3, 3] = cfg['grasp_to_tcp']['translation']
    return T


def segment_points(adapter: BaseEnvAdapter) -> tuple[np.ndarray, np.ndarray, Callable[[set], np.ndarray]]:
    """Camera point cloud (H, W, 3, world frame), its valid-depth mask, and a function from a set of segment
    names to the mask of valid points on those segments."""
    _, _, points, seg, seg_id_to_name = adapter.get_keypoint_detection_inputs()
    valid = np.linalg.norm(points, axis=-1) > 0  # back-projection marks invalid depth as 0
    return points, valid, lambda names: np.isin(seg, [i for i, n in seg_id_to_name.items() if n in names]) & valid


def score_tcp_pose(client: 'GraspGenXClient', robot_cfg: dict, adapter: BaseEnvAdapter,
                   objects: list[str]) -> dict[str, Optional[float]]:
    """Evaluation only: GraspGen-X discriminator confidence of the current TCP pose as a grasp on each object
    (segment `<name>` or `<name>_main`); None where fewer than MIN_OBJECT_POINTS are visible."""
    points, _, mask_of = segment_points(adapter)
    grasp_G = adapter.get_ee_pose_world().to_transformation_matrix() @ np.linalg.inv(grasp_to_tcp(robot_cfg))
    scores = {}
    for name in objects:
        object_pc = points[mask_of({name, f'{name}_main'})]
        if len(object_pc) < MIN_OBJECT_POINTS:
            scores[name] = None
            continue
        response = client.request({'action': 'score_poses', 'gripper_name': robot_cfg['gripper_name'],
                                   'object_pc': object_pc.astype(np.float32),
                                   'poses': grasp_G[None].astype(np.float32)})
        scores[name] = float(np.asarray(response['confidences']).reshape(-1)[0])
    return scores


def select_grasps(grasps_G: np.ndarray, conf: np.ndarray, cfg: dict, ee_pos: np.ndarray,
                  part_kp: Optional[np.ndarray] = None, approach: Optional[str] = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Filter, reweight and rank grasps; return (K, 4, 4) TCP targets and (K,) log scores.

    approach: VLM constraint, 'top_down' (approach within approach_max_deg of straight down) or
        'side' (approach within approach_max_deg of horizontal).
    part_kp: VLM part keypoint (e.g. a handle); scores are multiplied by exp(-|g - kp|^2 / sigma^2).
    """
    approach_z = grasps_G[:, 2, 2]  # world z component of each grasp's approach axis
    cos_max = np.cos(np.radians(cfg['approach_max_deg']))
    if approach == 'top_down':
        keep = -approach_z >= cos_max
    elif approach == 'side':
        keep = np.abs(approach_z) <= np.sin(np.radians(cfg['approach_max_deg']))
    else:
        keep = np.ones(len(grasps_G), dtype=bool)
    if not keep.any():
        log.warning(f"No grasp satisfies approach '{approach}'; ignoring the approach constraint")
        keep[:] = True
    tcp = grasps_G[keep] @ grasp_to_tcp(cfg)
    log_scores = np.log(np.clip(conf[keep], 1e-6, None))
    if part_kp is not None:
        log_scores = log_scores - ((tcp[:, :3, 3] - part_kp) ** 2).sum(-1) / cfg['keypoint_sigma'] ** 2

    order = np.argsort(-log_scores)[:cfg['top_k']]
    if cfg['mode'] == 'nearest':  # nearest of the same quality top-k the other modes steer toward
        order = order[np.argsort(((tcp[order, :3, 3] - ee_pos) ** 2).sum(-1))[:1]]
    return tcp[order], log_scores[order]


class GraspReward:
    """Grasp-stage reward over predicted TCP poses; drop-in for a VLM guidance function.

    `takes_poses` tells PI05PolicySteer to pass (B, T+1, 4, 4) poses instead of (B, T, 3) positions.
    """

    takes_poses = True

    def __init__(self, tcp_poses: np.ndarray, log_scores: np.ndarray, cfg: dict):
        self.targets = torch.as_tensor(tcp_poses, dtype=torch.float32)
        self.log_scores = torch.as_tensor(log_scores, dtype=torch.float32)
        self.log_coupling = torch.zeros_like(self.log_scores)
        self.committed: Optional[int] = None
        self.commit_reason: Optional[str] = None
        self.commit_step: Optional[int] = None
        self.cfg = cfg
        self.cost_kwargs = dict(
            pregrasp_dist=cfg['pregrasp_dist'], sigma_trans=cfg['sigma_trans'],
            rot_weight=rotation_weight(cfg['rot_deg_equiv']),
            gate_dist=cfg['gate_dist'], gate_tau=cfg['gate_tau'], rot_tol_deg=cfg['rot_tol_deg'],
        )

    def _active(self, device) -> tuple[Tensor, Tensor]:
        """Targets and log scores in play: all grasps, or only the committed one."""
        idx = slice(None) if self.committed is None else slice(self.committed, self.committed + 1)
        log_s = (self.log_scores + self.log_coupling)[idx]
        return self.targets[idx].to(device), log_s.to(device)

    def costs(self, poses: Tensor) -> Tensor:
        targets, _ = self._active(poses.device)
        return grasp_costs(poses, targets.to(poses.dtype), **self.cost_kwargs)

    def particle_rewards(self, poses: Tensor) -> Tensor:
        """(B,) soft-min reward per particle, in units where sigma_trans of error costs 1.

        Shifted by a constant so that reaching the best-scored grasp gives 0 and far away is strongly
        negative, like the VLM keypoint rewards that PI05PolicySteer's reward normalization and the
        stage triggers assume.
        """
        _, log_s = self._active(poses.device)
        log_s = log_s.to(poses.dtype)
        beta = self.cfg['beta']
        return soft_min_reward(self.costs(poses), log_s, beta) - log_s.max() / beta

    def guidance_cost(self, poses: Tensor) -> Tensor:
        """(B,) cost whose value is the nearest grasp's cost and whose gradient is the quality-weighted soft-min's
        (sum_k w_k grad c_k, the gradient of soft_min_reward). The soft-min value itself carries a
        +log(K)/beta bonus when weights are spread, which would make VLS think the stage is nearly done."""
        _, log_s = self._active(poses.device)
        costs = self.costs(poses)
        weights = soft_min_weights(costs, log_s.to(poses.dtype), self.cfg['beta']).detach()
        soft = (weights * costs).sum(-1)
        nearest = costs.min(dim=-1).values
        return soft - (soft - nearest).detach()

    def __call__(self, keypoints: Tensor, poses: Tensor) -> Tensor:
        """Guidance reward: -sqrt(cost), i.e. linear in distance (sigma units), 0 at a grasp.

        PI05PolicySteer eases guidance off as 1 - r / r_init rises (sigmoid centred at 0.75, stage trigger at
        0.8). With a squared cost that happens at ~half the starting distance (~12 cm from 25 cm), before the
        orientation gate opens; linear in distance it happens at ~1/4 of it, and orientation error keeps r low
        until the gripper is within rot_tol_deg of a grasp. Gradients are unit-normalized downstream, so the
        direction is unchanged.
        """
        del keypoints  # in the grasp stage keypoints only reweight grasps (at acquisition)
        return -torch.sqrt(self.guidance_cost(poses) + 1e-6).mean()

    def fkd_rewards(self, poses: Tensor) -> Tensor:
        """Per-particle reward for FKD resampling, rescaled to m^2 like the VLM keypoint rewards
        (-dist^2) so that fkd.lmbda keeps its meaning across stages."""
        return self.cfg['sigma_trans'] ** 2 * self.particle_rewards(poses)

    def weights(self, poses: Tensor) -> Tensor:
        _, log_s = self._active(poses.device)
        return soft_min_weights(self.costs(poses), log_s.to(poses.dtype), self.cfg['beta'])

    def end_distance(self, poses: Tensor) -> Tensor:
        """(B,) distance from each trajectory's final TCP position to the nearest active grasp."""
        targets, _ = self._active(poses.device)
        return torch.cdist(poses[:, -1, :3, 3], targets[:, :3, 3].to(poses.dtype)).min(dim=1).values

    def set_coupling(self, unsteered_poses: Tensor, beta: float) -> None:
        """D.1: s_k <- s_k * sum_n exp(-beta * c_nk) over unsteered VLA particles (in log space)."""
        if self.committed is not None:
            return
        with torch.no_grad():
            log_f = torch.logsumexp(-beta * self.costs(unsteered_poses), dim=0).cpu()
        self.log_coupling = log_f - log_f.max()

    def update_commit(self, executed_poses: Tensor, step: Optional[int] = None) -> Optional[str]:
        """Lock onto one grasp once it dominates the soft-min or the TCP is close to it, either way only if the TCP
        is within commit_rot_deg of its orientation (no mode switching)."""
        if self.committed is not None:
            return None
        with torch.no_grad():
            w = self.weights(executed_poses)[0]
            targets = self.targets.to(executed_poses)
            dist = torch.norm(executed_poses[0, -1, :3, 3] - targets[:, :3, 3], dim=-1)
            rot_ok = (orientation_cost(executed_poses[0, -1, :3, :3], targets[:, :3, :3])
                      <= angle_to_orientation_cost(self.cfg['commit_rot_deg']))
            dist = torch.where(rot_ok, dist, torch.full_like(dist, math.inf))
        if len(self.targets) == 1:
            self.committed, self.commit_reason = 0, 'single'
        elif w.max() > self.cfg['commit_weight'] and rot_ok[w.argmax()]:
            self.committed, self.commit_reason = int(w.argmax()), 'weight'
        elif dist.min() < self.cfg['commit_dist']:
            self.committed, self.commit_reason = int(dist.argmin()), 'distance'
        if self.committed is not None:
            self.commit_step = step
            return self.commit_reason
        return None


@dataclass
class Acquisition:
    """One GraspGen-X query for the target object."""
    object_pc: np.ndarray  # (N, 3) world
    scene_pc: np.ndarray  # (M, 3) world, cropped around the object
    anchor: np.ndarray  # (3,) mean of the target object's tracked keypoints at acquisition
    infer_ms: float


class GraspSteering:
    """Per-episode grasp-stage controller called from main.Main._run_episode."""

    def __init__(self, cfg: dict, adapter, policy):
        if cfg['mode'] not in MODES:
            raise ValueError(f"main.grasp.mode must be one of {MODES}, got {cfg['mode']!r}")
        if type(adapter).delta_actions_to_ee_poses is BaseEnvAdapter.delta_actions_to_ee_poses:
            raise ValueError(f"Grasp steering needs {type(adapter).__name__}.delta_actions_to_ee_poses "
                             f"(predicted TCP poses); only LIBERO implements it so far")
        if not hasattr(policy, '_sample_to_ee_poses'):
            raise ValueError(f"Grasp steering is wired into PI05PolicySteer only, not {type(policy).__name__} "
                             f"(CALVIN's diffusion policy needs the same hooks in core/diffusion_policy_steer.py)")
        self.cfg = cfg
        self.adapter = adapter
        self.policy = policy
        self.client = GraspGenXClient(cfg['endpoint'])
        self.client.request({'action': 'health'})  # fail fast if the server is not running
        self.rng = np.random.default_rng(0)
        self.reset_episode(None)

    def reset_episode(self, grasp_info: Optional[dict]) -> None:
        """grasp_info: the VLM's grasp fields from metadata.json (grasp_stage, grasp_object, ...)."""
        self.info = grasp_info or {}
        self.reward: Optional[GraspReward] = None
        self.acq: Optional[Acquisition] = None
        self.record: dict = {}
        self.engaged_chunks = 0  # chunks steered by GraspReward this episode (0 = ran as plain VLS)

    def is_grasp_stage(self, stage: int) -> bool:
        return self.info.get('grasp_stage') == stage and bool(self.info.get('grasp_object'))

    # ---------------------------------------------------------------- acquisition
    def _point_clouds(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        points, valid, mask_of = segment_points(self.adapter)
        obj_mask = mask_of({self.info['grasp_object']})
        object_pc = points[obj_mask]
        if len(object_pc) < MIN_OBJECT_POINTS:
            log.warning(f"Only {len(object_pc)} points on '{self.info['grasp_object']}'; skipping grasp guidance")
            return None
        # Scene for collision filtering: everything else near the object, minus the robot's own gripper
        scene = points[valid & ~obj_mask]
        ee_pos = self.adapter.get_ee_pose_world().position
        scene = scene[(np.linalg.norm(scene - object_pc.mean(0), axis=-1) < SCENE_CROP_RADIUS)
                      & (np.linalg.norm(scene - ee_pos, axis=-1) > self.cfg['robot_exclusion_radius'])]
        if len(scene) > MAX_SCENE_POINTS:
            scene = scene[self.rng.choice(len(scene), MAX_SCENE_POINTS, replace=False)]
        return object_pc.astype(np.float32), scene.astype(np.float32)

    def _query(self, action: str, **payload) -> tuple[np.ndarray, np.ndarray, float]:
        response = self.client.request({
            'action': action, 'gripper_name': self.cfg['gripper_name'],
            'object_pc': self.acq.object_pc, 'scene_pc': self.acq.scene_pc,
            'num_grasps': self.cfg['num_grasps'], 'collision_threshold': self.cfg['collision_threshold'], **payload,
        })
        return (np.asarray(response['grasps'], dtype=np.float64).reshape(-1, 4, 4),
                np.asarray(response['confidences'], dtype=np.float64).reshape(-1),
                float(response['timing']['infer_ms']))

    def _new_reward(self, grasps_G: np.ndarray, conf: np.ndarray, part_kp: Optional[np.ndarray]
                    ) -> Optional[GraspReward]:
        if len(grasps_G) == 0:
            return None
        tcp, log_s = select_grasps(grasps_G, conf, self.cfg, self.adapter.get_ee_pose_world().position,
                                   part_kp, self.info.get('grasp_approach'))
        return GraspReward(tcp, log_s, self.cfg)

    def _acquire(self, anchor: np.ndarray, part_kp: Optional[np.ndarray]) -> bool:
        clouds = self._point_clouds()
        if clouds is None:
            self.reward = None
            return False
        self.acq = Acquisition(*clouds, anchor=anchor, infer_ms=0.0)
        planner = 'diffusion' if self.cfg['mode'] == 'coupled_full' else self.cfg['planner']
        grasps_G, conf, self.acq.infer_ms = self._query('infer_object_filtered', planner=planner)
        log.info(f"GraspGen-X: {len(grasps_G)} collision-free grasps on '{self.info['grasp_object']}' "
                 f"in {self.acq.infer_ms:.0f} ms")
        self.record['acquired'] = True
        self.policy.reset_stage()  # new targets: the stage's reward baseline no longer applies
        # On a failed (re)acquisition the old grasps are stale (the object moved): fall back to the VLM function
        self.reward = self._new_reward(grasps_G, conf, part_kp)
        if self.reward is None:
            log.warning("GraspGen-X returned no collision-free grasps; using the VLM stage function")
        return self.reward is not None

    # ---------------------------------------------------------------- per chunk
    def guidance_fns(self, keypoints: np.ndarray, keypoint_names: list,
                     unsteered: Optional[Tensor]) -> Optional[list]:
        """Grasp-stage guidance for the next chunk, or None to fall back to the VLM stage function.

        keypoints: (K, 3) tracked keypoints; keypoint_names: object name per keypoint.
        unsteered: (B, H, 7) normalized unsteered action chunk (needed by the coupled modes).
        """
        self.record = {'mode': self.cfg['mode'], 'acquired': False}
        obj_kps = keypoints[[n == self.info['grasp_object'] for n in keypoint_names]]
        if len(obj_kps) == 0 and self.acq is None:
            log.warning(f"No tracked keypoints on '{self.info['grasp_object']}'; object motion cannot trigger a re-acquire")
        anchor = obj_kps.mean(0) if len(obj_kps) else np.full(3, np.nan)
        kp_idx = self.info.get('grasp_keypoint')
        part_kp = keypoints[kp_idx] if kp_idx is not None and 0 <= kp_idx < len(keypoints) else None

        moved = self.acq is not None and np.linalg.norm(anchor - self.acq.anchor) > self.cfg['reacquire_dist']
        if self.acq is None or moved:
            if moved:
                log.info(f"Target object moved {np.linalg.norm(anchor - self.acq.anchor):.3f} m; re-running GraspGen-X")
            if not self._acquire(anchor, part_kp):
                return None
        if self.reward is None:
            return None

        if self.reward.committed is None and unsteered is not None and self.cfg['mode'].startswith('coupled'):
            unsteered_poses = self.policy._sample_to_ee_poses(unsteered.to(self.policy.config.device))
            if self.cfg['mode'] == 'coupled_cheap':
                self.reward.set_coupling(unsteered_poses, self.cfg['coupling_beta'])
            else:
                targets_G = unsteered_poses[:, -1].detach().cpu().double().numpy() @ np.linalg.inv(
                    grasp_to_tcp(self.cfg))
                grasps_G, conf, infer_ms = self._query(
                    'infer_guided', target_poses=targets_G.astype(np.float32),
                    guide_scale=self.cfg['coupled_guide_scale'], sigma_trans=self.cfg['sigma_trans'],
                    rot_weight=rotation_weight(self.cfg['rot_deg_equiv']))
                self.record['guided_infer_ms'] = infer_ms
                guided = self._new_reward(grasps_G, conf, part_kp)
                if guided is not None:
                    self.reward = guided
                else:  # keep steering toward the previous grasp set rather than dropping to the VLM function
                    self.record['guided_empty'] = True
                    log.warning("Guided GraspGen-X query returned no collision-free grasps; keeping the previous set")
        self.engaged_chunks += 1
        return [self.reward]

    def fkd_extra_score(self, poses: Tensor) -> Tensor:
        """E: gradient-free discriminator term for FKD resampling, gated on near the grasp.

        disc_weight * gate(distance to nearest grasp, disc_gate_dist) * log(confidence of the terminal TCP pose as a grasp).
        """
        terminal_G = poses[:, -1].detach().cpu().double().numpy() @ np.linalg.inv(grasp_to_tcp(self.cfg))
        response = self.client.request({
            'action': 'score_poses', 'gripper_name': self.cfg['gripper_name'],
            'object_pc': self.acq.object_pc, 'poses': terminal_G.astype(np.float32),
        })
        conf = torch.as_tensor(np.array(response['confidences']), dtype=poses.dtype, device=poses.device)
        with torch.no_grad():
            gate = torch.sigmoid((self.cfg['disc_gate_dist'] - self.reward.end_distance(poses)) / self.cfg['gate_tau'])
        return self.cfg['disc_weight'] * gate * torch.log(conf.clamp_min(1e-3))

    def after_chunk(self, executed_sample: Tensor, step: Optional[int] = None) -> Optional[Tensor]:
        """Commit check on the executed particle's predicted poses; fill the chunk record.
        Returns those (1, H+1, 4, 4) poses so the chunk logger does not recompute them."""
        if self.reward is None:
            return None
        with torch.no_grad():
            poses = self.policy._sample_to_ee_poses(executed_sample)
            weights = self.reward.weights(poses)[0]
            # Predicted chunk-end error to the committed (else most-weighted) grasp: did orientation get steered?
            k = self.reward.committed if self.reward.committed is not None else int(weights.argmax())
            target = self.reward.targets[k].to(poses)
            end_dist = torch.norm(poses[0, -1, :3, 3] - target[:3, 3])
            end_rot_deg = orientation_cost_to_deg(orientation_cost(poses[0, -1, :3, :3], target[None, :3, :3])[0])
            # Tilt = angle between approach axes; the rest is twist about the approach axis, which does not
            # matter for objects that are symmetric about it (cans, bottles)
            end_tilt_deg = torch.rad2deg(torch.arccos((poses[0, -1, :3, 2] @ target[:3, 2]).clamp(-1, 1)))
            # Every candidate's approach-axis angle from the current TCP approach: are policy-compatible grasps on offer?
            cand_tilt_deg = torch.rad2deg(torch.arccos(
                (self.reward.targets[:, :3, 2].to(poses) @ poses[0, 0, :3, 2]).clamp(-1, 1)))
        reason = self.reward.update_commit(poses, step)
        if reason:
            log.info(f"Committed to grasp {self.reward.committed} ({reason})")
        self.record.update({
            'num_grasps': len(self.reward.targets),
            'grasp_log_scores': (self.reward.log_scores + self.reward.log_coupling).tolist(),
            'softmin_weights': weights.cpu().tolist(),
            'committed': self.reward.committed,
            'commit_reason': self.reward.commit_reason,
            'commit_step': self.reward.commit_step,
            'end_dist_cm': 100 * float(end_dist),  # executed particle's predicted chunk end vs its target grasp
            'end_rot_err_deg': float(end_rot_deg),  # jaw-symmetric
            'end_tilt_deg': float(end_tilt_deg),  # approach-axis part of end_rot_err_deg
            'candidate_tilt_deg': cand_tilt_deg.cpu().tolist(),
            'committed_pose': (self.reward.targets[self.reward.committed].tolist()
                               if self.reward.committed is not None else None),
            'acquire_infer_ms': self.acq.infer_ms if self.record.get('acquired') else None,
        })
        return poses
