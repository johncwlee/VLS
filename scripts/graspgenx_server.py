"""
GraspGenX ZMQ server for VLS grasp steering. Runs in envs/graspgenx (GraspGenX's torch/diffusers pins
conflict with envs/vls); launch it with ./run_graspgenx_server.sh from the outer repo.

Subclasses GraspGenX's own GraspGenXZMQServer, so its actions (health, metadata, infer_object, ...)
keep working, and adds three actions used by core/grasp_guidance.py:

    infer_object_filtered  object_pc (+ scene_pc) -> grasps, collision-filtered against scene_pc
    score_poses            object_pc + poses -> discriminator confidence of arbitrary grasp poses
    infer_guided           object_pc + target_poses -> diffusion-only grasps whose DDPM sampling is
                           guided toward the target poses (coupled steering, D.2)

GraspGenX itself is not modified: guidance wraps the generator's diffusion_head for the duration of one
request (classifier guidance on the predicted noise). All poses are in the caller's frame (VLS: world).
"""

import argparse
import contextlib
import importlib.util
import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from graspgenx.samplers import run_planner_on_object
from graspgenx.samplers.graspmoe import _score_grasps_with_discriminator
from graspgenx.serving.zmq_server import GraspGenXZMQServer
from graspgenx.utils.collision_filter import filter_colliding_grasps
from graspgenx.utils.point_cloud import point_cloud_outlier_removal
from graspgenx.utils.transformations import rt_to_matrix

# Load VLS's pure-torch cost module by path: importing the `utils` package would pull in the rest of VLS
_spec = importlib.util.spec_from_file_location("grasp_costs", Path(__file__).parent.parent / "utils" / "grasp_costs.py")
grasp_costs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grasp_costs)

log = logging.getLogger("vls_graspgenx_server")

# The GraspGenX commit this wrapper was written against (diffusion_head call signature, scheduler and
# attribute names, _score_grasps_with_discriminator). A different HEAD may still work; it is only warned about.
GRASPGENX_COMMIT = "b942909"
GUIDED_SOFTMIN_BETA = 1.0  # soft-min temperature over target poses inside the guided sampler


class _GuidedHead(nn.Module):
    """Wraps the generator's diffusion_head to add classifier guidance toward target grasp poses.

    eps' = eps + scale * sqrt(1 - alpha_bar_k) * g / |g|, with g = d cost(x0_hat) / d x_t and
    x0_hat = (x_t - sqrt(1 - alpha_bar_k) * eps) / sqrt(alpha_bar_k) per position / rotation block
    (each block has its own scheduler). eps is treated as constant, so no gradient flows through the
    network. The gradient is normalized per grasp so `scale` is the noise-space step size.
    """

    def __init__(self, inner: nn.Module, gen: nn.Module, targets: torch.Tensor, scale: float,
                 sigma_trans: float, rot_weight: float):
        super().__init__()
        self.inner = inner
        # Plain attribute, not a submodule: gen owns this head, so registering gen here would create a module cycle
        object.__setattr__(self, "gen", gen)
        self.targets, self.scale = targets, scale
        self.sigma_trans, self.rot_weight = sigma_trans, rot_weight

    def forward(self, obs_embedding, k, x_t):
        eps = self.inner(obs_embedding, k, x_t)
        gen = self.gen
        # sample() runs under inference_mode; leave it locally and work on normal-tensor copies
        with torch.inference_mode(False), torch.enable_grad():
            x = x_t.detach().clone().requires_grad_(True)
            e = eps.detach().clone()
            ab_pos = gen.noise_scheduler_pos.alphas_cumprod[k].to(x)
            ab_rot = gen.noise_scheduler_rot.alphas_cumprod[k].to(x)
            x0 = torch.cat([
                (x[:, :3] - (1 - ab_pos).sqrt() * e[:, :3]) / ab_pos.sqrt(),
                (x[:, 3:] - (1 - ab_rot).sqrt() * e[:, 3:]) / ab_rot.sqrt(),
            ], dim=-1)
            poses = rt_to_matrix(x0, gen.grasp_repr, gen.kappa)  # centered object frame, metres
            cost = grasp_costs.pose_distance_cost(poses, self.targets, self.sigma_trans, self.rot_weight)
            softmin = -torch.logsumexp(-GUIDED_SOFTMIN_BETA * cost, dim=1) / GUIDED_SOFTMIN_BETA
            g = torch.autograd.grad(softmin.sum(), x)[0]
            g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            shift = torch.cat([(1 - ab_pos).sqrt() * g[:, :3], (1 - ab_rot).sqrt() * g[:, 3:]], dim=-1)
        return eps + self.scale * shift.to(eps.dtype)


class VLSGraspServer(GraspGenXZMQServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gen = self._shared_model.grasp_generator
        # Preconditions for _GuidedHead; fail at startup rather than mid-episode after a GraspGenX update
        assert gen.pose_repr == "mlp", f"guided sampling needs pose_repr 'mlp', got {gen.pose_repr!r}"
        assert gen.compositional_schedular, "guided sampling expects separate position/rotation schedulers"
        assert gen.grasp_repr in ("r3_6d", "r3_so3"), f"unsupported grasp_repr {gen.grasp_repr!r}"
        assert isinstance(gen.diffusion_head, nn.Module)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _clean_object_pc(request: dict) -> tuple[torch.Tensor, np.ndarray]:
        """Outlier-removed object cloud on GPU and its centre (the sampler's own preprocessing)."""
        pc = torch.from_numpy(GraspGenXZMQServer._validate_object_pc(request.get("object_pc"))).cuda().float()
        filtered, _ = point_cloud_outlier_removal(pc)
        if len(filtered) >= 10:
            pc = filtered
        return pc, pc.mean(dim=0).cpu().numpy().astype(np.float64)

    @staticmethod
    def _collision_filter(sampler, request: dict, grasps: np.ndarray, conf: np.ndarray):
        scene = np.asarray(request.get("scene_pc", np.zeros((0, 3))), dtype=np.float32).reshape(-1, 3)
        if len(scene) == 0 or len(grasps) == 0:
            return grasps, conf
        free = filter_colliding_grasps(scene, grasps, sampler.gripper.collision_mesh,
                                       collision_threshold=float(request["collision_threshold"]))
        return grasps[free], conf[free]

    @staticmethod
    def _response(grasps, conf, t0) -> dict:
        return {
            "grasps": np.asarray(grasps, dtype=np.float32).reshape(-1, 4, 4),
            "confidences": np.asarray(conf, dtype=np.float32).reshape(-1),
            "timing": {"infer_ms": (time.monotonic() - t0) * 1000.0},
        }

    # ------------------------------------------------------------------ actions
    def _handle_infer_object_filtered(self, request: dict) -> dict:
        t0 = time.monotonic()
        sampler = self._get_sampler(request.get("gripper_name") or self.default_gripper)
        pc = self._validate_object_pc(request.get("object_pc"))
        grasps, conf, _, _ = run_planner_on_object(pc, sampler, **self._parse_planner_kwargs(request))
        grasps, conf = self._collision_filter(sampler, request, np.asarray(grasps), np.asarray(conf))
        return self._response(grasps, conf, t0)

    def _handle_score_poses(self, request: dict) -> dict:
        t0 = time.monotonic()
        sampler = self._get_sampler(request.get("gripper_name") or self.default_gripper)
        pc, center = self._clean_object_pc(request)
        poses = np.asarray(request["poses"], dtype=np.float32).reshape(-1, 4, 4)
        conf = _score_grasps_with_discriminator(poses, pc - pc.mean(dim=0), center, sampler)
        return self._response(poses, conf, t0)

    @contextlib.contextmanager
    def _guided(self, targets: torch.Tensor, request: dict):
        gen = self._shared_model.grasp_generator
        inner = gen.diffusion_head
        gen.diffusion_head = _GuidedHead(inner, gen, targets, float(request["guide_scale"]),
                                         float(request["sigma_trans"]), float(request["rot_weight"]))
        try:
            yield
        finally:
            gen.diffusion_head = inner

    def _handle_infer_guided(self, request: dict) -> dict:
        t0 = time.monotonic()
        sampler = self._get_sampler(request.get("gripper_name") or self.default_gripper)
        pc, center = self._clean_object_pc(request)
        targets = torch.as_tensor(np.asarray(request["target_poses"], dtype=np.float32).reshape(-1, 4, 4)).cuda()
        targets[:, :3, 3] -= torch.as_tensor(center, dtype=torch.float32, device=targets.device)  # sampler's centred frame
        with self._guided(targets, request):
            # The cloud is already cleaned, so sample() centres it on exactly `center`. Diffusion only:
            # GraspMoE's OBB grasps are heuristic and cannot be guided.
            grasps, conf, _ = sampler.sample(pc, threshold=-1.0, num_grasps=int(request.get("num_grasps", 200)),
                                             remove_outliers=False)
        if len(grasps) == 0:
            return self._response(np.zeros((0, 4, 4)), np.zeros(0), t0)
        grasps = grasps.cpu().numpy()
        grasps[:, 3, 3] = 1.0  # same fix-up as GraspGenXSampler.run_inference
        grasps, conf = self._collision_filter(sampler, request, grasps, conf.cpu().numpy())
        return self._response(grasps, conf, t0)

    def _dispatch(self, request: dict) -> dict:
        handler = {
            "infer_object_filtered": self._handle_infer_object_filtered,
            "score_poses": self._handle_score_poses,
            "infer_guided": self._handle_infer_guided,
        }.get(request.get("action"))
        return handler(request) if handler else super()._dispatch(request)


def _warn_if_graspgenx_moved():
    import graspgenx
    repo = Path(graspgenx.__file__).resolve().parent.parent
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    if head != GRASPGENX_COMMIT:
        log.warning("GraspGenX is at %s, this server was written against %s; check _GuidedHead and "
                    "score_poses if requests fail.", head, GRASPGENX_COMMIT)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", default=os.path.join(os.environ.get("GRASPGENX_CHECKPOINT_DIR", ""), "release"),
                        help="checkpoint root with gen/ and dis/ (default: $GRASPGENX_CHECKPOINT_DIR/release)")
    parser.add_argument("--assets_dir",
                        default=os.path.join(os.environ.get("GRASPGENX_GRIPPER_CFG_DIR", ""), "gripper_descriptions", "assets"),
                        help="gripper assets root containing x_grippers/ "
                             "(default: $GRASPGENX_GRIPPER_CFG_DIR/gripper_descriptions/assets)")
    parser.add_argument("--gripper", default="franka_panda", help="gripper to pre-load and use by default")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--tensorrt", action="store_true", help="TensorRT diffusion head (needs the tensorrt extra)")
    args = parser.parse_args()

    _warn_if_graspgenx_moved()
    VLSGraspServer(config_path=args.checkpoints, assets_dir=args.assets_dir, host=args.host, port=args.port,
                   default_gripper=args.gripper, use_tensorrt=args.tensorrt).serve_forever()


if __name__ == "__main__":
    main()
