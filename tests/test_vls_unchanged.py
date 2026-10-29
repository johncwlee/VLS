"""With grasp steering off, VLS must behave as before the feature branch (base commit b113a14).

Compares the keypoint-guidance gradient, the constant EE path model and the VLM prompt against the code
at that commit. Intended exceptions (bug fixes decided separately, see plans/graspgen-x_plan.md): FKD
resampling now runs for pi0.5, and main.start_ratio overrides now take effect.
"""

import importlib.util
import subprocess
import types
from pathlib import Path

import pytest
import torch

from core.env_adapters.libero_adapter import predict_ee_positions
from core.pi05_steer import PI05PolicySteer

BASE_COMMIT = "b113a14"
VLS_ROOT = Path(__file__).resolve().parent.parent


def base_file(path: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(VLS_ROOT), "show", f"{BASE_COMMIT}:{path}"],
                              capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"base commit {BASE_COMMIT} not available")


def load_base_module(path: str, tmp_path):
    src = tmp_path / Path(path).name
    src.write_text(base_file(path))
    spec = importlib.util.spec_from_file_location(f"base_{src.stem}", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeLiberoAdapter:
    """Stand-in with the original constant EE model, enough for the steering math."""

    start = torch.tensor([0.4, 0.1, 0.3])

    def env_postprocessor(self, transition):
        return transition

    def delta_actions_to_ee_trajectory(self, actions):
        return predict_ee_positions(actions, self.start.to(actions), {'ee_prediction': 'constant'}, {})

    def steerable_action_dims(self):
        return slice(0, 6)


def steer_stub(cls):
    stub = types.SimpleNamespace(_adapter=FakeLiberoAdapter(), _postprocessor=lambda x: 2.0 * x + 0.1,
                                 _action_chunk_horizon=10)
    for name in ('_sample_to_trajectory_3d', '_sample_to_ee_poses'):
        if hasattr(cls, name):
            setattr(stub, name, types.MethodType(getattr(cls, name), stub))
    return stub


def keypoint_reward(keypoints, traj):
    return -(torch.norm(traj - keypoints[0], dim=-1) ** 2).mean(dim=1).mean()


def test_keypoint_gradient_matches_base(tmp_path):
    base = load_base_module("core/pi05_steer.py", tmp_path).PI05PolicySteer
    sample = torch.randn(20, 50, 7)
    keypoints = torch.tensor([[0.5, 0.0, 0.1]])
    old_grad, old_reward = base._compute_keypoint_gradient(steer_stub(base), sample, keypoints, [keypoint_reward])
    new_grad, new_reward, _ = PI05PolicySteer._compute_keypoint_gradient(
        steer_stub(PI05PolicySteer), sample, keypoints, [keypoint_reward])
    assert new_reward == pytest.approx(old_reward)
    assert torch.allclose(new_grad, old_grad)
    assert torch.all(new_grad[..., 3:] == 0), "position-only rewards must not steer rotation"


def test_constant_ee_model_matches_base():
    actions = torch.randn(10, 7)
    start = torch.tensor([0.4, 0.1, 0.3])
    expected = torch.cat([start[None], start + torch.cumsum(actions[:, :3] * 0.01, dim=0)])  # base: ACTION_SCALE_POS
    assert torch.allclose(predict_ee_positions(actions, start, {'ee_prediction': 'constant', 'action_scale_pos': 0.01}, {}),
                          expected)


def test_prompt_unchanged_without_grasp_steering():
    template = (VLS_ROOT / "vlm_query" / "guidance_template.txt").read_text()
    assert template.replace("{grasp_fields}", "") == base_file("vlm_query/guidance_template.txt")


def test_new_features_default_off():
    """Every GraspGen-X-related switch must default to the pre-feature behaviour."""
    import yaml
    main = yaml.safe_load((VLS_ROOT / "configs" / "config.yaml").read_text())['main']
    libero = yaml.safe_load((VLS_ROOT / "configs" / "backend" / "libero.yaml").read_text())['libero']
    assert main['grasp']['enabled'] is False and main['grasp']['score_at_close'] is False
    assert main['ee_prediction'] is None and libero['ee_prediction'] == 'constant'
    assert libero['action_scale_pos'] == 0.01  # VLS's original EE scale
    assert not any(main[k] for k in ('analysis_logs', 'log_unsteered', 'log_steps'))
