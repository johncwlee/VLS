# RLinf pi0.5 in RPent's env: replay our handle-start OSC waypoints for libero_goal_task task 0 init k, then run only the VLA with the short prompt for 300 env steps.
# Usage (.venv-rpent, RPent dir): python <this file> INIT_INDEX POSES_JSON OUT_DIR
import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.getcwd())
from robots.libero.robot_spec import _init_runtime
from robots.libero.tools import LiberoPrimitives
from rpent.dashboard.events import NullDashboardEventSink

SHORT_PROMPT = 'pull the lowest gray drawer handle outward'  # same as the lerobot runs
VLA_ENV_STEPS = 300  # main.py episode length in our handle-start runs (steps.jsonl has 300 lines)
OSC_MAX_STEP_M = 0.05  # same gains as scratch/2026-11-25_handle_start_pose.py drive_osc
OSC_MAX_STEP_RAD = 0.5

init_index, poses_path, out_dir = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
with open(poses_path) as f:
    pose_record = json.load(f)[init_index]
assert pose_record['init_index'] == init_index

args = argparse.Namespace(suite='libero_goal_task', task=0, seed=init_index, max_episode_steps=10000,
                          env_endpoint=None, vla_endpoint=None, cuda_device=0, libero_type='pro',
                          collect_flywheel_data=False, planner='none')
daemons, runtime_kwargs = _init_runtime(args, out_dir, NullDashboardEventSink(), {'env', 'vla'})
primitives = LiberoPrimitives(env=runtime_kwargs['env'], model=runtime_kwargs['model'], sam3_client=None,
                              check_cancelled=lambda: None)
primitives.reset()
primitives.start_recording()
env = primitives.env


def eef_pose():
    raw = env.raw_obs()
    return np.asarray(raw['robot0_eef_pos'], dtype=np.float64), Rotation.from_quat(raw['robot0_eef_quat']).as_matrix()


def drive_osc(target_pos_m, R_target, max_steps, tol_m, tol_deg):
    """Same servo as drive_osc in 2026-11-25_handle_start_pose.py, on robot0_eef_quat. R_target @ R_now.T is the
    same world rotation in the hand-body and grip-site frames (they differ by a fixed right-multiplied offset)."""
    for _ in range(max_steps):
        pos_m, R_now = eef_pose()
        error_m = target_pos_m - pos_m
        error_rotvec_rad = Rotation.from_matrix(R_target @ R_now.T).as_rotvec()
        if np.linalg.norm(error_m) < tol_m and np.degrees(np.linalg.norm(error_rotvec_rad)) < tol_deg:
            break
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(error_m / OSC_MAX_STEP_M, -1.0, 1.0)
        action[3:6] = np.clip(error_rotvec_rad / OSC_MAX_STEP_RAD, -1.0, 1.0)
        action[6] = -1.0  # fingers open
        primitives._step_env(action)  # as RPent's move_to does (tools.py:398-405); execute_action's spec RPC fails


for waypoint in pose_record['waypoints']:
    drive_osc(np.array(waypoint['eef_pos']), Rotation.from_quat(waypoint['eef_quat_xyzw']).as_matrix(),
              waypoint['max_steps'], waypoint['tol_m'], waypoint['tol_deg'])
primitives._step_env(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))  # hold step, as the handle-start reset does
start_frames = primitives.recorded_frame_count()
pos_m, R_now = eef_pose()
start = {'reached_eef_pos': pos_m.tolist(),
         'pos_error_vs_our_start_cm': 100 * float(np.linalg.norm(pos_m - np.array(pose_record['reached_eef_pos']))),
         'rot_error_vs_our_start_deg': float(np.degrees(np.linalg.norm(Rotation.from_matrix(
             Rotation.from_quat(pose_record['reached_eef_quat_xyzw']).as_matrix() @ R_now.T).as_rotvec()))),
         'osc_steps': start_frames}
print('[start]', json.dumps(start), flush=True)

chunks = 0
while primitives.recorded_frame_count() - start_frames < VLA_ENV_STEPS and not (env.terminated or env.truncated):
    primitives._vlm_chunk(SHORT_PROMPT)
    chunks += 1
vla_steps = primitives.recorded_frame_count() - start_frames
result = {'init_index': init_index, 'prompt': SHORT_PROMPT, 'success': bool(env.terminated), 'vla_env_steps': vla_steps,
          'vla_chunks': chunks, 'steps_per_chunk': vla_steps / max(chunks, 1), 'start': start}
print('[result]', json.dumps(result), flush=True)
with open(out_dir / 'result.json', 'w') as f:
    json.dump(result, f, indent=1)
frames = primitives.stop_recording()
imageio.mimwrite(out_dir / f"episode_{'success' if result['success'] else 'fail'}.mp4", frames[start_frames:], fps=20,
                 macro_block_size=1)
imageio.mimwrite(out_dir / 'episode_with_approach.mp4', frames, fps=20, macro_block_size=1)
for daemon in daemons:
    daemon.stop()
