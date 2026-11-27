# Replays RPent's two task-0 commands (move_pose, then pi0_doubled) from t0_s{k}/states.json in RPent's env, with no planner, and a chosen pi0.5 noise seed.
# Usage (.venv-rpent, RPent dir): NOISE_SEED=n python <this file> INIT_INDEX OUT_DIR
import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, os.getcwd())
import robots.libero.robot_spec as robot_spec
from robots.libero.tools import LiberoPrimitives
from rpent.dashboard.events import NullDashboardEventSink

SCRATCH = os.path.dirname(os.path.abspath(__file__))
RPENT_RUN = os.path.join(SCRATCH, '..', 'outputs', 'rpent_goal_task_t01', '2026-11-26_18-42-23')

init_index, out_dir = int(sys.argv[1]), Path(sys.argv[2])
noise_seed = int(os.environ['NOISE_SEED'])
out_dir.mkdir(parents=True, exist_ok=True)
with open(os.path.join(RPENT_RUN, f't0_s{init_index}', 'states.json')) as f:
    commands = [step['command'] for step in json.load(f)['steps'][1:]]
assert [c['action'] for c in commands] == ['move_pose', 'pi0_doubled'], commands

# Start the VLA server through the seeding wrapper instead of directly
original_daemon_class = robot_spec.ProcessDaemon


class SeededDaemon(original_daemon_class):
    def __init__(self, *, name, cmd, **kwargs):
        if name == 'vla_server':
            cmd = [cmd[0], os.path.join(SCRATCH, '2026-11-26_seeded_vla_server.py')] + cmd[1:]
        super().__init__(name=name, cmd=cmd, **kwargs)


robot_spec.ProcessDaemon = SeededDaemon

args = argparse.Namespace(suite='libero_goal_task', task=0, seed=init_index, max_episode_steps=10000,
                          env_endpoint=None, vla_endpoint=None, cuda_device=0, libero_type='pro',
                          collect_flywheel_data=False, planner='none')
daemons, runtime_kwargs = robot_spec._init_runtime(args, out_dir, NullDashboardEventSink(), {'env', 'vla'})
primitives = LiberoPrimitives(env=runtime_kwargs['env'], model=runtime_kwargs['model'], sam3_client=None,
                              check_cancelled=lambda: None)
primitives.reset()
primitives.start_recording()

# First predicted chunk, to confirm different noise seeds give different actions
first_chunk = {}
gripper_commands = []  # last action dim of every predicted VLA action: -1 open, +1 close
original_predict = primitives.model.predict


def recording_predict(obs, options=None):
    actions = original_predict(obs, options)
    if 'actions' not in first_chunk:
        first_chunk['actions'] = np.asarray(actions).round(4).tolist()
    gripper_commands.extend(np.asarray(actions)[:, -1].tolist())
    return actions


primitives.model.predict = recording_predict

# Per recorded frame: finger qpos (states[6:8], m) and the wrist image
finger_qpos_per_frame = []
wrist_frames = []
original_record_frame = primitives.record_frame


def recording_record_frame(obs):
    original_record_frame(obs)
    finger_qpos_per_frame.append(np.asarray(obs['states'])[6:8].tolist())
    wrist_frames.append(np.ascontiguousarray(np.asarray(obs['wrist_images'])))


primitives.record_frame = recording_record_frame

move = {k: v for k, v in commands[0].items() if k != 'action'}
move_result = primitives.move_pose(**move).data
move_frames = primitives.recorded_frame_count()
vla = {k: v for k, v in commands[1].items() if k != 'action'}
vla_result = primitives.pi0_doubled(**vla).data
result = {'init_index': init_index, 'noise_seed': noise_seed, 'move_pose_args': move,
          'move_pose_final_eef_pos': move_result.get('final_eef_pos'), 'move_pose_final_dist_m': move_result.get('final_dist_m'),
          'pi0_doubled_args': vla, 'chunks_used': vla_result['chunks_used'], 'success': bool(primitives.env.terminated),
          'vla_env_steps': primitives.recorded_frame_count() - move_frames, 'first_chunk_first_action': first_chunk['actions'][0]}
print('[result]', json.dumps(result), flush=True)
with open(out_dir / 'result.json', 'w') as f:
    json.dump(result, f, indent=1)
with open(out_dir / 'gripper_log.json', 'w') as f:
    json.dump({'move_pose_frames': move_frames, 'vla_gripper_commands': gripper_commands,
               'finger_qpos_per_frame_m': finger_qpos_per_frame}, f)
frames = primitives.stop_recording()
imageio.mimwrite(out_dir / f"episode_{'success' if result['success'] else 'fail'}.mp4", frames, fps=20, macro_block_size=1)
imageio.mimwrite(out_dir / 'wrist.mp4', wrist_frames, fps=20, macro_block_size=1)
for daemon in daemons:
    daemon.stop()
