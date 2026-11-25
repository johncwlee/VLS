# Checks the 2026-11-24 pose-prompt change: other modes byte-identical to HEAD, pose text present, example fn runs and its barriers fire.
import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, REPO)
from closed_loop import prompt as new_prompt
from closed_loop.ee_input import check_on_dummy_poses, wrap_for_poses
from utils.guidance_utils import load_functions_from_txt

# HEAD copy of closed_loop/ in a temp package, to build the old prompts
old_dir = os.path.join(tempfile.mkdtemp(), 'closed_loop')
os.makedirs(old_dir)
for name in subprocess.run(['git', 'ls-tree', '--name-only', 'HEAD', 'closed_loop/'], cwd=REPO, capture_output=True,
                           text=True, check=True).stdout.split():
    with open(os.path.join(old_dir, os.path.basename(name)), 'w') as f:
        f.write(subprocess.run(['git', 'show', f'HEAD:{name}'], cwd=REPO, capture_output=True, text=True, check=True).stdout)
spec = importlib.util.spec_from_file_location('old_prompt', os.path.join(old_dir, 'prompt.py'))
old_prompt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_prompt)
old_prompt.TEMPLATE_DIR = old_dir

ee_pose_now = (np.array([-0.2, 0.0, 1.17]), np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, -1]]))
common = dict(instruction='open the bottom drawer of the cabinet', history=[], trigger_reason='initial query',
              gripper_state='open', key_points_objects_map={0: 'a', 1: 'b'}, keypoint_positions=np.zeros((2, 3)),
              task_specific_patterns='notes', ee_pose_now=ee_pose_now)
cases = [dict(mode='first', plan=[], plan_id=0, current_stage=1),
         dict(mode='check', plan=['reach handle', 'pull'], plan_id=1, current_stage=1)]
for ee_input in ('position', 'gripper_points'):
    for case in cases:
        old = old_prompt.build_prompt(**case, **common, ee_input=ee_input)
        new = new_prompt.build_prompt(**case, **common, ee_input=ee_input, table_height_m=0.9)
        print(f'{ee_input:14s} {case["mode"]:5s}: old {len(old)} chars -> new {len(new)} chars, expected identical -> '
              f'{"match" if old == new else "MISMATCH"}')

pose_text = new_prompt.build_prompt(**cases[0], **common, ee_input='pose', table_height_m=0.8999)
expected_bits = ['DO NOT add any obstacle avoidance', 'gripper body fills this box', 'Table surface height (estimated',
                 'z = 0.900 m', 'table_z = 0.900', 'Contact without closing the gripper', '{table_z}',
                 'route the gripper around them with stages', 'obstacle_cost']
for bit in expected_bits:
    want = bit not in ('DO NOT add any obstacle avoidance', '{table_z}', 'obstacle_cost')
    got = bit in pose_text
    print(f'pose prompt contains {bit!r}: {got}, expected {want} -> {"match" if got == want else "MISMATCH"}')

# Example function: run the code block exactly as the prompt shows it
example_src = new_prompt._read('example_pose.txt').replace('{table_z}', '0.900')
path = os.path.join(tempfile.mkdtemp(), 'stage1_guidance.txt')
with open(path, 'w') as f:
    f.write(example_src)
fn = load_functions_from_txt(path, validate=False)[0]
check_on_dummy_poses(fn)
print('example passes check_on_dummy_poses: True, expected True -> match')

# Barrier check: example's target is keypoint 0; put it at task 0's handle height, gripper level, fingers vertical
# (call 6's infeasible pose: hand 5.2 cm into the table), vs the same spot with fingers horizontal
# keypoint 4: obstacle side 10 cm behind the target in -Y, so the example's free direction is +Y
kp = torch.tensor([[0.056, -0.137, 0.952], [0.0, 0.0, 0.952], [0.0, 0.0, 1.052], [0.5, 0.5, 1.5], [0.056, -0.237, 0.952]],
                  dtype=torch.float64)
R_vertical = torch.tensor([[0.0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=torch.float64)   # x (finger) = -Z, z (approach) = -Y
R_horizontal = torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float64)  # x (finger) = +X, z (approach) = -Y
for name, R, want_positive in (('fingers vertical', R_vertical, True), ('fingers horizontal', R_horizontal, False)):
    pose = torch.eye(4, dtype=torch.float64)
    pose[:3, :3], pose[:3, 3] = R, kp[0]
    poses = pose.expand(1, 11, 4, 4)
    body_ee = torch.tensor([[x, y, z] for x in (-0.106, -0.053, 0.0, 0.053, 0.106)
                            for y in (-0.032, 0.032) for z in (-0.124, 0.012)], dtype=torch.float64)
    body = torch.einsum('btij,nj->btni', poses[..., :3, :3], body_ee) + poses[..., :3, 3].unsqueeze(2)
    table_cost = (torch.relu(0.900 + 0.01 - body[..., 2]) ** 2).sum(dim=-1)[0, 0].item()
    print(f'{name}: lowest body point z {body[..., 2].min().item():.3f} m, table_cost {table_cost:.2e}, '
          f'expected {"> 0" if want_positive else "0"} -> {"match" if (table_cost > 0) == want_positive else "MISMATCH"}; '
          f'full reward {fn(kp.float(), poses.float()).item():.2e}')

# Directional approach: same 5 cm from the target, on the approach line (out along free_dir = +Y) vs off it (+X)
rewards = {}
for name, offset in (('on line', [0.0, 0.05, 0.0]), ('off line', [0.05, 0.0, 0.0])):
    pose = torch.eye(4, dtype=torch.float64)
    pose[:3, :3], pose[:3, 3] = R_horizontal, kp[0] + torch.tensor(offset, dtype=torch.float64)
    rewards[name] = fn(kp.float(), pose.expand(1, 11, 4, 4).float()).item()
print(f'reward 5 cm from target: on line {rewards["on line"]:.2e}, off line {rewards["off line"]:.2e}, expected on > off -> '
      f'{"match" if rewards["on line"] > rewards["off line"] else "MISMATCH"}')
