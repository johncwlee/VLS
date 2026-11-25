# Renders the EE pose each task 0 pose-mode reward scores highest (held still over the chunk), placed by IK with physics off, next to the real pose.
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import core.env_adapters.libero_adapter as libero_adapter
from utils.guidance_utils import load_functions_from_txt

EP_DIR = 'outputs/t0i0_ee_pose_contacts/2026-11-24_04-11-02/episode_1'
CALLS = ['call_03_step0070', 'call_06_step0180', 'call_08_step0260']
REAL_STEP = 180  # real EE pose shown for comparison: when call 6's reward took over
OUT_PATH = 'outputs/task0_ee_videos_2026-11-24/target_poses_pose_contacts.png'
HORIZON_PLUS_1 = 11  # pose mode passes (B, T+1, 4, 4), T = 10 (closed_loop/ee_input.py:34)
OPEN_FINGER_QPOS = [0.0388, -0.0388]  # logged open width 7.8 cm (steps.jsonl finger_qpos_m)
SITE = 'gripper0_grip_site'  # OSC controller's EE site: what ee_pos / ee_poses measure
CAMERAS = ['agentview', 'sideview']


def axis_angle_to_matrix(rotvec):
    angle = torch.linalg.norm(rotvec) + 1e-12
    k = rotvec / angle
    K = torch.zeros(3, 3, dtype=rotvec.dtype)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, dtype=rotvec.dtype) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)


def matrix_to_axis_angle(R):
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    return axis * angle


def quat_wxyz_to_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def keypoints_from_prompt(prompt_path):
    text = open(prompt_path).read()
    block = text.split('Keypoint 3D positions [X, Y, Z] (current positions): "')[1].split('"')[0]
    rows = [r.strip(' []\n') for r in block.split(']') if r.strip(' []\n')]
    return np.array([[float(v) for v in r.split()] for r in rows])  # (K, 3) world, m


def reward_argmax_pose(fn, keypoints_world_m, start_pos_m, start_R):
    """Pose (pos (3,), R (3, 3)) maximizing fn when the EE holds it for all T+1 steps; gradient ascent from start."""
    kp = torch.tensor(keypoints_world_m, dtype=torch.float64)
    pos = torch.tensor(start_pos_m, dtype=torch.float64, requires_grad=True)
    delta_rotvec = torch.full((3,), 1e-4, dtype=torch.float64, requires_grad=True)  # rotation applied on top of start
    R0 = torch.tensor(start_R, dtype=torch.float64)
    opt = torch.optim.Adam([pos, delta_rotvec], lr=0.002)
    for _ in range(4000):
        T_world_ee = torch.eye(4, dtype=torch.float64)
        T_world_ee = T_world_ee.clone()
        T_world_ee[:3, :3] = axis_angle_to_matrix(delta_rotvec) @ R0
        T_world_ee[:3, 3] = pos
        ee_poses = T_world_ee.expand(1, HORIZON_PLUS_1, 4, 4)  # (B=1, T+1, 4, 4)
        loss = -fn(kp, ee_poses)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        R = (axis_angle_to_matrix(delta_rotvec) @ R0).numpy()
        return pos.detach().numpy(), R, -loss.item()


def place_ee_by_ik(sim, model, target_pos, target_R):
    """Damped least squares IK on the 7 arm joints (qpos 0-6) to put SITE at the target. Physics is not stepped."""
    site_id = model.site_name2id(SITE)
    for _ in range(500):
        sim.forward()
        pos = sim.data.site_xpos[site_id]
        R = sim.data.site_xmat[site_id].reshape(3, 3)
        err = np.concatenate([target_pos - pos, matrix_to_axis_angle(target_R @ R.T)])  # (6,) m and rad
        if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
            break
        J = np.concatenate([sim.data.get_site_jacp(SITE).reshape(3, -1), sim.data.get_site_jacr(SITE).reshape(3, -1)])[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        q = sim.data.qpos[:7] + 0.5 * dq
        sim.data.qpos[:7] = np.clip(q, model.jnt_range[:7, 0], model.jnt_range[:7, 1])
    sim.data.qpos[7:9] = OPEN_FINGER_QPOS
    sim.forward()
    pos = sim.data.site_xpos[site_id]
    R = sim.data.site_xmat[site_id].reshape(3, 3)
    return np.linalg.norm(target_pos - pos), np.degrees(np.linalg.norm(matrix_to_axis_angle(target_R @ R.T)))


def render(sim, label_lines):
    images = []
    for cam in CAMERAS:
        img = sim.render(width=480, height=480, camera_name=cam)[::-1]  # offscreen render is upside down
        img = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
        cv2.putText(img, cam, (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        images.append(img)
    header = np.full((70, 480 * len(CAMERAS), 3), 30, np.uint8)
    for k, line in enumerate(label_lines):
        cv2.putText(header, line, (8, 20 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([header, np.concatenate(images, axis=1)], axis=0)


steps = [json.loads(line) for line in open(os.path.join(EP_DIR, 'steps.jsonl'))]
env = libero_adapter.create_libero_envs('libero_goal_task', task_ids_filter=[0])[0]
env._env.reset()
env._env.set_init_state(env._init_states[0])
sim = env._env.env.sim
model = sim.model
handle_center_m = sim.data.geom_xpos[model.geom_name2id('wooden_cabinet_1_g40')].copy()  # bottom handle bar

poses = []  # (label lines, pos, R)
real = steps[REAL_STEP]
real_pos, real_R = np.array(real['ee_pos_before']), quat_wxyz_to_matrix(np.array(real['ee_quat_wxyz_before']))
poses.append(([f'REAL EE at step {REAL_STEP}', f'pos {np.round(real_pos, 3).tolist()}'], real_pos, real_R))
for call in CALLS:
    call_dir = os.path.join(EP_DIR, 'closed_loop', call)
    guidance_path = [os.path.join(call_dir, f) for f in os.listdir(call_dir) if f.endswith('_guidance.txt')][0]
    fn = load_functions_from_txt(guidance_path, validate=False)[0]
    kp = keypoints_from_prompt(os.path.join(call_dir, 'prompt.txt'))
    call_step = int(call.split('step')[1])
    start = steps[min(call_step, len(steps) - 1)]
    start_pos, start_R = np.array(start['ee_pos_before']), quat_wxyz_to_matrix(np.array(start['ee_quat_wxyz_before']))
    pos, R, reward = reward_argmax_pose(fn, kp, start_pos, start_R)
    finger_bar_deg = np.degrees(np.arccos(abs(R[:, 0] @ np.array([1.0, 0, 0]))))
    approach_err_deg = np.degrees(np.arccos(np.clip(R[:, 2] @ np.array([0, -1.0, 0]), -1, 1)))
    rel_cm = (pos - kp[13]) * 100
    print(f'{call} {os.path.basename(guidance_path)}: target pos {np.round(pos, 4).tolist()} = kp13 + '
          f'{np.round(rel_cm, 1).tolist()} cm, finger line {np.round(R[:, 0], 2).tolist()} ({finger_bar_deg:.0f} deg to bar), '
          f'approach {np.round(R[:, 2], 2).tolist()} ({approach_err_deg:.0f} deg from -Y), reward {reward:.2e}')
    poses.append(([f'TARGET of {call} ({os.path.basename(guidance_path)[:6]})',
                   f'kp13 {np.round(rel_cm, 1).tolist()} cm, finger-bar {finger_bar_deg:.0f} deg, approach err {approach_err_deg:.0f} deg'],
                  pos, R))

panels = []
for label, pos, R in poses:
    env._env.set_init_state(env._init_states[0])
    pos_err_m, rot_err_deg = place_ee_by_ik(sim, model, pos, R)
    label = label + [f'IK residual {pos_err_m * 1000:.1f} mm, {rot_err_deg:.1f} deg (physics off: overlaps are not resolved)']
    print(f'{label[0]}: IK residual {pos_err_m * 1000:.2f} mm {rot_err_deg:.2f} deg')
    panels.append(render(sim, label))
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
cv2.imwrite(OUT_PATH, np.concatenate(panels, axis=0))
print(f'handle bar center (sim) {np.round(handle_center_m, 3).tolist()}; wrote {OUT_PATH}')
