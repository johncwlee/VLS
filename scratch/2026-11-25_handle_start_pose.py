# Task 0 init 0: drive the gripper (OSC) to 5 cm from the bottom drawer handle, pointing diagonally down at it, then render it.
import os
import sys

import imageio
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import core.env_adapters.libero_adapter as libero_adapter

HANDLE_GEOM = 'wooden_cabinet_1_g40'  # bottom drawer handle bar; runs along world X, drawer face points +Y
OSC_MAX_STEP_M = 0.05  # controller output_max for position: action 1.0 = 5 cm per step
OSC_MAX_STEP_RAD = 0.5  # controller output_max for rotation
DOWN_ANGLES_DEG = [30, 45, 60]  # approach angle below horizontal
STANDOFF_M = 0.05  # grip site distance from the bar center along the approach line
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'rollouts_tmp')


def make_env_at_init_state(init_index):
    """libero_goal_task task 0 env, reset to a fixed LIBERO init state and settled as LiberoEnv.reset does."""
    env = libero_adapter.create_libero_envs('libero_goal_task', task_ids_filter=[0])[0]
    reset_to_init_state(env, init_index)
    return env


def reset_to_init_state(env, init_index):
    env._env.reset()
    env._env.set_init_state(env._init_states[init_index])
    for _ in range(env.num_steps_wait):
        env._env.step(libero_adapter.get_libero_dummy_action())


def rotation_angle_deg(R_a, R_b):
    cos_angle = (np.trace(R_a.T @ R_b) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))


def robot_contacts(sim):
    """Set of (robot body, other body) in contact, ignoring the table."""
    model = sim.model
    pairs = set()
    for k in range(sim.data.ncon):
        contact = sim.data.contact[k]
        body_1 = model.body_id2name(model.geom_bodyid[contact.geom1])
        body_2 = model.body_id2name(model.geom_bodyid[contact.geom2])
        is_robot_1 = body_1.startswith('robot0') or body_1.startswith('gripper0')
        is_robot_2 = body_2.startswith('robot0') or body_2.startswith('gripper0')
        if is_robot_1 and not is_robot_2 and 'table' not in body_2:
            pairs.add((body_1, body_2))
        if is_robot_2 and not is_robot_1 and 'table' not in body_1:
            pairs.add((body_2, body_1))
    return pairs


def drive_osc(env, target_world_m, R_target, max_steps, tol_m=0.005, tol_deg=1.0):
    """OSC servo to a world position and orientation, fingers open. Returns the contacts seen on the way."""
    controller = env._env.env.robots[0].controller
    contacts = set()
    for _ in range(max_steps):
        error_world_m = target_world_m - np.array(controller.ee_pos)
        R_error = R_target @ np.array(controller.ee_ori_mat).T
        error_rotvec_world_rad = Rotation.from_matrix(R_error).as_rotvec()  # OSC rotation deltas are world-frame axis-angle
        if np.linalg.norm(error_world_m) < tol_m and np.degrees(np.linalg.norm(error_rotvec_world_rad)) < tol_deg:
            break
        action = np.zeros(7)
        action[:3] = np.clip(error_world_m / OSC_MAX_STEP_M, -1.0, 1.0)
        action[3:6] = np.clip(error_rotvec_world_rad / OSC_MAX_STEP_RAD, -1.0, 1.0)
        action[6] = -1.0  # fingers open
        env._env.step(action.tolist())
        contacts |= robot_contacts(env._env.env.sim)
    return contacts


def drive_to_handle(env, down_deg, R_target_world=None):
    """Gripper z axis pointing at the bar from +Y, down_deg below horizontal; fingers straddle the bar (start finger
    axis, world Y, stays perpendicular to the bar). If R_target_world (3, 3) is given, that orientation is used
    instead. Grip site STANDOFF_M back from the bar along the gripper z axis.
    Returns (position error m, rotation from start deg, contacts, handle pos, target)."""
    sim = env._env.env.sim
    controller = env._env.env.robots[0].controller
    handle_pos_world_m = sim.data.geom_xpos[sim.model.geom_name2id(HANDLE_GEOM)].copy()
    R_start = np.array(controller.ee_ori_mat)
    if R_target_world is None:
        # Start gripper z is straight down (90 deg below horizontal); about world X, -(90 - down_deg) tips it toward -Y
        R_target = Rotation.from_rotvec([-np.radians(90 - down_deg), 0.0, 0.0]).as_matrix() @ R_start
    else:
        R_target = np.asarray(R_target_world)
    approach_dir_world = R_target[:, 2]  # gripper z, (0, -cos d, -sin d)
    target_world_m = handle_pos_world_m - STANDOFF_M * approach_dir_world
    # Rotate just above the start, cross over the bowl and plate 10 cm above the approach line, back off to 12 cm
    # out on the line, then come in along it
    above_start_world_m = np.array(controller.ee_pos) + np.array([0.0, 0.0, 0.05])
    contacts = drive_osc(env, above_start_world_m, R_target, max_steps=150)
    contacts |= drive_osc(env, handle_pos_world_m - 0.12 * approach_dir_world + np.array([0.0, 0.0, 0.10]), R_target,
                          max_steps=150)
    contacts |= drive_osc(env, handle_pos_world_m - 0.12 * approach_dir_world, R_target, max_steps=100)
    contacts |= drive_osc(env, target_world_m, R_target, max_steps=100)
    error_m = float(np.linalg.norm(target_world_m - np.array(controller.ee_pos)))
    rotation_deg = rotation_angle_deg(R_start, np.array(controller.ee_ori_mat))
    return error_m, rotation_deg, contacts, handle_pos_world_m, target_world_m


if __name__ == '__main__':
    env = make_env_at_init_state(0)
    os.makedirs(OUT_DIR, exist_ok=True)
    for down_deg in DOWN_ANGLES_DEG:
        reset_to_init_state(env, 0)
        error_m, rotation_deg, contacts, handle_pos, target = drive_to_handle(env, down_deg)
        controller = env._env.env.robots[0].controller
        finger_axis_world = np.array(controller.ee_ori_mat)[:, 0]  # fingers open along gripper x (measured at start)
        print(f'down {down_deg:2d} deg: position error {100 * error_m:.2f} cm  rotation from start {rotation_deg:.1f} deg  '
              f'gripper z {np.round(np.array(controller.ee_ori_mat)[:, 2], 2)}  finger axis {np.round(finger_axis_world, 2)}  '
              f'contacts {sorted(contacts)}')
        print(f'  handle center {np.round(handle_pos, 3)}  target {np.round(target, 3)}  ee {np.round(controller.ee_pos, 3)}')
        sim = env._env.env.sim
        for camera in ('agentview', 'frontview', 'robot0_eye_in_hand'):
            image = sim.render(camera_name=camera, height=640, width=640)[::-1]  # MuJoCo renders upside down
            path = os.path.join(OUT_DIR, f'handle_start_init0_down{down_deg}_{camera}.png')
            imageio.imwrite(path, image)
            print(f'  {path}')
