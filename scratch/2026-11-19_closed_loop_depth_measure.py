# Runs main.py (closed loop, prompts unchanged) and at each VLM call measures contact/placement from rendered depth + sim masks, next to ground truth.
import json
import os
import sys

import hydra
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import closed_loop.run as closed_loop_run
import main as main_module

CAMERAS = {'agentview': 640, 'robot0_eye_in_hand': 256}  # name -> render size (px); 640 as step-0 keypoints, 256 as the policy
GEOM_OBJTYPE = 5  # mujoco mjOBJ_GEOM, first channel of a segmentation render
runner = {}
original_start_closed_loop = main_module.Main._start_closed_loop
original_closed_loop_step = main_module.Main._closed_loop_step
original_build_prompt = closed_loop_run.build_prompt


def world_points(sim, camera_name, size_px):
    """Render depth + segmentation; return (H*W, 3) world points (m) and (H*W,) body ids (-1 = not a geom)."""
    rgb, depth01 = sim.render(camera_name=camera_name, width=size_px, height=size_px, depth=True)
    seg = sim.render(camera_name=camera_name, width=size_px, height=size_px, segmentation=True)  # (H, W, 2) [objtype, objid]
    extent = sim.model.stat.extent
    near, far = sim.model.vis.map.znear * extent, sim.model.vis.map.zfar * extent
    depth_m = near / (1.0 - depth01 * (1.0 - near / far))  # (H, W) distance along the optical axis, robosuite formula
    cam_id = sim.model.camera_name2id(camera_name)
    fovy_rad = np.radians(sim.model.cam_fovy[cam_id])
    focal_px = (size_px / 2.0) / np.tan(fovy_rad / 2.0)
    # Raw render rows start at the image bottom (OpenGL), so row index grows along camera +y (up).
    cols, rows = np.meshgrid(np.arange(size_px), np.arange(size_px))
    x_cam = (cols + 0.5 - size_px / 2.0) / focal_px * depth_m
    y_cam = (rows + 0.5 - size_px / 2.0) / focal_px * depth_m
    z_cam = -depth_m  # MuJoCo cameras look along their -z axis
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3)  # (H*W, 3) camera frame
    cam_pos_world = sim.data.cam_xpos[cam_id]           # (3,) live pose: right for the moving wrist camera too
    cam_rot_world = sim.data.cam_xmat[cam_id].reshape(3, 3)
    points_world = points_cam @ cam_rot_world.T + cam_pos_world
    objtype = seg[..., 0].reshape(-1)
    objid = seg[..., 1].reshape(-1)
    body_ids = np.where(objtype == GEOM_OBJTYPE, sim.model.geom_bodyid[np.clip(objid, 0, None)], -1)
    return points_world, body_ids, depth_m, seg


def object_mask(sim, body_ids, object_prefix):
    names = np.array([sim.model.body_id2name(b) or '' for b in range(sim.model.nbody)])
    ids = np.where(np.char.startswith(names, object_prefix))[0]
    return np.isin(body_ids, ids)


def measure(env, step, out_dir, call_index):
    sim = env.sim
    def body_pos_m(name):
        return sim.data.body_xpos[env.obj_body_id[name]].copy()
    record = {'step': step, 'call': call_index, 'cameras': {}}
    is_bottle_task = 'wine_bottle_1' in env.obj_of_interest
    if is_bottle_task:
        mover, support = 'wine_bottle_1', 'plate_1'
        bottle, plate = body_pos_m(mover), body_pos_m(support)
        record['gt'] = {'contact': bool(env.object_states_dict[mover].check_contact(env.object_states_dict[support])),
                        'origin_dz_cm': float((bottle[2] - plate[2]) * 100),
                        'origin_dxy_cm': float(np.linalg.norm(bottle[:2] - plate[:2]) * 100)}
    else:
        mover, support = 'cream_cheese_1', 'wine_rack_1'
        site = 'wine_rack_1_top_region'
        delta_m = sim.data.get_site_xmat(site) @ (body_pos_m(mover) - sim.data.get_site_xpos(site))  # as LIBERO under()
        half_m = np.array(env.object_sites_dict[site].size)
        record['gt'] = {'in_region': bool(abs(delta_m[0]) < half_m[0] and abs(delta_m[1]) < half_m[1]),
                        'past_long_edge_cm': float((abs(delta_m[0]) - half_m[0]) * 100),
                        'across_cm': float(abs(delta_m[1]) * 100)}
    mover_true_m = body_pos_m(mover)
    saved = {}
    for camera_name, size_px in CAMERAS.items():
        points, body_ids, depth_m, seg = world_points(sim, camera_name, size_px)
        saved[f'{camera_name}_depth_m'] = depth_m.astype(np.float32)
        saved[f'{camera_name}_seg'] = seg.astype(np.int32)
        mover_pts = points[object_mask(sim, body_ids, mover)]      # (N, 3) visible surface, world m
        support_pts = points[object_mask(sim, body_ids, support)]  # (M, 3)
        cam = {'n_mover': int(len(mover_pts)), 'n_support': int(len(support_pts))}
        if len(mover_pts) >= 20:
            # Sanity check of the back-projection: visible-surface xy centroid vs true body origin xy
            cam['mover_xy_err_cm'] = float(np.linalg.norm(mover_pts[:, :2].mean(0) - mover_true_m[:2]) * 100)
            cam['mover_min_z_m'] = float(np.percentile(mover_pts[:, 2], 2))  # 2nd percentile: ignore stray edge pixels
        if len(mover_pts) >= 20 and len(support_pts) >= 20:
            # Support surface under the mover: support points within 1.5 cm (xy) of any visible mover point
            tree = cKDTree(mover_pts[:, :2])
            near_dist_m, _ = tree.query(support_pts[:, :2], distance_upper_bound=0.015)
            under = support_pts[np.isfinite(near_dist_m)]
            cam['n_support_under'] = int(len(under))
            if len(under) >= 10:
                cam['support_top_z_m'] = float(np.percentile(under[:, 2], 90))
                cam['obs_gap_cm'] = float((cam['mover_min_z_m'] - cam['support_top_z_m']) * 100)
            # Overlap: fraction of visible mover points with a support point within 1.5 cm in xy
            support_tree = cKDTree(support_pts[:, :2])
            d_m, _ = support_tree.query(mover_pts[:, :2], distance_upper_bound=0.015)
            cam['overlap_frac'] = float(np.isfinite(d_m).mean())
        record['cameras'][camera_name] = cam
    np.savez_compressed(os.path.join(out_dir, f'call_{call_index:02d}_step{step:04d}.npz'), **saved)
    return record


def start_closed_loop_keep_self(self, *args, **kwargs):
    runner['main'] = self
    runner['step'] = 0
    runner['call_index'] = 0
    return original_start_closed_loop(self, *args, **kwargs)


def closed_loop_step_keep_step(self, stage_state, global_step, *args, **kwargs):
    runner['step'] = global_step
    return original_closed_loop_step(self, stage_state, global_step, *args, **kwargs)


def build_prompt_and_measure(*args, **kwargs):
    prompt = original_build_prompt(*args, **kwargs)  # returned unchanged: this test only measures
    main_obj = runner['main']
    env = main_obj.adapter._get_current_robosuite_env().env
    out_dir = os.path.join(os.path.dirname(main_obj.closed_loop['call_root']), 'depth_measure')
    os.makedirs(out_dir, exist_ok=True)
    runner['call_index'] += 1
    record = measure(env, runner['step'], out_dir, runner['call_index'])
    with open(os.path.join(out_dir, 'measure.jsonl'), 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(f'[depth_measure] {json.dumps(record)}', flush=True)
    return prompt


main_module.Main._start_closed_loop = start_closed_loop_keep_self
main_module.Main._closed_loop_step = closed_loop_step_keep_step
closed_loop_run.build_prompt = build_prompt_and_measure

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs'))


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name='config')  # main.py's relative path fails on import
def run(cfg):
    main_module.main.__wrapped__(cfg)


if __name__ == '__main__':
    run()
