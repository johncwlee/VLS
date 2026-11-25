# Annotated task 0 videos (11-19 original + 2026-11-24 position / pose / gripper_points runs): EE vs bottom handle, finger orientation, VLM calls.
import glob
import importlib.util
import json
import os
import re
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

SCRATCH = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('fv', os.path.join(SCRATCH, '2026-11-20_failure_videos.py'))
fv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fv)

RUNS = [  # (label, episode dir)
    ('position_2026-11-19', 'outputs/goal5fix_on_t0-4/2026-11-19_01-25-56/episode_1'),
    ('position_2026-11-24', 'outputs/t0i0_ee_position/2026-11-24_02-41-33/episode_1'),
    ('pose_2026-11-24', 'outputs/t0i0_ee_pose/2026-11-24_02-41-33/episode_1'),
    ('gripper_points_2026-11-24', 'outputs/t0i0_ee_gripper_points/2026-11-24_02-41-33/episode_1'),
    ('pose_contacts_2026-11-24', 'outputs/t0i0_ee_pose_contacts/2026-11-24_04-11-02/episode_1'),
    ('pose_clear_r1_2026-11-24', 'outputs/t0i0_pose_clear_r1/2026-11-24_04-36-56/episode_1'),
    ('pose_clear_r2_2026-11-24', 'outputs/t0i0_pose_clear_r2/2026-11-24_04-36-56/episode_1'),
    ('pose_clear_r3_2026-11-24', 'outputs/t0i0_pose_clear_r3/2026-11-24_04-36-56/episode_1'),
    ('open_loop_vls_2026-11-19', 'outputs/goal5fix_off_t0-4/2026-11-19_01-26-36/episode_1'),  # closed_loop_vls=false
    ('pose_route_r1_2026-11-24', 'outputs/t0i0_pose_route_r1/2026-11-24_04-49-09/episode_1'),
    ('pose_route_r2_2026-11-24', 'outputs/t0i0_pose_route_r2/2026-11-24_04-49-09/episode_1'),
    ('pose_route_r3_2026-11-24', 'outputs/t0i0_pose_route_r3/2026-11-24_05-00-38/episode_1'),  # 04-49-09 hung loading DINOv2
]
ONLY = ['pose_route_r1_2026-11-24', 'pose_route_r2_2026-11-24', 'pose_route_r3_2026-11-24']  # None = all
OUT_DIR = 'outputs/task0_ee_videos_2026-11-24'
KP_HANDLE = 13  # wooden_cabinet_1_cabinet_bottom
BAR_AXIS_WORLD = np.array([1.0, 0.0, 0.0])  # bottom handle bar runs along world X (notes/handoff.md)
PULL_DIR_WORLD = np.array([0.0, 1.0, 0.0])  # drawers open along world +Y, so approach should be -Y
PLOT_H = 220
WHITE, GREY, YELLOW, GREEN, RED, CYAN = fv.WHITE, fv.GREY, fv.YELLOW, fv.GREEN, fv.RED, fv.CYAN


def quat_wxyz_to_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def reward_comments(call_dir):
    """Comment lines of the stage guidance function: the VLM's own description of each reward term."""
    paths = glob.glob(os.path.join(call_dir, 'stage*_guidance.txt'))
    if not paths:
        return []
    return [line.strip()[1:].strip() for line in open(paths[0]) if line.strip().startswith('#')]


def draw_plot(dist_cm, angle_deg, approach_deg, calls, i, width):
    """Bottom strip: EE-handle distance (cm, 0-30), finger-bar angle and approach error (deg, 0-180) vs step."""
    strip = np.full((PLOT_H, width, 3), 20, np.uint8)
    n = len(dist_cm)
    x0, x1, y0, y1 = 50, width - 10, 15, PLOT_H - 25

    def px(step):
        return int(x0 + (x1 - x0) * step / max(n - 1, 1))

    def py(v, vmax):
        return int(y1 - (y1 - y0) * min(v, vmax) / vmax)

    for level_cm in (0, 5, 10, 20, 30):
        cv2.line(strip, (x0, py(level_cm, 30)), (x1, py(level_cm, 30)), (50, 50, 50), 1)
        cv2.putText(strip, f'{level_cm}', (5, py(level_cm, 30) + 4), fv.FONT, 0.38, CYAN, 1, cv2.LINE_AA)
    for c in calls:
        if c['step'] < n:
            color = fv.VERDICT_COLOR.get(c['parsed']['verdict'], WHITE)
            cv2.line(strip, (px(c['step']), y0), (px(c['step']), y1), color, 1)
            cv2.putText(strip, c['parsed']['verdict'][:4], (px(c['step']) + 2, y0 + 10), fv.FONT, 0.35, color, 1, cv2.LINE_AA)
    for series, vmax, color in ((dist_cm, 30, CYAN), (angle_deg, 180, GREEN), (approach_deg, 180, (200, 120, 255))):
        pts = np.array([[px(k), py(v, vmax)] for k, v in enumerate(series)], np.int32)
        cv2.polylines(strip, [pts], False, color, 1, cv2.LINE_AA)
    cv2.line(strip, (x0, py(90, 180)), (x1, py(90, 180)), (40, 90, 40), 1)  # 90 deg = fingers across the bar
    cv2.line(strip, (px(i), y0), (px(i), y1), WHITE, 1)
    cv2.putText(strip, 'cyan: EE-kp13 cm (left axis 0-30)   green: finger line vs bar deg (0-180, 90 = across, dark line)   '
                'purple: approach vs -Y deg (0 = good)', (x0, PLOT_H - 7), fv.FONT, 0.36, WHITE, 1, cv2.LINE_AA)
    return strip


def make_video(label, ep_dir):
    rows = [json.loads(line) for line in open(os.path.join(ep_dir, 'steps.jsonl'))]
    cap = cv2.VideoCapture(glob.glob(os.path.join(ep_dir, '*agentview.mp4'))[0])
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    assert len(frames) == len(rows), (len(frames), len(rows))
    trial = json.loads(open(os.path.join(os.path.dirname(ep_dir), 'trials.jsonl')).readline())
    assert trial['episode'] == 1 and trial['task_id'] == 0, trial
    open_loop = not os.path.isdir(os.path.join(ep_dir, 'closed_loop'))
    if open_loop:
        # Plain VLS: one VLM call at step 0 writes all stages; VLS's stage recognizer switches them (HUD "Stage:")
        calls = []
        vlm_dir = os.path.join(ep_dir, 'vlm_agent')
        kp_handle_world_m = np.array(json.load(open(os.path.join(vlm_dir, 'metadata.json')))['init_keypoint_positions'][KP_HANDLE])
        open_loop_lines = [('open-loop VLS: one VLM call at step 0; stages switch by VLS stage recognizer (HUD, top left)', GREY)]
        open_loop_lines += [(line[2:].strip(), YELLOW) for line in open(os.path.join(vlm_dir, 'output_raw.txt')) if line.startswith('# Stage')]
        for path in sorted(glob.glob(os.path.join(vlm_dir, 'stage*_guidance.txt'))):
            comments = [line.strip()[1:].strip() for line in open(path) if line.strip().startswith('#')]
            open_loop_lines += [(f'{os.path.basename(path)[:6]} reward terms:', GREY)] + [(f'  - {t}', YELLOW) for t in comments]
    else:
        calls = fv.load_calls(ep_dir)
        for c in calls:
            c['comments'] = reward_comments(os.path.join(ep_dir, 'closed_loop', c['name']))
        kp_handle_world_m = calls[0]['keypoints'][KP_HANDLE]  # keypoint at episode start; drawer stays closed
    ee_pos_world_m = np.array([r['ee_pos_before'] for r in rows])  # (N, 3)
    R_world_ee = np.array([quat_wxyz_to_matrix(np.array(r['ee_quat_wxyz_before'])) for r in rows])  # (N, 3, 3)
    finger_line_world = R_world_ee[:, :, 0]  # (N, 3) EE x axis = finger line
    approach_world = R_world_ee[:, :, 2]  # (N, 3) EE z axis = approach
    delta_cm = (ee_pos_world_m - kp_handle_world_m) * 100  # (N, 3) EE minus handle keypoint
    dist_cm = np.linalg.norm(delta_cm, axis=1)
    finger_bar_deg = np.degrees(np.arccos(np.clip(np.abs(finger_line_world @ BAR_AXIS_WORLD), 0, 1)))  # 90 = across
    approach_err_deg = np.degrees(np.arccos(np.clip(approach_world @ -PULL_DIR_WORLD, -1, 1)))  # 0 = pointing -Y

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'task0_init0_{label}.mp4')
    h, w = frames[0].shape[:2]
    width = w + 640
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, h + PLOT_H))
    last_call, active = None, None
    for i, (frame, row) in enumerate(zip(frames, rows)):
        call_now = [c for c in calls if c['step'] == i]
        if call_now:
            last_call = call_now[-1]
            if 'guidance_fn' in last_call:
                active = last_call
            elif last_call['parsed']['verdict'] == 'done' or not last_call['parsed'].get('guidance', True):
                active = None
        width_cm = (row['finger_qpos_m'][0] - row['finger_qpos_m'][1]) * 100
        d = delta_cm[i]
        lines = [(f'task 0 init 0 seed 10000   {label}   step {i}/{len(rows) - 1}', WHITE),
                 (f'outcome {trial["outcome"]}, stop {trial.get("stop_reason")}, '
                  f'{trial.get("closed_loop_calls", 1)} VLM calls', GREY),
                 ('', WHITE),
                 (f'EE - kp13 (bottom handle): {dist_cm[i]:.1f} cm   dx {d[0]:+.1f}  dy {d[1]:+.1f}  dz {d[2]:+.1f} cm', CYAN),
                 (f'  (min so far {dist_cm[:i + 1].min():.1f} cm; dy > 0 = in front of the drawer, dz > 0 = above)', GREY),
                 (f'finger line vs bar: {finger_bar_deg[i]:.0f} deg (need ~90)   finger line {np.round(finger_line_world[i], 2).tolist()}', GREEN),
                 (f'approach vs -Y: {approach_err_deg[i]:.0f} deg   approach {np.round(approach_world[i], 2).tolist()}', (200, 120, 255)),
                 (f'gripper width {width_cm:.1f} cm   success flag {row["success"]}', WHITE),
                 ('', WHITE)]
        if open_loop:
            lines += open_loop_lines
        elif active is not None:
            lines += [(f'active reward: call @ step {active["step"]} terms:', YELLOW)]
            lines += [(f'  - {t}', YELLOW) for t in active['comments']] or [('  (no comments in function)', GREY)]
        else:
            lines += [('active reward: none', GREY)]
        lines += [('', WHITE)]
        if last_call is not None:
            p = last_call['parsed']
            lines += [(f'VLM call @ step {last_call["step"]}  {p["verdict"].upper()}  stage {p["current_stage"]}',
                       fv.VERDICT_COLOR.get(p['verdict'], WHITE)),
                      (f'trigger: {last_call["trigger"]}', GREY),
                      (f'evidence: {p["evidence"]}', WHITE)]
            if p.get('plan'):
                for k, stage in enumerate(p['plan'], 1):
                    lines += [(f'  {k}. {stage}', YELLOW if k == p['current_stage'] else GREY)]
        panel = cv2.resize(fv.draw_panel(lines, highlight=bool(call_now)), (640, h))
        top = np.concatenate([frame, panel], axis=1)
        canvas = np.concatenate([top, draw_plot(dist_cm, finger_bar_deg, approach_err_deg, calls, i, width)], axis=0)
        for _ in range(fv.HOLD_FRAMES if call_now else 1):
            writer.write(canvas)
    after_last = [c for c in calls if c['step'] >= len(rows)]
    i_min = int(np.argmin(dist_cm))
    end_lines = [(f'END {label}: {trial["outcome"]}, {trial.get("stop_reason")}, '
                  f'{trial.get("closed_loop_calls", 1)} calls, {len(rows)} steps', RED),
                 ('', WHITE),
                 (f'closest to kp13: {dist_cm[i_min]:.1f} cm at step {i_min} (dx {delta_cm[i_min][0]:+.1f} dy '
                  f'{delta_cm[i_min][1]:+.1f} dz {delta_cm[i_min][2]:+.1f}), finger-bar {finger_bar_deg[i_min]:.0f} deg, '
                  f'approach err {approach_err_deg[i_min]:.0f} deg', CYAN)]
    for c in after_last:
        end_lines += [('', WHITE), (f'last VLM call @ step {c["step"]}: {c["parsed"]["verdict"].upper()}', YELLOW),
                      (f'evidence: {c["parsed"]["evidence"]}', WHITE)]
    end_panel = cv2.resize(fv.draw_panel(end_lines, highlight=False), (640, h))
    end_canvas = np.concatenate([np.concatenate([frames[-1], end_panel], axis=1),
                                 draw_plot(dist_cm, finger_bar_deg, approach_err_deg, calls, len(rows) - 1, width)], axis=0)
    for _ in range(fv.HOLD_FRAMES):
        writer.write(end_canvas)
    writer.release()
    h264_path = out_path.replace('.mp4', '_h264.mp4')  # cv2's mp4v doesn't play in VS Code / browsers
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-loglevel', 'error', '-y', '-i', out_path, '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', h264_path], check=True)
    os.replace(h264_path, out_path)
    print(f'{label}: {len(rows)} steps, {len(calls)} calls, min EE-kp13 {dist_cm[i_min]:.1f} cm at step {i_min}, '
          f'finger-bar {finger_bar_deg[i_min]:.0f} deg, approach err {approach_err_deg[i_min]:.0f} deg -> {out_path}')
    for c in calls:
        print(f'  call @ {c["step"]:3d} {c["parsed"]["verdict"]:8s} reward terms: {" | ".join(c["comments"]) or "-"}')


for label, ep_dir in RUNS:
    if ONLY is None or label in ONLY:
        make_video(label, ep_dir)
