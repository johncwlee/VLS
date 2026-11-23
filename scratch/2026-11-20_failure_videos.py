# Annotated flag-on failure videos (one per failure-heavy task) from the goal5fix runs: VLM reasoning + where steering pulls vs where the EE goes.
import ast
import glob
import json
import os
import re
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.guidance_utils import load_functions_from_txt

RUNS = {0: 'outputs/goal5fix_on_t0-4', 5: 'outputs/goal5fix_on_t5-9'}  # suite task offset -> run
TASKS = [0, 1, 3, 4, 9]  # flag-on successes 0-1 of 5 (LOG.md 2026-11-19)
OUT_DIR = 'outputs/failure_videos_2026-11-20'
OPEN_WIDTH_M = 0.06  # same threshold as main.py _closed_loop_step
HORIZON = 10  # pi05 action_chunk_horizon (configs/policy.yaml:25): guidance sees (B, 10, 3)
HOLD_FRAMES = 30  # 3 s at 10 fps on each VLM call
FONT = cv2.FONT_HERSHEY_SIMPLEX


def run_dir(name):
    return sorted(glob.glob(f'{name}/*/'))[-1]


def pick_episode(task):
    """Flag-on failure with the most VLM calls for a suite task; ties -> lowest episode."""
    offset = 5 if task >= 5 else 0
    base = run_dir(RUNS[offset])
    trials = [json.loads(line) for line in open(os.path.join(base, 'trials.jsonl'))]
    fails = [t for t in trials if t['task_id'] == task - offset and t['outcome'] == 'fail']
    assert fails, f'task {task}: no failures'
    best = sorted(fails, key=lambda t: (-t['closed_loop_calls'], t['episode']))[0]
    ep_dirs = [os.path.join(base, f"episode_{best['episode']}{suffix}") for suffix in ('', '_failed')]
    ep_dirs = [d for d in ep_dirs if os.path.isdir(d)]
    assert len(ep_dirs) == 1, ep_dirs
    ep_dir = ep_dirs[0]
    return best, ep_dir


def parse_prompt(prompt_path):
    text = open(prompt_path).read()
    task_text = re.search(r'Query Task: "(.*)"', text).group(1)
    trigger = re.search(r'This call was triggered because: (.*)', text).group(1)
    kp_map = ast.literal_eval(re.search(r'Keypoints to object name map: "(\{.*\})"', text).group(1))
    kp_block = re.search(r'Keypoint 3D positions \[X, Y, Z\] \(current positions\): "(\[\[.*?\]\])"', text, re.S).group(1)
    rows = re.findall(r'\[([^\[\]]+)\]', kp_block)
    keypoints_world_m = np.array([[float(v) for v in row.split()] for row in rows])  # (K, 3) world frame, meters
    assert keypoints_world_m.shape == (len(kp_map), 3), keypoints_world_m.shape
    return task_text, trigger, kp_map, keypoints_world_m


def reward_argmax(fn, keypoints_world_m, start_world_m):
    """Point p maximizing fn(keypoints, path that sits at p for all T steps), by gradient ascent from start.

    Returns (p (3,), reward at p). A held-still path at p is the path the reward likes most when the
    reward only scores distances of path points to a target (the common pattern in these functions)."""
    kp = torch.tensor(keypoints_world_m, dtype=torch.float32)
    p = torch.tensor(start_world_m, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([p], lr=0.005)
    for _ in range(3000):
        path = p.view(1, 1, 3).expand(1, HORIZON, 3)  # (B=1, T, 3)
        loss = -fn(kp, path)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        r = fn(kp, p.view(1, 1, 3).expand(1, HORIZON, 3)).item()
    return p.detach().numpy().astype(np.float64), r


def reward_is_flat(fn, keypoints_world_m, start_world_m):
    """True if the reward gradient is zero at the start and 3 nearby held-still paths (e.g. a `return 0` stage)."""
    kp = torch.tensor(keypoints_world_m, dtype=torch.float32)
    for offset in ([0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]):
        p = torch.tensor(start_world_m + np.array(offset), dtype=torch.float32, requires_grad=True)
        r = fn(kp, p.view(1, 1, 3).expand(1, HORIZON, 3))
        if r.requires_grad:
            (grad,) = torch.autograd.grad(r, p)
            if grad.norm().item() > 1e-8:
                return False
    return True


def load_calls(ep_dir):
    calls = []
    for call_dir in sorted(glob.glob(os.path.join(ep_dir, 'closed_loop', 'call_*'))):
        step = int(re.search(r'step(\d+)', call_dir).group(1))
        parsed = json.load(open(os.path.join(call_dir, 'parsed.json')))
        task_text, trigger, kp_map, kp = parse_prompt(os.path.join(call_dir, 'prompt.txt'))
        call = {'step': step, 'name': os.path.basename(call_dir), 'parsed': parsed, 'trigger': trigger,
                'task_text': task_text, 'kp_map': kp_map, 'keypoints': kp, 'target': None, 'target_desc': ''}
        guidance_paths = glob.glob(os.path.join(call_dir, 'stage*_guidance.txt'))
        if parsed['verdict'] in ('plan', 'advance', 'replan') and parsed.get('guidance') and guidance_paths:
            assert len(guidance_paths) == 1, guidance_paths
            call['guidance_fn'] = load_functions_from_txt(guidance_paths[0], validate=False)[0]
        calls.append(call)
    return calls


def nearest_keypoint_desc(target, kp, kp_map):
    d_cm = np.linalg.norm(kp - target, axis=1) * 100
    i = int(np.argmin(d_cm))
    return f'kp {i} {kp_map[i].replace("_main", "")} {d_cm[i]:.1f} cm away'


def wrap(text, width):
    words, lines, line = text.split(), [], ''
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = f'{line} {w}'.strip()
    if line:
        lines.append(line)
    return lines


def draw_panel(lines, highlight):
    panel = np.full((640, 640, 3), 30, np.uint8)
    if highlight:
        cv2.rectangle(panel, (0, 0), (639, 639), (0, 200, 255), 6)
    y = 22
    for text, color in lines:
        for sub in wrap(text, 78) if text else ['']:
            cv2.putText(panel, sub, (10, y), FONT, 0.45, color, 1, cv2.LINE_AA)
            y += 17
            if y > 630:
                return panel
    return panel


WHITE, GREY, YELLOW, GREEN, RED, CYAN = (235, 235, 235), (150, 150, 150), (0, 220, 255), (90, 220, 90), (80, 80, 255), (255, 220, 120)
VERDICT_COLOR = {'plan': CYAN, 'continue': WHITE, 'advance': GREEN, 'replan': YELLOW, 'done': RED}


def make_video(task, trial, ep_dir):
    rows = [json.loads(line) for line in open(os.path.join(ep_dir, 'steps.jsonl'))]
    video_path = glob.glob(os.path.join(ep_dir, '*agentview.mp4'))[0]
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    assert len(frames) == len(rows), (len(frames), len(rows))
    calls = load_calls(ep_dir)
    assert len({c['task_text'] for c in calls}) == 1, 'calls disagree on the task text'

    ee_world_m = np.array([r['ee_pos_before'] for r in rows])  # (N, 3) world frame, meters
    for call in calls:
        if 'guidance_fn' in call and call['step'] < len(rows):  # a call after the last step never steered
            start = ee_world_m[call['step']]
            if reward_is_flat(call['guidance_fn'], call['keypoints'], start):
                call['flat'] = True
                continue
            call['target'], call['target_reward'] = reward_argmax(call['guidance_fn'], call['keypoints'], start)
            call['target_desc'] = nearest_keypoint_desc(call['target'], call['keypoints'], call['kp_map'])
            # Second start 10-20 cm away: if it ends elsewhere, the reward leaves some direction free.
            other, _ = reward_argmax(call['guidance_fn'], call['keypoints'], start + np.array([0.1, -0.1, 0.2]))
            gap_cm = np.linalg.norm(other - call['target']) * 100
            if gap_cm > 1.0:
                call['target_desc'] += f' | NOT UNIQUE (2nd start ends {gap_cm:.0f} cm away)'

    # Which call's guidance is active at each step: last call that wrote a function, until a call ends steering.
    active_call = [None] * len(rows)
    current = None
    for i in range(len(rows)):
        for call in calls:
            if call['step'] == i:
                if 'guidance_fn' in call:
                    current = call
                elif call['parsed']['verdict'] == 'done' or not call['parsed'].get('guidance', True):
                    current = None
        active_call[i] = current

    summary = {'task': task, 'episode': trial['episode'], 'init_state': (trial['episode'] - 1) % 5,
               'stop_reason': trial['stop_reason'], 'n_steps': len(rows), 'calls': []}
    for k, call in enumerate(calls):
        seg_end = calls[k + 1]['step'] if k + 1 < len(calls) else len(rows)
        entry = {'step': call['step'], 'verdict': call['parsed']['verdict'], 'stage': call['parsed']['current_stage'],
                 'trigger': call['trigger'], 'evidence': call['parsed']['evidence']}
        if call.get('flat'):
            entry['target_desc'] = 'flat reward: no steering'
        elif call['target'] is not None:
            d_cm = np.linalg.norm(ee_world_m[call['step']:seg_end] - call['target'], axis=1) * 100
            entry.update(target_world_m=np.round(call['target'], 3).tolist(), target_desc=call['target_desc'],
                         ee_to_target_start_cm=round(float(d_cm[0]), 1), ee_to_target_min_cm=round(float(d_cm.min()), 1),
                         ee_to_target_end_cm=round(float(d_cm[-1]), 1), steps_in_segment=int(seg_end - call['step']))
        summary['calls'].append(entry)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'task{task}_ep{trial["episode"]}_init{summary["init_state"]}.mp4')
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280, 640))
    task_text = calls[0]['task_text']
    last_call = None
    for i, (frame, row) in enumerate(zip(frames, rows)):
        width_cm = (row['finger_qpos_m'][0] - row['finger_qpos_m'][1]) * 100
        gripper = 'open' if width_cm >= OPEN_WIDTH_M * 100 else 'closed'
        contacts = ', '.join(f'{k}: {v}' for k, v in row['obj_contact'].items()) or 'none'
        call_now = [c for c in calls if c['step'] == i]
        if call_now:
            last_call = call_now[-1]
        act = active_call[i]
        lines = [(f'task {task}: "{task_text}"   ep {trial["episode"]} init {summary["init_state"]}', WHITE),
                 (f'step {i}/{len(rows) - 1}   stop: {trial["stop_reason"]}', GREY),
                 ('', WHITE),
                 (f'EE world (m): {np.round(ee_world_m[i], 3).tolist()}', WHITE),
                 (f'gripper width {width_cm:.1f} cm ({gripper})   contacts: {contacts}', WHITE),
                 (f'success flag: {row["success"]}', WHITE)]
        if act is not None and act.get('flat'):
            lines += [(f'steering: none (call @ step {act["step"]} wrote a reward that is flat, e.g. return 0)', GREY)]
        elif act is not None:
            d_cm = np.linalg.norm(ee_world_m[i] - act['target']) * 100
            delta_cm = (act['target'] - ee_world_m[i]) * 100
            lines += [(f'steering target (reward argmax, from call @ step {act["step"]}): {np.round(act["target"], 3).tolist()}', CYAN),
                      (f'  nearest: {act["target_desc"]}', CYAN),
                      (f'  EE -> target: {d_cm:.1f} cm   (dx {delta_cm[0]:+.1f}, dy {delta_cm[1]:+.1f}, dz {delta_cm[2]:+.1f} cm)', CYAN)]
        else:
            lines += [('steering: none active', GREY)]
        lines += [('', WHITE)]
        if last_call is not None:
            p = last_call['parsed']
            lines += [(f'VLM call @ step {last_call["step"]}  verdict: {p["verdict"].upper()}  stage {p["current_stage"]}', VERDICT_COLOR.get(p['verdict'], WHITE)),
                      (f'trigger: {last_call["trigger"]}', GREY),
                      (f'evidence: {p["evidence"]}', WHITE)]
            if p.get('plan'):
                lines += [('plan:', GREY)]
                for k, stage in enumerate(p['plan'], 1):
                    lines += [(f'  {k}. {stage}', YELLOW if k == p['current_stage'] else GREY)]
        panel = draw_panel(lines, highlight=bool(call_now))
        canvas = np.concatenate([frame, panel], axis=1)
        for _ in range(HOLD_FRAMES if call_now else 1):
            writer.write(canvas)
    after_last = [c for c in calls if c['step'] >= len(rows)]
    end_lines = [(f'END task {task} ep {trial["episode"]}: outcome {trial["outcome"]}, stop {trial["stop_reason"]}, '
                  f'{len(calls)} VLM calls, {len(rows)} steps', RED), ('', WHITE),
                 ('final object positions (world, m):', GREY)]
    for name, pos in rows[-1]['obj_pos_m'].items():
        end_lines += [(f'  {name}: {np.round(pos, 3).tolist()}  (start {np.round(rows[0]["obj_pos_m"][name], 3).tolist()})', WHITE)]
    for c in after_last:
        end_lines += [('', WHITE), (f'last VLM call @ step {c["step"]} (after the last step): {c["parsed"]["verdict"].upper()}', YELLOW),
                      (f'trigger: {c["trigger"]}', GREY), (f'evidence: {c["parsed"]["evidence"]}', WHITE)]
    end_panel = draw_panel(end_lines, highlight=False)
    for _ in range(HOLD_FRAMES):
        writer.write(np.concatenate([frames[-1], end_panel], axis=1))
    writer.release()
    # cv2 writes mp4v, which browsers and VS Code don't play; re-encode to h264 like the source videos.
    h264_path = out_path.replace('.mp4', '_h264.mp4')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-loglevel', 'error', '-y', '-i', out_path, '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', h264_path], check=True)
    os.replace(h264_path, out_path)
    summary['video'] = out_path
    summary['source'] = ep_dir
    return summary


def check_argmax():
    """Known case: episode 7 call 2 target is 0.8 * kp6 + 0.2 * kp5 (written in its guidance)."""
    call_dir = 'outputs/goal5fix_on_t0-4/2026-11-19_01-25-56/episode_7/closed_loop/call_02_step0030'
    _, _, _, kp = parse_prompt(os.path.join(call_dir, 'prompt.txt'))
    fn = load_functions_from_txt(os.path.join(call_dir, 'stage1_guidance.txt'), validate=False)[0]
    expected = 0.8 * kp[6] + 0.2 * kp[5]
    got, _ = reward_argmax(fn, kp, kp[6] + np.array([0.1, -0.1, 0.2]))
    err_cm = np.linalg.norm(got - expected) * 100
    print(f'argmax check: input ep7 call2 -> got {np.round(got, 4).tolist()} expected {np.round(expected, 4).tolist()} '
          f'err {err_cm:.2f} cm -> {"match" if err_cm < 0.1 else "MISMATCH"}')


if __name__ == '__main__':
    check_argmax()
    summaries = []
    for task in TASKS:
        trial, ep_dir = pick_episode(task)
        summaries.append(make_video(task, trial, ep_dir))
    with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summaries, f, indent=1)
    for s in summaries:
        print(f"\ntask {s['task']} ep {s['episode']} init {s['init_state']} stop {s['stop_reason']} steps {s['n_steps']} -> {s['video']}")
        for c in s['calls']:
            line = f"  step {c['step']:3d} {c['verdict']:8s} st{c['stage']} | {c['trigger'][:45]}"
            if c.get('target_desc') == 'flat reward: no steering':
                line += ' | flat reward: no steering'
            elif 'target_desc' in c:
                line += (f" | target {c['target_desc']} | EE->target start {c['ee_to_target_start_cm']} "
                         f"min {c['ee_to_target_min_cm']} end {c['ee_to_target_end_cm']} cm over {c['steps_in_segment']} steps")
            print(line)
            print(f"      evidence: {c['evidence']}")
