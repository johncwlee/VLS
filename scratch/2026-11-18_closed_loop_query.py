# Sends 4 real closed-loop VLM calls on saved scenes and prints, per case, what we expected vs what the VLM answered.
import json
import os
import re
import sys
from datetime import datetime

import textwrap

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from closed_loop.prompt import build_prompt
from closed_loop.query import run_call

RUN = 'outputs/vlsmain_equiv_opus_low/2026-11-07_03-26-40'
MODEL = 'claude-opus-5-5'  # same as configs/perception.yaml
EFFORT = 'low'
out_root = f'outputs/closed_loop_step3/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

with open('vlm_query/guidance_template_libero.txt') as f:
    libero_notes = f.read()


def load_episode(episode):
    vlm_dir = f'{RUN}/episode_{episode}/vlm_agent'
    with open(f'{vlm_dir}/metadata.json') as f:
        metadata = json.load(f)
    with open(f'{vlm_dir}/prompt.txt') as f:
        instruction = re.search(r'Query Task: "(.*)"', f.read()).group(1)
    with open(f'{vlm_dir}/output_raw.txt') as f:
        saved_plan = [re.sub(r'^# Stage \d+:\s*', '', l) for l in f.read().split('\n') if re.match(r'# Stage \d+:', l)]
    start_img = cv2.imread(f'{vlm_dir}/query_img.png')[..., ::-1].copy()  # RGB, keypoints drawn at t=0
    return metadata, instruction, saved_plan, start_img


def video_frame(episode, outcome, index):
    """RGB frame from the saved video, with the status box and SUCCESS/FAIL label painted black."""
    cap = cv2.VideoCapture(f'{RUN}/episode_{episode}/episode_{episode}_{outcome}_agentview.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    assert ok, f'could not read frame {index} of episode {episode}'
    frame[0:150, 0:260] = 0    # status box: "Step Stage Guide Grip ..." (pixels in the 640x640 frame)
    frame[0:45, 470:640] = 0   # SUCCESS / FAIL label
    return frame[..., ::-1].copy()


def check_prompt(metadata, instruction, plan, plan_id, stage, history, trigger, gripper_state):
    text = build_prompt(mode='check', instruction=instruction, plan=plan, plan_id=plan_id, current_stage=stage,
                        history=history, trigger_reason=trigger, gripper_state=gripper_state,
                        key_points_objects_map=metadata['key_points_objects_map'],
                        keypoint_positions=metadata['init_keypoint_positions'], task_specific_patterns=libero_notes)
    # Offline only: video frames have no keypoints drawn, and the saved positions are from t=0
    old = 'Image 2: the scene now, with keypoints at their current positions.'
    assert text.count(old) == 1
    text = text.replace(old, 'Image 2: the scene now (keypoints are not drawn on this image).')
    old = 'Keypoint 3D positions [X, Y, Z] (current positions)'
    assert text.count(old) == 1
    return text.replace(old, 'Keypoint 3D positions [X, Y, Z] (at the start of the episode; objects may have moved since)')


def history_entry(step, plan, plan_id, stage, chunks, now, best, trigger, verdict, evidence):
    return dict(step=step, plan_id=plan_id, stage=stage, stage_name=plan[stage - 1], chunks=chunks, reward_now=now,
                reward_best=best, trigger=trigger, verdict=verdict, evidence=evidence)


def text_panel(lines, height, width=760):
    """White panel with wrapped black text, as RGB uint8."""
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    y = 28
    for line in lines:
        for chunk in textwrap.wrap(line, 70) or ['']:
            cv2.putText(panel, chunk, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            y += 22
    return panel


def summary_image(images, lines):
    """The images the VLM saw (with captions) next to the case text."""
    tiles = []
    for i, rgb in enumerate(images, start=1):
        tile = cv2.resize(rgb, (480, 480))
        caption = 'Image 1: start, keypoints drawn' if i == 1 and len(images) > 1 else ('Image 2: now' if i == 2 else 'Image: now')
        tile = np.vstack([text_panel([caption], 40, 480), tile])
        tiles.append(tile)
    return np.hstack(tiles + [text_panel(lines, 520)])


summaries = []


def report(name, trigger, expected, call_dir, prompt_text, images):
    gripper_line = re.search(r'^Gripper: .*$', prompt_text, re.MULTILINE).group(0)
    lines = [f'CASE: {name}', f'TRIGGER: {trigger}  |  {gripper_line}', f'EXPECTED: {expected}']
    try:
        p = run_call(prompt_text, images, call_dir, MODEL, EFFORT)
        lines.append(f'GOT: verdict={p["verdict"]}  done={p["completed_stages"]}  undone={p["undone_stages"]}  '
                     f'stage={p["current_stage"]}  guidance={p["guidance"]}  function={p["function_name"]}  '
                     f'loaded={len(p["guidance_fns"])}  ({p["seconds"]:.0f}s)')
        lines.append(f'EVIDENCE: {p["evidence"]}')
        for i, stage in enumerate(p['plan'], start=1):
            lines.append(f'PLAN stage {i}: {stage}')
        if p['function_source']:
            targets = [l.strip() for l in p['function_source'].split('\n') if 'torch.tensor([' in l and 'long' in l]
            lines.append(f'TARGET KEYPOINTS: {targets}')
    except Exception as e:
        p = None
        if not os.path.exists(call_dir):
            os.makedirs(call_dir)
        lines.append(f'ERROR: {type(e).__name__}: {e}')
    lines.append(f'FILES: {call_dir}')
    for line in lines:
        print(line)
    print()
    image = summary_image(images, lines)
    cv2.imwrite(f'{call_dir}/summary.png', image[..., ::-1])
    summaries.append(image)
    return p


if __name__ == '__main__':
    os.makedirs(out_root)

    # Case A: first call, episode 3 start image
    meta3, task3, saved_plan3, start3 = load_episode(3)
    prompt_a = build_prompt(mode='first', instruction=task3, plan=[], plan_id=0, current_stage=1, history=[],
                            trigger_reason='initial query', gripper_state='open', key_points_objects_map=meta3['key_points_objects_map'],
                            keypoint_positions=meta3['init_keypoint_positions'], task_specific_patterns=libero_notes)
    a = report(f'A. first call, "{task3}"', 'initial query',
               'verdict=plan, 2 stages (grasp bottle, move to bowl), stage=1, stage1_guidance aims at bottle keypoints (3/4), loads',
               f'{out_root}/A_first', prompt_a, [start3])
    plan3 = a['plan'] if a is not None else saved_plan3
    hist3 = [history_entry(0, plan3, 1, 1, 0, 0.0, 0.0, 'initial query', 'plan', a['evidence'] if a else 'plan made')]

    # Case B: nothing has happened yet (frame 0), checked on a stage time limit
    prompt_b = check_prompt(meta3, task3, plan3, 1, 1, hist3, 'stage 1 time limit reached (4 chunks)', 'open')  # frame 0: Grip:O
    report('B. nothing happened yet (episode 3, frame 0)', 'stage 1 time limit reached (4 chunks)',
           'verdict=continue, stage=1, no function. Wrong: advance or done',
           f'{out_root}/B_nothing_happened', prompt_b, [start3, video_frame(3, 'success', 0)])

    # Case C: bottle grasped and lifted (frame 40)
    prompt_c = check_prompt(meta3, task3, plan3, 1, 1, hist3, 'gripper closed', 'closed')  # frame 40: Grip:C
    report('C. good grasp (episode 3, frame 40)', 'gripper closed',
           'verdict=advance, done=[1], stage=2, stage2_guidance aims at bowl keypoints (0/1), loads',
           f'{out_root}/C_good_grasp', prompt_c, [start3, video_frame(3, 'success', 40)])

    # Case C2: same frame as C, but a time-limit trigger, so only the "Gripper:" line says it closed
    prompt_c2 = check_prompt(meta3, task3, plan3, 1, 1, hist3, 'stage 1 time limit reached (4 chunks)', 'closed')
    report('C2. good grasp, time-limit trigger (episode 3, frame 40)', 'stage 1 time limit reached (4 chunks)',
           'verdict=advance, stage=2 (the grasp is known only from the "Gripper: closed" line)',
           f'{out_root}/C2_good_grasp_time_limit', prompt_c2, [start3, video_frame(3, 'success', 40)])

    # Case D: episode 9 failure, bottle held over the plate and never released (frame 200)
    meta9, task9, plan9, start9 = load_episode(9)
    hist9 = [history_entry(0, plan9, 1, 1, 0, 0.0, 0.0, 'initial query', 'plan', 'nothing done yet'),
             history_entry(40, plan9, 1, 1, 4, 0.85, 0.85, 'gripper closed', 'advance', 'the bottle is in the gripper')]
    prompt_d = check_prompt(meta9, task9, plan9, 1, 2, hist9, 'stage 2 time limit reached (16 chunks)', 'closed')  # frame 200: Grip:C
    report(f'D. stuck: bottle held over the plate, never released (episode 9, frame 200), "{task9}"',
           'stage 2 time limit reached (16 chunks)',
           'NOT done and NOT advance (the bottle is not placed). replan or continue',
           f'{out_root}/D_stuck', prompt_d, [start9, video_frame(9, 'fail', 200)])

    # Case E: episode 3 really finished (frame 60, bottle released in the bowl, env says success)
    hist3_e = hist3 + [history_entry(40, plan3, 1, 1, 4, 0.85, 0.85, 'gripper closed', 'advance', 'the bottle is grasped')]
    prompt_e = check_prompt(meta3, task3, plan3, 1, 2, hist3_e, 'gripper opened', 'open')  # frame 60: Grip:O
    report(f'E. task finished (episode 3, frame 60, env says success), "{task3}"', 'gripper opened',
           'verdict=done, stage=0 (the "no longer interacting" line must not block a real finish)',
           f'{out_root}/E_finished', prompt_e, [start3, video_frame(3, 'success', 60)])

    # One overview image: the case rows stacked, padded to the same width
    width = max(s.shape[1] for s in summaries)
    rows = [np.hstack([s, np.full((s.shape[0], width - s.shape[1], 3), 255, dtype=np.uint8)]) for s in summaries]
    cv2.imwrite(f'{out_root}/summary_all.png', np.vstack(rows)[..., ::-1])
    print(f'overview: {out_root}/summary_all.png')
