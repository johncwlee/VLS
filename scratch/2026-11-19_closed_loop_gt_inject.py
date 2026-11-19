# Re-sends saved closed-loop calls with and without a ground-truth contact/height block, and prints both verdicts.
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from closed_loop.query import run_call

RUN = 'outputs/closed_loop_step5_objlog_on/2026-11-18_23-49-17'
MODEL = 'claude-opus-5-5'  # same as the run
EFFORT = 'low'
out_root = f'outputs/closed_loop_gt_inject/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

# wine_rack_1_top_region, read from the sim by scratch/2026-11-19_rack_top_region.py (world frame, m)
RACK_SITE_POS_M = np.array([-0.265, -0.18, 1.137])
RACK_SITE_MAT = np.array([[-1.0, 0.0, 0.0], [0.0, -0.462, -0.887], [0.0, -0.887, 0.462]])
RACK_HALF_SIZE_M = np.array([0.101, 0.022, 0.003])

# (episode folder, call folder, what a right answer looks like)
CASES = [
    ('episode_9_failed', 'call_05_step0100', 'not "on the plate"; bottle must still be lowered'),
    ('episode_9_failed', 'call_06_step0140', 'not "on the plate"; bottle must still be lowered'),
    ('episode_9_failed', 'call_07_step0180', 'not "on the plate"; bottle must still be lowered'),
    ('episode_10', 'call_05_step0110', 'not done; cheese is not on the rack'),
]


def step_row(episode_dir, step):
    with open(f'{RUN}/{episode_dir}/steps.jsonl') as f:
        for line in f:
            row = json.loads(line)
            if row['step'] == step:
                return row
    raise ValueError(f'step {step} not in {episode_dir}/steps.jsonl')


def ground_truth_text(episode_dir, call_step):
    row = step_row(episode_dir, call_step - 1)  # state after the last executed step, as the VLM's image shows
    pos = {name: np.array(xyz) for name, xyz in row['obj_pos_m'].items()}
    if 'wine_bottle_1' in pos:
        bottle, plate = pos['wine_bottle_1'], pos['plate_1']
        dz_cm = (bottle[2] - plate[2]) * 100
        dxy_cm = np.linalg.norm(bottle[:2] - plate[:2]) * 100
        touching = 'yes' if row['obj_contact']['wine_bottle_1|plate_1'] else 'no'
        return (f'Measured from the simulator (ground truth): wine bottle touching the plate: {touching}. '
                f'The bottle\'s base is {dz_cm:.1f} cm above the plate\'s base (when the bottle stood on the table '
                f'it was -0.4 cm, so a bottle resting on the plate is about 0 cm); horizontal distance between their '
                f'centres: {dxy_cm:.1f} cm.')
    cheese = pos['cream_cheese_1']
    delta_m = RACK_SITE_MAT @ (cheese - RACK_SITE_POS_M)  # same formula as LIBERO site_object.py under()
    along_past_edge_cm = (abs(delta_m[0]) - RACK_HALF_SIZE_M[0]) * 100
    across_cm = abs(delta_m[1]) * 100
    inside = 'yes' if (abs(delta_m[0]) < RACK_HALF_SIZE_M[0] and abs(delta_m[1]) < RACK_HALF_SIZE_M[1]) else 'no'
    return (f'Measured from the simulator (ground truth): cream cheese inside the rack\'s top region (the area that '
            f'counts as "on the rack"): {inside}. Along the rack\'s long side it is {along_past_edge_cm:.1f} cm '
            f'past the region\'s edge; across the rack it is {across_cm:.1f} cm from the centre line '
            f'(region half-width {RACK_HALF_SIZE_M[1] * 100:.1f} cm).')


results = []
for episode_dir, call_name, expected in CASES:
    saved = f'{RUN}/{episode_dir}/closed_loop/{call_name}'
    with open(f'{saved}/prompt.txt') as f:
        prompt = f.read()
    images = [cv2.imread(f'{saved}/image_{i}.png')[..., ::-1].copy() for i in (1, 2)]  # RGB
    with open(f'{saved}/parsed.json') as f:
        original = json.load(f)
    call_step = int(call_name[-4:])
    truth = ground_truth_text(episode_dir, call_step)
    gripper_line = next(line for line in prompt.split('\n') if line.startswith('Gripper: '))
    assert prompt.count(gripper_line + '\n') == 1, f'{call_name}: gripper line not unique'
    prompt_gt = prompt.replace(gripper_line + '\n', gripper_line + '\n' + truth + '\n')

    answers = {}
    for variant, text in (('plain', prompt), ('with_gt', prompt_gt)):
        call_dir = f'{out_root}/{episode_dir}_{call_name}_{variant}'
        try:
            parsed = run_call(text, images, call_dir, MODEL, EFFORT)
            answers[variant] = (parsed['verdict'], parsed['evidence'])
        except Exception as e:
            answers[variant] = ('ERROR', f'{type(e).__name__}: {e}')
    results.append((episode_dir, call_name, expected, truth, original, answers))

print(f'output: {out_root}\n')
for episode_dir, call_name, expected, truth, original, answers in results:
    print(f'=== {episode_dir} {call_name}')
    print(f'  injected:   {truth}')
    print(f'  expected:   {expected}')
    print(f'  run (orig): {original["verdict"]} | {original["evidence"]}')
    for variant in ('plain', 'with_gt'):
        verdict, evidence = answers[variant]
        print(f'  {variant:10s}: {verdict} | {evidence}')
    print()
