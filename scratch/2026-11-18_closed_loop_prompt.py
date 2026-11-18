# Renders the closed-loop VLS prompt for 4 cases (first, check after a good grasp, after a drop, after a replan) from a saved episode.
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from closed_loop.prompt import build_prompt

episode_vlm_dir = 'outputs/vlsmain_equiv_opus_low/2026-11-07_03-26-40/episode_3/vlm_agent'
out_dir = sys.argv[1]

with open(os.path.join(episode_vlm_dir, 'metadata.json')) as f:
    metadata = json.load(f)
with open(os.path.join(episode_vlm_dir, 'prompt.txt')) as f:
    instruction = re.search(r'Query Task: "(.*)"', f.read()).group(1)
with open(os.path.join(episode_vlm_dir, 'output_raw.txt')) as f:
    plan = [re.sub(r'^# Stage \d+:\s*', '', line) for line in f.read().split('\n') if re.match(r'# Stage \d+:', line)]
with open('vlm_query/guidance_template_libero.txt') as f:
    libero_notes = f.read()

first_call = {'step': 0, 'plan_id': 1, 'stage': 1, 'stage_name': plan[0], 'chunks': 0, 'reward_now': 0.0,
              'reward_best': 0.0, 'trigger': 'initial query', 'verdict': 'plan',
              'evidence': 'nothing is done yet; the bottle is upright on the table'}
grasp_ok = {'step': 40, 'plan_id': 1, 'stage': 1, 'stage_name': plan[0], 'chunks': 4, 'reward_now': 0.85,
            'reward_best': 0.85, 'trigger': 'gripper closed', 'verdict': 'advance', 'evidence': 'the bottle is in the gripper'}
dropped = {'step': 70, 'plan_id': 1, 'stage': 2, 'stage_name': plan[1], 'chunks': 3, 'reward_now': 0.30,
           'reward_best': 0.70, 'trigger': 'gripper opened', 'verdict': 'replan',
           'evidence': 'the bottle fell on the table next to the bowl'}
new_plan = ['Move gripper to grasp the fallen wine bottle again (keypoints 3, 4)',
            'Move gripper with wine bottle to above the bowl (keypoints 0, 1)']
cases = {
    'first': dict(mode='first', plan=[], plan_id=0, current_stage=1, history=[], trigger_reason='initial query', gripper_state='open'),
    'check_good_grasp': dict(mode='check', plan=plan, plan_id=1, current_stage=1, history=[first_call],
                             trigger_reason='gripper closed', gripper_state='closed'),
    'check_drop': dict(mode='check', plan=plan, plan_id=1, current_stage=2, history=[first_call, grasp_ok],
                       trigger_reason='gripper opened', gripper_state='open'),
    'check_after_replan': dict(mode='check', plan=new_plan, plan_id=2, current_stage=1,
                               history=[first_call, grasp_ok, dropped], trigger_reason='gripper closed',
                               gripper_state='closed'),
}

print(f"task: {instruction}")
print(f"plan from saved output: {plan}")
for name, case in cases.items():
    text = build_prompt(instruction=instruction, key_points_objects_map=metadata['key_points_objects_map'],
                        keypoint_positions=metadata['init_keypoint_positions'],
                        task_specific_patterns=libero_notes, **case)
    leftover = re.findall(r'\{(mode_instructions|instruction|trigger_reason|plan|history|key_points_objects_map|'
                          r'keypoint_positions|task_specific_patterns|image_description|current_stage|plan_id)\}', text)
    path = os.path.join(out_dir, f'prompt_{name}.txt')
    with open(path, 'w') as f:
        f.write(text)
    print(f"{name}: {len(text)} chars, unfilled placeholders: {leftover}, saved {path}")
