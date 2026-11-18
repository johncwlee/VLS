# Re-runs case D (stuck over plate) and adds case E (task really finished) after the "no longer interacting" line in the done rule.
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import importlib
step3 = importlib.import_module('2026-11-18_closed_loop_query')

previous = 'outputs/closed_loop_step3/2026-11-18_04-49-53'
out_root = f'outputs/closed_loop_step3/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_done_rule'
os.makedirs(out_root)

meta9, task9, plan9, start9 = step3.load_episode(9)
hist9 = [step3.history_entry(0, plan9, 1, 1, 0, 0.0, 0.0, 'initial query', 'plan', 'nothing done yet'),
         step3.history_entry(40, plan9, 1, 1, 4, 0.85, 0.85, 'gripper closed', 'advance', 'the bottle is in the gripper')]
trigger = 'stage 2 time limit reached (16 chunks)'
prompt = step3.check_prompt(meta9, task9, plan9, 1, 2, hist9, trigger, 'closed')  # frame 200: Grip:C
step3.report(f'D. stuck over plate (episode 9, frame 200), "{task9}"', trigger + '  |  Gripper: closed',
             'NOT done (was done twice before). replan or continue',
             f'{out_root}/D_done_rule', prompt, [start9, step3.video_frame(9, 'fail', 200)])

with open(f'{previous}/A_first/parsed.json') as f:
    a = json.load(f)
meta3, task3, _, start3 = step3.load_episode(3)
hist3 = [step3.history_entry(0, a['plan'], 1, 1, 0, 0.0, 0.0, 'initial query', 'plan', a['evidence']),
         step3.history_entry(40, a['plan'], 1, 1, 4, 0.85, 0.85, 'gripper closed', 'advance', 'the bottle is grasped')]
trigger = 'gripper opened'
prompt = step3.check_prompt(meta3, task3, a['plan'], 1, 2, hist3, trigger, 'open')  # frame 60: Grip:O
step3.report(f'E. task finished (episode 3, frame 60, env says success), "{task3}"', trigger + '  |  Gripper: open',
             'done (the new line must not block a real finish)',
             f'{out_root}/E_finished', prompt, [start3, step3.video_frame(3, 'success', 60)])

width = max(s.shape[1] for s in step3.summaries)
rows = [np.hstack([s, np.full((s.shape[0], width - s.shape[1], 3), 255, dtype=np.uint8)]) for s in step3.summaries]
cv2.imwrite(f'{out_root}/summary_all.png', np.vstack(rows)[..., ::-1])
print(f'overview: {out_root}/summary_all.png')
