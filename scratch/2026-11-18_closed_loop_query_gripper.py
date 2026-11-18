# Re-runs step 3 cases C2 (grasp, time-limit trigger) and D (stuck over plate) with the new "Gripper:" line; all else as before.
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
out_root = f'outputs/closed_loop_step3/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_gripper_state'
os.makedirs(out_root)

with open(f'{previous}/A_first/parsed.json') as f:
    a = json.load(f)
meta3, task3, _, start3 = step3.load_episode(3)
hist3 = [step3.history_entry(0, a['plan'], 1, 1, 0, 0.0, 0.0, 'initial query', 'plan', a['evidence'])]
trigger = 'stage 1 time limit reached (4 chunks)'
prompt = step3.check_prompt(meta3, task3, a['plan'], 1, 1, hist3, trigger, 'closed')  # frame 40: Grip:C
step3.report('C2 + gripper. grasp, time-limit trigger (episode 3, frame 40)', trigger + '  |  Gripper: closed',
             'advance (was continue without the gripper line)',
             f'{out_root}/C2_gripper', prompt, [start3, step3.video_frame(3, 'success', 40)])

meta9, task9, plan9, start9 = step3.load_episode(9)
hist9 = [step3.history_entry(0, plan9, 1, 1, 0, 0.0, 0.0, 'initial query', 'plan', 'nothing done yet'),
         step3.history_entry(40, plan9, 1, 1, 4, 0.85, 0.85, 'gripper closed', 'advance', 'the bottle is in the gripper')]
trigger = 'stage 2 time limit reached (16 chunks)'
prompt = step3.check_prompt(meta9, task9, plan9, 1, 2, hist9, trigger, 'closed')  # frame 200: Grip:C
step3.report(f'D + gripper. stuck over plate (episode 9, frame 200), "{task9}"', trigger + '  |  Gripper: closed',
             'NOT done (was done without the gripper line). replan or continue',
             f'{out_root}/D_gripper', prompt, [start9, step3.video_frame(9, 'fail', 200)])

width = max(s.shape[1] for s in step3.summaries)
rows = [np.hstack([s, np.full((s.shape[0], width - s.shape[1], 3), 255, dtype=np.uint8)]) for s in step3.summaries]
cv2.imwrite(f'{out_root}/summary_all.png', np.vstack(rows)[..., ::-1])
print(f'overview: {out_root}/summary_all.png')
