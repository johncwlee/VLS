# Re-runs step 3 case C (good grasp, frame 40) with a time-limit trigger instead of "gripper closed", same plan and history as run 2026-11-18_04-49-53.
import json
import os
import sys
from datetime import datetime

import cv2

sys.path.insert(0, os.path.dirname(__file__))
import importlib
step3 = importlib.import_module('2026-11-18_closed_loop_query')

previous = 'outputs/closed_loop_step3/2026-11-18_04-49-53'
out_root = f'outputs/closed_loop_step3/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_c_no_hint'
os.makedirs(out_root)

with open(f'{previous}/A_first/parsed.json') as f:
    a = json.load(f)
meta3, task3, _, start3 = step3.load_episode(3)
hist3 = [step3.history_entry(0, a['plan'], 1, 1, 0, 0.0, 0.0, 'initial query', 'plan', a['evidence'])]

trigger = 'stage 1 time limit reached (4 chunks)'
prompt_c2 = step3.check_prompt(meta3, task3, a['plan'], 1, 1, hist3, trigger)
step3.report('C2. good grasp, NO gripper hint (episode 3, frame 40)', trigger,
             'verdict=advance from the image alone. If continue: C passed because of the trigger text',
             f'{out_root}/C2_good_grasp_no_hint', prompt_c2, [start3, step3.video_frame(3, 'success', 40)])
cv2.imwrite(f'{out_root}/summary_all.png', step3.summaries[0][..., ::-1])
