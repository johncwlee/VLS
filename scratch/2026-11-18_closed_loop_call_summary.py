# Draws one picture per closed-loop call of an episode: the images the VLM saw, the trigger, its answer and evidence.
import json
import os
import re
import textwrap

import cv2
import numpy as np

EPISODE_DIR = 'outputs/closed_loop_step5_flag_on/2026-11-18_20-51-42/episode_10'
OUT_DIR = 'outputs/closed_loop_step5_review/episode_10'


def text_panel(lines, height, width=760):
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    y = 28
    for line in lines:
        for chunk in textwrap.wrap(line, 70) or ['']:
            cv2.putText(panel, chunk, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            y += 22
    return panel


os.makedirs(OUT_DIR, exist_ok=True)
call_root = os.path.join(EPISODE_DIR, 'closed_loop')
pictures = []
for call in sorted(os.listdir(call_root)):
    call_dir = os.path.join(call_root, call)
    with open(os.path.join(call_dir, 'prompt.txt')) as f:
        prompt = f.read()
    with open(os.path.join(call_dir, 'parsed.json')) as f:
        parsed = json.load(f)
    trigger = re.search(r'^This call was triggered because: (.*)$', prompt, re.MULTILINE).group(1)
    gripper = re.search(r'^Gripper: .*$', prompt, re.MULTILINE).group(0)
    lines = [f'{call}', f'TRIGGER: {trigger}  |  {gripper}',
             f'ANSWER: verdict={parsed["verdict"]}  stage={parsed["current_stage"]}  done={parsed["completed_stages"]}  '
             f'undone={parsed["undone_stages"]}  guidance={parsed["guidance"]}  function={parsed["function_name"]}  '
             f'({parsed["seconds"]:.0f}s)',
             f'EVIDENCE: {parsed["evidence"]}']
    for i, stage in enumerate(parsed['plan'], start=1):
        lines.append(f'PLAN stage {i}: {stage}')

    tiles = []
    image_paths = sorted(p for p in os.listdir(call_dir) if re.match(r'image_\d+\.png', p))
    for name in image_paths:
        caption = 'Image 1: when the plan was made' if name == 'image_1.png' and len(image_paths) > 1 else 'Image: now'
        if name == 'image_2.png':
            caption = 'Image 2: now'
        tile = cv2.resize(cv2.imread(os.path.join(call_dir, name)), (480, 480))
        tiles.append(np.vstack([text_panel([caption], 40, 480), tile]))
    picture = np.hstack(tiles + [text_panel(lines, 520)])
    cv2.imwrite(os.path.join(OUT_DIR, f'{call}.png'), picture)
    pictures.append(picture)
    print(f'{call}: {trigger} -> {parsed["verdict"]}')

width = max(p.shape[1] for p in pictures)
padded = [np.hstack([p, np.full((p.shape[0], width - p.shape[1], 3), 255, np.uint8)]) for p in pictures]
cv2.imwrite(os.path.join(OUT_DIR, 'all_calls.png'), np.vstack(padded))
