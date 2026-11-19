# Per suite task of the goal5 runs: flag on vs off outcomes, VLM calls and errors, and per-failure end state from steps.jsonl.
import collections
import glob
import json
import os
import sys

import numpy as np

PREFIX = sys.argv[2] if len(sys.argv) > 2 else 'goal5fix'  # goal5 = random layouts (stopped), goal5fix = fixed init states
RUNS = {('on', 0): f'{PREFIX}_on_t0-4', ('on', 5): f'{PREFIX}_on_t5-9', ('off', 0): f'{PREFIX}_off_t0-4', ('off', 5): f'{PREFIX}_off_t5-9'}
OPEN_WIDTH_M = 0.06  # same threshold as main.py _closed_loop_step


def run_dir(name):
    return sorted(glob.glob(f'outputs/{name}/*/'))[-1]


def episode_dir(base, episode):
    for suffix in ('', '_failed'):
        path = os.path.join(base, f'episode_{episode}{suffix}')
        if os.path.isdir(path):
            return path
    return None


def end_state(ep_dir):
    """Short description of the episode from steps.jsonl: grasps, max lift, final object state."""
    path = os.path.join(ep_dir, 'steps.jsonl')
    if not os.path.exists(path):
        return 'no steps.jsonl'
    rows = [json.loads(line) for line in open(path)]
    widths_cm = np.array([(r['finger_qpos_m'][0] - r['finger_qpos_m'][1]) * 100 for r in rows])
    closed = widths_cm < OPEN_WIDTH_M * 100
    n_closes = int(np.sum(closed[1:] & ~closed[:-1]))
    empty_closes = int(np.sum((widths_cm[1:] < 1.0) & (widths_cm[:-1] >= 1.0)))
    parts = [f'{len(rows)} steps', f'{n_closes} closes ({empty_closes} on nothing)']
    names = list(rows[0]['obj_pos_m'])
    for name in names:
        z0 = rows[0]['obj_pos_m'][name][2]
        lift_cm = max(r['obj_pos_m'][name][2] - z0 for r in rows) * 100
        moved_cm = np.linalg.norm(np.array(rows[-1]['obj_pos_m'][name][:2]) - np.array(rows[0]['obj_pos_m'][name][:2])) * 100
        parts.append(f'{name}: max lift {lift_cm:.1f} cm, xy moved {moved_cm:.1f} cm')
    contacts = [k for k, v in rows[-1]['obj_contact'].items() if v]
    parts.append(f'final width {widths_cm[-1]:.1f} cm, contacts {contacts or "none"}')
    return '; '.join(parts)


def calls_summary(ep_dir):
    verdicts = []
    for call in sorted(glob.glob(os.path.join(ep_dir, 'closed_loop', 'call_*'))):
        parsed = os.path.join(call, 'parsed.json')
        verdicts.append(json.load(open(parsed))['verdict'] if os.path.exists(parsed) else 'ERR')
    return ' '.join(v[:3] for v in verdicts)


task = int(sys.argv[1])
offset = 0 if task < 5 else 5
for flag in ('on', 'off'):
    base = run_dir(RUNS[(flag, offset)])
    trials = [json.loads(line) for line in open(os.path.join(base, 'trials.jsonl'))]
    trials = [t for t in trials if t['task_id'] + offset == task]
    outcomes = collections.Counter(t['outcome'] for t in trials)
    print(f'=== task {task} flag {flag}: {dict(outcomes)}  ({base})')
    for t in trials:
        ep_dir = episode_dir(base, t['episode'])
        line = f"  ep {t['episode']:2d} {t['outcome']:8s}"
        if flag == 'on':
            line += f" calls {t.get('closed_loop_calls')} stop {t.get('stop_reason')} | {calls_summary(ep_dir)}"
        if t.get('vlm_errors'):
            line += f" vlm_errors {len(t['vlm_errors'])}"
        if t.get('error'):
            line += f" ERROR {t['error'][:120]}"
        print(line)
        if t['outcome'] != 'success':
            print(f'      {end_state(ep_dir)}')
