"""
Summarize run_grasp_experiments.sh results (outer repo) into per-suite, per-arm tables.

Reads results/vls/grasp_<arm>/<suite>/<timestamp>/{trials.jsonl, episode_*/chunks.jsonl}:
  task success      trials.jsonl outcome == 'success' (errors count as failures; also reported separately)
  grasp success     any object of interest lifted >= --lift (default 5 cm) at some point (max_lift, eval-only)
  engagement        grasp arms only: share of episodes where GraspReward steered at least one chunk
                    (trials.jsonl grasp_engaged_chunks > 0), and task success among those episodes. Episodes with
                    no usable VLM grasp target run as plain VLS and dilute the arm's overall success.
  steering displacement (chunks.jsonl, chunks with main.log_unsteered), stage-1 chunks only (the grasp stage in
                    LIBERO pick tasks; VLS arms have no grasp-stage label): twin_l2 is the L2 distance in normalized
                    action space between the executed chunk and its unsteered twin (same initial noise, matched
                    through FKD ancestry); twin_tcp_dist the same for the chunk-end TCP position (m). Lower = closer
                    to the base policy; the coupled-steering claim is that coupled arms are lower than one_way at
                    equal or better success.
  guidance_energy   sum over flow steps of |scale * grad|^2 * dt. Gradients are unit-normalized, so this mostly
                    tracks the guidance schedule (scale), not displacement; use twin_* for displacement claims
  grasp diagnostics fraction of grasp-stage chunks committed, GraspGen-X acquisition latency
  twist check       per episode, at the last grasp-stage chunk (the hand-over to the policy): predicted angle error
                    to the target grasp (end_rot_err) and its approach-axis part (end_tilt). twist_frac = episodes
                    whose approach is right (tilt < --tilt_ok) but whose total error is large (> --rot_bad): the
                    remaining error is twist about the approach axis, which only matters for asymmetric objects.
                    A high twist_frac means steering fights the policy over twist; see plans/graspgen-x_experiments_next.md
                    (check 3) for the grouped-commit fix
  gripper closes    trials.jsonl gripper_closes (main.analysis_logs), any arm: closes = close commands per episode
                    (> 1 = regrasps); at the first close, close_speed (TCP cm/step just before), close_quality
                    (GraspGen-X discriminator confidence of that TCP pose, best object of interest; needs
                    main.grasp.score_at_close) and close_grasp_rot (grasp arms: angle to the nearest GraspGen-X target)
  runs              timestamped run directories pooled per (suite, arm); > 1 means repeated or resumed runs
Each arm's success and grasp rates are compared with every --baselines arm in the same suite (two-proportion
z-test p-values, columns *_p_vs_<baseline>). Default baselines: vls_lag first, because grasp arms also switch the
EE path model to `lag`, so one_way vs vls_lag isolates the grasp reward (the P2 pass bar); vls (as released) second.

Usage (outer repo):
    envs/vls/bin/python modules/VLS/scripts/summarize_grasp_experiments.py results/vls [--lift 0.05] [--json out.json]
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np


def two_proportion_p(k1, n1, k2, n2):
    if min(n1, n2) == 0:
        return float('nan')
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k1 / n1 - k2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def collect(root, lift, tilt_ok, rot_bad):
    stats = defaultdict(lambda: defaultdict(list))  # (suite, arm) -> metric -> values
    for trials_path in glob.glob(os.path.join(root, 'grasp_*', '*', '*', 'trials.jsonl')):
        run_dir = os.path.dirname(trials_path)
        arm = os.path.basename(os.path.dirname(os.path.dirname(run_dir)))[len('grasp_'):]
        suite = os.path.basename(os.path.dirname(run_dir))
        s = stats[(suite, arm)]
        s['runs'].append(run_dir)
        for t in read_jsonl(trials_path):
            s['success'].append(t.get('outcome') == 'success')
            s['error'].append(t.get('outcome') not in ('success', 'fail'))
            max_lift = t.get('max_lift') or {}
            if max_lift:
                s['grasp'].append(max(max_lift.values()) >= lift)
            if 'gripper_closes' in t:
                closes = t['gripper_closes']
                s['closes'].append(len(closes))
                first = closes[0] if closes else {}
                if first.get('speed_cm') is not None:
                    s['close_speed'].append(first['speed_cm'])
                quality = [q for q in (first.get('quality') or {}).values() if q is not None]
                if quality:
                    s['close_quality'].append(max(quality))
                if 'nearest_grasp_rot_deg' in first:
                    s['close_grasp_rot'].append(first['nearest_grasp_rot_deg'])
            if 'grasp_engaged_chunks' in t:
                engaged = t['grasp_engaged_chunks'] > 0
                s['engaged'].append(engaged)
                if engaged:
                    s['engaged_success'].append(t.get('outcome') == 'success')
        for chunks_path in glob.glob(os.path.join(run_dir, 'episode_*', 'chunks.jsonl')):
            chunks = read_jsonl(chunks_path)
            last = next((c['grasp'] for c in reversed(chunks) if 'end_tilt_deg' in c.get('grasp', {})), None)
            if last is not None:
                s['end_rot_err'].append(last['end_rot_err_deg'])
                s['end_tilt'].append(last['end_tilt_deg'])
                s['twist'].append(last['end_tilt_deg'] < tilt_ok and last['end_rot_err_deg'] > rot_bad)
            for c in chunks:
                if 'guidance_energy' in c:
                    s['guidance_energy'].append(c['guidance_energy'])
                if c.get('stage') == 1:
                    for key in ('twin_l2', 'twin_tcp_dist'):
                        if key in c:
                            s[key].append(c[key])
                if 'grasp' in c:
                    s['committed'].append(c['grasp'].get('committed') is not None)
                    if c['grasp'].get('acquire_infer_ms'):
                        s['acquire_ms'].append(c['grasp']['acquire_infer_ms'])
    return stats


def summarize(stats, baselines):
    rows = []
    for (suite, arm), s in sorted(stats.items()):
        row = {'suite': suite, 'arm': arm, 'episodes': len(s['success']), 'runs': len(s['runs'])}
        for key in ('success', 'grasp'):
            if s[key]:
                row[f'{key}_rate'] = float(np.mean(s[key]))
                for b in baselines:
                    base = stats.get((suite, b), {})
                    if arm != b and base.get(key):
                        row[f'{key}_p_vs_{b}'] = two_proportion_p(sum(s[key]), len(s[key]), sum(base[key]), len(base[key]))
        row['error_rate'] = float(np.mean(s['error'])) if s['error'] else 0.0
        if s['engaged']:
            row['engaged_rate'] = float(np.mean(s['engaged']))
        if s['engaged_success']:
            row['engaged_success_rate'] = float(np.mean(s['engaged_success']))
        for key in ('twin_l2', 'twin_tcp_dist', 'guidance_energy', 'acquire_ms', 'closes', 'close_speed',
                    'close_quality', 'close_grasp_rot'):
            if s[key]:
                row[f'{key}_median'] = float(np.median(s[key]))
        row['close_quality_n'] = len(s['close_quality'])  # episodes scored (None when the object was occluded)
        if s['committed']:
            row['committed_frac'] = float(np.mean(s['committed']))
        if s['end_rot_err']:
            row['end_rot_err_median'] = float(np.median(s['end_rot_err']))
            row['end_tilt_median'] = float(np.median(s['end_tilt']))
            row['twist_frac'] = float(np.mean(s['twist']))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', help='results root containing grasp_<arm>/<suite>/<timestamp>/')
    parser.add_argument('--lift', type=float, default=0.05, help='grasp success lift threshold (m)')
    parser.add_argument('--tilt_ok', type=float, default=15.0, help='twist check: approach-axis error counted as fine (deg)')
    parser.add_argument('--rot_bad', type=float, default=30.0, help='twist check: total angle error counted as large (deg)')
    parser.add_argument('--baselines', nargs='+', default=['vls_lag', 'vls'],
                        help='arms to compare every other arm against (p-value columns), first = primary')
    parser.add_argument('--json', default=None, help='also write the rows as JSON')
    args = parser.parse_args()

    rows = summarize(collect(args.root, args.lift, args.tilt_ok, args.rot_bad), args.baselines)
    if not rows:
        raise SystemExit(f"no grasp_<arm>/<suite>/<timestamp>/trials.jsonl under {args.root}")
    cols = (['suite', 'arm', 'runs', 'episodes', 'success_rate'] + [f'success_p_vs_{b}' for b in args.baselines]
            + ['grasp_rate'] + [f'grasp_p_vs_{b}' for b in args.baselines] + [
            'engaged_rate', 'engaged_success_rate', 'error_rate', 'twin_l2_median', 'twin_tcp_dist_median',
            'guidance_energy_median', 'committed_frac', 'acquire_ms_median', 'end_rot_err_median', 'end_tilt_median',
            'twist_frac', 'closes_median', 'close_speed_median', 'close_quality_median', 'close_quality_n',
            'close_grasp_rot_median'])
    print('\t'.join(cols))
    for r in rows:
        print('\t'.join(f"{r[c]:.3g}" if isinstance(r.get(c), float) else str(r.get(c, '-')) for c in cols))
    for r in rows:
        if r.get('twist_frac', 0) > 0.5:
            print(f"note: {r['arm']}/{r['suite']}: {r['twist_frac']:.0%} of episodes hand over with the approach right but "
                  f"> {args.rot_bad:g} deg total error (twist). See plans/graspgen-x_experiments_next.md, check 3.")
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(rows, f, indent=2)


if __name__ == '__main__':
    main()
