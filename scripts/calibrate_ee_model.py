"""
Fit and check LIBERO's action -> predicted end-effector model (backend.libero.ee_lag_model).

Why this exists: VLS predicts where the gripper will be by adding up pi0.5's delta-EE actions at a fixed
0.01 m per unit. The real robosuite OSC arm lags its commands and keeps moving (about 3.7 cm error at the
end of a 10-step chunk). Keypoint steering only uses the direction, but grasp steering measures real
centimetres (sigma 1.5 cm, 3 cm commit, 6 cm orientation gate), so it needs an accurate model. The lag
model is motion_t = gain * action_t + lag * motion_(t-1) per axis (core/env_adapters/libero_adapter.py,
_ee_step_motion). This script:

  1. drives the arm with bounded random actions in several suites (robot only, no policy, no GPU model),
  2. least-squares fits gain / lag per axis for position and rotation (pooled, and per suite to check that
     the numbers are a property of the arm, not of one scene),
  3. evaluates the *adapter's own* prediction code on held-out 10-step chunks for the constant and the
     fitted lag model, and prints the YAML block to paste into configs/backend/libero.yaml.

Pass criterion (plan P0(c)): median chunk-end error < 1 cm and < 5 deg for the lag model in every suite.
Per-suite gains that differ by more than ~20% would mean the model depends on the scene or arm pose.

First result (2026-10-27, random actions, 4 suites): gains agree across suites (an arm property), but the
lag model only cut chunk-end error from ~2.0 to ~1.6 cm. Even a full 6x6 linear model left ~3.7 mm/step,
so the arm's response to large random commands is not linear. pi0.5's real chunks are smoother; the
decisive check is --from_logs on real rollouts (main.log_steps=true), which fits on part of the episodes
and scores the rest with the same predict_ee_poses the adapter uses.

Run from modules/VLS with vls.env loaded:
    set -a; . ../../.env; . ../../vls.env; set +a
    ../../envs/vls/bin/python scripts/calibrate_ee_model.py --out outputs/ee_calibration.json        # random actions
    ../../envs/vls/bin/python scripts/calibrate_ee_model.py --from_logs 'outputs/**/steps.jsonl'     # real rollouts
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.env_adapters import create_adapter  # noqa: E402
from core.env_adapters.libero_adapter import predict_ee_poses  # noqa: E402

CHUNK = 10  # pi0.5 executes 10 actions per chunk
HOME_RADIUS = 0.12  # m; actions are biased back toward the start pose beyond this, to avoid table contact


def make_adapter(suite: str):
    cfg = OmegaConf.to_container(OmegaConf.load("configs/backend/libero.yaml").libero, resolve=True)
    cfg.update(suite_name=suite, episode_num=1, start_episode=0, task_ids_filter=[0])
    adapter = create_adapter("libero", cfg)
    adapter.reset(seed=0)
    return adapter


def random_actions(rng, adapter, home, n):
    """Piecewise-constant random deltas (held 1-3 steps, like a policy's smooth chunks), pulled home if far."""
    actions, held = [], 0
    for _ in range(n):
        if held == 0:
            current = np.r_[rng.uniform(-0.7, 0.7, 3), rng.uniform(-0.5, 0.5, 3)]
            held = rng.integers(1, 4)
        held -= 1
        a = current.copy()
        offset = home - adapter.get_ee_pose_world().position
        if np.linalg.norm(offset) > HOME_RADIUS:
            a[:3] = np.clip(offset / np.linalg.norm(offset), -1, 1) * 0.7
        actions.append(np.r_[a, -1.0])  # gripper stays open
    return actions


def collect(adapter, rng, steps):
    """Per-step (action, measured motion) rows; motion is what LiberoAdapter.step records."""
    home = adapter.get_ee_pose_world().position.copy()
    rows = []
    prev = {'pos': np.zeros(3), 'rot': np.zeros(3)}
    for a in random_actions(rng, adapter, home, steps):
        adapter.step(torch.tensor(a, dtype=torch.float32))
        motion = adapter._last_ee_motion
        rows.append((a, prev, motion))
        prev = motion
    return rows


def fit(rows):
    """Per-axis least squares motion_t = gain * a_t + lag * motion_(t-1); returns params and per-step RMS."""
    params, rms = {}, {}
    for kind, sl in (('pos', slice(0, 3)), ('rot', slice(3, 6))):
        gains, lags, errs = [], [], []
        for i in range(3):
            X = np.array([[a[sl][i], prev[kind][i]] for a, prev, _ in rows])
            y = np.array([m[kind][i] for _, _, m in rows])
            (g, l), *_ = np.linalg.lstsq(X, y, rcond=None)
            gains.append(float(g)); lags.append(float(l)); errs.append(X @ [g, l] - y)
        params[f'{kind}_gain'], params[f'{kind}_lag'] = gains, lags
        rms[kind] = float(np.sqrt((np.array(errs) ** 2).sum(0).mean()))
    return params, rms


def evaluate(adapter, rng, chunks, model_cfgs):
    """Chunk-end errors of the adapter's own prediction code for each EE model on fresh chunks."""
    home = adapter.get_ee_pose_world().position.copy()
    errors = {name: {'pos_cm': [], 'rot_deg': []} for name in model_cfgs}
    for _ in range(chunks):
        actions = np.array(random_actions(rng, adapter, home, CHUNK))
        predicted = {}
        for name, overrides in model_cfgs.items():
            saved = dict(adapter._env_config)
            adapter._env_config.update(overrides)
            predicted[name] = adapter.delta_actions_to_ee_poses(torch.tensor(actions, dtype=torch.float32)).numpy()
            adapter._env_config.clear(); adapter._env_config.update(saved)
        for a in actions:
            adapter.step(torch.tensor(a, dtype=torch.float32))
        pos, quat = adapter.get_ee_pose_world().position, adapter.get_ee_pose_world().quaternion
        R_actual = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        for name, P in predicted.items():
            errors[name]['pos_cm'].append(100 * float(np.linalg.norm(P[-1, :3, 3] - pos)))
            errors[name]['rot_deg'].append(float(np.degrees(Rotation.from_matrix(P[-1, :3, :3].T @ R_actual).magnitude())))
    return {name: {k: {'median': float(np.median(v)), 'p90': float(np.percentile(v, 90))} for k, v in e.items()}
            for name, e in errors.items()}


def _quat_wxyz_to_matrix(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def load_logged_episodes(patterns: list) -> list:
    """steps.jsonl files (paths or globs) from runs with main.log_steps=true, one list of per-step dicts per episode."""
    episodes = []
    for path in sorted(p for pattern in patterns for p in glob.glob(pattern, recursive=True)):
        with open(path) as f:
            steps = sorted((json.loads(line) for line in f if line.strip()), key=lambda r: r['step'])
        # The last step ends the episode; LIBERO then resets, so its 'after' pose is the reset pose, not arm motion
        steps = steps[:-1]
        for s in steps:
            s['motion'] = {
                'pos': np.subtract(s['ee_pos_after'], s['ee_pos_before']),
                'rot': Rotation.from_matrix(_quat_wxyz_to_matrix(s['ee_quat_wxyz_after'])
                                            @ _quat_wxyz_to_matrix(s['ee_quat_wxyz_before']).T).as_rotvec(),
            }
        episodes.append(steps)
    return episodes


def logged_rows(episodes):
    rows = []
    for steps in episodes:
        prev = {'pos': np.zeros(3), 'rot': np.zeros(3)}
        for s in steps:
            rows.append((np.asarray(s['action']), prev, s['motion']))
            prev = s['motion']
    return rows


def evaluate_logged(episodes, model_cfgs):
    """Chunk-end errors on logged chunks (10 consecutive executed steps starting at chunk_index 0)."""
    errors = {name: {'pos_cm': [], 'rot_deg': []} for name in model_cfgs}
    for steps in episodes:
        for i, s in enumerate(steps):
            chunk = steps[i:i + CHUNK]
            if s['chunk_index'] != 0 or len(chunk) < CHUNK or chunk[-1]['step'] - s['step'] != CHUNK - 1:
                continue
            seed = steps[i - 1]['motion'] if i > 0 and steps[i - 1]['step'] == s['step'] - 1 \
                else {'pos': np.zeros(3), 'rot': np.zeros(3)}
            actions = torch.tensor([c['action'] for c in chunk], dtype=torch.float64)
            start_pos = torch.tensor(s['ee_pos_before'], dtype=torch.float64)
            start_rot = torch.tensor(_quat_wxyz_to_matrix(s['ee_quat_wxyz_before']), dtype=torch.float64)
            end_pos, end_rot = np.asarray(chunk[-1]['ee_pos_after']), _quat_wxyz_to_matrix(chunk[-1]['ee_quat_wxyz_after'])
            for name, env_cfg in model_cfgs.items():
                P = predict_ee_poses(actions, start_pos, start_rot, env_cfg, seed).numpy()
                errors[name]['pos_cm'].append(100 * float(np.linalg.norm(P[-1, :3, 3] - end_pos)))
                errors[name]['rot_deg'].append(float(np.degrees(Rotation.from_matrix(P[-1, :3, :3].T @ end_rot).magnitude())))
    return {name: {k: {'median': float(np.median(v)), 'p90': float(np.percentile(v, 90)), 'n': len(v)}
                   for k, v in e.items()} for name, e in errors.items()}


def main_from_logs(args, base):
    """Fit on the first 70% of logged episodes, score all models on the remaining 30%."""
    episodes = load_logged_episodes(args.from_logs)
    if len(episodes) < 2:
        raise SystemExit(f"need >= 2 logged episodes for a held-out split, found {len(episodes)} at {args.from_logs}")
    split = max(1, int(0.7 * len(episodes)))
    fitted, rms = fit(logged_rows(episodes[:split]))
    models = {
        'constant': {**base, 'ee_prediction': 'constant'},
        'lag_configured': {**base, 'ee_prediction': 'lag'},
        'lag_refit': {**base, 'ee_prediction': 'lag', 'ee_lag_model': fitted},
    }
    evaluation = evaluate_logged(episodes[split:], models)
    report = {'refit_on_logs': fitted, 'refit_step_rms': rms, 'held_out_chunk_end_errors': evaluation,
              'episodes': {'fit': split, 'held_out': len(episodes) - split}, 'args': vars(args)}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"{len(episodes)} logged episodes ({split} fit, {len(episodes) - split} held out); "
          f"refit per-step RMS pos {rms['pos'] * 1000:.2f} mm, rot {np.degrees(rms['rot']):.3f} deg")
    for name, e in evaluation.items():
        print(f"  {name:15s} chunk-end pos {e['pos_cm']['median']:.2f} ({e['pos_cm']['p90']:.2f}) cm, "
              f"rot {e['rot_deg']['median']:.1f} ({e['rot_deg']['p90']:.1f}) deg   [median (p90), n={e['pos_cm']['n']}]")
    print("refit ee_lag_model:", {k: [round(x, 5) for x in v] for k, v in fitted.items()})
    print(f"wrote {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suites", nargs="+", default=["libero_object", "libero_spatial", "libero_goal", "libero_10"])
    parser.add_argument("--fit_steps", type=int, default=300, help="random steps per suite for fitting")
    parser.add_argument("--eval_chunks", type=int, default=20, help="held-out 10-step chunks per suite")
    parser.add_argument("--out", default="outputs/ee_calibration.json")
    parser.add_argument("--from_logs", nargs="+", default=None,
                        help="steps.jsonl paths or globs from runs with main.log_steps=true (e.g. 'outputs/**/steps.jsonl')")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.from_logs:
        main_from_logs(args, OmegaConf.to_container(OmegaConf.load("configs/backend/libero.yaml").libero, resolve=True))
        return

    rng = np.random.default_rng(0)
    adapters = {s: make_adapter(s) for s in args.suites}
    rows = {s: collect(a, rng, args.fit_steps) for s, a in adapters.items()}
    pooled, pooled_rms = fit([r for rs in rows.values() for r in rs])
    per_suite = {s: dict(zip(('params', 'rms'), fit(rs))) for s, rs in rows.items()}

    base = OmegaConf.to_container(OmegaConf.load("configs/backend/libero.yaml").libero, resolve=True)
    models = {
        'constant': {'ee_prediction': 'constant'},
        'lag_configured': {'ee_prediction': 'lag', 'ee_lag_model': base.get('ee_lag_model')},
        'lag_fitted': {'ee_prediction': 'lag', 'ee_lag_model': pooled},
    }
    if models['lag_configured']['ee_lag_model'] is None:
        del models['lag_configured']
    evaluation = {s: evaluate(a, rng, args.eval_chunks, models) for s, a in adapters.items()}

    report = {'pooled_fit': pooled, 'pooled_step_rms': pooled_rms, 'per_suite_fit': per_suite,
              'chunk_end_errors': evaluation, 'args': vars(args)}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nper-step RMS of the pooled fit: pos {pooled_rms['pos'] * 1000:.2f} mm, "
          f"rot {np.degrees(pooled_rms['rot']):.3f} deg")
    print("per-suite fits (gain / lag):")
    for s, r in per_suite.items():
        p = r['params']
        print(f"  {s:16s} pos gain {np.round(p['pos_gain'], 4)} lag {np.round(p['pos_lag'], 3)} | "
              f"rot gain {np.round(p['rot_gain'], 4)} lag {np.round(p['rot_lag'], 3)}")
    print("chunk-end error, median (p90):")
    for s, ev in evaluation.items():
        print("  " + s.ljust(16) + " | ".join(
            f"{m}: {e['pos_cm']['median']:.2f} ({e['pos_cm']['p90']:.2f}) cm, {e['rot_deg']['median']:.1f} "
            f"({e['rot_deg']['p90']:.1f}) deg" for m, e in ev.items()))
    r = lambda v: [round(x, 5) for x in v]
    print("\nYAML for configs/backend/libero.yaml:\n  ee_lag_model:")
    for k in ('pos_gain', 'pos_lag', 'rot_gain', 'rot_lag'):
        print(f"    {k}: {r(pooled[k])}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
