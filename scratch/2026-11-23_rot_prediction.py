# Checks how well the constant and lag action->rotation models predict the EE orientation 10 steps ahead on logged task 0 rollouts.
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

RUN = sys.argv[1]  # an outputs/<exp>/<timestamp> folder with episode_*/steps.jsonl
HORIZON = 10  # steps per chunk on LIBERO (configs/policy.yaml action_chunk_horizon)
ROT_SCALE = 0.1  # rad per action unit, configs/backend/libero.yaml action_scale_rot
LAG_GAIN = np.array([0.04673, 0.04018, 0.04987])  # configs/backend/libero.yaml ee_lag_model
LAG_LAG = np.array([0.6021, 0.5899, 0.54252])


def rot_from_wxyz(q):
    return R.from_quat([q[1], q[2], q[3], q[0]])


def angle_deg(r_a, r_b):
    return np.degrees((r_a.inv() * r_b).magnitude())


rows = []
for ep in range(1, 6):
    steps = [json.loads(line) for line in open(f"{RUN}/episode_{ep}/steps.jsonl")]
    steps = [s for s in steps if s["ee_quat_wxyz_before"] is not None]
    for start in range(1, len(steps) - HORIZON, HORIZON):
        window = steps[start:start + HORIZON]
        if [s["step"] for s in window] != list(range(window[0]["step"], window[0]["step"] + HORIZON)):
            continue  # logged steps are not consecutive here
        r_start = rot_from_wxyz(window[0]["ee_quat_wxyz_before"])
        r_true = rot_from_wxyz(window[-1]["ee_quat_wxyz_after"])
        prev = steps[start - 1]
        # last executed step's measured world-frame rotation motion, seeds the lag model
        motion = (rot_from_wxyz(prev["ee_quat_wxyz_after"]) * rot_from_wxyz(prev["ee_quat_wxyz_before"]).inv()).as_rotvec()
        r_const, r_lag = r_start, r_start
        for s in window:
            a_rot = np.array(s["action"][3:6])  # normalized axis-angle delta, world frame
            r_const = R.from_rotvec(ROT_SCALE * a_rot) * r_const
            motion = LAG_GAIN * a_rot + LAG_LAG * motion
            r_lag = R.from_rotvec(motion) * r_lag
        rows.append((angle_deg(r_start, r_true), angle_deg(r_const, r_true), angle_deg(r_lag, r_true)))

rows = np.array(rows)  # (N, 3) degrees
print(f"{len(rows)} windows of {HORIZON} steps, error vs measured orientation at window end (deg)")
for name, col in (("no-rotation baseline", 0), ("constant model", 1), ("lag model", 2)):
    print(f"  {name:22s} median {np.median(rows[:, col]):5.1f}  p90 {np.percentile(rows[:, col], 90):5.1f}")
