"""Integration tests against a running GraspGenX server (./run_graspgenx_server.sh in the outer repo).

Synthetic scene: an 4x4x8 cm box standing on a table plane at z=0 (world frame, metres).
Skipped when the server is not reachable.
"""

import numpy as np
import pytest

from core.grasp_guidance import GraspGenXClient
from utils.grasp_costs import rotation_weight

ENDPOINT = "tcp://127.0.0.1:5556"
BOX_CENTER = np.array([0.5, 0.0, 0.04])
BOX_HALF = np.array([0.02, 0.02, 0.04])


@pytest.fixture(scope="module")
def client():
    try:
        GraspGenXClient(ENDPOINT, timeout_ms=3000).request({"action": "health"})
    except TimeoutError:
        pytest.skip(f"GraspGenX server not running at {ENDPOINT}")
    return GraspGenXClient(ENDPOINT, timeout_ms=120000)


def box_and_table(rng):
    faces = []
    for axis in range(3):
        for sign in (-1, 1):
            if axis == 2 and sign == -1:
                continue  # bottom face rests on the table
            pts = rng.uniform(-1, 1, (800, 3)) * BOX_HALF
            pts[:, axis] = sign * BOX_HALF[axis]
            faces.append(pts + BOX_CENTER)
    table = np.c_[rng.uniform(0.2, 0.8, (6000, 2)) - [0, 0.3], np.zeros(6000)]
    table = table[np.any(np.abs(table[:, :2] - BOX_CENTER[:2]) > BOX_HALF[:2], axis=1)]
    return np.concatenate(faces).astype(np.float32), table.astype(np.float32)


def query(client, action, obj, scene, **kw):
    r = client.request({"action": action, "gripper_name": "franka_panda", "object_pc": obj, "scene_pc": scene,
                        "num_grasps": 100, "collision_threshold": 0.01, **kw})
    return np.asarray(r["grasps"]).reshape(-1, 4, 4), np.asarray(r["confidences"]), r["timing"]["infer_ms"]


def test_filtered_grasps_avoid_table(client):
    obj, table = box_and_table(np.random.default_rng(0))
    grasps, conf, ms = query(client, "infer_object_filtered", obj, table, planner="graspmoe")
    print(f"graspmoe: {len(grasps)} collision-free grasps in {ms:.0f} ms")
    assert len(grasps) > 0
    # Check the grasps steering would use (top_k by score). Rare near-zero-confidence diffusion samples can
    # sit entirely under the table, where no scene point exists for the collision filter to hit.
    grasps = grasps[np.argsort(-conf)[:16]]
    # Fingertip centre (0.1034 m along approach for franka_panda) must be on the box, not under the table
    tips = grasps[:, :3, 3] + 0.1034 * grasps[:, :3, 2]
    assert np.all(tips[:, 2] > 0.0)
    assert np.median(np.linalg.norm(tips - BOX_CENTER, axis=1)) < 0.08
    assert np.all(grasps[:, 2, 2] < 0.5), "no grasp should approach from below the table"


def test_score_poses_prefers_real_grasps(client):
    obj, table = box_and_table(np.random.default_rng(1))
    grasps, conf, _ = query(client, "infer_object_filtered", obj, table, planner="diffusion")
    best = grasps[np.argsort(-conf)[:10]]
    shifted = best.copy()
    shifted[:, :3, 3] += [0.0, 0.0, 0.25]  # same orientation, 25 cm above the object: grasps nothing
    r = client.request({"action": "score_poses", "gripper_name": "franka_panda", "object_pc": obj,
                        "poses": np.concatenate([best, shifted]).astype(np.float32)})
    scores = np.asarray(r["confidences"])
    assert scores[:10].mean() > scores[10:].mean() + 0.2


def test_guided_sampling_moves_toward_targets(client):
    obj, table = box_and_table(np.random.default_rng(2))
    unguided, _, _ = query(client, "infer_object_filtered", obj, np.zeros((0, 3), np.float32), planner="diffusion")
    target = unguided[:1]  # pull every sample toward one particular grasp
    guided, conf, ms = query(client, "infer_guided", obj, np.zeros((0, 3), np.float32),
                             target_poses=target.astype(np.float32), guide_scale=10.0, sigma_trans=0.015,
                             rot_weight=rotation_weight(15))
    print(f"infer_guided: {len(guided)} grasps in {ms:.0f} ms, median confidence {np.median(conf):.2f}")

    def median_dist(g):
        return np.median(np.linalg.norm(g[:, :3, 3] - target[0, :3, 3], axis=1))

    assert median_dist(guided) < 0.5 * median_dist(unguided)
    assert np.median(conf) > 0.3
