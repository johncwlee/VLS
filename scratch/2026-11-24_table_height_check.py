# Checks closed_loop/table_height.py on agentview point clouds of libero_goal_task init states against the sim table top.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import core.env_adapters.libero_adapter as libero_adapter
from closed_loop.table_height import estimate_table_height_m

TOL_M = 0.005

for task_id in (0, 5):
    env = libero_adapter.create_libero_envs('libero_goal_task', task_ids_filter=[task_id])[0]
    adapter = libero_adapter.LiberoAdapter.__new__(libero_adapter.LiberoAdapter)  # only what get_keypoint_detection_inputs reads
    adapter._env = [env]
    adapter.current_task_idx = 0
    adapter._env_config = {}
    adapter.vlm_camera = 'agentview'
    for init_state in (0, 3):
        env._env.reset()
        env._env.set_init_state(env._init_states[init_state])
        sim = env._env.env.sim  # reset() builds a new sim
        table_gid = sim.model.geom_name2id('table_collision')
        _, _, points, _, _ = adapter.get_keypoint_detection_inputs()
        estimate = estimate_table_height_m(points)
        truth = sim.data.geom_xpos[table_gid][2] + sim.model.geom_size[table_gid][2]  # check only, never prompted
        print(f'task {task_id} init {init_state}: points {points.shape} -> estimate {estimate:.4f} m, sim table top {truth:.4f} m, '
              f'err {1000 * (estimate - truth):+.1f} mm -> {"match" if abs(estimate - truth) < TOL_M else "MISMATCH"}')
