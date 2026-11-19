# Checks whether LIBERO starting layouts depend on the reset seed when set_init_state runs before vs after reset.
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'libero_pro'))  # as libero_adapter.py
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

BDDL = os.path.join(get_libero_path("bddl_files"), "libero_goal_task", "put_the_wine_bottle_on_the_rack.bddl")
INIT_FILE = os.path.join(get_libero_path("init_states"), "libero_goal", "put_the_wine_bottle_on_the_rack.pruned_init")

env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=128, camera_widths=128)
init_states = torch.load(INIT_FILE, weights_only=False)
print(f"init states in file: {len(init_states)}")
problem_env = env.env
names = ["cream_cheese_1", "akita_black_bowl_1", "plate_1", "wine_bottle_1"]


def positions_cm():
    return np.array([problem_env.sim.data.body_xpos[problem_env.obj_body_id[n]] for n in names]) * 100  # (4, 3) cm


def layout(order, seed, init_index):
    env.seed(seed)
    if order == "set_then_reset":  # what libero_adapter.py LiberoEnv.reset does
        env.set_init_state(init_states[init_index])
        env.reset()
    else:  # LIBERO's own evaluation order
        env.reset()
        env.set_init_state(init_states[init_index])
    return positions_cm()


for order in ("set_then_reset", "reset_then_set"):
    a = layout(order, seed=1, init_index=0)
    b = layout(order, seed=2, init_index=0)
    c = layout(order, seed=1, init_index=0)
    d = layout(order, seed=1, init_index=1)
    print(f"\n{order}: cream cheese start (cm) seed1 {np.round(a[0], 1)}  seed2 {np.round(b[0], 1)}")
    print(f"  max object shift, same init state, seed 1 vs 2:     {np.abs(a - b).max():.2f} cm  (expected 0 if seed doesn't matter)")
    print(f"  max object shift, same init state, seed 1 vs 1:     {np.abs(a - c).max():.2f} cm  (expected 0)")
    print(f"  max object shift, init state 0 vs 1, same seed:     {np.abs(a - d).max():.2f} cm  (expected > 0 if init state is used)")
env.close()
