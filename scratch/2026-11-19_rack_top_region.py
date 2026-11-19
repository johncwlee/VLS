# Checks the logged ep 10 cream cheese positions against LIBERO's wine_rack_1_top_region test (site_object.py under()).
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "libero_pro"))  # as libero_adapter.py
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

BDDL = os.path.join(get_libero_path("bddl_files"), "libero_goal_task", "put_the_wine_bottle_on_the_rack.bddl")
RUNS = {
    "width6 flag on (VLM: replan, 'on cabinet top')": "outputs/closed_loop_step5_width6/2026-11-18_23-23-10/episode_10",
    "objlog flag on (VLM: done, 'on the rack')": "outputs/closed_loop_step5_objlog_on/2026-11-18_23-49-17/episode_10",
    "objlog flag off": "outputs/closed_loop_step5_objlog_off/2026-11-18_23-44-34/episode_10",
}

env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=128, camera_widths=128)
env.reset()
problem_env = env.env
site_name = "wine_rack_1_top_region"
site = problem_env.object_sites_dict[site_name]
site_pos_m = problem_env.sim.data.get_site_xpos(site_name).copy()        # (3,) world, m
site_mat = problem_env.sim.data.get_site_xmat(site_name).copy()          # (3, 3)
half_size_m = np.array(site.size)                                        # (3,) as used by under()
OTHER_HEIGHT_M = 0.10  # under() default
print(f"site {site_name}: pos {np.round(site_pos_m, 3)}  size {np.round(half_size_m, 3)}")
print(f"site xmat:\n{np.round(site_mat, 3)}")
z_lo, z_hi = half_size_m[2] - 0.005, half_size_m[2] + OTHER_HEIGHT_M
print(f"under() passes when |dx| < {half_size_m[0]:.3f}, |dy| < {half_size_m[1]:.3f}, {z_lo:.3f} < dz < {z_hi:.3f} "
      f"(d = xmat @ (cheese - site), m); plus rack-cheese contact")

# Cross-check: the same formula as under(), against LIBERO's own function at the site position
for name, run_dir in RUNS.items():
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "steps.jsonl"))]
    has_obj = "obj_pos_m" in rows[0]
    print(f"\n{name}: {run_dir}")
    last = rows[-1]
    if has_obj:
        cheese_m = np.array(last["obj_pos_m"]["cream_cheese_1"])
        source = "logged body origin"
    else:
        # width6 run predates obj_pos_m: use keypoint 2 from the last closed-loop prompt (tracked on the cheese body)
        prompts = sorted(glob.glob(os.path.join(run_dir, "closed_loop", "call_*", "prompt.txt")))
        text = open(prompts[-1]).read()
        block = text.split("Keypoint 3D positions [X, Y, Z] (current positions):")[1]
        numbers = [float(x) for x in block.replace("[", " ").replace("]", " ").replace('"', " ").split()[:9]]
        cheese_m = np.array(numbers[6:9])
        source = f"keypoint 2 from {os.path.basename(os.path.dirname(prompts[-1]))} (not the body origin)"
    delta_m = site_mat @ (cheese_m - site_pos_m)
    libero_says = site.under(site_pos_m, site_mat, cheese_m)
    print(f"  step {last['step']} cheese {np.round(cheese_m, 3)} ({source})")
    print(f"  d = {np.round(delta_m * 100, 1)} cm   under(): {libero_says}"
          f"   x ok {abs(delta_m[0]) < half_size_m[0]}, y ok {abs(delta_m[1]) < half_size_m[1]}, z ok {z_lo < delta_m[2] < z_hi}")
    if has_obj:
        print(f"  logged success at any step: {any(r['success'] for r in rows)}")
env.close()
