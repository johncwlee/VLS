# Runs main.py with closed-loop prompts that also carry a simulator ground-truth contact/placement line (test only).
import os
import sys

import hydra
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import closed_loop.run as closed_loop_run
import main as main_module

runner = {}  # the live main.Main instance, set on its first closed-loop step
original_start_closed_loop = main_module.Main._start_closed_loop
original_build_prompt = closed_loop_run.build_prompt
injected_log = []


def ground_truth_text(env):
    """Same wording as scratch/2026-11-19_closed_loop_gt_inject.py; env is the LIBERO problem env."""
    def body_pos_m(name):
        return env.sim.data.body_xpos[env.obj_body_id[name]].copy()  # (3,) world, m

    if 'wine_bottle_1' in env.obj_of_interest and 'plate_1' in env.obj_of_interest:
        bottle, plate = body_pos_m('wine_bottle_1'), body_pos_m('plate_1')
        dz_cm = (bottle[2] - plate[2]) * 100
        dxy_cm = np.linalg.norm(bottle[:2] - plate[:2]) * 100
        touching = env.object_states_dict['wine_bottle_1'].check_contact(env.object_states_dict['plate_1'])
        return (f'Measured from the simulator (ground truth): wine bottle touching the plate: {"yes" if touching else "no"}. '
                f'The bottle\'s base is {dz_cm:.1f} cm above the plate\'s base (when the bottle stood on the table '
                f'it was -0.4 cm, so a bottle resting on the plate is about 0 cm); horizontal distance between their '
                f'centres: {dxy_cm:.1f} cm.')
    if 'cream_cheese_1' in env.obj_of_interest and 'wine_rack_1_top_region' in env.obj_of_interest:
        site_name = 'wine_rack_1_top_region'
        site_pos_m = env.sim.data.get_site_xpos(site_name)
        site_mat = env.sim.data.get_site_xmat(site_name)
        half_size_m = np.array(env.object_sites_dict[site_name].size)
        delta_m = site_mat @ (body_pos_m('cream_cheese_1') - site_pos_m)  # as LIBERO site_object.py under()
        inside = abs(delta_m[0]) < half_size_m[0] and abs(delta_m[1]) < half_size_m[1]
        along_past_edge_cm = (abs(delta_m[0]) - half_size_m[0]) * 100
        return (f'Measured from the simulator (ground truth): cream cheese inside the rack\'s top region (the area that '
                f'counts as "on the rack"): {"yes" if inside else "no"}. Along the rack\'s long side it is '
                f'{along_past_edge_cm:.1f} cm past the region\'s edge; across the rack it is {abs(delta_m[1]) * 100:.1f} cm '
                f'from the centre line (region half-width {half_size_m[1] * 100:.1f} cm).')
    raise ValueError(f'no ground-truth text for objects {env.obj_of_interest}')


def build_prompt_with_gt(*args, **kwargs):
    prompt = original_build_prompt(*args, **kwargs)
    env = runner['main'].adapter._get_current_robosuite_env().env
    truth = ground_truth_text(env)
    gripper_line = next(line for line in prompt.split('\n') if line.startswith('Gripper: '))
    assert prompt.count(gripper_line + '\n') == 1, 'gripper line not unique'
    injected_log.append(truth)
    print(f'[gt_inject] {truth}', flush=True)
    return prompt.replace(gripper_line + '\n', gripper_line + '\n' + truth + '\n')


def start_closed_loop_keep_self(self, *args, **kwargs):
    runner['main'] = self  # the first call (start_episode) builds a prompt before any _closed_loop_step
    return original_start_closed_loop(self, *args, **kwargs)


main_module.Main._start_closed_loop = start_closed_loop_keep_self
closed_loop_run.build_prompt = build_prompt_with_gt

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs'))


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name='config')  # main.py's relative path fails on import
def run(cfg):
    main_module.main.__wrapped__(cfg)


if __name__ == '__main__':
    run()
