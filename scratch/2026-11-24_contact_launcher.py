# Fixed-init launcher plus per-step gripper contacts and cabinet drawer joints in steps.jsonl (runtime patch, main.py unchanged).
import importlib.util
import json
import os

SCRATCH = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('fixed_init_launcher', os.path.join(SCRATCH, '2026-11-19_fixed_init_launcher.py'))
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)  # applies the fixed-init reset patches
main_module = launcher.main_module

original_log_step = main_module.Main._log_step
GRIPPER_BODY_KEYS = ('gripper', 'finger', 'hand')


def contacts_and_drawers(sim):
    model = sim.model
    pairs = set()
    for k in range(sim.data.ncon):
        c = sim.data.contact[k]
        names = []
        for gid in (c.geom1, c.geom2):
            body = model.body_id2name(model.geom_bodyid[gid])
            names.append((body, f'{body}/{model.geom_id2name(gid) or gid}'))
        a_grip = any(key in names[0][0] for key in GRIPPER_BODY_KEYS)
        b_grip = any(key in names[1][0] for key in GRIPPER_BODY_KEYS)
        if a_grip and not b_grip and not names[1][0].startswith('robot0'):
            pairs.add((names[0][1], names[1][1]))
        elif b_grip and not a_grip and not names[0][0].startswith('robot0'):
            pairs.add((names[1][1], names[0][1]))
    joints = [model.joint_id2name(j) for j in range(model.njnt) if 'cabinet' in (model.joint_id2name(j) or '')]
    drawer_qpos_m = {j: float(sim.data.qpos[model.get_joint_qpos_addr(j)]) for j in joints}
    return sorted(pairs), drawer_qpos_m


def log_step_with_contacts(self, episode_dir, global_step, chunk_index, action, ee_before):
    original_log_step(self, episode_dir, global_step, chunk_index, action, ee_before)
    sim = self.adapter._get_current_robosuite_env().env.sim
    contacts, drawer_qpos_m = contacts_and_drawers(sim)
    with open(os.path.join(episode_dir, 'contacts.jsonl'), 'a') as f:
        f.write(json.dumps({'step': global_step, 'contacts': contacts, 'drawer_qpos_m': drawer_qpos_m}) + '\n')


main_module.Main._log_step = log_step_with_contacts

if __name__ == '__main__':
    launcher.run()
