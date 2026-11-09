"""Init-state protocol of the LIBERO adapter (`init_state_mode` in configs/backend/libero.yaml).

official: the episode's layout is the task's stored init state (episode i -> state i % 50), whatever the seed.
seeded_random (legacy VLS): reset() re-samples placements after set_init_state(), so the layout follows the seed.
Slow: builds LIBERO envs.
"""

import numpy as np
from omegaconf import OmegaConf

from core.env_adapters import create_adapter


def make_adapter(mode: str, episodes: int):
    cfg = OmegaConf.to_container(OmegaConf.load("configs/backend/libero.yaml").libero, resolve=True)
    cfg.update(suite_name="libero_object", episode_num=episodes, start_episode=0, task_ids_filter=[0],
               init_state_mode=mode)
    return create_adapter("libero", cfg)


def layout(adapter, seed: int) -> tuple[np.ndarray, dict]:
    _, info = adapter.reset(seed=seed)
    env = adapter._env[0]._env
    qpos = [env.sim.data.get_joint_qpos(obj.joints[-1]) for obj in env.env.objects_dict.values() if obj.joints]
    return np.concatenate(qpos), info


def test_official_mode_uses_stored_init_states():
    one = make_adapter("official", episodes=1)  # every episode is episode 0 of the task
    a, info_a = layout(one, seed=1)
    b, info_b = layout(one, seed=2)
    assert info_a["init_state_id"] == info_b["init_state_id"] == 0
    np.testing.assert_allclose(a, b, atol=1e-6)

    two = make_adapter("official", episodes=2)  # episodes 0 and 1 -> init states 0 and 1
    c, info_c = layout(two, seed=1)
    d, info_d = layout(two, seed=1)
    assert (info_c["init_state_id"], info_d["init_state_id"]) == (0, 1)
    np.testing.assert_allclose(c, a, atol=1e-6)
    assert np.abs(c - d).max() > 1e-3


def test_seeded_random_mode_follows_seed():
    legacy = make_adapter("seeded_random", episodes=1)
    a, info = layout(legacy, seed=1)
    b, _ = layout(legacy, seed=2)
    assert info["init_state_id"] is None
    assert np.abs(a - b).max() > 1e-3
