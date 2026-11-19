# Runs main.py with LIBERO's fixed init states (reset, then set_init_state k = episode within task) and a per-episode torch/numpy seed (test only).
import os
import random
import sys

import gymnasium as gym
import hydra
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import core.env_adapters.libero_adapter as libero_adapter
import main as main_module

EPISODE_SEED_BASE = 10_000
current = {'init_index': None}
original_adapter_reset = libero_adapter.LiberoAdapter.reset


def env_reset_fixed_init(self, seed=None, **kwargs):
    """LiberoEnv.reset with LIBERO's evaluation order: reset() re-samples the layout, set_init_state overwrites it."""
    gym.Env.reset(self, seed=seed)
    self._env.seed(seed)
    self._env.reset()
    assert self._init_states is not None, 'no init states loaded for this task'
    init_index = current['init_index']
    assert init_index is not None and init_index < len(self._init_states), f'init index {init_index}'
    raw_obs = self._env.set_init_state(self._init_states[init_index])
    for _ in range(self.num_steps_wait):  # settle, as the original reset does
        raw_obs, _, _, _ = self._env.step(libero_adapter.get_libero_dummy_action())
    observation = self._format_raw_obs(raw_obs)
    return observation, {"is_success": False}


def adapter_reset_fixed_init(self, **kwargs):
    next_episode_idx = self.current_episode_idx + 1  # the adapter's reset increments to this
    current['init_index'] = next_episode_idx % self.episodes_per_task  # 0..4: k-th episode of its task
    # Same seed for the same episode in flag-on and flag-off runs, so policy noise starts the same
    episode_seed = EPISODE_SEED_BASE + next_episode_idx
    torch.manual_seed(episode_seed)
    torch.cuda.manual_seed(episode_seed)
    np.random.seed(episode_seed)
    random.seed(episode_seed)
    print(f'[fixed_init] episode_idx {next_episode_idx} init_state {current["init_index"]} seed {episode_seed}', flush=True)
    return original_adapter_reset(self, **kwargs)


libero_adapter.LiberoEnv.reset = env_reset_fixed_init
libero_adapter.LiberoAdapter.reset = adapter_reset_fixed_init

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs'))


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name='config')  # main.py's relative path fails on import
def run(cfg):
    main_module.main.__wrapped__(cfg)


if __name__ == '__main__':
    run()
