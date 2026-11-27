# Handle-start launcher (gripper 5 cm from the task's drawer handle), but pi0.5 gets a short prompt instead of the task text.
import importlib.util
import os

SCRATCH = os.path.dirname(os.path.abspath(__file__))
# RPent's pi0_doubled prompt for libero_goal_task task 0 (all 3 RPent episodes, outputs/rpent_goal_task_t01/2026-11-26_18-42-23)
SHORT_PROMPT = 'pull the lowest gray drawer handle outward'

spec = importlib.util.spec_from_file_location('handle_start_launcher', os.path.join(SCRATCH, '2026-11-25_handle_start_launcher.py'))
handle_start = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handle_start)  # applies the fixed-init and drive-to-handle patches
libero_adapter = handle_start.libero_adapter

handle_reset = libero_adapter.LiberoEnv.reset


def reset_with_short_prompt(self, seed=None, **kwargs):
    # The handle-start reset finds the drawer from the original task text, so restore it before each reset
    if not hasattr(self, 'original_task_description'):
        self.original_task_description = self.task_description
    self.task_description = self.original_task_description
    result = handle_reset(self, seed=seed, **kwargs)
    self.task_description = SHORT_PROMPT  # read by the policy obs (libero_adapter.py:901-902)
    print(f'[short_prompt] {self.original_task_description!r} -> {SHORT_PROMPT!r}', flush=True)
    return result


libero_adapter.LiberoEnv.reset = reset_with_short_prompt

if __name__ == '__main__':
    handle_start.launcher.run()
