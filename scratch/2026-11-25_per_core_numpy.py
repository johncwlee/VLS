# Repeats a fixed single-thread numpy matmul+tanh on each online core (taskset) and checks the output hash matches CPU 0's.
import os
import subprocess
import sys

REPEATS_PER_CORE = 20
WORKER = """
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import hashlib
import numpy as np
rng = np.random.default_rng(0)
a = rng.standard_normal((512, 512))
for _ in range(200):
    a = np.tanh(a @ a / 22.6)  # 22.6 ~ sqrt(512), keeps values out of tanh saturation
print(hashlib.sha256(a.tobytes()).hexdigest())
"""

env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
online = sorted(os.sched_getaffinity(0))


def run_on_core(core):
    result = subprocess.run(['taskset', '-c', str(core), sys.executable, '-c', WORKER],
                            capture_output=True, text=True, env=env)
    if result.returncode != 0:
        return f'CRASH code {result.returncode}: {result.stderr.strip().splitlines()[-1:] }'
    return result.stdout.strip()


reference = run_on_core(online[0])
assert len(reference) == 64, f'reference run failed: {reference}'
print(f'online cores: {online}')
print(f'reference hash (core {online[0]}): {reference[:16]}')
bad_cores = []
for core in online:
    outputs = [run_on_core(core) for _ in range(REPEATS_PER_CORE)]
    n_match = sum(output == reference for output in outputs)
    wrong = [output[:40] for output in outputs if output != reference]
    status = 'ok' if n_match == REPEATS_PER_CORE else 'BAD'
    print(f'core {core:2d}: {n_match}/{REPEATS_PER_CORE} match  {status}  {wrong[:3]}', flush=True)
    if status == 'BAD':
        bad_cores.append(core)
print(f'bad cores: {bad_cores}')
