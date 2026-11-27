# Starts RPent's pi05_vla_server after seeding torch with $NOISE_SEED, so pi0.5's sampling noise differs per run (RPent never seeds it).
import os
import runpy
import sys

import torch

noise_seed = int(os.environ['NOISE_SEED'])
torch.manual_seed(noise_seed)
torch.cuda.manual_seed_all(noise_seed)
server_path = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(server_path, run_name='__main__')
