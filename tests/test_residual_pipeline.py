"""End-to-end residual pipeline on one LIBERO task (GPU, slow: ~8 min on an RTX 5090, measured 2026-11-12).

1. Train with a warm-up gate that cannot pass: round 1 collects with exploration noise and a zero residual, the run
   stops cleanly at round 2. The round-1 eval of the zero residual must match the π0.5 baseline episode for episode.
2. Resume with a gate any held-out data passes: the actor turns on and trains; Stage 1 is not collected again.
3. Evaluate the checkpoint twice in mode=eval: identical per-episode tables, baseline outcomes reused.
Each launch is `python -m residual.train` in a subprocess, as scripts/run/run_residual.sh runs it, writing only under
pytest's tmp dir. Needs envs/vls.env sourced (see README).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or "LIBERO_CONFIG_PATH" not in os.environ,
                                reason="needs CUDA and envs/vls.env (LIBERO_CONFIG_PATH)")

VLS = Path(__file__).resolve().parents[1]
TIMEOUT_S = 3600
# libero_spatial_task t1: π0.5 succeeds on ~66% of inits (2026-11-10 baseline), so outcomes and episode lengths vary,
# and its cabinet is a fixture LIBERO re-draws at every reset, so the paired-eval seeding is exercised.
# heldout_frac 0.5 so that a 4-episode round holds out online episodes for the gate/* action metrics.
SMALL = ["logging.wandb.mode=offline", "env.suites=[libero_spatial]", "env.task_ids=[1]", "offline.episodes_per_task=2",
         "offline.updates=20", "online.episodes_per_task=4", "online.updates_per_round=50", "online.heldout_frac=0.5",
         "eval.every_rounds=1", "eval.episodes_per_task=2", "eval.final_episodes_per_task=2",
         "train.buffer_capacity_per_task=2000", "logging.log_every_updates=10", "logging.video_eval_per_task=0"]


def launch(hydra_dir: Path, *overrides: str) -> str:
    proc = subprocess.run([sys.executable, "-m", "residual.train", *SMALL, *overrides, f"hydra.run.dir={hydra_dir}"],
                          cwd=VLS, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=TIMEOUT_S,
                          check=False)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-5000:]
    return out


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_warmup_resume_actor_and_repeatable_eval(tmp_path):
    run_dir = tmp_path / "run"

    launch(run_dir, "online.rounds=1", "actor.warmup_rounds=1")
    state = torch.load(run_dir / "last.pt", weights_only=True)["state"]
    assert state["round"] == 1 and not state["actor_on"]
    online = [e for e in jsonl(run_dir / "episodes.jsonl") if e.get("stage") == 3 and "edit" in e]
    assert online and all(e["edit"] == 0 for e in online)            # residual still zero
    assert all(e["exec_vs_base_trans"] > 0 for e in online)          # exploration noise on
    assert all(e["exec_vs_base_grip"] == 0 for e in online)          # never on the gripper
    rows = jsonl(run_dir / "metrics.jsonl")
    assert any("gate/q_action_diff" in r for r in rows) and any("progress/chunks" in r for r in rows)
    assert not any(k.startswith("update/actor/") for r in rows for k in r)
    warmup_eval = [r for r in rows if "eval/policy/flip_frac" in r]
    assert warmup_eval and warmup_eval[0]["eval/policy/flip_frac"] == 0  # zero residual == π0.5, same seeds
    wandb_id = (run_dir / "wandb_id.txt").read_text()

    launch(tmp_path / "hydra_resume", f"train.resume_dir={run_dir}", "online.rounds=4", "actor.warmup_rounds=1")
    state = torch.load(run_dir / "last.pt", weights_only=True)["state"]
    assert state["round"] == 4 and state["actor_on"]
    assert sum(e.get("stage") == 1 for e in jsonl(run_dir / "episodes.jsonl")) == 2  # Stage 1 not re-collected
    rows = jsonl(run_dir / "metrics.jsonl")
    assert any("update/actor/q_loss" in r for r in rows)
    assert any("eval/policy/paired_se" in r for r in rows) and any("eval/policy_train/paired_se" in r for r in rows)
    assert (run_dir / "best.pt").exists() and (run_dir / "wandb_id.txt").read_text() == wandb_id

    tables = []
    for name in ("eval_a", "eval_b"):
        eval_dir = tmp_path / name
        launch(eval_dir, "mode=eval", f"policy.residual_ckpt={run_dir / 'last.pt'}")
        assert not list((eval_dir / "tables").glob("baseline*"))  # baseline outcomes came with the checkpoint
        assert {"policy_round0", "policy_train_round0"} <= json.loads((eval_dir / "eval_summary.json").read_text()).keys()
        tables.append({p.name: p.read_text() for p in sorted((eval_dir / "tables").glob("policy*"))})
    assert tables[0] and tables[0] == tables[1]
