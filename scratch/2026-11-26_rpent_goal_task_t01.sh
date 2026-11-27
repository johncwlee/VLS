#!/bin/bash
# Runs RPent (HF memory, claude_code planner, opus-5-5 low) on SUITE x TASKS x SEEDS (default libero_goal_task, 0-1 x 0-2 = our init states 0-2), one episode each, with a per-tool-call frame log.
set -u
VLS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RPENT_DIR="$VLS_DIR/../RPent"
SUITE="${SUITE:-libero_goal_task}"
TASKS="${TASKS:-0 1}"
SEEDS="${SEEDS:-0 1 2}"
RUN_NAME="${RUN_NAME:-rpent_goal_task_t01}"
RUN_DIR="$VLS_DIR/outputs/$RUN_NAME/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$RUN_DIR"

if [ "$(cat /sys/devices/system/cpu/offline)" != "2-3" ]; then
    echo "CPUs 2-3 are not offline (faulty core 2); stopping"; exit 1
fi

source "$VLS_DIR/../.venv-rpent/bin/activate"
cd "$RPENT_DIR"
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export LIBERO_TYPE=pro
export PI05_CHECKPOINT_PATH="$RPENT_DIR/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT"
export SAM3_CHECKPOINT_PATH="$RPENT_DIR/checkpoints/sam3/sam3.pt"

COMMON_ARGS=(--robot libero --libero-type pro --suite "$SUITE"
    --planner claude_code --model claude-opus-5-5 --reasoning-effort low
    --memory-profile hf --max-episode-steps 10000 --cuda-device 0)

EPISODES="tasks [$TASKS] x seeds [$SEEDS]" python - "$RUN_DIR/run_info.json" "${COMMON_ARGS[@]}" <<'EOF'
import json, os, subprocess, sys
info = {
    "rpent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "rpent_git_status": subprocess.check_output(["git", "status", "--short"], text=True),
    "vls_commit": subprocess.check_output(["git", "-C", "../VLS", "rev-parse", "HEAD"], text=True).strip(),
    "common_args": sys.argv[2:],
    "episodes": os.environ["EPISODES"] + "; seed k = LIBERO-PRO init state k (checked for goal_task, scratch/2026-11-26_rpent_init_state_match.py)",
    "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip(),
    "memory": "hf profile, version auto -> GPT_5.5_xhigh (no mapping for claude models)",
}
json.dump(info, open(sys.argv[1], "w"), indent=1)
EOF

for task in $TASKS; do
    for seed in $SEEDS; do
        episode_dir="$RUN_DIR/t${task}_s${seed}"
        mkdir -p "$episode_dir"
        RPENT_TOOL_LOG="$episode_dir/tool_log.jsonl" timeout 3h \
            python "$VLS_DIR/scratch/2026-11-26_rpent_tool_log_launcher.py" "${COMMON_ARGS[@]}" \
            --task "$task" --seed "$seed" --output-dir "$episode_dir" > "$episode_dir/stdout.log" 2>&1
        echo "EPISODE t${task}_s${seed} exit $?"
    done
done
echo "RUN_DIR $RUN_DIR"
echo "ALL_DONE"
