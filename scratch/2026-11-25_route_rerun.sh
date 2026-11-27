#!/bin/bash
# Reruns the 11-25 route-prompt setting on tasks 0 1 3 4 9 x init states 0-1, time limit 8 chunks, one GPU, tasks in sequence.
set -u
cd "$(dirname "$0")/.."
source ../.venv-vls/bin/activate
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
mkdir -p rollouts_tmp

for task in 0 1 3 4 9; do
    name="route5b_t${task}"
    python scratch/2026-11-19_fixed_init_launcher.py \
        backend=libero backend.libero.suite_name=libero_goal_task main.episode_num=2 \
        "backend.libero.task_ids_filter=[${task}]" main.use_fkd=false \
        perception.keypoint_detector.feature_extractor=dinov2_vits14 main.use_guidance=true \
        main.sample_batch_size=20 main.log_steps=true main.closed_loop_vls=true \
        main.closed_loop_ee_input=pose main.closed_loop_time_limit_chunks=8 \
        experiment.name="${name}" > "rollouts_tmp/${name}.log" 2>&1
    echo "RUN_EXIT task ${task} code $?"
    run_dir=$(ls -d outputs/${name}/*/ | tail -1)
    echo "RUN_DIR task ${task} ${run_dir}"
    cat "${run_dir}trials.jsonl" | sed "s/^/RESULT task ${task} /"
    for video in "${run_dir}"episode_*/episode_*_agentview.mp4; do
        episode=$(basename "$(dirname "${video}")" | sed 's/episode_//')
        init_state=$((episode - 1))
        outcome=$(basename "${video}" | sed 's/episode_[0-9]*_\(.*\)_agentview.mp4/\1/')
        cp "${video}" "rollouts_tmp/t${task}_init${init_state}_${outcome}.mp4"
        echo "VIDEO rollouts_tmp/t${task}_init${init_state}_${outcome}.mp4"
    done
done
echo "ALL_DONE"
