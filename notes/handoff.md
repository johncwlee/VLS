# Handoff (2026-11-26, end of day)

Where to run: the lab desktop (main copy; CPUs 2-3 offline, check `/sys/devices/system/cpu/offline` says 2-3).
Turing only for long jobs (see CLAUDE.md). Branch `closed_loop` at 288cfa7; today's scratch files are uncommitted.

## Goal for next session (user)
Task 0 ("open the bottom drawer") is parked: it may be too hard. Next: go through the main failure cases on OTHER
tasks and try to improve them, and look across all LIBERO-PRO suites (spatial/object/goal/10 x task/swap/lan).
Start point: per-task results of the 11-19 flag on/off run (24/50 vs 20/50, LOG.md 2026-11-19) and the 11-25 route
run (0/11, LOG.md 2026-11-25); nobody has looked at why each episode failed yet.

## What we learned (details: LOG.md 2026-11-26)
- RPent (agent planner + scripted moves + pi0.5) solves goal task 0 by replaying a stored seed-0 recipe: one
  move_pose to a fixed pose ~8 cm above the bottom handle, angled down into the cabinet face, then pi0.5 with
  "pull the lowest gray drawer handle outward". The planner adds little on this task.
- Same start pose: our VLA 1/3, RLinf 1/3. From our handle start: both 0/3. From RPent's own move_pose: RLinf 13/18.
  So the start pose matters more than the checkpoint here (n small).
- Neither VLA grasps the handle; both hook and pull with open fingers.
- VLS's reward targets a different pose (handle height, horizontal approach). Untested idea: steer to a pose the VLA
  can finish from, like RPent's start.
- The orientation-in-reward and waypoint/obstacle prompt work (11-23 to 11-25) is built but not shown to help.

## How to run RPent things (if needed again)
- Venv `~/VLS_check/.venv-rpent`, repo `~/VLS_check/RPent` (a3340f4), checkpoints in `RPent/checkpoints/`.
- Full RPent episodes: `SUITE=... TASKS="0 1" SEEDS="0 1 2" RUN_NAME=... bash scratch/2026-11-26_rpent_goal_task_t01.sh`,
  then `python scratch/2026-11-26_rpent_annotate_video.py EPISODE_DIR` for the annotated video.
- Planner-free replay with a pi0.5 noise seed: `scratch/2026-11-26_rpent_replay_t0.py` (+ `2026-11-26_seeded_vla_server.py`).
- RPent-env videos are flipped 180 deg vs ours.
