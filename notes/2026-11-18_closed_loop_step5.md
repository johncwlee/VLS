# Closed-loop VLS, step 5 (2026-11-18): first sim runs and two open problems

Code: `closed_loop/run.py` + hooks in `main.py` behind `main.closed_loop_vls` (default false). Uncommitted at time of writing.
Tests: `tests/test_closed_loop_run.py` (fake VLM), printer `scratch/2026-11-18_closed_loop_triggers.py`: 35/35 chunks match;
`python -m pytest tests -q`: 56 passed, 3 skipped.

## Runs (smoke tests, 1 episode each: not rates)
Overrides for both: `backend=libero backend.libero.suite_name=libero_goal_task main.episode_num=10 main.use_fkd=false
perception.keypoint_detector.feature_extractor=dinov2_vits14 main.use_guidance=true main.sample_batch_size=20`,
claude-opus-5-5, effort low. LIBERO needs episode_num divisible by 10, so `main.start_episode` picks the episodes.

| run | flag | episodes | GPU | result |
|---|---|---|---|---|
| `outputs/closed_loop_step5_flag_off/2026-11-18_20-51-42` | off | 10 (task 9) | 0 | fail; original path (vlm_agent/, 4 stage-recognizer checks), no new trial fields: behaves as before |
| `outputs/closed_loop_step5_flag_on/2026-11-18_20-51-42` | on | 9, 10 (tasks 8, 9) | 1 | both fail, no crash, no VLM errors; 10 and 11 calls; stop_reason null |

Earlier empty folders (`*_flag_off|on/2026-11-18_20-51-03`) crashed at startup (episode_num=1/2 not divisible by 10). Kept.
The tracebacks at the end of each log are robosuite EGL cleanup at exit ("Exception ignored in __del__"), not run errors.

### Episode 9 (task 8, "Put the wine bottle on the plate"): the known stuck case
| call | step | trigger | verdict | evidence (short) |
|---|---|---|---|---|
| 1 | 0 | initial query | plan | bottle upright, gripper empty |
| 2 | 20 | reward > 80% | continue | hovering above neck |
| 3 | 30 | gripper closed | advance -> 2 | closed on the neck |
| 4 | 60 | reward > 80% | continue | holding bottle over plate |
| 5-6 | 100, 140 | time limit | continue | on plate ~4 cm from center, still held |
| 7 | 180 | time limit | replan | same, still held (replanned on its own; forced text never sent) |
| 8 | 220 | time limit | replan | "scene looks the same as 4 chunks ago" |
| 9 | 260 | reward < 60% | continue | ~1.5 cm from plate center |
| 10 | 270 | gripper opened | **done** | "gripper has opened" -- FALSE, see problem 2 |
Time-limit checks caught the stuck hold (as offline case D). Ended at 300 steps, fail.

### Episode 10 (task 9, "Put the cream cheese on the rack")
Pictures per call: `outputs/closed_loop_step5_review/episode_10/` (`all_calls.png`), made by
`scratch/2026-11-18_closed_loop_call_summary.py`. ("≈" renders as "???" in the pictures; files are fine.)
- Calls 1-5: plan, grasp (call 3 advance), carry; call 5 gripper opened -> continue (released at front edge of rack).
- Call 6 (step 150, time limit) -> replan: "dropped short of the rack", cheese on cabinet top ~12 cm in front of it.
- Calls 7-9: re-grasp attempt; call 9 (step 240) gripper closed -> advance.
- Call 10 (step 250) gripper opened -> replan: re-grasp missed, cheese still on cabinet top.
- Call 11 (step 290) continue; timeout at 300, fail. Advance between the two replans reset the give-up count (as designed).
- Gripper triggers here were real: commands stay ~-1.0 for 10+ steps (108-111, 240-251).

## Problem 1: re-grasp in episode 10 misses in depth (OPEN, parked until decided)
- From the camera, the fingers come down in front of the cheese; they should move back to reach it (steps 170-239).
- The guidance targets the right object: call 6 `stage1_guidance` pulls the planned path to keypoint 2
  (-0.036, -0.214, 1.146 m), xy weighted 3x over z.
- But the reward said it was close: norm_r 0.82-0.88 at steps 170-200, so steering strength (Sig_Str) fell to 4-16%.
- Hypothesis (unverified): the action -> end-effector path model (`main.ee_prediction`, default) mispredicts where the
  gripper really goes; John's config note says `lag` "measures real centimetres". Test: rerun with `main.log_steps=true`
  and compare measured EE position with keypoint 2. Not run. Not a closed-loop bug, but limits every recovery.

## Problem 2: gripper state read from one command (OPEN, needs a decision)
Closed loop uses gripper open/closed for (a) the gripper trigger and (b) the `Gripper:` line in the prompt (the VLM
can't see it from agentview). `main.py:_closed_loop_step` reads one action: the last of each chunk.

```
chunk = 10 actions (steps 260..269); last value of each action = gripper command (-1 open, +1 close)
                                   read only this one --v
gripper cmd:  +0.95 +0.98 +0.99 ... +1.00 +1.00   -1.01   | next chunk: +0.99 +1.00 ...
```
Episode 9, video status box + frames (`ep9` steps 266-285): command closed at 266-268, a single open command (-1.01)
at 269, closed again from 270. Fingers stay on the bottle neck in every frame. Our hook read "open" -> trigger
"gripper opened" -> prompt said `Gripper: open` -> VLM answered `done` -> guidance off, robot kept holding -> timeout.
The original `_update_stage` also reads a single action (index 0 of the previous chunk, main.py:765) -- same weakness, not tested.
Not checked: whether one -1 step moves the fingers at all (the overlay shows commands, not finger positions).

Options:
1. Measured finger opening from robot state (gripper joint positions in the policy observation). Reports what the
   fingers did; a blip that doesn't open them can't fire. (Recommended)
2. Majority of the chunk's 10 commands. Simpler, still a command.

## Next
1. Pick the gripper reading (problem 2), fix, rerun episodes 9-10.
2. Decide whether to investigate problem 1 now (log_steps rerun) or park it.
3. Commit step 5 (closed_loop/run.py, main.py hooks, config keys, tests, scratch scripts), update PLAN_closed_loop.md.
