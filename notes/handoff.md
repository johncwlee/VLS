# Handoff (2026-11-23, end of day; updated after the commit + desktop session)

Where to run: NOT the lab desktop for now. Its CPU core 2 is faulty (i9-13900KF; torch import 0/5 on CPU 2, 15/15 on
other cores; numpy matmul crashes 3/20 on CPU 2; dmesg segfaults in apt/libc since Nov 16). See LOG.md 2026-11-23.
Use Turing when a GPU frees up (`nvidia-smi`; another user's vLLM container held 44/49 GB per GPU).
All code is committed and pushed: `closed_loop` at 51fd3a5 on origin (johncwlee/VLS).

Goal: closed-loop VLS on libero_goal_task. Now going through failure modes one at a time. Current one: task 0
("open the bottom drawer of the cabinet"; pi0.5 was trained on the middle drawer). Steering never sets the gripper
orientation, so even when the EE gets near the bottom handle it can't pull it.

## What we know (task 0, init state 0 = episode 1 of outputs/goal5fix_on_t0-4/2026-11-19_01-25-56)
- VLM rewards get (B, T, 3) positions only (core/pi05_steer.py:445-449); rotation dims are steered
  (libero_adapter.py:1484-1486) but get zero gradient.
- Bottom handle (sim geoms 201-203): bar along world X, center (0.037, -0.129, 0.946) m, sticks out 1.8 cm in +Y.
  Drawers slide along world +Y (joint axis -Y, range [-0.16, 0.01]).
- Ep 1 closest approach: 3.6 cm from keypoint 13 at step 144, finger line [-0.85, -0.34, -0.39] (mostly along X, i.e.
  parallel to the bar: can't pinch it). Episodes 2-5 got no closer than 6.9-14.1 cm.
- The VLM guessed the pull direction both ways: call 4 used a +X pre-grasp offset (wrong), call 5 used +Y (right).
- Keypoint 13 height varies 0.95-1.13 m across init states (perception may put it on different handles; unverified).
- EE frame (LIBERO Panda): x = finger line, z = approach (starts [0, 0, -1]). Fingertip body origins at EE
  (+-0.0475, 0, 0.0114) m open, (+-0.0097, 0, 0.0114) closed.
- Rotation prediction is usable: 10-step chunk error median 2.5 deg (constant model) vs 11.7 deg if assuming no rotation
  (`scratch/2026-11-23_rot_prediction.py`).

## Code written today (committed d7ea5b0 + fix 51fd3a5; not run in sim)
New `main.closed_loop_ee_input` (configs/config.yaml): what the VLM's guidance function takes.
- `position` (default): unchanged. Prompt is byte-identical to before (checked, first + check calls).
- `pose`: fn(keypoints, ee_poses (B, T+1, 4, 4)).
- `gripper_points`: fn(keypoints, gripper_points (B, T+1, 4, 3)): center, fingertip A, fingertip B, wrist (11 cm back).
Files: `closed_loop/ee_input.py` (new: points from poses, wrapper with takes_poses, dummy-pose check),
`closed_loop/example_{position,pose,gripper_points}.txt` (new; position = VLS's example moved out of the template),
`closed_loop/prompt_template.txt` (3 placeholders + `{gripper_pose}` line), `closed_loop/prompt.py`,
`closed_loop/query.py`, `closed_loop/run.py`, `main.py` (passes `adapter._measure_ee()`; main.py:901 grasp_active
now also needs grasp_steering, else pose fns would call grasp_steering.after_chunk on None).
Both examples encode the same generic grasp (keypoint 0 from above, fingers along kp1->kp2). The handle direction is
NOT given to the VLM (no sim ground truth): the user didn't pick that yet.

## Broken / not done
- pytest: 56 passed, 3 skipped (FakeVLM signature fixed). test_libero_kinematics passes only with a GPU visible.
- Fixed: `check_on_dummy_poses` imported a missing pytorch_kinematics function, so pose / gripper_points modes raised
  ImportError on the first VLM function. No pytest covers these modes yet.
- The port note's step 0 and step 4 expectations are stale (see TODO.md); `import libero` alone always fails (the adapter
  adds third_party/libero_pro to sys.path).
- Not checked: gripper_points_from_poses against sim fingertip bodies at several orientations; offline VLM calls with
  pose / gripper_points prompts on saved task 0 frames; any sim run.
- FKD path: pose fns would hit `fn.fkd_rewards` (GraspReward only). Runs use use_fkd=false, so not handled.

## Try next
1. (done) FakeVLM signature.
2. Scratch test: sim fingertips vs gripper_points_from_poses (input pose -> computed -> measured -> match).
3. Offline: one first-call per mode on episode 1's start image; read the functions it writes.
4. Sim: task 0 init state 0, one episode each for position / pose / gripper_points, same launcher overrides as
   goal5fix (`notes/2026-11-23_desktop_port.md` step 5 + `main.closed_loop_ee_input=<mode>`). 1 episode each: not rates.
5. Then: stricter stage-completion check (user asked: be SURE a stage is done before advance). Choice still open:
   (a) prompt only, or (b) add measured numbers (EE-target distance, drawer joint = sim ground truth?).

## Known issues carried over (TODO.md)
- LIBERO adapter ignores init states; use `scratch/2026-11-19_fixed_init_launcher.py` for fair comparisons.
- Give-up rule ends many episodes early; advance->replan loops not caught.
