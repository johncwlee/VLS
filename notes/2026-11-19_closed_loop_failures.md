# Closed loop on libero_goal_task ep 9-10 (2026-11-18/19): why it fails

Commit 3cf682c (code), claude-opus-5-5 effort low, 20 particles, use_fkd=false, dinov2_vits14. 1 episode per setup: not rates.
Ep 9 = task 8 "Put the wine bottle on the plate" (On wine_bottle_1 plate_1).
Ep 10 = task 9 "Put the cream cheese on the rack" (On cream_cheese_1 wine_rack_1_top_region).

## Gripper reading (fixed)
Measured finger width (q1 - q2, Panda finger joints) from `outputs/gripper_width_measure/2026-11-18_22-55-11`:
open 7.63-8.01 cm; on bottle neck 1.50; on cream cheese 4.26-4.57; closing on nothing 0.3.
A single -1 command inside a grasp (ep 9 step 209) moves the fingers 0.01 cm. Closed loop now reads open = width >= 6 cm.
`adapter.get_gripper_state()` returns q1 + q2 (about 0 always); upstream VLS code, no callers.

## LIBERO success
On(A, B) for objects: B body z <= A body z, A-B contact, body-origin xy distance < 3 cm. No gripper condition.
Region (rack top): site box 20 x 4.4 cm, tilted; object point inside it + rack contact.

## Failures found
| run | ep | what happened |
|---|---|---|
| objlog_off/2026-11-18_23-44-34 | 9 | base policy holds bottle 3.3-4.4 cm above plate (xy ok), never touches, 300 steps |
| objlog_on/2026-11-18_23-49-17 | 9 | VLM says "on the plate" while 3.3-3.8 cm above; vlm_error at call 8 (comment after `guidance = "no"`) |
| objlog_on/2026-11-18_23-49-17 | 10 | VLM says done "on the rack"; cheese 6.5 cm past region edge |
| gt_sim/2026-11-19_00-31-41 (GT in prompt) | 9 | VLM right (not touching, replan "lower"); its targets plate + 13.5 / 12.5 cm are 1.8 / 0.8 cm too high; EE stays 3-4 cm above even that; gave up |
| gt_sim/2026-11-19_00-31-41 | 10 | VLM tracks drops right; policy drops cheese twice; release 0.5 cm past edge at end; one `advance` on an empty close (width 0.3 cm) |
| depth_measure/2026-11-19_00-48-12 | 9 | SUCCESS at step 277: replan at 260 aimed gripper at current neck z - 2 cm; VLM still thought bottle was on the plate |
| depth_measure/2026-11-19_00-48-12 | 10 | cheese dropped on cabinet top, open gripper hovers over it 160 steps |

Offline GT injection (`outputs/closed_loop_gt_inject/2026-11-19_00-21-54`, 4 saved calls, 1 call each): with GT the
evidence was right 4/4, verdict changed to the right action 2/4; plain re-runs repeated the original wrong reading 4/4.
The injected "resting = about 0 cm" hint was off: the bottle touched at origin dz 0.9 cm.

## Depth + sim masks at each call (scratch/2026-11-19_closed_loop_depth_measure.py)
Bottle over plate: agentview gap 2.6-3.8 cm vs true origin dz 3.2-4.2 (offset 0.4-0.7); wrist 3.3-4.6. Works.
Cheese near rack: gap -17 cm, overlap 0.52-0.58 while 8.4 cm past edge: the "support under it" rule picks up the
rack's upright parts. Fails. Wrist sees 57-72 rack points. "Touching" case never measured.

## Known logging issue
`_log_step` on the success step logs the state after the env reset (ep 9 step 277 shows the bottle at its start).
