"""Builds the closed-loop VLS prompt text for the "first" call and for "check" calls."""
import os

import numpy as np
import torch

from closed_loop.ee_input import EE_INPUTS, gripper_points_from_poses

TEMPLATE_DIR = os.path.dirname(__file__)

# What the guidance function gets, per main.closed_loop_ee_input (closed_loop/ee_input.py). "position" is VLS's text.
EE_INPUT_SUMMARY = {
    "position": "- Each guidance function takes a sequence of 3D points as the action sequence and a set of keypoints "
                "as input, and returns a scalar reward.",
    "pose": "- Each guidance function takes a sequence of proposed end-effector poses (position and orientation) and a "
            "set of keypoints as input, and returns a scalar reward.",
    "gripper_points": "- Each guidance function takes a sequence of proposed gripper points (fingertips and wrist) and "
                      "a set of keypoints as input, and returns a scalar reward.",
}
EE_INPUT_DESCRIPTION = {
    "position": "  - `action_sequence`: torch.tensor of shape `(B, T, 3)` representing the proposed end-effector "
                "trajectory in world frame, with first item in each batch representing current end-effector position.",
    "pose": "\n".join([
        "  - `ee_poses`: torch.tensor of shape `(B, T, 4, 4)`: the proposed end-effector poses in world frame, one 4x4 "
        "homogeneous matrix per time step. The first time step is the current pose.",
        "    - `ee_poses[..., :3, 3]`: gripper position (between the fingertips), shape `(B, T, 3)`.",
        "    - `ee_poses[..., :3, 0]`: unit vector along the finger line, from fingertip A to fingertip B. "
        "The fingers close along this line.",
        "    - `ee_poses[..., :3, 2]`: approach direction, unit vector from the wrist toward the fingertips.",
        "    - Steering moves and rotates the gripper. A reward on position only leaves the orientation to the policy; "
        "add an orientation term when the stage needs the gripper in a particular orientation.",
    ]),
    "gripper_points": "\n".join([
        "  - `gripper_points`: torch.tensor of shape `(B, T, 4, 3)`: four points on the gripper in world frame, "
        "for each proposed time step. The first time step is the current state.",
        "    - `gripper_points[:, :, 0]`: gripper center, midway between the fingertips.",
        "    - `gripper_points[:, :, 1]` and `gripper_points[:, :, 2]`: fingertips A and B, 9.5 cm apart (fingers open). "
        "The fingers close along the line from A to B.",
        "    - `gripper_points[:, :, 3]`: wrist, 11 cm behind the gripper center, on the side away from the fingertips.",
        "    - Steering moves and rotates the gripper. A reward on the center only leaves the orientation to the policy; "
        "place the fingertips or the wrist relative to keypoints when the stage needs a particular orientation.",
    ]),
}


def _read(name):
    """Template text without the lines starting with %% (our notes marking changes vs VLS; never sent)."""
    with open(os.path.join(TEMPLATE_DIR, name), 'r') as f:
        lines = f.read().split("\n")
    kept = [line for line in lines if not line.startswith("%%")]
    return "\n".join(kept)


def format_plan(plan):
    """plan: list of stage descriptions, stage 1 first."""
    if len(plan) == 0:
        return "(no plan yet)"
    lines = []
    for i, description in enumerate(plan, start=1):
        lines.append(f"Stage {i}: {description}")
    return "\n".join(lines)


def format_history(history):
    """history: list of dicts, one per earlier VLM call, oldest first."""
    if len(history) == 0:
        return "(no earlier decisions)"
    lines = []
    for entry in history:
        # reward_now / reward_best are the policy's normalized reward (0 = start of stage, 1 = at target)
        # plan_id goes up by 1 on every plan/replan, so "stage 1" of plan 1 and of plan 2 are not confused
        lines.append(f"- step {entry['step']}: plan {entry['plan_id']}, stage {entry['stage']} ({entry['stage_name']}), "
                     f"worked on it for {entry['chunks']} chunks, "
                     f"reward now {entry['reward_now']:.2f} (best {entry['reward_best']:.2f}); "
                     f"trigger: {entry['trigger']}; you decided: {entry['verdict']}; evidence: {entry['evidence']}")
    return "\n".join(lines)


def format_gripper_pose(ee_input, ee_pose_now):
    """The extra text after "Gripper: open/closed" in the Query section: none for "position"."""
    if ee_input == "position":
        return ""
    ee_pos_world_m, R_world_ee = ee_pose_now
    if ee_input == "pose":
        return (f"\nGripper pose now: position {np.round(ee_pos_world_m, 3).tolist()}, "
                f"finger line {np.round(R_world_ee[:, 0], 2).tolist()}, "
                f"approach direction {np.round(R_world_ee[:, 2], 2).tolist()}")
    pose = np.eye(4)
    pose[:3, :3] = R_world_ee
    pose[:3, 3] = ee_pos_world_m
    points = gripper_points_from_poses(torch.as_tensor(pose)).numpy()  # (4, 3)
    return (f"\nGripper points now: center {np.round(points[0], 3).tolist()}, fingertip A {np.round(points[1], 3).tolist()}, "
            f"fingertip B {np.round(points[2], 3).tolist()}, wrist {np.round(points[3], 3).tolist()}")


def build_prompt(mode, instruction, plan, plan_id, current_stage, history, trigger_reason, gripper_state,
                 key_points_objects_map, keypoint_positions, task_specific_patterns, ee_input="position", ee_pose_now=None):
    """
    Args:
        mode: "first" (make a plan) or "check" (judge the current stage).
        plan: list of stage descriptions. Empty for "first".
        plan_id: number of the current plan (1 = the first plan). Ignored for "first".
        current_stage: 1-based stage the robot is working on. Ignored for "first".
        history: list of dicts with keys step, plan_id, stage, stage_name, chunks, reward_now, reward_best,
            trigger, verdict, evidence.
        gripper_state: "open" or "closed", from the measured finger width (main.py _closed_loop_step).
            The camera often can't show it, so without this the VLM can't tell a grasp from a hover.
        key_points_objects_map: {keypoint index: object name}.
        keypoint_positions: (K, 3) keypoint positions in world frame, meters, at the time of this call.
        task_specific_patterns: environment notes, e.g. vlm_query/guidance_template_libero.txt.
        ee_input: what the guidance function gets (closed_loop/ee_input.py EE_INPUTS).
        ee_pose_now: (ee_pos_world_m (3,), R_world_ee (3, 3)) measured now; needed unless ee_input is "position".
    """
    assert ee_input in EE_INPUTS, f"unknown ee_input {ee_input!r}"
    assert ee_input == "position" or ee_pose_now is not None, f"ee_input {ee_input!r} needs ee_pose_now"
    assert mode in ("first", "check"), f"unknown mode {mode!r}"
    assert gripper_state in ("open", "closed"), f"gripper_state must be 'open' or 'closed', got {gripper_state!r}"
    if mode == "check":
        assert len(plan) > 0, "a check call needs the current plan"
        assert 1 <= current_stage <= len(plan), f"current_stage {current_stage} is outside the plan (1..{len(plan)})"
        mode_instructions = _read('mode_check.txt').replace("{current_stage}", str(current_stage))
        image_description = ("Two images follow.\n"
                             "Image 1: the scene when the current plan was made, with keypoints marked with their indices.\n"
                             "Image 2: the scene now, with keypoints at their current positions.")
    else:
        mode_instructions = _read('mode_first.txt')
        image_description = "Query Image (the scene now, with keypoints marked with their indices):"

    # replace() rather than format(): the template holds Python code with curly braces
    prompt_text = _read('prompt_template.txt')
    replacements = {
        "{mode_instructions}": mode_instructions,
        "{instruction}": str(instruction),
        "{trigger_reason}": str(trigger_reason),
        "{gripper_state}": gripper_state,
        "{plan_id}": str(plan_id) if mode == "check" else "none yet",
        "{plan}": format_plan(plan),
        "{history}": format_history(history),
        "{key_points_objects_map}": str(key_points_objects_map),
        "{keypoint_positions}": str(keypoint_positions),
        "{task_specific_patterns}": task_specific_patterns,
        "{image_description}": image_description,
        "{ee_input_summary}": EE_INPUT_SUMMARY[ee_input],
        "{ee_input_description}": EE_INPUT_DESCRIPTION[ee_input],
        "{example_function}": _read(f"example_{ee_input}.txt").rstrip("\n"),
        "{gripper_pose}": format_gripper_pose(ee_input, ee_pose_now),
    }
    for placeholder, value in replacements.items():
        prompt_text = prompt_text.replace(placeholder, value)
    return prompt_text
