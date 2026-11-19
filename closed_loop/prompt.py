"""Builds the closed-loop VLS prompt text for the "first" call and for "check" calls."""
import os

TEMPLATE_DIR = os.path.dirname(__file__)


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


def build_prompt(mode, instruction, plan, plan_id, current_stage, history, trigger_reason, gripper_state,
                 key_points_objects_map, keypoint_positions, task_specific_patterns):
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
    """
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
    }
    for placeholder, value in replacements.items():
        prompt_text = prompt_text.replace(placeholder, value)
    return prompt_text
