"""Runs closed-loop VLS inside an episode: the first call, then a check call whenever a trigger fires (PLAN_closed_loop.md step 5)."""
import os

from closed_loop.episode_state import apply_answer, new_episode_state, trigger_text
from closed_loop.prompt import build_prompt
from closed_loop.query import run_call
from vlm_query.vlm_agent import VLMCallError

LIBERO_NOTES_PATH = os.path.join(os.path.dirname(__file__), '..', 'vlm_query', 'guidance_template_libero.txt')


def new_closed_loop(call_root, instruction, start_image_kp, key_points_objects_map, model, reasoning_effort,
                    time_limit_chunks, query_limit, reward_upper, reward_lower, ee_input="position"):
    """
    The dict that start_episode and on_new_chunk update. No VLM call.

    Args:
        call_root: folder that gets one sub-folder per VLM call.
        start_image_kp: (H, W, 3) uint8 RGB scene when the current plan is made, keypoints drawn with their indices.
        key_points_objects_map: {keypoint index: object name}.
        time_limit_chunks: chunks without a check before a time-limit check fires (main.closed_loop_time_limit_chunks).
        query_limit: main.vlm_query_limit; closed-loop calls count toward it.
        reward_upper, reward_lower: normalized-reward thresholds of the reward triggers (main.schmitt_upper / lower).
        ee_input: what the guidance function takes (main.closed_loop_ee_input, closed_loop/ee_input.py).
    """
    with open(LIBERO_NOTES_PATH) as f:
        libero_notes = f.read()
    return {
        "state": new_episode_state(query_limit),
        "guidance_fns": [],
        "call_root": call_root,
        "instruction": instruction,
        "start_image_kp": start_image_kp,
        "key_points_objects_map": key_points_objects_map,
        "task_notes": libero_notes,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "ee_input": ee_input,
        "time_limit_chunks": time_limit_chunks,
        "reward_upper": reward_upper,
        "reward_lower": reward_lower,
        "prev_gripper_open": None,   # None until the first action is known
        "prev_reward": 0.0,
        "chunks_since_check": 0,
        "chunks_in_stage": 0,
        "reward_best": 0.0,          # best normalized reward in the current stage
    }


def _call(cl, prompt_text, images, step, trigger, time_limit, reward_now):
    """One VLM call: send, read, update the episode state. Returns True if the plan or stage changed."""
    state = cl["state"]
    before = (state["plan_id"], state["current_stage"])
    call_dir = os.path.join(cl["call_root"], f"call_{state['query_count'] + 1:02d}_step{step:04d}")
    try:
        answer = run_call(prompt_text, images, call_dir, cl["model"], cl["reasoning_effort"], cl["ee_input"])
        apply_answer(state, answer, step, trigger, time_limit, cl["chunks_in_stage"], reward_now, cl["reward_best"])
    except ValueError as e:
        # ParseError (a ValueError) or a guidance function that fails its dummy-trajectory check: the VLM's fault,
        # so it is reported as a VLM error (trials.jsonl outcome 'vlm_error'), not as a code crash
        raise VLMCallError(f"closed-loop call {call_dir}: {e}") from e

    if answer["verdict"] in ("plan", "advance", "replan"):
        cl["guidance_fns"] = answer["guidance_fns"]
    elif answer["verdict"] == "done":
        cl["guidance_fns"] = []
    if answer["verdict"] == "replan":
        # The prompt's image 1 is "the scene when the current plan was made"
        cl["start_image_kp"] = images[-1]

    cl["chunks_since_check"] = 0
    changed = (state["plan_id"], state["current_stage"]) != before
    if changed:
        cl["chunks_in_stage"] = 0
        cl["reward_best"] = 0.0
        # The new stage's reward starts over; main.py:507 also resets it, so the old stage's reward can't
        # make a "reward dropped" trigger fire on the next chunk
        cl["prev_reward"] = 0.0
    return changed


def start_episode(cl, keypoints, ee_pose_now=None):
    """
    First call: the VLM writes the plan and the stage-1 guidance function.
    keypoints: (K, 3) keypoint positions in world frame, meters.
    ee_pose_now: (ee_pos_world_m (3,), R_world_ee (3, 3)) measured; needed unless cl["ee_input"] is "position".
    Raises VLMCallError if the call fails or the answer is unusable.
    """
    # Assumption: every LIBERO episode starts with the gripper open
    prompt_text = build_prompt(mode="first", instruction=cl["instruction"], plan=[], plan_id=0, current_stage=1,
                               history=[], trigger_reason="initial query", gripper_state="open",
                               key_points_objects_map=cl["key_points_objects_map"], keypoint_positions=keypoints,
                               task_specific_patterns=cl["task_notes"], ee_input=cl["ee_input"],
                               ee_pose_now=ee_pose_now)
    _call(cl, prompt_text, [cl["start_image_kp"]], step=0, trigger="initial query", time_limit=False, reward_now=0.0)


def pick_trigger(cl, step, gripper_open, reward):
    """
    Called once per new chunk. Updates the chunk counters and returns (trigger text, is_time_limit), or None.
    Priority as in main.py _update_stage: gripper > reward > time limit.

    gripper_open: measured finger width >= 6 cm after the chunk (main.py _closed_loop_step); None before the first action.
    reward: the policy's normalized reward of the current stage (0 = start, 1 = at target).
    """
    if step > 0:
        cl["chunks_since_check"] += 1
        cl["chunks_in_stage"] += 1
    cl["reward_best"] = max(cl["reward_best"], reward)
    prev_gripper_open = cl["prev_gripper_open"]
    prev_reward = cl["prev_reward"]
    cl["prev_gripper_open"] = gripper_open
    cl["prev_reward"] = reward

    state = cl["state"]
    if state["stop_reason"] is not None or state["current_stage"] == 0:
        return None
    if prev_gripper_open is not None and gripper_open is not None and gripper_open != prev_gripper_open:
        if gripper_open:
            return "gripper opened", False
        return "gripper closed", False
    # Reward triggers only while guidance is on, as in main.py:470-471 (the reward is only computed then)
    if state["guidance"] and prev_reward < cl["reward_upper"] <= reward:
        return f"reward rose above {cl['reward_upper']:.0%}", False
    if state["guidance"] and prev_reward > cl["reward_lower"] >= reward:
        return f"reward dropped below {cl['reward_lower']:.0%}", False
    if cl["chunks_since_check"] >= cl["time_limit_chunks"]:
        return f"stage {state['current_stage']} time limit reached ({cl['time_limit_chunks']} chunks)", True
    return None


def on_new_chunk(cl, step, gripper_open, reward, current_image_kp, keypoints, ee_pose_now=None):
    """
    Called at the start of every chunk. Makes a check call if a trigger fires.

    Args:
        current_image_kp: (H, W, 3) uint8 RGB scene now, current keypoints drawn with their indices.
        keypoints: (K, 3) current keypoint positions in world frame, meters.
        ee_pose_now: (ee_pos_world_m (3,), R_world_ee (3, 3)) measured; needed unless cl["ee_input"] is "position".
    Returns (current_stage, use_guidance, guidance_fns, stage_changed). current_stage is 0 after `done`.
    The episode must end when cl["state"]["stop_reason"] is set.
    """
    state = cl["state"]
    trigger = pick_trigger(cl, step, gripper_open, reward)
    stage_changed = False
    if trigger is not None:
        reason, is_time_limit = trigger
        assert gripper_open is not None, "a check call needs the gripper state"
        text = trigger_text(state, reason)
        prompt_text = build_prompt(mode="check", instruction=cl["instruction"], plan=state["plan"],
                                   plan_id=state["plan_id"], current_stage=state["current_stage"],
                                   history=state["history"], trigger_reason=text,
                                   gripper_state="open" if gripper_open else "closed",
                                   key_points_objects_map=cl["key_points_objects_map"], keypoint_positions=keypoints,
                                   task_specific_patterns=cl["task_notes"], ee_input=cl["ee_input"],
                                   ee_pose_now=ee_pose_now)
        stage_changed = _call(cl, prompt_text, [cl["start_image_kp"], current_image_kp], step, text, is_time_limit,
                              reward)
    use_guidance = state["guidance"] and len(cl["guidance_fns"]) > 0
    return state["current_stage"], use_guidance, cl["guidance_fns"], stage_changed
