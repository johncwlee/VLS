"""Keeps track of one closed-loop episode: plan, current stage, history, and the stop rules (PLAN_closed_loop.md step 4)."""
from closed_loop.parse import ParseError

# Both decided 2026-11-18 (PLAN_closed_loop.md, step 4)
MAX_TIME_LIMIT_CONTINUES = 3     # time-limit `continue` answers in one stage; then the next check must not continue
MAX_REPLANS_WITHOUT_ADVANCE = 3  # plans 2 and 3 get tried; the 3rd replan ends the episode ('vls_gave_up')


def new_episode_state(query_limit):
    """query_limit: main.vlm_query_limit; closed-loop calls count toward it, and reaching it ends the episode."""
    return {
        "plan": [],                  # stage descriptions of the current plan, stage 1 first
        "plan_id": 0,                # 0 = no plan yet; +1 on every plan/replan
        "current_stage": 0,          # 1-based in plan; 0 before the first plan and after `done`
        "guidance": False,
        "function_source": None,     # guidance function for current_stage, as text
        "history": [],               # one dict per VLM call, the format closed_loop.prompt.format_history reads
        "query_count": 0,
        "query_limit": query_limit,
        "time_limit_continues": 0,   # in the current stage; reset on advance/replan
        "replans_since_advance": 0,  # stages get renumbered after a replan, so we count replans, not stage names
        "stop_reason": None,         # None, 'vls_gave_up' or 'vlm_query_limit'
    }


def replan_required(state):
    return state["time_limit_continues"] >= MAX_TIME_LIMIT_CONTINUES


def trigger_text(state, reason):
    """Trigger line for the next call. Once the stage has used up its time limits, it says a re-plan is required."""
    if not replan_required(state):
        return reason
    return (f"{reason}; stage {state['current_stage']} reached {MAX_TIME_LIMIT_CONTINUES} time limits without "
            f"finishing, so `continue` is not allowed: re-plan unless the stage is finished")


def apply_answer(state, answer, step, trigger, time_limit, chunks, reward_now, reward_best):
    """
    Updates `state` in place with one parsed VLM answer (closed_loop.parse.parse_output).

    Args:
        step: sim step of the call.
        trigger: the trigger text sent in the prompt.
        time_limit: True when the call was a stage time-limit check (only these count toward the continue limit).
        chunks: chunks executed in the current stage before this call.
        reward_now, reward_best: normalized reward of the current stage (0 = start, 1 = at target).
    Raises ParseError when the answer breaks a rule that depends on the episode (not on the answer text alone).
    """
    assert state["stop_reason"] is None, f"episode already stopped ({state['stop_reason']}); no more calls"
    verdict = answer["verdict"]
    state["query_count"] += 1

    first_call = state["plan_id"] == 0
    if first_call and verdict != "plan":
        raise ParseError(f"first call must answer `plan`, got {verdict!r}")
    if not first_call and verdict == "plan":
        raise ParseError("`plan` is only for the first call; a check call must answer continue, advance, replan or done")
    if replan_required(state) and verdict == "continue":
        raise ParseError(f"stage {state['current_stage']} reached {MAX_TIME_LIMIT_CONTINUES} time limits, "
                         f"so `continue` is not allowed")

    if first_call:
        state["plan"] = answer["plan"]
        state["plan_id"] = 1
        state["current_stage"] = answer["current_stage"]

    # The entry describes the stage that was checked, so it is written before the stage changes below
    state["history"].append({
        "step": step,
        "plan_id": state["plan_id"],
        "stage": state["current_stage"],
        "stage_name": state["plan"][state["current_stage"] - 1],
        "chunks": chunks,
        "reward_now": reward_now,
        "reward_best": reward_best,
        "trigger": trigger,
        "verdict": verdict,
        "evidence": answer["evidence"],
    })

    if verdict == "plan":
        state["guidance"] = answer["guidance"]
        state["function_source"] = answer["function_source"]
    elif verdict == "continue":
        # The prompt says `continue` keeps the same guidance, so the answer's guidance flag is not used
        if time_limit:
            state["time_limit_continues"] += 1
    elif verdict == "advance":
        state["current_stage"] = answer["current_stage"]
        state["guidance"] = answer["guidance"]
        state["function_source"] = answer["function_source"]
        state["time_limit_continues"] = 0
        state["replans_since_advance"] = 0
    elif verdict == "replan":
        state["plan"] = answer["plan"]
        state["plan_id"] += 1
        state["current_stage"] = answer["current_stage"]
        state["guidance"] = answer["guidance"]
        state["function_source"] = answer["function_source"]
        state["time_limit_continues"] = 0
        state["replans_since_advance"] += 1
        if state["replans_since_advance"] >= MAX_REPLANS_WITHOUT_ADVANCE:
            state["stop_reason"] = "vls_gave_up"
    elif verdict == "done":
        state["current_stage"] = 0
        state["guidance"] = False
        state["function_source"] = None

    if state["stop_reason"] is None and state["query_count"] >= state["query_limit"]:
        state["stop_reason"] = "vlm_query_limit"
