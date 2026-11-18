"""Cases for closed_loop/episode_state.py. scratch/2026-11-18_closed_loop_state.py prints the same cases step by step."""
import pytest

from closed_loop.episode_state import apply_answer, new_episode_state, trigger_text
from closed_loop.parse import ParseError

PLAN_1 = ["grasp the wine bottle", "move the bottle above the bowl"]
PLAN_2 = ["grasp the fallen wine bottle", "move the bottle above the bowl"]
PLAN_3 = ["grasp the wine bottle by its body", "move the bottle above the bowl"]


def ans(verdict, plan=PLAN_1, stage=1, guidance=True):
    """A parsed answer, as closed_loop.parse.parse_output returns it."""
    has_function = verdict in ("plan", "advance", "replan") and guidance
    return {"verdict": verdict, "plan": plan, "completed_stages": [], "undone_stages": [],
            "current_stage": 0 if verdict == "done" else stage, "guidance": guidance and verdict != "done",
            "evidence": f"made-up {verdict}",
            "function_name": f"stage{stage}_guidance" if has_function else None,
            "function_source": f"def stage{stage}_guidance(...)" if has_function else None}


def exp(plan_id, stage, tl_continues=0, replans=0, queries=1, stop=None, replan_required=False):
    """Expected state after a step, in the same shape as summarize()."""
    return dict(plan_id=plan_id, stage=stage, tl_continues=tl_continues, replans=replans, queries=queries,
                stop=stop, replan_required=replan_required)


TL = True    # stage time-limit check
GRIP = False  # gripper/reward-triggered check

# (name, query_limit, steps); a step is (answer, time_limit, expected state) or (answer, time_limit, error words)
SEQUENCES = [
    ("normal episode: plan, advance, done", 50, [
        (ans("plan"), GRIP, exp(1, 1)),
        (ans("advance", stage=2), GRIP, exp(1, 2, queries=2)),
        (ans("done"), GRIP, exp(1, 0, queries=3)),
    ]),
    ("gripper-triggered continues don't count toward the limit", 50, [
        (ans("plan"), GRIP, exp(1, 1)),
        (ans("continue"), GRIP, exp(1, 1, queries=2)),
        (ans("continue"), GRIP, exp(1, 1, queries=3)),
        (ans("continue"), GRIP, exp(1, 1, queries=4)),
        (ans("continue"), GRIP, exp(1, 1, queries=5)),
    ]),
    ("3 time-limit continues, then the forced check replans", 50, [
        (ans("plan"), GRIP, exp(1, 1)),
        (ans("continue"), TL, exp(1, 1, tl_continues=1, queries=2)),
        (ans("continue"), TL, exp(1, 1, tl_continues=2, queries=3)),
        (ans("continue"), TL, exp(1, 1, tl_continues=3, queries=4, replan_required=True)),
        (ans("replan", plan=PLAN_2), TL, exp(2, 1, replans=1, queries=5)),
    ]),
    ("forced check answered continue -> error", 50, [
        (ans("plan"), GRIP, exp(1, 1)),
        (ans("continue"), TL, exp(1, 1, tl_continues=1, queries=2)),
        (ans("continue"), TL, exp(1, 1, tl_continues=2, queries=3)),
        (ans("continue"), TL, exp(1, 1, tl_continues=3, queries=4, replan_required=True)),
        (ans("continue"), TL, "`continue` is not allowed"),
    ]),
    ("forced check answered advance -> allowed, counters reset", 50, [
        (ans("plan"), GRIP, exp(1, 1)),
        (ans("continue"), TL, exp(1, 1, tl_continues=1, queries=2)),
        (ans("continue"), TL, exp(1, 1, tl_continues=2, queries=3)),
        (ans("continue"), TL, exp(1, 1, tl_continues=3, queries=4, replan_required=True)),
        (ans("advance", stage=2), TL, exp(1, 2, queries=5)),
    ]),
    ("plans 2 and 3 are tried; the 3rd replan without an advance -> give up", 50, [
        (ans("plan"), GRIP, exp(1, 1)),
        (ans("replan", plan=PLAN_2), GRIP, exp(2, 1, replans=1, queries=2)),
        (ans("replan", plan=PLAN_3), GRIP, exp(3, 1, replans=2, queries=3)),
        (ans("replan", plan=PLAN_2), GRIP, exp(4, 1, replans=3, queries=4, stop="vls_gave_up")),
    ]),
    ("an advance between replans resets the count", 50, [
        (ans("plan"), GRIP, exp(1, 1)),
        (ans("replan", plan=PLAN_2), GRIP, exp(2, 1, replans=1, queries=2)),
        (ans("advance", plan=PLAN_2, stage=2), GRIP, exp(2, 2, queries=3)),
        (ans("replan", plan=PLAN_3), GRIP, exp(3, 1, replans=1, queries=4)),
    ]),
    ("query limit 3 is reached", 3, [
        (ans("plan"), GRIP, exp(1, 1)),
        (ans("continue"), GRIP, exp(1, 1, queries=2)),
        (ans("continue"), GRIP, exp(1, 1, queries=3, stop="vlm_query_limit")),
    ]),
    ("first call that is not a plan -> error", 50, [
        (ans("continue"), GRIP, "first call must answer `plan`"),
    ]),
    ("check call answered plan -> error", 50, [
        (ans("plan"), GRIP, exp(1, 1)),
        (ans("plan"), GRIP, "`plan` is only for the first call"),
    ]),
]


def summarize(state):
    return dict(plan_id=state["plan_id"], stage=state["current_stage"], tl_continues=state["time_limit_continues"],
                replans=state["replans_since_advance"], queries=state["query_count"], stop=state["stop_reason"],
                replan_required="re-plan" in trigger_text(state, "x"))


def run_step(state, step_index, answer, time_limit):
    trigger = "stage time limit" if time_limit else "gripper closed"
    apply_answer(state, answer, step=10 * step_index, trigger=trigger_text(state, trigger), time_limit=time_limit,
                 chunks=4, reward_now=0.1, reward_best=0.2)


@pytest.mark.parametrize("name, query_limit, steps", SEQUENCES, ids=[s[0] for s in SEQUENCES])
def test_sequence(name, query_limit, steps):
    state = new_episode_state(query_limit)
    for i, (answer, time_limit, expected) in enumerate(steps):
        if isinstance(expected, str):
            with pytest.raises(ParseError, match=expected.replace("(", r"\(")):
                run_step(state, i, answer, time_limit)
        else:
            run_step(state, i, answer, time_limit)
            assert summarize(state) == expected, f"step {i}"


def test_history_records_the_checked_stage():
    # The advance entry must name stage 1 (the stage that was checked), not stage 2
    state = new_episode_state(50)
    run_step(state, 0, ans("plan"), GRIP)
    run_step(state, 1, ans("advance", stage=2), GRIP)
    assert [(e["stage"], e["stage_name"], e["verdict"]) for e in state["history"]] == [
        (1, PLAN_1[0], "plan"), (1, PLAN_1[0], "advance")]
