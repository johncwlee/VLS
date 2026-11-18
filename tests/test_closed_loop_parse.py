"""Cases for closed_loop/parse.py. scratch/2026-11-18_closed_loop_parse.py prints the same cases one by one."""
import os

import pytest

from closed_loop.parse import ParseError, parse_output

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
OLD_ANSWER_PATH = os.path.join(REPO_ROOT, "outputs/vlsmain_equiv_opus_low/2026-11-07_03-26-40/episode_3/vlm_agent/output_raw.txt")

PLAN_1 = """# Plan:
# Stage 1: Move gripper to grasp the wine bottle (keypoints 3, 4)
# Stage 2: Move gripper with wine bottle to above the bowl (keypoints 0, 1)"""

PLAN_2 = """# Plan:
# Stage 1: Move gripper to grasp the fallen wine bottle again (keypoints 3, 4)
# Stage 2: Move gripper with wine bottle to above the bowl (keypoints 0, 1)"""


def guidance_function(stage):
    return f"""def stage{stage}_guidance(keypoints, action_sequence):
    target_idx = torch.tensor([3], dtype=torch.long, device=keypoints.device)
    target_pos = keypoints[target_idx][0]
    all_dists = torch.norm(action_sequence - target_pos, dim=-1)
    reward = -(all_dists ** 2).mean(dim=1)
    return reward.mean()"""


def answer(plan, verdict, completed, undone, stage, guidance, evidence, function=""):
    return f"""```python
{plan}

verdict = {verdict}
completed_stages = {completed}
undone_stages = {undone}
current_stage = {stage}
guidance = {guidance}
evidence = "{evidence}"

{function}
```"""


# Good answers: (name, answer text, expected short summary)
GOOD_CASES = [
    ("first call -> plan",
     answer(PLAN_1, "plan", "[]", "[]", 1, "yes", "nothing is done yet; the bottle is upright", guidance_function(1)),
     dict(verdict="plan", done=[], undone=[], stage=1, guidance=True, function="stage1_guidance", plan_stages=2)),
    ("good grasp -> advance",
     answer(PLAN_1, "advance", "[1]", "[]", 2, "yes", "the bottle is in the gripper", guidance_function(2)),
     dict(verdict="advance", done=[1], undone=[], stage=2, guidance=True, function="stage2_guidance", plan_stages=2)),
    ("drop -> replan",
     answer(PLAN_2, "replan", "[]", "[1]", 1, "yes", "the bottle fell on the table", guidance_function(1)),
     dict(verdict="replan", done=[], undone=[1], stage=1, guidance=True, function="stage1_guidance", plan_stages=2)),
    ("still going -> continue",
     answer(PLAN_1, "continue", "[]", "[]", 1, "yes", "the gripper is moving toward the bottle"),
     dict(verdict="continue", done=[], undone=[], stage=1, guidance=True, function=None, plan_stages=2)),
    ("task finished -> done",
     answer(PLAN_1, "done", "[1, 2]", "[]", 0, "no", "the bottle is in the bowl"),
     dict(verdict="done", done=[1, 2], undone=[], stage=0, guidance=False, function=None, plan_stages=2)),
]

# Broken answers: (name, answer text, words the error message must contain)
BROKEN_CASES = [
    ("no verdict line",
     answer(PLAN_1, "plan", "[]", "[]", 1, "yes", "x", guidance_function(1)).replace("verdict = plan\n", ""),
     "no `verdict"),
    ("made-up verdict",
     answer(PLAN_1, "maybe", "[]", "[]", 1, "yes", "x", guidance_function(1)),
     "verdict 'maybe' is not one of"),
    ("advance but no function",
     answer(PLAN_1, "advance", "[1]", "[]", 2, "yes", "the bottle is in the gripper"),
     "needs stage2_guidance"),
    ("function name doesn't match the stage",
     answer(PLAN_1, "advance", "[1]", "[]", 2, "yes", "the bottle is in the gripper", guidance_function(1)),
     "needs stage2_guidance"),
    ("current_stage past the end of the plan",
     answer(PLAN_1, "advance", "[1, 2]", "[]", 3, "yes", "x", guidance_function(3)),
     "outside the plan"),
]


def summarize(parsed):
    """Short form of parse_output's result, in the same shape as the expected dicts above."""
    return dict(verdict=parsed["verdict"], done=parsed["completed_stages"], undone=parsed["undone_stages"],
                stage=parsed["current_stage"], guidance=parsed["guidance"], function=parsed["function_name"],
                plan_stages=len(parsed["plan"]))


@pytest.mark.parametrize("name, text, expected", GOOD_CASES, ids=[c[0] for c in GOOD_CASES])
def test_good_answer(name, text, expected):
    assert summarize(parse_output(text)) == expected


@pytest.mark.parametrize("name, text, error_words", BROKEN_CASES, ids=[c[0] for c in BROKEN_CASES])
def test_broken_answer(name, text, error_words):
    with pytest.raises(ParseError, match=error_words.replace("(", r"\(")):
        parse_output(text)


@pytest.mark.skipif(not os.path.exists(OLD_ANSWER_PATH), reason="saved run not on this machine")
def test_old_format_answer_is_rejected():
    # Real answer to the old prompt: it has num_stages and two functions, but no verdict
    with open(OLD_ANSWER_PATH) as f:
        old_answer = f.read()
    with pytest.raises(ParseError, match="no `verdict"):
        parse_output(old_answer)
