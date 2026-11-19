"""Cases for closed_loop/run.py with a fake VLM. scratch/2026-11-18_closed_loop_triggers.py prints them chunk by chunk."""
import numpy as np
import pytest

import closed_loop.run as run
from test_closed_loop_episode_state import PLAN_2, PLAN_3, ans

TIME_LIMIT_CHUNKS = 4
OPEN, CLOSED = True, False


def image(value):
    """A fake (8, 8, 3) uint8 RGB image filled with `value`, so tests can tell images apart."""
    return np.full((8, 8, 3), value, dtype=np.uint8)


class FakeVLM:
    """Stands in for closed_loop.query.run_call: hands out canned answers and records what it was sent."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.sent = []  # (trigger line, [image fill values]) per call

    def __call__(self, prompt_text, images, call_dir, model, reasoning_effort):
        assert self.answers, f"unexpected VLM call: {call_dir}"
        trigger = [l for l in prompt_text.split("\n") if l.startswith("This call was triggered because:")][0]
        self.sent.append((trigger.split(": ", 1)[1], [int(img[0, 0, 0]) for img in images]))
        answer = dict(self.answers.pop(0))
        answer["guidance_fns"] = ["fn"] if answer["function_source"] is not None else []
        return answer


def new_cl(tmp_path):
    cl = run.new_closed_loop(str(tmp_path), "put the wine bottle in the bowl", image(1), {0: "bowl", 3: "bottle"},
                             model="fake", reasoning_effort="low", time_limit_chunks=TIME_LIMIT_CHUNKS,
                             query_limit=50, reward_upper=0.8, reward_lower=0.6)
    return cl


# (name, first answer, chunks); a chunk is (step, gripper_open, reward, answer if a call should happen,
#  expected (trigger reason or None, stage after, stage_changed))
SEQUENCES = [
    ("quiet chunks -> time-limit check every 4 chunks", ans("plan"), [
        (0, None, 0.0, None, (None, 1, False)),
        (10, OPEN, 0.1, None, (None, 1, False)),
        (20, OPEN, 0.2, None, (None, 1, False)),
        (30, OPEN, 0.3, None, (None, 1, False)),
        (40, OPEN, 0.4, ans("continue"), ("stage 1 time limit reached (4 chunks)", 1, False)),
        (50, OPEN, 0.4, None, (None, 1, False)),
        (60, OPEN, 0.4, None, (None, 1, False)),
        (70, OPEN, 0.4, None, (None, 1, False)),
        (80, OPEN, 0.4, ans("continue"), ("stage 1 time limit reached (4 chunks)", 1, False)),
    ]),
    ("gripper closes -> check, advance to stage 2", ans("plan"), [
        (0, None, 0.0, None, (None, 1, False)),
        (10, OPEN, 0.5, None, (None, 1, False)),
        (20, CLOSED, 0.7, ans("advance", stage=2), ("gripper closed", 2, True)),
        (30, CLOSED, 0.1, None, (None, 2, False)),
    ]),
    ("reward rises above 80% -> check", ans("plan"), [
        (0, None, 0.0, None, (None, 1, False)),
        (10, OPEN, 0.5, None, (None, 1, False)),
        (20, OPEN, 0.85, ans("continue"), ("reward rose above 80%", 1, False)),
    ]),
    ("guidance off -> reward is ignored", ans("plan", guidance=False), [
        (0, None, 0.0, None, (None, 1, False)),
        (10, OPEN, 0.5, None, (None, 1, False)),
        (20, OPEN, 0.85, None, (None, 1, False)),
    ]),
    ("after done -> no more calls, even at the time limit", ans("plan"), [
        (0, None, 0.0, None, (None, 1, False)),
        (10, OPEN, 0.5, None, (None, 1, False)),
        (20, CLOSED, 0.5, ans("done"), ("gripper closed", 0, True)),
        (30, CLOSED, 0.5, None, (None, 0, False)),
        (40, CLOSED, 0.5, None, (None, 0, False)),
        (50, CLOSED, 0.5, None, (None, 0, False)),
        (60, CLOSED, 0.5, None, (None, 0, False)),
        (70, OPEN, 0.5, None, (None, 0, False)),
    ]),
    ("3 replans -> gave up, no more calls", ans("plan"), [
        (0, None, 0.0, None, (None, 1, False)),
        (10, OPEN, 0.5, None, (None, 1, False)),
        (20, CLOSED, 0.5, ans("replan", plan=PLAN_2), ("gripper closed", 1, True)),
        (30, OPEN, 0.5, ans("replan", plan=PLAN_3), ("gripper opened", 1, True)),
        (40, CLOSED, 0.5, ans("replan", plan=PLAN_2), ("gripper closed", 1, True)),
        (50, OPEN, 0.5, None, (None, 1, False)),
    ]),
    ("gripper unknown at step 0 -> a close at step 10 is not a change (as in main.py)", ans("plan"), [
        (0, None, 0.0, None, (None, 1, False)),
        (10, CLOSED, 0.5, None, (None, 1, False)),
    ]),
]


def run_sequence(cl, fake, first_answer, chunks):
    """Yields (step, expected, got) per chunk; the image of chunk i is filled with 10 + i."""
    fake.answers.append(first_answer)
    run.start_episode(cl, keypoints=np.zeros((2, 3)))
    for i, (step, gripper_open, reward, answer, expected) in enumerate(chunks):
        if answer is not None:
            fake.answers.append(answer)
        calls_before = len(fake.sent)
        stage, use_guidance, fns, stage_changed = run.on_new_chunk(cl, step, gripper_open, reward, image(10 + i),
                                                                   keypoints=np.zeros((2, 3)))
        trigger = fake.sent[-1][0] if len(fake.sent) > calls_before else None
        yield step, expected, (trigger, stage, stage_changed)


@pytest.mark.parametrize("name, first_answer, chunks", SEQUENCES, ids=[s[0] for s in SEQUENCES])
def test_sequence(name, first_answer, chunks, tmp_path, monkeypatch):
    fake = FakeVLM([])
    monkeypatch.setattr(run, "run_call", fake)
    cl = new_cl(tmp_path)
    for step, expected, got in run_sequence(cl, fake, first_answer, chunks):
        assert got == expected, f"step {step}"
    assert fake.answers == [], "an expected VLM call did not happen"


def test_give_up_sets_stop_reason(tmp_path, monkeypatch):
    fake = FakeVLM([])
    monkeypatch.setattr(run, "run_call", fake)
    cl = new_cl(tmp_path)
    gave_up = [s for s in SEQUENCES if s[0].startswith("3 replans")][0]
    list(run_sequence(cl, fake, *gave_up[1:]))
    assert cl["state"]["stop_reason"] == "vls_gave_up"


def test_replan_swaps_image_1(tmp_path, monkeypatch):
    # Image 1 must be the scene when the current plan was made: the start image, then the replan-time image
    fake = FakeVLM([])
    monkeypatch.setattr(run, "run_call", fake)
    cl = new_cl(tmp_path)
    chunks = [(0, None, 0.0, None, None),
              (10, OPEN, 0.5, None, None),
              (20, CLOSED, 0.5, ans("replan", plan=PLAN_2), None),   # image 12
              (30, OPEN, 0.5, ans("continue"), None)]                # image 13
    list(run_sequence(cl, fake, ans("plan"), chunks))
    assert [images for _, images in fake.sent] == [[1], [1, 12], [12, 13]]


def test_bad_answer_is_a_vlm_error(tmp_path, monkeypatch):
    fake = FakeVLM([])
    monkeypatch.setattr(run, "run_call", fake)
    cl = new_cl(tmp_path)
    fake.answers.append(ans("continue"))  # first call must be a plan
    with pytest.raises(run.VLMCallError, match="first call must answer `plan`"):
        run.start_episode(cl, keypoints=np.zeros((2, 3)))
