# Prints each episode-state sequence step by step: the answer fed in, the state after it, what we expected, match or not.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# tests/ has no __init__.py and another installed package is named "tests", so import the file directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from closed_loop.episode_state import new_episode_state, trigger_text
from closed_loop.parse import ParseError
from test_closed_loop_episode_state import SEQUENCES, run_step, summarize


def short(summary):
    return "  ".join(f"{k}={v}" for k, v in summary.items())


matches = 0
total = 0
for name, query_limit, steps in SEQUENCES:
    print(f"SEQUENCE: {name}  (query limit {query_limit})")
    state = new_episode_state(query_limit)
    for i, (answer, time_limit, expected) in enumerate(steps):
        total += 1
        kind = "time limit" if time_limit else "gripper"
        print(f"  step {i}: {kind:10s} answer={answer['verdict']} stage={answer['current_stage']}")
        print(f"    TRIGGER:  {trigger_text(state, kind)}")
        try:
            run_step(state, i, answer, time_limit)
            got = short(summarize(state))
        except ParseError as e:
            got = f"ERROR: {e}"
        if isinstance(expected, str):
            want = f"ERROR containing {expected!r}"
            ok = expected in got
        else:
            want = short(expected)
            ok = got == want
        matches += ok
        print(f"    STATE:    {got}")
        print(f"    EXPECTED: {want}")
        print(f"    RESULT:   {'match' if ok else 'MISMATCH'}")
    print()

print(f"{matches} of {total} steps match")
