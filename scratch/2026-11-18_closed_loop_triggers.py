# Prints each closed-loop trigger sequence chunk by chunk (fake VLM): what fired, the stage after, what we expected.
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# tests/ has no __init__.py and another installed package is named "tests", so import the file directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
import closed_loop.run as run
from test_closed_loop_run import SEQUENCES, FakeVLM, new_cl, run_sequence

matches = 0
total = 0
for name, first_answer, chunks in SEQUENCES:
    print(f"SEQUENCE: {name}")
    fake = FakeVLM([])
    run.run_call = fake
    with tempfile.TemporaryDirectory() as tmp:
        cl = new_cl(tmp)
        inputs = {c[0]: c for c in chunks}
        for step, expected, got in run_sequence(cl, fake, first_answer, chunks):
            total += 1
            _, gripper_open, reward, answer, _ = inputs[step]
            gripper = {None: "-", True: "open", False: "closed"}[gripper_open]
            vlm = answer["verdict"] if answer is not None else "-"
            ok = got == expected
            matches += ok
            print(f"  step {step:3d}  gripper={gripper:6s} reward={reward:.2f}  fake VLM answer={vlm:8s}")
            print(f"    GOT:      trigger={got[0]}  stage={got[1]}  stage_changed={got[2]}")
            print(f"    EXPECTED: trigger={expected[0]}  stage={expected[1]}  stage_changed={expected[2]}")
            print(f"    RESULT:   {'match' if ok else 'MISMATCH'}")
    print(f"  stop_reason={cl['state']['stop_reason']}  calls={cl['state']['query_count']}\n")

print(f"{matches} of {total} chunks match")
