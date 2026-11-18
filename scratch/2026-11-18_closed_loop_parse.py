# Prints each closed-loop parser case: the answer fed in, what the parser read, what we expected, and match or not.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# tests/ has no __init__.py and another installed package is named "tests", so import the file directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from closed_loop.parse import ParseError, parse_output
from test_closed_loop_parse import BROKEN_CASES, GOOD_CASES, OLD_ANSWER_PATH, summarize


def answer_fields(text):
    """The answer's top-level fields and function names, on one line."""
    kept = []
    for line in text.split("\n"):
        if line.startswith(("verdict", "completed_stages", "undone_stages", "current_stage", "guidance", "def ")):
            kept.append(line.split("(")[0] if line.startswith("def ") else line)
    return " / ".join(kept)


def short(summary):
    return "  ".join(f"{k}={v}" for k, v in summary.items())


matches = 0
for name, text, expected in GOOD_CASES:
    read = summarize(parse_output(text))
    ok = read == expected
    matches += ok
    print(f"CASE:        {name}")
    print(f"ANSWER:      {answer_fields(text)}")
    print(f"PARSER READ: {short(read)}")
    print(f"EXPECTED:    {short(expected)}")
    print(f"RESULT:      {'match' if ok else 'MISMATCH'}\n")

with open(OLD_ANSWER_PATH) as f:
    old_answer = f.read()
broken = BROKEN_CASES + [("real old-format answer (episode 3)", old_answer, "no `verdict")]
for name, text, error_words in broken:
    print(f"CASE:        {name}  (should raise an error)")
    print(f"ANSWER:      {answer_fields(text)}")
    try:
        read = summarize(parse_output(text))
        print(f"PARSER READ: {short(read)}")
        print("RESULT:      NO ERROR (wrong)\n")
    except ParseError as e:
        ok = error_words in str(e)
        matches += ok
        print(f"ERROR:       {e}")
        print(f"EXPECTED:    error containing {error_words!r}")
        print(f"RESULT:      {'match' if ok else 'WRONG ERROR'}\n")

print(f"{matches} of {len(GOOD_CASES) + len(broken)} cases match")
