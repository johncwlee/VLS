# Runs the `rpent` CLI unchanged, but logs every tool call with the episode-video frame range it covered (for annotated videos).
# Usage (.venv-rpent, RPent dir): RPENT_TOOL_LOG=/path/tool_log.jsonl python <this file> <rpent CLI args...>
import json
import os
import sys
import time

from rpent.cli.main import main as rpent_main
from rpent.tools.toolkit import Toolkit

TOOL_LOG_PATH = os.environ["RPENT_TOOL_LOG"]
ARG_PREVIEW_CHARS = 300
original_execute_tool = Toolkit.execute_tool
call_counter = {"n": 0}


def frame_count(toolkit):
    """Frames in the episode.mp4 buffer so far (one per env step), or None before the env exists."""
    primitives = getattr(toolkit, "_primitives", None)
    if primitives is None or not hasattr(primitives, "recorded_frame_count"):
        return None
    return primitives.recorded_frame_count()


def result_summary(data):
    """Scalar fields of a tool result (drops arrays, images, long text)."""
    summary = {}
    for key, value in data.items():
        if isinstance(value, (bool, int, float)) or value is None:
            summary[key] = value
        elif isinstance(value, str) and len(value) <= 200:
            summary[key] = value
    return summary


def logged_execute_tool(self, name, input_dict):
    call_index = call_counter["n"]
    call_counter["n"] += 1
    frame_start = frame_count(self)
    wall_start_s = time.time()
    result = original_execute_tool(self, name, input_dict)
    record = {
        "call_index": call_index,
        "tool": name,
        "input": json.dumps(input_dict, default=str)[:ARG_PREVIEW_CHARS],
        "frame_start": frame_start,
        "frame_end": frame_count(self),
        "wall_start_s": wall_start_s,
        "duration_s": round(time.time() - wall_start_s, 2),
        "is_error": bool(result.is_error),
        "result": result_summary(result.data or {}),
    }
    with open(TOOL_LOG_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return result


Toolkit.execute_tool = logged_execute_tool

if __name__ == "__main__":
    sys.argv = ["rpent"] + sys.argv[1:]
    sys.exit(rpent_main())
