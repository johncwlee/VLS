# Overlays which RPent tool (VLA vs scripted primitive) produced each frame of episode.mp4, plus planner cards between motions.
# Usage (.venv-rpent): python scratch/2026-11-26_rpent_annotate_video.py EPISODE_DIR
import glob
import json
import os
import sys
import textwrap

import cv2
import imageio.v2 as imageio
import numpy as np

VLA_TOOLS = {"pi0_pick", "pi0_doubled"}
FPS = 20  # RPent saves episode.mp4 at 20 fps (robots/libero/toolkit.py:229)
SCALE = 3  # 256 px frames -> 768 px so the text is readable
BANNER_H = 150
CARD_FRAMES = 20  # 1 s planner card before each motion primitive
ORANGE = (255, 150, 0)
BLUE = (80, 170, 255)
WHITE = (255, 255, 255)
GREY = (170, 170, 170)


def draw_lines(image, lines, top, color_per_line, scale=0.6):
    y = top
    for text, color in zip(lines, color_per_line):
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += int(28 * scale / 0.6)
    return image


def with_banner(frame, lines, colors):
    big = cv2.resize(frame, (frame.shape[1] * SCALE, frame.shape[0] * SCALE), interpolation=cv2.INTER_LINEAR)
    banner = np.zeros((BANNER_H, big.shape[1], 3), dtype=np.uint8)
    draw_lines(banner, lines, 28, colors)
    return np.concatenate([big, banner], axis=0)


def wrap(text, width=68):
    return textwrap.wrap(text, width) or [""]


def short_input(call):
    return call["input"].replace('"', "")


def main(episode_dir):
    video_paths = glob.glob(os.path.join(episode_dir, "**", "episode.mp4"), recursive=True)
    assert len(video_paths) == 1, f"expected one episode.mp4 under {episode_dir}, found {video_paths}"
    frames = [np.asarray(f) for f in imageio.mimread(video_paths[0], memtest=False)]
    with open(os.path.join(episode_dir, "tool_log.jsonl")) as f:
        calls = [json.loads(line) for line in f]
    with open(os.path.join(episode_dir, "states.json")) as f:
        steps = json.load(f)["steps"]
    # After a motion the tool result is replaced by the env-state view; the primitive's own result is in states.json
    primitive_result_by_step = {step["step_idx"]: (step.get("result") or {}) for step in steps}
    transcripts = glob.glob(os.path.join(episode_dir, "**", "transcript_*.json"), recursive=True)
    outcome = "no transcript found"
    if transcripts:
        with open(transcripts[0]) as f:
            transcript = json.load(f)
        finish = transcript.get("finish") or {}
        # transcript environment_success is None in claude_code runs; LIBERO's goal check is states.json 'terminated'
        libero_success = any(step["terminated"] for step in steps)
        outcome = (f"LIBERO success={libero_success}  agent finish={finish.get('status')}  "
                   f"error={transcript.get('error')}")

    motion_calls = [c for c in calls if c["frame_start"] is not None and c["frame_end"] > c["frame_start"]]
    covered = sum(c["frame_end"] - c["frame_start"] for c in motion_calls)
    print(f"{len(frames)} video frames, {len(calls)} tool calls, {len(motion_calls)} with motion covering {covered} frames")

    out_frames = []
    pending_planner_calls = []
    motion_number = 0
    for call in calls:
        is_motion = call in motion_calls
        if not is_motion:
            pending_planner_calls.append(call)
            continue
        motion_number += 1
        is_vla = call["tool"] in VLA_TOOLS
        kind = "VLA (pi0.5)" if is_vla else "SCRIPTED"
        color = ORANGE if is_vla else BLUE
        start, end = call["frame_start"], min(call["frame_end"], len(frames))

        # Planner card: what the agent looked at / decided since the last motion
        card_lines = [f"PLANNER  (before motion #{motion_number})"]
        card_colors = [WHITE]
        names = [p["tool"] for p in pending_planner_calls]
        card_lines += wrap("calls without motion: " + (", ".join(names) if names else "none"))[:2]
        card_colors += [GREY] * (len(card_lines) - 1)
        card_lines += wrap(f"next: {kind} {call['tool']} {short_input(call)}")[:2]
        card_colors += [color] * (len(card_lines) - len(card_colors))
        dim = (frames[min(start, len(frames) - 1)] * 0.35).astype(np.uint8)
        card = with_banner(dim, card_lines, card_colors)
        out_frames += [card] * CARD_FRAMES
        pending_planner_calls = []

        primitive_result = primitive_result_by_step.get(call["result"].get("step"), {})
        result = {k: v for k, v in primitive_result.items()
                  if k in ("chunks_used", "steps_used", "steps", "terminated", "error", "reason")}
        for i in range(start, end):
            lines = [f"#{motion_number}  {kind}: {call['tool']}"]
            lines += wrap(short_input(call))[:2]
            lines.append(f"frame {i}/{len(frames)}  step {i - start + 1}/{end - start}  result {json.dumps(result)[:70]}")
            colors = [color] + [WHITE] * (len(lines) - 2) + [GREY]
            out_frames.append(with_banner(frames[i], lines, colors))

    uncovered = len(frames) - covered
    if uncovered != 0:
        print(f"WARNING: {uncovered} video frames not matched to a tool call")
    final_lines = ["END"] + wrap(outcome)[:2] + [f"tool calls {len(calls)}, motion primitives {len(motion_calls)}"]
    out_frames += [with_banner(frames[-1], final_lines, [WHITE] * len(final_lines))] * (2 * FPS)

    out_path = os.path.join(episode_dir, "episode_annotated.mp4")
    imageio.mimwrite(out_path, out_frames, fps=FPS, macro_block_size=1)
    print(f"wrote {out_path} ({len(out_frames)} frames)")
    print(outcome)


if __name__ == "__main__":
    main(sys.argv[1])
