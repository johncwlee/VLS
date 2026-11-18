"""Sends one closed-loop prompt to the VLM, saves everything to a call folder, and returns the parsed answer."""
import base64
import json
import os
import time

import cv2

from closed_loop.parse import parse_output
from utils.guidance_utils import load_functions_from_txt
from vlm_query.claude_code_client import query_claude_code
from vlm_query.vlm_agent import VLMCallError


def _image_part(rgb):
    """(H, W, 3) uint8 RGB -> OpenAI-style image part, the format query_claude_code takes."""
    assert rgb.dtype.name == "uint8" and rgb.ndim == 3 and rgb.shape[2] == 3, f"expected (H, W, 3) uint8, got {rgb.shape} {rgb.dtype}"
    ok, png = cv2.imencode(".png", rgb[..., ::-1])
    assert ok, "PNG encoding failed"
    data = base64.b64encode(png.tobytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def run_call(prompt_text, images, call_dir, model, reasoning_effort):
    """
    Args:
        prompt_text: from closed_loop.prompt.build_prompt.
        images: list of (H, W, 3) uint8 RGB images, in the order the prompt describes them.
        call_dir: folder for this call's files (created; must not exist yet, so no run is overwritten).
    Returns:
        parse_output's dict, plus "guidance_fns" (loaded and checked on a dummy trajectory, or [] when no function)
        and "seconds" (VLM wall time).
    Raises VLMCallError if the VLM call fails, ParseError if the answer breaks the format,
    ValueError if the guidance function fails the dummy-trajectory check.
    """
    os.makedirs(call_dir, exist_ok=False)
    with open(os.path.join(call_dir, "prompt.txt"), "w") as f:
        f.write(prompt_text)
    for i, rgb in enumerate(images, start=1):
        cv2.imwrite(os.path.join(call_dir, f"image_{i}.png"), rgb[..., ::-1])

    parts = [{"type": "text", "text": prompt_text}] + [_image_part(rgb) for rgb in images]
    start = time.time()
    try:
        output = query_claude_code(parts, model, reasoning_effort)
    except Exception as e:
        raise VLMCallError(f"claude_code closed-loop query to {model} failed: {e}") from e
    seconds = time.time() - start
    if not output.strip():
        raise VLMCallError(f"claude_code closed-loop query to {model} returned no text")
    with open(os.path.join(call_dir, "output_raw.txt"), "w") as f:
        f.write(output)

    parsed = parse_output(output)

    guidance_fns = []
    if parsed["function_source"] is not None:
        function_path = os.path.join(call_dir, f"{parsed['function_name']}.txt")
        with open(function_path, "w") as f:
            f.write(parsed["function_source"] + "\n")
        # Runs the function on a dummy trajectory and raises if it crashes or returns a non-tensor
        guidance_fns = load_functions_from_txt(function_path, validate=True)

    saved = {k: v for k, v in parsed.items() if k != "function_source"}
    saved["model"] = model
    saved["reasoning_effort"] = reasoning_effort
    saved["seconds"] = round(seconds, 1)
    with open(os.path.join(call_dir, "parsed.json"), "w") as f:
        json.dump(saved, f, indent=2)

    parsed["guidance_fns"] = guidance_fns
    parsed["seconds"] = seconds
    return parsed
