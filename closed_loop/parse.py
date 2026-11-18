"""Reads the VLM's answer to a closed-loop prompt (closed_loop/prompt_template.txt)."""
import re

VERDICTS = ("plan", "continue", "advance", "replan", "done")
# Verdicts that start a new stage, so they need a guidance function when guidance = yes
NEEDS_FUNCTION = ("plan", "advance", "replan")


class ParseError(ValueError):
    """The VLM's answer is missing a field or breaks a rule of the output format."""


def _field(output, name):
    """Value of a top-level `name = value` line, without a trailing comment or quotes. None if absent."""
    for line in output.split("\n"):
        # Top-level lines only: indented assignments are locals inside the guidance function
        match = re.match(rf"{name}\s*=\s*(.*)", line)
        if match is None:
            continue
        value = match.group(1)
        if not value.lstrip().startswith('"'):
            value = value.split("#")[0]
        return value.strip().strip('"\'')
    return None


def _required(output, name):
    value = _field(output, name)
    if value is None or value == "" or value == "?":
        raise ParseError(f"no `{name} = ...` line in the answer")
    return value


def _int_list(value, name):
    """'[1, 2]' -> [1, 2]; '[]' -> []."""
    if not (value.startswith("[") and value.endswith("]")):
        raise ParseError(f"`{name}` should be a list like [1, 2], got {value!r}")
    numbers = []
    for item in value[1:-1].split(","):
        item = item.strip()
        if item == "":
            continue
        if not item.isdigit():
            raise ParseError(f"`{name}` should hold stage numbers, got {item!r}")
        numbers.append(int(item))
    return numbers


def _plan(output):
    """'# Stage N: description' lines -> list of descriptions. Stages must be numbered 1, 2, 3, ..."""
    plan = []
    for line in output.split("\n"):
        match = re.match(r"#\s*Stage\s+(\d+):\s*(.*)", line.strip())
        if match is None:
            continue
        number = int(match.group(1))
        if number != len(plan) + 1:
            raise ParseError(f"plan stages must be numbered 1, 2, 3, ...; found stage {number} after {len(plan)} stages")
        plan.append(match.group(2).strip())
    if len(plan) == 0:
        raise ParseError("no plan: expected lines like `# Stage 1: ...`")
    return plan


def _functions(output):
    """{name: source} for each `def name(...)` block, ending at its last `    return` line.

    Same block split as VLMAgent._parse_and_save_guidance_functions (vlm_query/vlm_agent.py:160).
    """
    lines = output.split("\n")
    functions = {}
    name = None
    start = None
    for i, line in enumerate(lines):
        if line.startswith("def "):
            start = i
            name = line.split("(")[0].split("def ")[1]
        if line.startswith("    return ") and name is not None:
            functions[name] = "\n".join(lines[start:i + 1])
    return functions


def parse_output(output):
    """
    Returns a dict:
        verdict: one of VERDICTS
        plan: list of stage descriptions (the plan in the answer, stage 1 first)
        completed_stages, undone_stages: lists of stage numbers in the plan the VLM was checking
        current_stage: stage to work on next, 1-based in `plan`; 0 when verdict is "done"
        guidance: True = steer with the function, False = let the policy act alone
        evidence: one sentence from the VLM
        function_name, function_source: the guidance function for current_stage, or None
    Raises ParseError when a field is missing or a rule is broken.
    """
    verdict = _required(output, "verdict")
    if verdict not in VERDICTS:
        raise ParseError(f"verdict {verdict!r} is not one of {VERDICTS}")

    plan = _plan(output)
    completed_stages = _int_list(_required(output, "completed_stages"), "completed_stages")
    undone_stages = _int_list(_required(output, "undone_stages"), "undone_stages")

    current_stage_text = _required(output, "current_stage")
    if not current_stage_text.isdigit():
        raise ParseError(f"current_stage should be a number, got {current_stage_text!r}")
    current_stage = int(current_stage_text)
    if verdict == "done":
        if current_stage != 0:
            raise ParseError(f"verdict is done, so current_stage should be 0, got {current_stage}")
    elif not 1 <= current_stage <= len(plan):
        raise ParseError(f"current_stage {current_stage} is outside the plan (stages 1..{len(plan)})")

    guidance_text = _required(output, "guidance").lower()
    if guidance_text not in ("yes", "no"):
        raise ParseError(f"guidance should be yes or no, got {guidance_text!r}")
    guidance = guidance_text == "yes"

    evidence = _required(output, "evidence")

    function_name = None
    function_source = None
    if verdict in NEEDS_FUNCTION and guidance:
        functions = _functions(output)
        function_name = f"stage{current_stage}_guidance"
        if function_name not in functions:
            found = list(functions) if functions else "none"
            raise ParseError(f"verdict is {verdict} with guidance = yes, so the answer needs {function_name}; "
                             f"functions found: {found}")
        function_source = functions[function_name]

    return {
        "verdict": verdict,
        "plan": plan,
        "completed_stages": completed_stages,
        "undone_stages": undone_stages,
        "current_stage": current_stage,
        "guidance": guidance,
        "evidence": evidence,
        "function_name": function_name,
        "function_source": function_source,
    }
