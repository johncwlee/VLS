"""
Query Claude models through Claude Code (claude-agent-sdk, local Claude Code login) or the Messages API (ANTHROPIC_API_KEY).

Takes OpenAI-style content parts so it can stand in for the OpenAI chat calls.
"""

import asyncio
import logging
import os

import claude_agent_sdk as sdk

log = logging.getLogger(__name__)


def _to_claude_block(part: dict) -> dict:
    if part["type"] == "text":
        return {"type": "text", "text": part["text"]}
    # OpenAI image part: data:<media_type>;base64,<data>
    header, data = part["image_url"]["url"].split(",", 1)
    media_type = header[len("data:"):].split(";")[0]
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def query_claude_code(parts: list, model: str, reasoning_effort: str | None = None) -> str:
    """Send one user message to Claude via Claude Code and return the reply text.

    reasoning_effort: low/medium/high/xhigh/max, "none" to disable thinking, or None for the default.
    """
    options = sdk.ClaudeAgentOptions(
        model=model,
        tools=[],  # plain completion: no file, shell or web access
        max_turns=1,
        setting_sources=[],  # ignore user/project .claude configuration
        strict_mcp_config=True,
        thinking={"type": "disabled"} if reasoning_effort == "none" else None,
        effort=reasoning_effort if reasoning_effort not in (None, "none") else None,
        # An ANTHROPIC_API_KEY (e.g. from a sourced .env) would override the login and bill the API; the
        # 'anthropic' provider is the API path
        env={"ANTHROPIC_API_KEY": ""},
    )

    async def prompt():
        content = [_to_claude_block(part) for part in parts]
        yield {"type": "user", "message": {"role": "user", "content": content}}

    async def run():
        result = None
        async for message in sdk.query(prompt=prompt(), options=options):
            if isinstance(message, sdk.ResultMessage):
                result = message
        return result

    result = asyncio.run(run())
    if result is None or result.is_error:
        detail = result.result or result.subtype if result is not None else "no result message"
        raise RuntimeError(f"Claude Code query failed: {detail}")
    return result.result or ""


# USD per million tokens (input, output); cache writes and reads are not priced here because each call is uncached.
_PRICES = {"claude-opus-5-5": (4.0, 20.0)}
# Process-wide token and cost totals across Anthropic API calls (read them to cost an episode or a run).
api_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
# USD cap on API spend for this process (main.py sets it from main.api_budget_usd); once spend reaches it, the
# 'anthropic' provider falls back to Claude Code (subscription). None = no cap.
api_budget_usd = None


def query_anthropic_api(parts: list, model: str, reasoning_effort: str | None = None, max_tokens: int = 8000) -> str:
    """Send one user message to the Claude Messages API (ANTHROPIC_API_KEY) and return the reply text.

    reasoning_effort: low/medium/high/xhigh/max (adaptive thinking), "none" to disable thinking, or None for the default.
    """
    import httpx

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (needed by the anthropic VLM provider)")
    body = {"model": model, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [_to_claude_block(p) for p in parts]}]}
    if reasoning_effort == "none":
        body["thinking"] = {"type": "disabled"}
    elif reasoning_effort is not None:
        body["thinking"] = {"type": "adaptive"}
        body["output_config"] = {"effort": reasoning_effort}
    response = httpx.post("https://api.anthropic.com/v1/messages", json=body, timeout=600,
                          headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"})
    if response.status_code != 200:
        raise RuntimeError(f"Anthropic API returned HTTP {response.status_code}: {response.text[:500]}")
    data = response.json()

    usage = data["usage"]
    in_price, out_price = _PRICES.get(model, (0.0, 0.0))
    cost = (usage["input_tokens"] * in_price + usage["output_tokens"] * out_price) / 1e6
    api_usage["calls"] += 1
    api_usage["input_tokens"] += usage["input_tokens"]
    api_usage["output_tokens"] += usage["output_tokens"]
    api_usage["cost_usd"] += cost
    log.info(f"[Anthropic API] {model} in={usage['input_tokens']} out={usage['output_tokens']} "
             f"(thinking={usage.get('output_tokens_details', {}).get('thinking_tokens', 0)}) "
             f"cost=${cost:.4f} | total calls={api_usage['calls']} cost=${api_usage['cost_usd']:.4f}")
    if data.get("stop_reason") == "max_tokens":
        log.warning(f"[Anthropic API] reply hit max_tokens={max_tokens}")
    return "".join(block["text"] for block in data["content"] if block["type"] == "text")


def query_claude(provider: str, parts: list, model: str, reasoning_effort: str | None = None,
                 max_tokens: int = 8000) -> str:
    """Route to Claude Code (provider 'claude_code') or the Messages API (provider 'anthropic').

    Past api_budget_usd, 'anthropic' calls go to Claude Code instead; the check runs before each call, so spend can
    end one call over the cap.
    """
    if provider == "anthropic":
        if api_budget_usd is None or api_usage["cost_usd"] < api_budget_usd:
            return query_anthropic_api(parts, model, reasoning_effort, max_tokens)
        if not api_usage.get("budget_hit"):
            api_usage["budget_hit"] = True
            log.warning(f"[Anthropic API] budget ${api_budget_usd:.2f} reached (spent ${api_usage['cost_usd']:.4f}); "
                        "using Claude Code (subscription) for the rest of the run")
    return query_claude_code(parts, model, reasoning_effort)
