"""
Query Claude models through Claude Code (claude-agent-sdk), using the local Claude Code login.

Takes OpenAI-style content parts so it can stand in for the OpenAI chat calls.
"""

import asyncio

import claude_agent_sdk as sdk


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
