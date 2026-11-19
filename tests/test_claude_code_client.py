"""API budget fallback in vlm_query.claude_code_client.query_claude."""

import pytest

from vlm_query import claude_code_client as client


@pytest.fixture
def calls(monkeypatch):
    """Stub both backends; each API call costs $1. Returns the list of backends hit, in order."""
    hits = []

    def fake_api(*args):
        client.api_usage["cost_usd"] += 1.0
        hits.append("api")
        return "api"

    monkeypatch.setattr(client, "query_anthropic_api", fake_api)
    monkeypatch.setattr(client, "query_claude_code", lambda *args: hits.append("claude_code") or "claude_code")
    monkeypatch.setattr(client, "api_usage", {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
    return hits


def test_anthropic_falls_back_to_claude_code_once_budget_spent(calls, monkeypatch):
    monkeypatch.setattr(client, "api_budget_usd", 2.0)
    for _ in range(4):
        client.query_claude("anthropic", [], "m")
    assert calls == ["api", "api", "claude_code", "claude_code"]


def test_no_budget_means_no_cap(calls, monkeypatch):
    monkeypatch.setattr(client, "api_budget_usd", None)
    for _ in range(3):
        client.query_claude("anthropic", [], "m")
    assert calls == ["api"] * 3


def test_claude_code_provider_ignores_budget(calls, monkeypatch):
    monkeypatch.setattr(client, "api_budget_usd", 0.0)
    client.query_claude("claude_code", [], "m")
    assert calls == ["claude_code"]
