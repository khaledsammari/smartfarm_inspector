"""LLM provider.

Extends the original ClaudeProvider with two things it needed to be usable:

  * a subscription backend, so the project runs on a Claude Pro/Max plan
    through Claude Code's OAuth login rather than consuming API credits;
  * an offline backend, so the whole pipeline and the ROS wiring can be
    tested with no credentials and no network.

The `call()` signature is unchanged, so nothing downstream had to be
rewritten.

Backend choice: SMARTFARM_BACKEND=agent_sdk|api|offline, or autodetect
(subscription first, then API key, then offline).

Billing note for the business case: cost the LLM at API rates per mission
even when running on a subscription. "Free because it is on a subscription"
is not what per-mission cost looks like at customer scale, and the token
counts logged per mission give you the real number.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = os.environ.get("SMARTFARM_MODEL", "claude-sonnet-5")


@dataclass
class LLMResponse:
    """Standardized LLM response. Unchanged from the original."""

    content: str
    stop_reason: str = "end_turn"
    usage: dict = field(default_factory=lambda: {"input_tokens": 0,
                                                 "output_tokens": 0})


def extract_json(text: str) -> dict | None:
    """Pull a JSON object out of a model response.

    The original used a bare json.loads, so any markdown fence or stray
    sentence made the whole parse fail and returned an error to the farmer.
    Models add fences often enough that this must be tolerated -- the
    validator downstream is the real gate, so a permissive read here costs
    nothing.
    """
    if not text:
        return None
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


class ClaudeProvider:
    """Claude Developer Platform. Requires ANTHROPIC_API_KEY. Billed per token."""

    name = "api"
    uses_subscription = False

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 client=None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key and client is None:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Either export it, or use the "
                "subscription backend: SMARTFARM_BACKEND=agent_sdk")
        self.model = model
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def call(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        resp = self.client.messages.create(**kwargs)
        return LLMResponse(
            content="".join(getattr(b, "text", "") for b in resp.content),
            stop_reason=resp.stop_reason,
            usage={"input_tokens": resp.usage.input_tokens,
                   "output_tokens": resp.usage.output_tokens},
        )


class SubscriptionProvider:
    """Claude Agent SDK over Claude Code's OAuth login -- no API credits.

    Setup:
        npm install -g @anthropic-ai/claude-code
        claude login
        pip install claude-agent-sdk
        unset ANTHROPIC_API_KEY

    That last line matters. If ANTHROPIC_API_KEY is set anywhere in the
    environment it silently wins over the OAuth token, and you are billed per
    token while believing you are on the subscription. Nothing warns you, so
    this class raises rather than let it happen quietly.

    The OAuth token is licensed for individual use. Right for a demo and for
    development; wrong for a multi-user or scheduled production deployment,
    which should use ClaudeProvider with an API key.
    """

    name = "agent_sdk"
    uses_subscription = True

    def __init__(self, model: str = DEFAULT_MODEL,
                 allow_api_key_fallback: bool = False):
        if os.environ.get("ANTHROPIC_API_KEY") and not allow_api_key_fallback:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is set and would override the Claude Code "
                "OAuth token, billing this run to the API account instead of "
                "your subscription. Run `unset ANTHROPIC_API_KEY` first.")
        self.model = model

    @staticmethod
    def check_auth() -> tuple[bool, str]:
        if os.environ.get("ANTHROPIC_API_KEY"):
            return False, "ANTHROPIC_API_KEY is set: API billing would be used"
        creds = os.path.expanduser("~/.claude/.credentials.json")
        if os.path.exists(creds):
            return True, "Claude Code OAuth found: subscription billing"
        return False, "no OAuth credentials; run `claude login`"

    def call(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        import asyncio

        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            TextBlock,
            query,
        )

        async def run() -> str:
            chunks: list[str] = []
            async for msg in query(
                prompt=prompt,
                options=ClaudeAgentOptions(
                    system_prompt=system_prompt or "",
                    model=self.model,
                    # No tools. The planner returns text; the ROS layer acts.
                    allowed_tools=[],
                    max_turns=1,
                ),
            ):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            chunks.append(block.text)
            return "".join(chunks)

        return LLMResponse(content=asyncio.run(run()))


class OfflineProvider:
    """Canned responses. No network, no credentials, no cost.

    Used by the test suite and by the ROS node's dry-run mode, so the whole
    pipeline including mission publication can be exercised before spending
    a single token.
    """

    name = "offline"
    uses_subscription = False

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or [json.dumps({
            "field_ids": ["north_field"],
            "crops": ["tomato"],
            "inspection_types": ["disease_detection"],
            "urgency": "routine",
            "confidence": 0.9,
            "clarification_needed": None,
        })]
        self.calls = 0

    def call(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return LLMResponse(content=r)


def get_llm_provider(provider_name: str | None = None, **kwargs):
    """Factory. Kept for compatibility with the original call sites.

    provider_name of None (or "auto") autodetects: subscription, then API
    key, then offline.
    """
    name = (provider_name or os.environ.get("SMARTFARM_BACKEND") or "auto").lower()

    if name in ("claude", "api"):
        return ClaudeProvider(**kwargs)
    if name in ("agent_sdk", "subscription"):
        return SubscriptionProvider(**{k: v for k, v in kwargs.items()
                                       if k in ("model", "allow_api_key_fallback")})
    if name == "offline":
        return OfflineProvider(**{k: v for k, v in kwargs.items()
                                  if k == "responses"})
    if name != "auto":
        raise ValueError(f"Unknown provider: {provider_name}")

    model = kwargs.get("model", DEFAULT_MODEL)
    ok, _ = SubscriptionProvider.check_auth()
    if ok:
        return SubscriptionProvider(model=model)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ClaudeProvider(model=model)
    return OfflineProvider()
