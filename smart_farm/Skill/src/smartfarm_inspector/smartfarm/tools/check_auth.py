"""Report which Claude auth method is active, without spending anything.

    python -m smartfarm.tools.check_auth

The failure mode this exists to prevent: ANTHROPIC_API_KEY set somewhere in
your shell profile silently overrides the Claude Code OAuth token, so you get
billed per token while believing you are on your subscription. Nothing warns
you; the calls just work.
"""

from __future__ import annotations

import os
import shutil
import sys

from ..llm.llm_provider import DEFAULT_MODEL, SubscriptionProvider


def main() -> int:
    print("Claude auth check\n" + "-" * 40)

    key = os.environ.get("ANTHROPIC_API_KEY")
    print(f"ANTHROPIC_API_KEY : {'set (' + key[:12] + '...)' if key else 'not set'}")

    cli = shutil.which("claude")
    print(f"claude CLI        : {cli or 'not found on PATH'}")

    creds = os.path.expanduser("~/.claude/.credentials.json")
    print(f"OAuth credentials : {'present' if os.path.exists(creds) else 'absent'}")

    try:
        import claude_agent_sdk  # noqa: F401
        sdk = "installed"
    except ImportError:
        sdk = "not installed (pip install claude-agent-sdk)"
    print(f"claude-agent-sdk  : {sdk}")

    print(f"model             : {os.environ.get('DEPOT_PLANNER_MODEL', DEFAULT_MODEL)}")
    print(f"SMARTFARM_BACKEND: {os.environ.get('SMARTFARM_BACKEND', 'unset (autodetect)')}")

    ok, reason = SubscriptionProvider.check_auth()
    print("-" * 40)
    print(f"verdict: {reason}")

    if key and os.path.exists(creds):
        print("\nWARNING: both are present. The API key wins and you will be")
        print("billed per token. Run `unset ANTHROPIC_API_KEY` to use the")
        print("subscription instead.")
        return 2
    if not ok and not key:
        print("\nNo working credentials. Either:")
        print("  claude login                      (subscription, no API cost)")
        print("  export ANTHROPIC_API_KEY=sk-ant-  (pay-as-you-go)")
        print("Without either, the planner falls back to the offline backend.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
