"""Offline demo of the full pipeline. No credentials, no network, no cost.

    python demo.py              # canned replies, exercises every stage
    python demo.py --live       # real Claude (subscription or API key)
    python demo.py --live --backend agent_sdk "check the north field"
"""

from __future__ import annotations

import argparse
import json
import sys

from smartfarm.llm.llm_provider import OfflineProvider, get_llm_provider
from smartfarm.orchestrator import InspectionOrchestrator
from smartfarm.planning.field_registry import RobotState

# Four scenarios that exercise the stages the original pipeline lacked.
SCENARIOS = [
    ("Check the north field for powdery mildew on tomatoes",
     {"field_ids": ["north field"], "crops": ["tomato"],
      "inspection_types": ["disease_detection"], "urgency": "routine",
      "confidence": 0.95, "clarification_needed": None}),

    ("Check the tomatoes for blight",
     {"field_ids": [], "crops": ["tomato"],
      "inspection_types": ["disease_detection"], "urgency": "routine",
      "confidence": 0.8, "clarification_needed": None}),

    ("Inspect the west orchard",          # field that does not exist
     {"field_ids": ["west orchard"], "crops": [],
      "inspection_types": ["quick_scan"], "urgency": "routine",
      "confidence": 0.4, "clarification_needed": None}),

    ("Urgent full-farm scan for crop stress",   # too big for the battery
     {"field_ids": ["north_field", "central_field", "south_field"],
      "crops": [], "inspection_types": ["crop_stress"], "urgency": "urgent",
      "confidence": 0.85, "clarification_needed": None}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("request", nargs="?")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--backend", choices=["agent_sdk", "api", "offline"])
    ap.add_argument("--battery", type=float, default=0.95)
    args = ap.parse_args()

    if args.live:
        provider = get_llm_provider(args.backend)
        billing = ("your Claude subscription"
                   if getattr(provider, "uses_subscription", False)
                   else "API credits")
        print(f"[live] backend={provider.name} -> billed to {billing}")
        o = InspectionOrchestrator(provider=provider)
        req = args.request or SCENARIOS[0][0]
        print(f"\n{'=' * 72}\n{req}\n{'=' * 72}")
        o.display_result(o.process_request(req, RobotState(battery_soc=args.battery)))
        return 0

    print("[offline] canned replies -- no credentials, no network, no cost")
    for req, canned in SCENARIOS:
        soc = 0.30 if "full-farm" in req else args.battery
        o = InspectionOrchestrator(provider=OfflineProvider([json.dumps(canned)]))
        print(f"\n{'=' * 72}\n{req}\n{'=' * 72}")
        o.display_result(o.process_request(req, RobotState(battery_soc=soc)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
