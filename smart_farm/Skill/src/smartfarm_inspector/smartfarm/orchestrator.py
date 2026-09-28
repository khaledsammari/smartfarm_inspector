"""Main pipeline orchestrator.

Same shape as the original -- parse, generate, display -- with two stages
inserted that the original did not have:

    parse  ->  generate task  ->  VALIDATE  ->  COMPILE to MissionData

Validation is the answer to "what happens when the LLM produces something
wrong". Compilation is what makes the task executable rather than descriptive.
"""

from __future__ import annotations

from dataclasses import asdict

from .llm.llm_parser import LLMRequestParser
from .planning.field_registry import FieldMap, RobotState
from .planning.task_generator import TaskGenerator
from .planning.validator import TaskValidator
from .ros.mission_compiler import MissionCompiler


class InspectionOrchestrator:
    def __init__(self, field_map: FieldMap | None = None, provider=None,
                 config_dir=None):
        self.map = field_map or FieldMap()
        self.parser = LLMRequestParser(field_map=self.map, provider=provider,
                                       config_dir=config_dir)
        self.task_generator = TaskGenerator(field_map=self.map,
                                            config_dir=config_dir)
        self.validator = TaskValidator(field_map=self.map)
        self.compiler = MissionCompiler(field_map=self.map)

    def process_request(self, farmer_request: str,
                        state: RobotState | None = None) -> dict:
        state = state or RobotState()

        parsed = self.parser.parse(farmer_request)
        if parsed.error:
            return {"status": "error", "error": parsed.error,
                    "parsed": asdict(parsed)}
        if parsed.clarification_needed:
            return {"status": "clarification_needed",
                    "clarification": parsed.clarification_needed,
                    "parsed": asdict(parsed)}

        try:
            task = self.task_generator.generate_task(parsed)
        except ValueError as exc:
            return {"status": "error", "error": str(exc),
                    "parsed": asdict(parsed)}

        result = self.validator.validate(task, state)
        if not result.ok:
            return {
                "status": "rejected",
                "parsed": asdict(parsed),
                "task": asdict(task),
                "errors": [str(e) for e in result.errors],
                "warnings": [str(w) for w in result.warnings],
            }

        mission = self.compiler.compile(task)
        return {
            "status": "success",
            "parsed": asdict(parsed),
            "task": asdict(task),
            "mission": mission,
            "warnings": [str(w) for w in result.warnings],
            "task_json": self.task_generator.task_to_json(task),
        }

    def display_result(self, result: dict) -> None:
        status = result.get("status", "unknown")
        print(f"\nStatus: {status.upper()}")

        if status == "clarification_needed":
            print(f"\n  Clarification needed: {result['clarification']}")
            return
        if status == "error":
            print(f"\n  Error: {result.get('error')}")
            return

        p = result["parsed"]
        print(f"\nPARSED REQUEST")
        print(f"  Fields      : {p['field_ids']}")
        print(f"  Crops       : {p['crops']}")
        print(f"  Inspections : {p['inspection_types']}")
        print(f"  Urgency     : {p['urgency']}")
        print(f"  Confidence  : {p['confidence']:.2f}")
        if p.get("dropped_crops"):
            print(f"  Ignored     : {p['dropped_crops']} (not grown here)")

        t = result["task"]
        print(f"\nGENERATED TASK")
        print(f"  Task ID     : {t['task_id']}")
        print(f"  Rows        : {t['rows']}")
        print(f"  Waypoints   : {len(t['waypoints'])}")
        print(f"  Route       : {t['estimated_distance_m']} m / "
              f"{t['estimated_energy_wh']} Wh")
        print(f"  Duration    : {t['duration_estimate_minutes']} min")
        print(f"  Speed       : {t['robot_config']['navigation']['max_speed_mps']} m/s")
        print(f"  Worker dist : {t['safety_config']['min_worker_distance_m']} m")

        if status == "rejected":
            print(f"\nVALIDATION FAILED -- mission not published")
            for e in result["errors"]:
                print(f"  {e}")
            return

        for w in result.get("warnings", []):
            print(f"  {w}")

        m = result["mission"]
        print(f"\nCOMPILED MissionData (published to /mission_dataa)")
        print(f"  {MissionCompiler.summarize(m)}")
        for g in m["goals"][:3]:
            pos = g["goal_pose"]["position"]
            acts = [a["action_type"] for a in g["goal_actions"]]
            print(f"  {g['goal_id']:26s} ({pos['x']:6.2f}, {pos['y']:6.2f})  {acts}")
        if len(m["goals"]) > 3:
            print(f"  ... {len(m['goals']) - 3} more goals")
