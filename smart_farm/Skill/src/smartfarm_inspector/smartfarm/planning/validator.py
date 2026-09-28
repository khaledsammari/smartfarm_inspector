"""Task validator: the deterministic gate between the LLM and the robot.

No LLM calls, no network. The original pipeline had no equivalent stage -- a
parsed request went straight to task generation and would have gone straight
to the robot. That is the single biggest gap for a jury question about what
happens when the model produces something wrong.

Errors reject the task. Warnings execute but are surfaced to the operator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .field_registry import FieldMap, RobotState

ENERGY_MARGIN_LIMIT = 0.90
ENERGY_WARN_LIMIT = 0.70

@dataclass
class Issue:
    severity: str      # "error" | "warning"
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.code}: {self.message}"


@dataclass
class ValidationResult:
    ok: bool
    issues: list = field(default_factory=list)

    @property
    def errors(self) -> list:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        return "\n".join(f"  {i}" for i in self.issues) or "  no issues"


class TaskValidator:
    def __init__(self, field_map: FieldMap | None = None):
        self.map = field_map or FieldMap()

    def validate(self, task, state: RobotState | None = None) -> ValidationResult:
        state = state or RobotState()
        issues: list[Issue] = []

        # --- structure ---------------------------------------------------
        if not task.waypoints:
            issues.append(Issue("error", "NO_WAYPOINTS",
                                "task has no waypoints; nothing to execute"))

        if task.map_name != self.map.map_name:
            issues.append(Issue(
                "error", "MAP_MISMATCH",
                f"task targets map '{task.map_name}' but the field map is "
                f"'{self.map.map_name}'"))

        for row in task.rows:
            if row not in self.map.rows:
                issues.append(Issue("error", "UNKNOWN_ROW",
                                    f"row '{row}' is not on this farm"))

        # --- geometry ----------------------------------------------------
        for wp in task.waypoints:
            pt = (wp["x"], wp["y"])
            if not self.map.in_geofence(pt):
                issues.append(Issue(
                    "error", "GEOFENCE_VIOLATION",
                    f"{wp['waypoint_id']} at ({pt[0]:.1f}, {pt[1]:.1f}) is "
                    "outside the operating envelope"))
            zone = self.map.violated_no_go(pt)
            if zone:
                issues.append(Issue(
                    "error", "NO_GO_VIOLATION",
                    f"{wp['waypoint_id']} at ({pt[0]:.1f}, {pt[1]:.1f}) is "
                    f"inside no-go zone '{zone}'"))

        # --- size --------------------------------------------------------
        max_goals = int(self.map.limits.get("max_goals_per_mission", 40))
        if len(task.waypoints) > max_goals:
            issues.append(Issue(
                "error", "MISSION_TOO_LARGE",
                f"{len(task.waypoints)} waypoints exceeds the limit of "
                f"{max_goals}. Increase the capture interval or inspect "
                "fewer fields."))

        # --- speed -------------------------------------------------------
        cap = float(self.map.robot_cfg["max_speed_mps"])
        commanded = task.robot_config.get("navigation", {}).get("max_speed_mps", 0)
        if commanded > cap:
            issues.append(Issue(
                "error", "SPEED_LIMIT",
                f"commanded speed {commanded} m/s exceeds the robot cap of "
                f"{cap} m/s"))
        elif commanded > 0.4 and "disease_detection" in task.inspection_types:
            issues.append(Issue(
                "warning", "SPEED_QUALITY",
                f"{commanded} m/s may blur imagery; <=0.4 recommended for "
                "disease detection"))

        # --- robot state -------------------------------------------------
        if state.pose_covariance >= 0.5:
            issues.append(Issue("error", "NOT_LOCALIZED",
                                f"pose covariance {state.pose_covariance} >= 0.5"))
        if not state.camera_stream_active:
            issues.append(Issue("error", "CAMERA_DOWN",
                                "camera stream inactive; imagery is the whole "
                                "point of this mission"))
        if state.free_storage_mb <= 200:
            issues.append(Issue("error", "NO_STORAGE",
                                f"only {state.free_storage_mb:.0f} MB free"))

        # --- energy ------------------------------------------------------
        cfg = self.map.robot_cfg
        capacity = float(cfg["battery_capacity_wh"])
        reserve = float(cfg["reserve_soc"]) * capacity
        usable = state.remaining_energy_wh(capacity) - reserve
        if task.estimated_energy_wh > usable:
            issues.append(Issue(
                "error", "ENERGY_BUDGET",
                f"task needs ~{task.estimated_energy_wh} Wh but only "
                f"{usable:.1f} Wh is usable above the "
                f"{float(cfg['reserve_soc']):.0%} reserve"))
        elif usable > 0 and task.estimated_energy_wh > ENERGY_MARGIN_LIMIT * usable:
            issues.append(Issue(
                "error", "ENERGY_MARGIN",
                f"task uses ~{task.estimated_energy_wh / usable:.0%} of usable "
                f"energy, over the {ENERGY_MARGIN_LIMIT:.0%} limit. The energy "
                "model is an uncalibrated estimate and leaves no room for a "
                "replan. Split this into separate missions, one field at a time."))
        elif usable > 0 and task.estimated_energy_wh > ENERGY_WARN_LIMIT * usable:
            issues.append(Issue(
                "warning", "ENERGY_TIGHT",
                f"task uses ~{task.estimated_energy_wh / usable:.0%} of "
                "usable energy"))

        # --- safety config sanity ---------------------------------------
        if not task.safety_config.get("obstacle_avoidance"):
            issues.append(Issue("error", "SAFETY_DISABLED",
                                "obstacle avoidance must not be disabled"))
        if task.safety_config.get("min_worker_distance_m", 0) < 2.0:
            issues.append(Issue(
                "error", "WORKER_DISTANCE",
                "minimum worker distance below 2.0 m is not permitted"))

        return ValidationResult(
            ok=not any(i.severity == "error" for i in issues), issues=issues)
