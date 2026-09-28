"""Generate executable ROS 2 tasks from parsed requests.

The original ROS2Task described *how* to configure the robot (sensors, camera
resolution, lidar range) but never *where to go*. Nothing in it could be
executed: there were no poses. This version keeps every original field and
adds the missing one, `waypoints`, computed from the field map.

Sensor and safety config are preserved so the original design intent survives,
but two values are now taken from the field map rather than the JSON config,
because the JSON values were not achievable on this robot:

  * max_speed_mps was 1.5. The Go2 under this stack's path follower is not
    stable there, and imagery at 1.5 m/s is unusable for disease detection.
    Inspection speed is 0.3 m/s, hard-capped at 0.8.
  * lidar_range_m of 5.0 is a sensor property, not a mission parameter; it is
    reported rather than commanded.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config_paths import config_file
from .field_registry import FieldMap


@dataclass
class Waypoint:
    """One navigable pose with the actions to run on arrival."""

    waypoint_id: str
    x: float
    y: float
    yaw: float
    row_id: str
    field_id: str
    actions: list = field(default_factory=list)


@dataclass
class ROS2Task:
    """ROS 2 compatible task. Original fields preserved; waypoints added."""

    task_id: str
    timestamp: str
    field_id: str
    inspection_types: list
    target_crops: list
    urgency: str
    robot_config: dict
    expected_outputs: list
    duration_estimate_minutes: int
    safety_config: dict
    # --- added: without these the task cannot be executed ---
    waypoints: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    map_name: str = ""
    frame_id: str = "map"
    estimated_distance_m: float = 0.0
    estimated_energy_wh: float = 0.0


class TaskGenerator:
    """Generate executable ROS 2 tasks from parsed requests."""

    def __init__(self, inspection_types_path: str | Path | None = None,
                 field_map: FieldMap | None = None,
                 config_dir: str | Path | None = None):
        if inspection_types_path:
            path = Path(inspection_types_path)
        elif config_dir:
            path = Path(config_dir) / "inspection_types.json"
        else:
            path = config_file("inspection_types.json")
        try:
            self.inspection_db = json.loads(path.read_text())
        except FileNotFoundError:
            print(f"Warning: {path} not found")
            self.inspection_db = {"inspection_types": {}}
        self.map = field_map or FieldMap()

    # ------------------------------------------------------------------
    def generate_task(self, parsed) -> ROS2Task:
        if not parsed.field_ids:
            raise ValueError(
                "cannot generate a task with no resolved field. The parser "
                "should have returned clarification_needed instead.")

        field_id = parsed.field_ids[0]
        rows = []
        for fid in parsed.field_ids:
            rows += self.map.field_rows(fid)

        limits = self.map.limits
        max_rows = int(limits.get("max_rows_per_mission", 6))
        rows = rows[:max_rows]

        interval = self._capture_interval(parsed.urgency)
        waypoints, distance = self._build_waypoints(rows, interval)

        specs = self._get_inspection_specs(parsed.inspection_types)
        energy = distance * float(self.map.robot_cfg["consumption_wh_per_m"])

        return ROS2Task(
            task_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            field_id=field_id,
            inspection_types=parsed.inspection_types,
            target_crops=parsed.crops,
            urgency=parsed.urgency,
            robot_config=self._build_robot_config(specs, parsed.urgency),
            expected_outputs=self._get_expected_outputs(parsed.inspection_types),
            duration_estimate_minutes=self._calculate_duration(
                distance, parsed.inspection_types, parsed.urgency),
            safety_config=self._build_safety_config(parsed.urgency),
            waypoints=[asdict(w) for w in waypoints],
            rows=rows,
            map_name=self.map.map_name,
            frame_id=self.map.frame_id,
            estimated_distance_m=round(distance, 1),
            estimated_energy_wh=round(energy, 1),
        )

    # ------------------------------------------------------------------
    def _capture_interval(self, urgency: str) -> float:
        """Urgent missions trade image density for speed. This is the honest
        place to express urgency -- not by driving faster, which degrades the
        imagery the inspection depends on."""
        default = float(self.map.limits.get("default_capture_interval_m", 4.0))
        return {"urgent": default * 2, "medium": default,
                "routine": default}[urgency]

    def _build_waypoints(self, rows: list[str],
                         interval: float) -> tuple[list[Waypoint], float]:
        camera = self.map.devices.get("camera", "realsense")
        waypoints: list[Waypoint] = []
        cursor = self.map.base_pose
        total = 0.0
        idx = 0

        for row_id in rows:
            entry, exit_ = self.map.row_endpoints(row_id)
            length = self.map.row_length(row_id)
            field_id = self.map.rows[row_id]["field"]
            n = max(2, int(length // interval) + 1)
            yaw = math.atan2(exit_[1] - entry[1], exit_[0] - entry[0])

            total += math.dist(cursor, entry) + length
            cursor = exit_

            for i in range(n):
                t = i / (n - 1)
                waypoints.append(Waypoint(
                    waypoint_id=f"wp{idx:03d}_{row_id}_{i}",
                    x=entry[0] + t * (exit_[0] - entry[0]),
                    y=entry[1] + t * (exit_[1] - entry[1]),
                    yaw=yaw,
                    row_id=row_id,
                    field_id=field_id,
                    actions=[{
                        "action_type": "take_picture",
                        "target_device": camera,
                        "config_json": json.dumps(
                            {"row_id": row_id, "field_id": field_id, "seq": i}),
                    }],
                ))
                idx += 1

        # return leg to the dock
        total += math.dist(cursor, self.map.base_pose)
        return waypoints, total

    # ------------------------------------------------------------------
    def _get_inspection_specs(self, inspection_types: list) -> dict:
        specs = self.inspection_db.get("inspection_types", {})
        return {t: specs[t] for t in inspection_types if t in specs}

    def _build_robot_config(self, specs: dict, urgency: str) -> dict:
        sensors = set()
        for spec in specs.values():
            sensors.update(spec.get("required_sensors", []))

        cfg = self.map.robot_cfg
        # Urgency raises speed toward the cap, never past it.
        speed = float(cfg["nominal_speed_mps"]) if urgency == "urgent" \
            else float(cfg["inspection_speed_mps"])
        speed = min(speed, float(cfg["max_speed_mps"]))

        return {
            "enabled_sensors": sorted(sensors),
            "navigation": {
                "mode": "waypoint_mission",
                "max_speed_mps": speed,
                # The sim has no GPS. AMCL against the car_tree map is what
                # actually localizes the robot.
                "use_gps": False,
                "localization": "amcl",
                "use_lidar": True,
            },
            "camera_settings": {"resolution": "1080p", "fps": 30, "format": "RGB"},
            "lidar_settings": {"range_m": 5.0, "scan_rate_hz": 10},
            "ai_models": sorted({
                spec.get("ai_model") for spec in specs.values()
                if spec.get("ai_model") not in (None, "none")
            }),
        }

    def _get_expected_outputs(self, inspection_types: list) -> list:
        outputs = ["camera_images", "lidar_scans"]
        extra = {
            "disease_detection": ["disease_map", "confidence_scores"],
            "weed_assessment": ["weed_density_map"],
            "irrigation_check": ["water_stress_indicators"],
            "crop_stress": ["stress_zones", "visual_anomalies"],
        }
        for t in inspection_types:
            outputs += extra.get(t, [])
        return outputs

    def _calculate_duration(self, distance_m: float, inspection_types: list,
                            urgency: str) -> int:
        """Duration from the actual route, not a fixed per-type constant.

        The original summed a `duration_minutes` field per inspection type and
        clamped to 60. That produced the same estimate whether the robot walked
        one row or six, which made the number meaningless for scheduling.
        """
        speed = float(self.map.robot_cfg["inspection_speed_mps"])
        drive_min = distance_m / speed / 60.0
        db = self.inspection_db.get("inspection_types", {})
        analysis_min = sum(
            db.get(t, {}).get("duration_minutes", 30) / 10.0
            for t in inspection_types)
        total = drive_min + analysis_min
        if urgency == "urgent":
            total *= 0.7
        return max(2, int(round(total)))

    def _build_safety_config(self, urgency: str) -> dict:
        constraints = self.inspection_db.get("field_constraints", {})
        cfg = self.map.robot_cfg
        # Worker distance does NOT shrink for urgent missions. The original
        # dropped it from 3.0 m to 2.5 m when urgency rose, which trades a
        # safety margin for speed -- exactly the wrong direction, and the
        # safety layer would override it anyway.
        return {
            "obstacle_avoidance": True,
            "emergency_stop_enabled": True,
            "max_speed_mps": float(cfg["max_speed_mps"]),
            "min_worker_distance_m": float(
                constraints.get("min_worker_distance_m", 3.0)),
            "battery_reserve_percent": int(float(cfg["reserve_soc"]) * 100),
            "geofence_enforced": True,
            "no_go_zones_enforced": True,
            "enable_lidar": True,
            "lidar_range_m": constraints.get("lidar_range_m", 5.0),
        }

    @staticmethod
    def task_to_json(task: ROS2Task) -> str:
        return json.dumps(asdict(task), indent=2)
