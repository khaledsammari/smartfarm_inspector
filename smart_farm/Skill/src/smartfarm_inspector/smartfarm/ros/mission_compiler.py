"""Compile a validated ROS2Task into navigation_interfaces/MissionData.

This is the stage the original pipeline was missing entirely. `ROS2Task` was
a JSON description of a mission; `MissionData` is what mission_manager
actually subscribes to.

    navigation_interfaces/msg/MissionData
      mission_id, mission_model, project_id
      GoalData[]  goals        (goal_id, geometry_model, goal_pose, map_name,
                                ActionData[] goal_actions)
      Area[]      geofencing   (area_id, Pose[] vertex_pose, last_edit)
      Area[]      danger_zones

Emitting the geofence and no-go zones here is what lets the stack's existing
geofencing_node and danger_zone_manager supervise the mission. They were
built and tested before the LLM existed and cannot be addressed or disabled
by it -- the LLM produces a plan, they hold the veto.

This module emits plain dicts, not ROS messages, so it is importable and
testable without a sourced ROS environment. `to_ros_msg` does the final
conversion inside the node.

Two constraints learned from reading the stack, both easy to get wrong:

  * Area polygons need at least 4 vertices AND must be closed -- first vertex
    repeated at the end. A rectangle therefore needs 5 poses. geofencing_node
    rejects malformed areas with a log warning rather than an error, so a bad
    geofence produces a mission that runs with NO FENCE AT ALL.
  * The mission topic is "mission_dataa", with the double 'a', marked
    "# PEZZA" in nav_nodes/utils/constants.py. Publishing to "/mission_data"
    silently does nothing.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from ..planning.field_registry import FieldMap

MIN_POLYGON_VERTICES = 4
DEFAULT_MISSION_MODEL = "Inspection"


def _pose(x: float, y: float, yaw: float = 0.0) -> dict:
    return {
        "position": {"x": float(x), "y": float(y), "z": 0.0},
        "orientation": {
            "x": 0.0, "y": 0.0,
            "z": float(math.sin(yaw / 2.0)),
            "w": float(math.cos(yaw / 2.0)),
        },
    }


def _closed_polygon(points: list) -> list[dict]:
    pts = [tuple(p) for p in points]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    if len(pts) < MIN_POLYGON_VERTICES + 1:
        raise ValueError(
            f"polygon has {len(pts)} vertices after closure; the stack needs "
            f"at least {MIN_POLYGON_VERTICES} distinct vertices plus closure")
    return [_pose(x, y) for x, y in pts]


def _area(area_id: str, points: list) -> dict:
    return {
        "area_id": area_id,
        "vertex_pose": _closed_polygon(points),
        "last_edit": datetime.now(timezone.utc).isoformat(),
    }


class MissionCompiler:
    def __init__(self, field_map: FieldMap | None = None):
        self.map = field_map or FieldMap()

    def compile(self, task, project_id: str = "smartfarm",
                append_return_to_base: bool = True) -> dict:
        """Lower a *validated* task. Assumes the validator has run."""
        goals = [
            {
                "goal_id": wp["waypoint_id"],
                "geometry_model": "",
                "goal_pose": _pose(wp["x"], wp["y"], wp["yaw"]),
                "map_name": task.map_name or self.map.map_name,
                "goal_actions": [
                    {
                        "action_type": a["action_type"],
                        "target_device": a.get("target_device", ""),
                        "config_json": a.get("config_json", "{}"),
                    }
                    for a in wp.get("actions", [])
                ],
            }
            for wp in task.waypoints
        ]

        if append_return_to_base:
            bx, by = self.map.base_pose
            goals.append({
                "goal_id": f"wp{len(goals):03d}_return_to_base",
                "geometry_model": "",
                "goal_pose": _pose(bx, by, 0.0),
                "map_name": task.map_name or self.map.map_name,
                "goal_actions": [],
            })

        return {
            "mission_id": task.task_id,
            "mission_model": DEFAULT_MISSION_MODEL,
            "project_id": project_id,
            "goals": goals,
            "geofencing": [_area("farm_envelope", self.map.geofence)],
            "danger_zones": [_area(z["name"], z["polygon"])
                             for z in self.map.no_go_zones],
        }

    @staticmethod
    def summarize(mission: dict) -> str:
        n_pic = sum(1 for g in mission["goals"] for a in g["goal_actions"]
                    if a["action_type"] == "take_picture")
        return (f"{mission['mission_id']}: {len(mission['goals'])} goals, "
                f"{n_pic} captures, {len(mission['geofencing'])} geofence "
                f"area(s), {len(mission['danger_zones'])} danger zone(s)")


def to_ros_msg(mission: dict) -> Any:
    """dict -> navigation_interfaces/msg/MissionData.

    Imported lazily so this module stays usable without a ROS environment.
    """
    from geometry_msgs.msg import Point, Pose, Quaternion
    from navigation_interfaces.msg import ActionData, Area, GoalData, MissionData

    def pose(d: dict) -> Pose:
        p, o = d["position"], d["orientation"]
        return Pose(position=Point(x=p["x"], y=p["y"], z=p["z"]),
                    orientation=Quaternion(x=o["x"], y=o["y"],
                                           z=o["z"], w=o["w"]))

    def area(d: dict) -> Area:
        return Area(area_id=d["area_id"],
                    vertex_pose=[pose(v) for v in d["vertex_pose"]],
                    last_edit=d["last_edit"])

    def goal(d: dict) -> GoalData:
        return GoalData(
            goal_id=d["goal_id"],
            geometry_model=d["geometry_model"],
            goal_pose=pose(d["goal_pose"]),
            map_name=d["map_name"],
            goal_actions=[
                ActionData(action_type=a["action_type"],
                           target_device=a.get("target_device", ""),
                           config_json=a.get("config_json", "{}"))
                for a in d["goal_actions"]
            ])

    return MissionData(
        mission_id=mission["mission_id"],
        mission_model=mission["mission_model"],
        project_id=mission["project_id"],
        goals=[goal(g) for g in mission["goals"]],
        geofencing=[area(a) for a in mission["geofencing"]],
        danger_zones=[area(a) for a in mission["danger_zones"]],
    )
