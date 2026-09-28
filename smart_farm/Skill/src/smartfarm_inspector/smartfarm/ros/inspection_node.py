"""Smart Farm inspection node.

Sits on top of the existing Eagle navigation stack. It does not replace
mission_manager, global_planner or the safety nodes -- it feeds them.

    /farm/inspection_request  (std_msgs/String)   a farmer sentence
            |
      LLMRequestParser  ---- Claude (subscription, API, or offline)
            |
      TaskGenerator     ---- waypoints from the field map
            |
      TaskValidator     ---- deterministic, no network
            |
      MissionCompiler
            |
    /mission_dataa  (navigation_interfaces/MissionData, TRANSIENT_LOCAL)
            |
    /mission/launch (LaunchCurrentMission)  --> the existing stack executes,
                                                geofencing_node and
                                                danger_zone_manager enforce
                                                the areas we published

Every outcome, including every rejection, is published on
/farm/inspection_result as JSON and written to the KPI log.

    ros2 run smartfarm_inspector inspection_node
    ros2 topic pub --once /farm/inspection_request std_msgs/msg/String \
      "data: 'check the north field for powdery mildew on tomatoes'"
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from dataclasses import asdict
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseWithCovarianceStamped
from navigation_interfaces.action import LaunchCurrentMission
from navigation_interfaces.msg import MissionData
from navigation_interfaces.srv import GetRobotStatus
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import String

from ..llm.llm_provider import get_llm_provider
from ..orchestrator import InspectionOrchestrator
from ..planning.field_registry import FieldMap, RobotState
from ..ros.mission_compiler import to_ros_msg

# Mirrors nav_nodes/utils/constants.py. Duplicated so this package does not
# build-depend on nav_nodes, but it MUST stay in sync.
# The topic name really is "mission_dataa" (marked "# PEZZA" upstream).
MISSION_DATA_TOPIC = "/mission_dataa"
LAUNCH_MISSION_ACTION = "/mission/launch"
GET_ROBOT_STATUS_SERVICE = "/robot/get_status"
AMCL_TOPIC = "/amcl_pose"

REQUEST_TOPIC = "/farm/inspection_request"
RESULT_TOPIC = "/farm/inspection_result"

QOS_R_KL_TL = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    depth=1,
)


class InspectionNode(Node):
    def __init__(self) -> None:
        super().__init__("smartfarm_inspection_node")

        share = Path(get_package_share_directory("smartfarm_inspector"))
        self.declare_parameter("config_dir", str(share / "config"))
        self.declare_parameter("backend", os.environ.get("SMARTFARM_BACKEND", ""))
        self.declare_parameter("model", os.environ.get("SMARTFARM_MODEL",
                                                       "claude-sonnet-5"))
        self.declare_parameter("project_id", "smartfarm")
        self.declare_parameter("auto_launch", True)
        self.declare_parameter("default_battery_soc", 0.95)

        cfg_dir = Path(self.get_parameter("config_dir").value)
        self.field_map = FieldMap(cfg_dir / "field_map.yaml")

        backend = self.get_parameter("backend").value or None
        model = self.get_parameter("model").value
        try:
            provider = get_llm_provider(backend, model=model)
        except RuntimeError as exc:
            # Most commonly: agent_sdk requested while ANTHROPIC_API_KEY is
            # set, which would silently bill the API instead of the
            # subscription. Surface it rather than choose for the user.
            self.get_logger().error(str(exc))
            raise

        self.orchestrator = InspectionOrchestrator(
            field_map=self.field_map, provider=provider, config_dir=cfg_dir)
        self.provider = provider

        self.cb = ReentrantCallbackGroup()
        self.state = RobotState(
            battery_soc=float(self.get_parameter("default_battery_soc").value))
        self._busy = threading.Lock()

        self.mission_pub = self.create_publisher(
            MissionData, MISSION_DATA_TOPIC, QOS_R_KL_TL)
        self.result_pub = self.create_publisher(String, RESULT_TOPIC, 10)
        self.create_subscription(String, REQUEST_TOPIC, self.on_request, 10,
                                 callback_group=self.cb)
        self.create_subscription(PoseWithCovarianceStamped, AMCL_TOPIC,
                                 self.on_pose, 10, callback_group=self.cb)

        self.launch_client = ActionClient(
            self, LaunchCurrentMission, LAUNCH_MISSION_ACTION,
            callback_group=self.cb)
        self.status_client = self.create_client(
            GetRobotStatus, GET_ROBOT_STATUS_SERVICE, callback_group=self.cb)

        billing = ("your Claude subscription" if
                   getattr(provider, "uses_subscription", False)
                   else "API credits" if provider.name == "api"
                   else "nothing (offline canned responses)")
        self.get_logger().info(
            f"smartfarm inspection node ready | backend={provider.name} "
            f"billed to {billing} | farm={self.field_map.farm_name} "
            f"| fields={sorted(self.field_map.fields)} "
            f"| listening on {REQUEST_TOPIC}")

    # ------------------------------------------------------------------
    def on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        p = msg.pose.pose.position
        self.state.pose = (p.x, p.y)
        cov = msg.pose.covariance
        self.state.pose_covariance = max(cov[0], cov[7])

    def on_request(self, msg: String) -> None:
        request = msg.data.strip()
        if not request:
            return
        if not self._busy.acquire(blocking=False):
            self._emit(status="busy", error="a request is already being processed")
            return
        # Parsing does a network round-trip; never block the executor.
        threading.Thread(target=self._handle, args=(request,), daemon=True).start()

    def _handle(self, request: str) -> None:
        try:
            self.get_logger().info(f'request: "{request}"')

            if self._robot_busy():
                self._emit(status="busy", request=request,
                           error="robot is already running a mission")
                return

            result = self.orchestrator.process_request(request, self.state)
            status = result["status"]

            if status != "success":
                self.get_logger().warn(f"not executing: {status}")
                for e in result.get("errors", []) or [result.get("error", "")]:
                    if e:
                        self.get_logger().warn(f"  {e}")
                self._emit(status=status, request=request,
                           error=result.get("error"),
                           clarification=result.get("clarification"),
                           errors=result.get("errors", []))
                return

            task = result["task"]
            mission = result["mission"]
            self.get_logger().info(
                f"task {task['task_id']}: {len(task['waypoints'])} waypoints, "
                f"{task['estimated_distance_m']} m, "
                f"{task['duration_estimate_minutes']} min")
            self.mission_pub.publish(to_ros_msg(mission))

            self._emit(status="success", request=request,
                       task_id=task["task_id"],
                       fields=result["parsed"]["field_ids"],
                       rows=task["rows"],
                       waypoints=len(task["waypoints"]),
                       distance_m=task["estimated_distance_m"],
                       energy_wh=task["estimated_energy_wh"],
                       confidence=result["parsed"]["confidence"],
                       attempts=result["parsed"]["attempts"],
                       usage=result["parsed"]["usage"],
                       warnings=result.get("warnings", []))
            self._log(result)

            if self.get_parameter("auto_launch").value:
                self._launch(task["task_id"])
            else:
                self.get_logger().info(
                    "auto_launch is false; start it with: ros2 action send_goal "
                    f"{LAUNCH_MISSION_ACTION} "
                    "navigation_interfaces/action/LaunchCurrentMission "
                    "'{request: {}}'")
        except Exception as exc:  # noqa: BLE001 - the node must survive faults
            self.get_logger().error(f"unhandled error: {exc}\n"
                                    f"{traceback.format_exc()}")
            self._emit(status="error", request=request, error=str(exc))
        finally:
            self._busy.release()

    # ------------------------------------------------------------------
    def _robot_busy(self) -> bool:
        if not self.status_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(
                f"{GET_ROBOT_STATUS_SERVICE} unavailable; assuming idle")
            return False
        future = self.status_client.call_async(GetRobotStatus.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        res = future.result()
        return bool(res and res.is_busy)

    def _launch(self, task_id: str) -> None:
        if not self.launch_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                f"{LAUNCH_MISSION_ACTION} not available; mission published "
                "but not started")
            return
        send = self.launch_client.send_goal_async(
            LaunchCurrentMission.Goal(),
            feedback_callback=lambda fb: self._on_feedback(task_id, fb))
        send.add_done_callback(lambda f: self._on_accepted(task_id, f))

    def _on_accepted(self, task_id: str, future) -> None:
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("launch rejected by mission_manager")
            self._emit(status="launch_rejected", task_id=task_id)
            return
        self.get_logger().info(f"mission {task_id} accepted, executing")
        handle.get_result_async().add_done_callback(
            lambda f: self._on_result(task_id, f))

    def _on_feedback(self, task_id: str, feedback) -> None:
        status = feedback.feedback.mission_status
        self.get_logger().info(f"[{task_id}] {status}")
        self._emit(status="running", task_id=task_id, mission_status=status)

    def _on_result(self, task_id: str, future) -> None:
        r = future.result().result
        ok = r.error_code == 0
        (self.get_logger().info if ok else self.get_logger().error)(
            f"[{task_id}] finished: code={r.error_code} {r.error_msg}")
        self._emit(status="completed" if ok else "failed", task_id=task_id,
                   terminal=True, error_code=int(r.error_code),
                   error=r.error_msg)

    # ------------------------------------------------------------------
    def _emit(self, **payload) -> None:
        self.result_pub.publish(String(data=json.dumps(payload, default=str)))

    def _log(self, result: dict) -> None:
        """One JSON per mission. This is the D5 KPI raw data."""
        log_dir = Path.home() / ".ros" / "smartfarm_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        tid = result["task"]["task_id"]
        (log_dir / f"{tid}.json").write_text(
            json.dumps({k: v for k, v in result.items() if k != "task_json"},
                       indent=2, default=str))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InspectionNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
