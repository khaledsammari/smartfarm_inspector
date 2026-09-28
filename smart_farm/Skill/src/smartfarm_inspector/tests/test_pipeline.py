"""Tests for the adapted pipeline.

Grouped by the stage they protect. Several encode bugs found while adapting
the original code, so a regression would be caught rather than shipped.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartfarm.llm.llm_parser import LLMRequestParser
from smartfarm.llm.llm_provider import OfflineProvider, extract_json
from smartfarm.orchestrator import InspectionOrchestrator
from smartfarm.planning.field_registry import FieldMap, RobotState
from smartfarm.planning.task_generator import TaskGenerator
from smartfarm.planning.validator import TaskValidator
from smartfarm.ros.mission_compiler import MIN_POLYGON_VERTICES, MissionCompiler


@pytest.fixture(scope="module")
def fmap():
    return FieldMap()


def reply(**kw):
    base = {"field_ids": ["north_field"], "crops": ["tomato"],
            "inspection_types": ["disease_detection"], "urgency": "routine",
            "confidence": 0.9, "clarification_needed": None}
    base.update(kw)
    return OfflineProvider([json.dumps(base)])


# --- JSON extraction -------------------------------------------------------


def test_extract_json_handles_markdown_fences():
    """The original used a bare json.loads, so a fenced reply -- which models
    produce routinely -- failed the whole request."""
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure!\n{"a": 1}\nHope that helps.') == {"a": 1}
    assert extract_json("no json here") is None


# --- field resolution ------------------------------------------------------


@pytest.mark.parametrize("phrase,expected", [
    ("north_field", "north_field"),
    ("north field", "north_field"),
    ("the north field", "north_field"),
    ("field 1", "north_field"),
    ("2", "central_field"),
    ("sector 3", "south_field"),
    ("bottom field", "south_field"),
])
def test_farmer_phrasing_resolves_to_mapped_fields(fmap, phrase, expected):
    r = fmap.resolve_field(phrase)
    assert r.ok and r.field_id == expected, r.reason


def test_unknown_field_is_refused_not_guessed(fmap):
    """Silently inspecting the wrong field is worse than refusing."""
    r = fmap.resolve_field("the west orchard")
    assert not r.ok
    assert "not a field on this farm" in r.reason


def test_crop_only_request_infers_the_field():
    """'check the tomatoes' is a normal way for a farmer to speak."""
    p = LLMRequestParser(provider=reply(field_ids=[]))
    result = p.parse("check the tomatoes for blight")
    assert result.field_ids == ["north_field"]


def test_unresolvable_field_becomes_a_clarification():
    """The original fell back to ['unspecified'], producing an inexecutable
    task with no explanation of why."""
    p = LLMRequestParser(provider=reply(field_ids=["west orchard"], crops=[]))
    result = p.parse("check the west orchard")
    assert result.clarification_needed
    assert result.field_ids == []


def test_invalid_crop_is_reported_not_silently_dropped():
    p = LLMRequestParser(provider=reply(crops=["tomato", "bananas"]))
    result = p.parse("check tomatoes and bananas")
    assert result.crops == ["tomato"]
    assert result.dropped_crops == ["bananas"]


def test_malformed_json_is_retried():
    provider = OfflineProvider(["not json at all",
                                json.dumps({"field_ids": ["north_field"],
                                            "crops": [], "inspection_types": [],
                                            "urgency": "routine",
                                            "confidence": 0.8})])
    result = LLMRequestParser(provider=provider).parse("check north field")
    assert result.field_ids == ["north_field"]
    assert result.attempts == 2


def test_few_shot_examples_reach_the_prompt():
    """They were loaded from config and then never used."""
    p = LLMRequestParser(provider=reply())
    prompt = p._build_system_prompt()
    assert "powdery mildew" in prompt
    assert "north_field" in prompt, "the model must be told which fields exist"


# --- task generation -------------------------------------------------------


def test_task_has_executable_waypoints(fmap):
    """The original ROS2Task had no poses, so nothing could run it."""
    p = LLMRequestParser(provider=reply())
    task = TaskGenerator(field_map=fmap).generate_task(p.parse("check north"))
    assert len(task.waypoints) == 14
    assert task.rows == ["north_1", "north_2"]
    assert all("x" in w and "y" in w and "yaw" in w for w in task.waypoints)
    assert task.estimated_distance_m > 0


def test_every_waypoint_carries_a_capture(fmap):
    p = LLMRequestParser(provider=reply())
    task = TaskGenerator(field_map=fmap).generate_task(p.parse("check north"))
    for w in task.waypoints:
        assert w["actions"][0]["action_type"] == "take_picture"
        assert w["actions"][0]["target_device"] == "realsense"
        assert json.loads(w["actions"][0]["config_json"])["row_id"] in task.rows


def test_speed_is_capped_at_the_robot_limit(fmap):
    """The original config allowed 1.5 m/s, above what the Go2 does stably
    under this stack and far above what yields usable imagery."""
    p = LLMRequestParser(provider=reply(urgency="urgent"))
    task = TaskGenerator(field_map=fmap).generate_task(p.parse("urgent check"))
    assert task.robot_config["navigation"]["max_speed_mps"] <= 0.8


def test_worker_distance_does_not_shrink_when_urgent(fmap):
    """The original dropped it 3.0 -> 2.5 m for urgent missions, trading a
    safety margin for speed."""
    p = LLMRequestParser(provider=reply(urgency="urgent"))
    task = TaskGenerator(field_map=fmap).generate_task(p.parse("urgent"))
    assert task.safety_config["min_worker_distance_m"] >= 3.0


def test_duration_scales_with_the_actual_route(fmap):
    """The original summed a constant per inspection type, so one row and six
    rows produced the same estimate."""
    gen = TaskGenerator(field_map=fmap)
    one = gen.generate_task(LLMRequestParser(
        provider=reply(field_ids=["north_field"])).parse("x"))
    three = gen.generate_task(LLMRequestParser(
        provider=reply(field_ids=["north_field", "central_field",
                                  "south_field"])).parse("x"))
    assert three.duration_estimate_minutes > one.duration_estimate_minutes


def test_gps_is_not_claimed(fmap):
    """The sim has no GPS; AMCL against car_tree is what localizes."""
    p = LLMRequestParser(provider=reply())
    task = TaskGenerator(field_map=fmap).generate_task(p.parse("x"))
    assert task.robot_config["navigation"]["use_gps"] is False


# --- validation ------------------------------------------------------------


def good_task(fmap, **kw):
    p = LLMRequestParser(provider=reply(**kw))
    return TaskGenerator(field_map=fmap).generate_task(p.parse("check north"))


def codes(result):
    return {i.code for i in result.errors}


def test_valid_task_passes(fmap):
    r = TaskValidator(fmap).validate(good_task(fmap), RobotState())
    assert r.ok, r.summary()


def test_geofence_violation_rejected(fmap):
    task = good_task(fmap)
    task.waypoints[3]["x"] = 200.0
    assert "GEOFENCE_VIOLATION" in codes(TaskValidator(fmap).validate(task))


def test_no_go_violation_rejected(fmap):
    task = good_task(fmap)
    task.waypoints[2]["x"], task.waypoints[2]["y"] = 14.5, 6.0
    assert "NO_GO_VIOLATION" in codes(TaskValidator(fmap).validate(task))


def test_overspeed_rejected(fmap):
    task = good_task(fmap)
    task.robot_config["navigation"]["max_speed_mps"] = 1.5
    assert "SPEED_LIMIT" in codes(TaskValidator(fmap).validate(task))


def test_disabled_safety_rejected(fmap):
    task = good_task(fmap)
    task.safety_config["obstacle_avoidance"] = False
    assert "SAFETY_DISABLED" in codes(TaskValidator(fmap).validate(task))


def test_degraded_localization_rejected(fmap):
    r = TaskValidator(fmap).validate(good_task(fmap),
                                     RobotState(pose_covariance=0.9))
    assert "NOT_LOCALIZED" in codes(r)


def test_camera_offline_rejected(fmap):
    r = TaskValidator(fmap).validate(good_task(fmap),
                                     RobotState(camera_stream_active=False))
    assert "CAMERA_DOWN" in codes(r)


def test_energy_budget_rejected_on_low_battery(fmap):
    task = good_task(fmap, field_ids=["north_field", "central_field",
                                      "south_field"])
    r = TaskValidator(fmap).validate(task, RobotState(battery_soc=0.25))
    assert "ENERGY_BUDGET" in codes(r)


def test_map_mismatch_rejected(fmap):
    task = good_task(fmap)
    task.map_name = "some_other_map"
    assert "MAP_MISMATCH" in codes(TaskValidator(fmap).validate(task))


# --- compilation -----------------------------------------------------------


def test_mission_message_shape(fmap):
    m = MissionCompiler(fmap).compile(good_task(fmap))
    assert set(m) == {"mission_id", "mission_model", "project_id",
                      "goals", "geofencing", "danger_zones"}
    assert set(m["goals"][0]) == {"goal_id", "geometry_model", "goal_pose",
                                  "map_name", "goal_actions"}


def test_polygons_are_closed_and_long_enough(fmap):
    """geofencing_node rejects unclosed or short polygons with a log WARNING,
    so a malformed geofence yields a mission that runs with no fence at all.
    A rectangle needs 5 poses."""
    m = MissionCompiler(fmap).compile(good_task(fmap))
    for area in m["geofencing"] + m["danger_zones"]:
        verts = area["vertex_pose"]
        assert len(verts) >= MIN_POLYGON_VERTICES + 1
        assert verts[0]["position"] == verts[-1]["position"]


def test_mission_ends_at_base(fmap):
    m = MissionCompiler(fmap).compile(good_task(fmap))
    last = m["goals"][-1]
    assert "return_to_base" in last["goal_id"]
    assert last["goal_pose"]["position"]["x"] == pytest.approx(fmap.base_pose[0])


def test_map_name_propagates(fmap):
    """Must match GO2_MAP_NAME in go2_sim.launch.py or the planner rejects."""
    m = MissionCompiler(fmap).compile(good_task(fmap))
    assert all(g["map_name"] == "car_tree" for g in m["goals"])


def test_no_go_zones_are_published(fmap):
    m = MissionCompiler(fmap).compile(good_task(fmap))
    assert {a["area_id"] for a in m["danger_zones"]} == {
        "irrigation_pump_house", "equipment_yard"}


# --- end to end ------------------------------------------------------------


def test_orchestrator_success_path(fmap):
    o = InspectionOrchestrator(field_map=fmap, provider=reply())
    r = o.process_request("Check the north field for powdery mildew on tomatoes")
    assert r["status"] == "success"
    assert len(r["mission"]["goals"]) == 15


def test_orchestrator_asks_rather_than_guessing(fmap):
    o = InspectionOrchestrator(
        field_map=fmap, provider=reply(field_ids=["west orchard"], crops=[]))
    r = o.process_request("check the west orchard")
    assert r["status"] == "clarification_needed"


def test_field_map_self_consistency(fmap):
    assert len(fmap.fields) == 3
    covered = {r for f in fmap.fields.values() for r in f["rows"]}
    assert covered == set(fmap.rows)


# --- config resolution -----------------------------------------------------


def test_configs_actually_load():
    """Regression: setup.py installs configs to share/, but the defaults
    resolved relative to the module, which after install is site-packages/
    -- a directory that does not exist. The node ran with an empty crop
    database and no few-shot examples, warning but not failing."""
    p = LLMRequestParser(provider=reply())
    assert p.valid_crops, "crops.json did not load"
    assert "tomato" in p.valid_crops
    assert p.few_shot_data.get("few_shot_examples"), "few_shot_examples.json did not load"

    gen = TaskGenerator()
    assert gen.inspection_db.get("inspection_types"), "inspection_types.json did not load"


def test_explicit_config_dir_is_honoured(tmp_path):
    """The node passes its resolved share directory; that must win."""
    import json as _json
    (tmp_path / "crops.json").write_text(_json.dumps({"crops": {"kiwi": {}}}))
    (tmp_path / "few_shot_examples.json").write_text(
        _json.dumps({"few_shot_examples": []}))
    p = LLMRequestParser(provider=reply(), config_dir=tmp_path)
    assert p.valid_crops == ["kiwi"]
