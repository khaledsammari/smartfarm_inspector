# SKILL_CARD — Smart Farm Asset Inspector

**Use case (Challenge Rules §3): asset inspection.**

The skill exposed to the LLM is a single natural-language interface:
*interpret a farmer's inspection request and turn it into an executable robot
mission.* Everything below documents that skill — what the model is told, what
it may return, what is checked before anything moves, and what happens when it
gets it wrong.

---

## 1. Skill name

`interpret_inspection_request`

## 2. Natural-language description (as exposed to the LLM)

> You are an agricultural inspection coordinator for a real robot working a
> real farm. Parse the farmer's request into JSON. Never invent a field id —
> only ids from the farm listing are valid. If the farmer names a field that
> is not on this farm, leave `field_ids` empty and set
> `clarification_needed`. If the request is too vague to act on, ask rather
> than guessing.

The prompt is assembled at runtime from three sources, so the model can never
be told about a field, crop or inspection type the system cannot execute:

| Source | Supplies |
|---|---|
| `config/field_map.yaml` | fields that exist, their crops, rows, aliases |
| `config/crops.json` | valid crop names |
| `config/few_shot_examples.json` | worked input/output examples |

## 3. Input parameters (LLM output contract)

| Field | Type | Range / values | Required |
|---|---|---|---|
| `field_ids` | list[string] | must resolve against `field_map.fields` | yes |
| `crops` | list[string] | `tomato`, `lettuce`, `corn`, `cucumber` | no |
| `inspection_types` | list[string] | `disease_detection`, `weed_assessment`, `irrigation_check`, `crop_stress`, `quick_scan` | no (defaults to `quick_scan`) |
| `urgency` | string | `routine`, `medium`, `urgent` | no (defaults `routine`) |
| `confidence` | float | 0.0 – 1.0 | no (defaults 0.5) |
| `clarification_needed` | string or null | a question for the farmer | no |

**Field resolution.** `field_ids` are accepted as a farmer would say them and
resolved against the map's alias table. `"north field"`, `"the north field"`,
`"field 1"`, `"1"` and `"north_field"` all resolve to `north_field`. An
unresolvable reference becomes a clarification question — never a guess.

## 4. Return values

`LLMParsedRequest` → `ROS2Task` → `MissionData`.

`ROS2Task` (units in field names):

| Field | Type | Notes |
|---|---|---|
| `task_id` | string | 8-char uuid, also the `mission_id` |
| `waypoints` | list[Waypoint] | `x`, `y` (m, map frame), `yaw` (rad), `row_id`, `field_id`, `actions` |
| `rows` | list[string] | crop rows to walk |
| `map_name` | string | must equal `GO2_MAP_NAME` in `go2_sim.launch.py` |
| `estimated_distance_m` | float | simulated route length |
| `estimated_energy_wh` | float | distance × 0.35 Wh/m |
| `duration_estimate_minutes` | int | derived from route, not a constant |
| `robot_config` | dict | sensors, navigation mode, speed |
| `safety_config` | dict | see §6 |

## 5. Preconditions

Checked by `TaskValidator` before publication. All must hold.

| Precondition | Check |
|---|---|
| Localized | `pose_covariance < 0.5` (from `/amcl_pose`) |
| Camera online | `camera_stream_active == true` |
| Storage | `free_storage_mb > 200` |
| Battery | `estimated_energy_wh ≤ usable energy above the 20% reserve` |
| Field known | every `row_id` exists in the field map |
| Map matches | `task.map_name == field_map.map_name` |
| Inside envelope | every waypoint inside `geofence` |
| Outside no-go | no waypoint inside any `no_go_zones` polygon |
| Mission size | `waypoints ≤ 40` |
| Speed | `commanded ≤ 0.8 m/s` |
| Safety intact | `obstacle_avoidance` enabled, worker distance ≥ 2.0 m |

## 6. Postconditions

- A `navigation_interfaces/MissionData` published on `/mission_dataa`
  (`TRANSIENT_LOCAL`, `RELIABLE`), carrying goals, geofence and danger zones.
- Robot visits every waypoint, capturing one image per waypoint to
  `~/.ros/media/realsense/`, tagged with `row_id` and `field_id`.
- Mission terminates at `charging_dock` — a return goal is always appended.
- One KPI record at `~/.ros/smartfarm_logs/{task_id}.json`.

## 7. Safety constraints

Two layers, and the second is the one that matters.

**Declared in the task:**

| Constraint | Value |
|---|---|
| `max_speed_mps` | 0.8 hard cap; 0.3 inspection speed |
| `min_worker_distance_m` | 3.0, and it does **not** shrink for urgent missions |
| `battery_reserve_percent` | 20 |
| `obstacle_avoidance` | always true; cannot be disabled |
| `geofence_enforced` | true |
| `no_go_zones_enforced` | true |

**Enforced by the existing stack.** The geofence and no-go polygons travel
inside `MissionData`, so `geofencing_node` and `danger_zone_manager` supervise
the mission at runtime, and `obstacle_avoidance` holds veto over `/cmd_vel`.
Those nodes were built and tested before the LLM existed. The LLM emits no
pose, no velocity and no ROS message — it returns a parsed request, and every
stage after it is deterministic code. **The safety layer has the final say,
always.**

Urgency raises image spacing, not speed. Going faster degrades the imagery the
inspection depends on, so urgency is spent on capture density instead.

## 8. Error modes and behaviour

| Mode | Cause | Behaviour |
|---|---|---|
| `clarification_needed` | field not on this farm, or request too vague | Ask the farmer. Nothing executes. Counts as a human intervention. |
| malformed JSON | model returned prose or a fenced block | Fenced blocks are recovered. Otherwise retried once with the error fed back; then reported. |
| `NOT_LOCALIZED` | AMCL covariance ≥ 0.5 | Reject. Operator sets an initial pose. |
| `CAMERA_DOWN` | camera stream inactive | Reject — imagery is the point of the mission. |
| `NO_STORAGE` | < 200 MB free | Reject. |
| `ENERGY_BUDGET` | route exceeds usable battery | Reject with the numbers, so the farmer can split the mission. |
| `GEOFENCE_VIOLATION` / `NO_GO_VIOLATION` | a waypoint outside the envelope or inside a no-go zone | Reject. |
| `MISSION_TOO_LARGE` | > 40 waypoints | Reject; suggests a wider capture interval. |
| `SPEED_LIMIT` | commanded above 0.8 m/s | Reject. |
| `SAFETY_DISABLED` / `WORKER_DISTANCE` | safety config weakened | Reject. |
| `MAP_MISMATCH` | task targets a different map | Reject. |
| `SPEED_QUALITY` | > 0.4 m/s with disease detection | **Warning** — executes, surfaced to the operator. |
| `ENERGY_TIGHT` | > 70% of usable energy | **Warning** — executes. |

Nothing partially validated is executed. A task is accepted or rejected
atomically, and every rejection carries a specific, actionable reason.

## 9. What this skill does *not* do

Stated plainly, because leaving it implicit would misrepresent the guarantee.

- **No vision analysis.** `disease_detection` names what the mission is *for*.
  Missions capture imagery; nothing analyses it. Detection false
  positive/negative rates are unmeasured.
- **No path feasibility check.** `global_planner`'s `CalculatePathToPose` is
  the authority. The validator only confirms waypoints sit in free space on
  the static map.
- **No dynamic obstacle or people handling** at plan time — runtime only, via
  `obstacle_avoidance`.
- **No calibrated energy model.** 0.35 Wh/m is an estimate, never checked
  against real Go2 telemetry.

## 10. Run instructions

See `README.md`. Summary:

```bash
pip install -r requirements.txt
python -m pytest tests -q          # 37 passing
python demo.py                     # offline, no credentials
python -m smartfarm.tools.verify_field_map --map <car_tree.yaml>
```

ROS 2 Humble, Gazebo Classic. Build into `~/eagle_robotics_ws/src/`,
`colcon build --packages-select smartfarm_inspector --symlink-install`.

Credentials: none in this repo. `.env.example` lists the variable names;
`python -m smartfarm.tools.check_auth` reports which auth method is active.
