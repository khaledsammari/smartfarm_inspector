# Smart Farm Asset Inspector

Autonomous crop inspection with a Unitree Go2 on the Eagle ROS 2 stack. A
farmer types a request in plain English; Claude parses it; a deterministic
validator accepts or rejects the resulting task; the robot walks the crop rows
capturing georeferenced imagery.

**Use case (Challenge Rules §3): asset inspection.**
Target stack: **ROS 2 Humble + Gazebo Classic**, map `nav_launch/maps/car_tree.yaml`.

---

## First: rotate your API key

The uploaded project contained a live `ANTHROPIC_API_KEY` in `.env`, with an
empty `.gitignore` and a `.git` directory present. If that key was ever
committed it is in git history and `git rm` will not remove it.

1. Rotate the key at <https://console.anthropic.com/settings/keys>
2. Use the `.env.example` here; `.env` is now gitignored
3. Before pushing: `git grep -nE "sk-ant-" -- . | grep -v example`

The challenge rules make "no API keys or credentials in the project" the one
binding requirement for D3.

## What was kept and what was added

The original three-stage pipeline — parse, generate task, orchestrate — is
intact, along with its dataclasses, config files and vocabulary. Two stages
were inserted, because the pipeline stopped short of anything executable.

```
 farmer sentence
       │
       ▼  llm_parser.py        Claude, grounded in the field map
 LLMParsedRequest
       │
       ▼  task_generator.py    now emits WAYPOINTS, not just sensor config
 ROS2Task
       │
       ▼  validator.py         NEW — deterministic gate, no network
 validated task ──reject──▶ operator, with a specific reason
       │
       ▼  mission_compiler.py  NEW — ROS2Task → MissionData
 /mission_dataa
       │
       ▼  the existing Eagle stack, unchanged
 mission_manager → global_planner → path_follower
 geofencing_node + danger_zone_manager + obstacle_avoidance   ← final say
```

**The central claim for D2:** the LLM has capability, the layers below it have
authority. It never emits a pose, a velocity or a ROS message — it produces a
parsed request, and everything downstream is deterministic code. The geofence
and no-go zones ride inside `MissionData`, so the safety nodes that existed
before the LLM supervise the mission and cannot be addressed or disabled by it.

## The gap that mattered most

`field_ids` were free text: `"north field"`, `"sector 3"`, `"2"`. Nothing can
navigate to free text. `config/field_map.yaml` is the bridge — it gives every
field real coordinates, crop rows verified against the occupancy grid, aliases
a farmer would actually say, and the geofence. An unresolvable field now
produces a clarification question instead of `['unspecified']`.

## Fixes worth knowing about

| Issue | Effect |
|---|---|
| `few_shot_examples.json` loaded but never used in the prompt | The examples did nothing |
| `_extract_json` was a bare `json.loads` | Any markdown fence failed the whole request |
| Fields never described to the model | Nothing stopped it inventing them |
| `max_speed_mps: 1.5` | Above stable Go2 speed here, and blurs inspection imagery. Now capped at 0.8 |
| Worker distance dropped 3.0 → 2.5 m when urgent | Traded a safety margin for speed. Now fixed at 3.0 |
| `duration_minutes` summed per inspection type | One row and six rows gave the same estimate. Now derived from the route |
| `use_gps: True` | The sim has no GPS; AMCL localizes |
| Flat imports (`from llm_provider import`) | Only worked from inside `src/llm/` |
| `datetime.utcnow()` | Deprecated in Python 3.12+ |
| No retry on malformed JSON | One stumble returned an error to the farmer |
| Invalid crops dropped silently | Farmer never told what was ignored |

A field-map conflict was also caught by the new validator during development:
the irrigation channel was placed on top of row `north_2`, so every mission
over that row was rejected with `NO_GO_VIOLATION`. Found at desk, not in the
field. Both no-go zones now sit outside the row band, and `verify_field_map`
checks this.

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest tests -q      # 37 passing
python demo.py                 # 4 scenarios, no credentials needed

# Claude on your subscription — no API credits
npm install -g @anthropic-ai/claude-code && claude login
pip install claude-agent-sdk
unset ANTHROPIC_API_KEY
python -m smartfarm.tools.check_auth
python demo.py --live --backend agent_sdk "check the north field for blight"
```

Verify the field map against the simulator's occupancy grid after any map or
world change:

```bash
python -m smartfarm.tools.verify_field_map \
  --map ~/eagle_robotics_ws/src/robot_navigation/nav_launch/maps/car_tree.yaml
```

## Running on the robot

```bash
cd ~/eagle_robotics_ws
cp -r smartfarm_inspector src/
colcon build --packages-select smartfarm_inspector --symlink-install
source install/setup.bash
```

```bash
# terminal 1
ros2 launch sim_bringup go2_sim.launch.py

# terminal 2 — auto_launch:=false lets you review the plan in RViz first
ros2 launch smartfarm_inspector smartfarm.launch.py auto_launch:=false

# terminal 3
ros2 topic pub --once /farm/inspection_request std_msgs/msg/String \
  "data: 'check the north field for powdery mildew on tomatoes'"
ros2 topic echo /farm/inspection_result

# start it
ros2 action send_goal /mission/launch \
  navigation_interfaces/action/LaunchCurrentMission '{request: {}}'
```

Backends: `agent_sdk` (subscription, no API credits), `api` (pay-as-you-go),
`offline` (canned, no network). Unset autodetects. The subscription OAuth token
is licensed for individual use — right for a demo, wrong for multi-user or
production, which should use `api`.

## Traps in the host stack

1. **The mission topic is `mission_dataa`**, with the double `a`, marked
   `# PEZZA` in `nav_nodes/utils/constants.py`. Publishing to `/mission_data`
   silently does nothing.
2. **`Area` polygons must be closed and have ≥4 vertices** — a rectangle needs
   5 poses. `geofencing_node` rejects malformed areas with a log *warning*, so
   a bad geofence yields a mission that runs **with no fence at all**. There is
   a test pinning this.
3. **`/mission_dataa` needs `TRANSIENT_LOCAL` + `RELIABLE`** QoS or
   `mission_manager` misses messages published before it subscribes.

## Known limitations

- **No vision stage.** Missions capture imagery to `~/.ros/media/realsense/`;
  nothing analyses it yet. Detection false positive/negative rates are
  therefore unmeasured. `disease_detection` describes what the mission is
  *for*, not something the code performs.
- Simulation only; not run on hardware.
- The energy model uses an uncalibrated 0.35 Wh/m constant.
- The field map is hand-authored per site.
- The default Gazebo world has no crop models; rows are the clear aisles of
  the `car_tree` obstacle grid.

## KPI data

Every mission writes `~/.ros/smartfarm_logs/{task_id}.json` with the request,
parsed result, confidence, attempts, token usage, validator verdict and route
estimate. Mission completion rate, human intervention rate and first-pass
parse validity come straight out of those files.

---

## Test world with crop plants

The default world has no crops — rows are just the clear aisles of the
`car_tree` obstacle grid. To generate a world with actual plants to
photograph, each with a known health state:

```bash
python -m smartfarm.tools.make_farm_world \
  --base ~/eagle_robotics_ws/src/robot_simulation/go2/go2_config/worlds/smaller_map.world \
  --out  ~/eagle_robotics_ws/src/robot_simulation/go2/go2_config/worlds/farm.world
```

Produces 61 plants along the six rows (roughly 28% showing stress or disease)
plus `farm_ground_truth.json` recording the true state of every one.

Launch it instead of the default world:

```bash
ros2 launch sim_bringup go2_sim.launch.py \
  world:=$HOME/eagle_robotics_ws/src/robot_simulation/go2/go2_config/worlds/farm.world
```

### The plants have no collision geometry, deliberately

`go2_config/launch/simulation.launch.py` runs `pointcloud_to_laserscan` with
`min_height: -1.0, max_height: 2.0`, so anything solid between the ground and
about 2.4 m is flattened into `/scan`. Solid plants would corrupt `/scan`
against the `car_tree` map, degrade AMCL, fill the local costmap along every
row, and force a fresh SLAM run to rebuild the map.

Gazebo ray sensors trace *collision* geometry; cameras render *visual*
geometry. Visual-only plants are therefore invisible to the lidar and normal
to the RealSense. **The existing `car_tree` map, AMCL and the costmap all keep
working with no changes.** The 21 original models keep their collision
geometry, so walls and vehicles still block navigation.

The trade-off: the robot can walk through a plant. It drives the aisle centre
and the plants sit 1.3 m to one side — in camera view, off the path. Mention
this in the demo rather than letting a viewer spot it first.

The plants are built from Gazebo primitives, not meshes. No external assets
means no third-party licence question in a corporate deliverable, and nothing
to install before the world will open.

### Ground truth and KPIs

`farm_ground_truth.json` maps every plant to its row, position and true health
state. Because the seed is fixed, the same `--seed` always regenerates the
same world. That is what makes detection false positives and false negatives
*countable* rather than estimated — the D5 KPI becomes a number.

Do not regenerate the world between a run and its scoring.

Health colours are chosen to separate under the Excess Green Index
(`ExG = 2G - R - B`), a standard RGB vegetation index used in precision
agriculture. It measures colour physics rather than learned features, so there
is no sim-to-real domain gap to argue about — but note that the vision stage
that would consume these images **is not implemented yet**.
