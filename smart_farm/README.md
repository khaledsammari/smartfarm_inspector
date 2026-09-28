# Smart Farm Asset Inspector

**Team:** Team 2
**Members:**

| Name | Email |
|---|---|
| _Yosra Mahfoudh_ | _ymahfoudh@eagleprojects.tn_ |
| _Khaled Sammari_ | _ksammari@eagleprojects.tn_ |

**Demo video:**[▶ Watch the demonstration video](https://eagleprojecttunisia-my.sharepoint.com/personal/ksammari_eagleprojects_tn/_layouts/15/stream.aspx?id=%2Fpersonal%2Fksammari%5Feagleprojects%5Ftn%2FDocuments%2FMicrosoft%20Teams%20Chat%20Files%2FBeauty%5FContest%5Fdemo%2Ewebm&referrer=StreamWebApp%2EWeb&referrerScenario=AddressBarCopied%2Eview%2Edf8d578f%2D6868%2D4edb%2D88bd%2D73e22410a167&ga=1)**

 

---

## What it does

A farmer types an inspection request in plain English. Claude interprets it,
a deterministic validator accepts or rejects the resulting task, and a Unitree
Go2 walks the crop rows capturing georeferenced imagery. A vision stage then
classifies plant health from that imagery and reports anomalies.

```
"check the north field for powdery mildew on tomatoes"
        │
        ▼  Claude (subscription auth, no API key)     3.7 s
LLMParsedRequest: north_field · tomato · disease_detection · 0.95
        │
        ▼  TaskGenerator          waypoints from the field map
ROS2Task: 14 waypoints · 109 m · 9 min
        │
        ▼  TaskValidator          deterministic, no network
        │                         reject ──▶ operator, with a specific reason
        ▼  MissionCompiler
/mission_dataa (MissionData: goals + geofence + danger zones)
        │
        ▼  the existing Eagle stack, unchanged
mission_manager → global_planner → path_follower
geofencing_node + danger_zone_manager + obstacle_avoidance  ← final say
        │
        ▼  vision stage (offboard, after the walk)
plant health classification + anomaly report
```

**The central claim:** the LLM has capability; the layers below it have
authority. It never emits a pose, a velocity or a ROS message — it produces a
parsed request, and everything after it is deterministic code.

## How to run it

```bash
# 1. Install into the Eagle workspace
cd skill/src/smartfarm_inspector
./install.sh ~/eagle_robotics_ws

# 2. Claude auth — subscription, no API key needed
nvm install 20 && nvm use 20
npm install -g @anthropic-ai/claude-code
claude                       # option 1 (subscription), sign in via incognito
pip install claude-agent-sdk
unset ANTHROPIC_API_KEY      # critical: it silently overrides the OAuth token
python3 -m smartfarm.tools.check_auth   # expect "subscription billing"

# 3. Offline test — no credentials, no network
python3 -m pytest tests -q -p no:launch_testing -p no:anyio
python3 demo.py

# 4. Live LLM test
python3 demo.py --live --backend agent_sdk "check the north field for blight"
```

### Full mission in simulation

```bash
# terminal 1 — simulator
source ~/team-2-proposal/Skill/install/setup.bash
ros2 launch sim_bridge2 go2_sim.launch.py

# terminal 2 — the LLM bridge
source ~/team-2-proposal/Skill/install/setup.bash
ros2 launch smartfarm_inspector smartfarm.launch.py auto_launch:=false

# terminal 3 — send a mission in English
source ~/team-2-proposal/Skill/install/setup.bash
ros2 topic pub --once /farm/inspection_request std_msgs/msg/String \
  "data: 'check the north field for powdery mildew on tomatoes'"

ros2 action send_goal /mission/launch \
  navigation_interfaces/action/LaunchCurrentMission '{request: {}}'

# terminal 4 — results 
python3 -m smartfarm.vision.cli plants /team-2-proposal/Skill/media/realsense --outdir ~/vision_out
 ```


Results: in ~/vision_out

**Dependencies:** ROS 2 Humble, Gazebo Classic, Python 3.10, Node 20 (only to
install the Claude Code CLI that produces the OAuth token — nothing in the
Python pipeline uses Node at runtime).

## Deliverables

| ID | Deliverable | Location |
|---|---|---|
| D0 | This README | `README.md` |
| D1 | Initiative description | `initiative/Initiative_Brief.pdf` |
| D2 | Solution architecture | `architecture/Architecture.pdf`, `architecture.mmd` |
| D3 | Documented skill | `skill/smartfarm_inspector/` (see its `SKILL_CARD.md`) |
| D4 | End-to-end demo | `demo/Demo_Notes.md` |
| D5 | KPIs and results | `kpi/KPI_Report.xlsx`, `05_kpi/raw/` |
| D6 | Business case | `business_case/Business_Case.xlsx` |

## Results summary

| KPI | Value |
|---|---|
| Mission completion rate | 1/1 (15/15 waypoints reached) |
| Human intervention rate | 0 |
| First-pass parse validity | 7/7 live requests, 0 repair rounds |
| Plan latency | 3.7 s |
| Safety interventions | 9, all auto-recovered |
| Images captured and analysed | 42 (36 judged, 6 no plants in frame) |
| Anomalous clusters | 45 / 170 (26.5%) |
| Automated tests | 39 passing |

