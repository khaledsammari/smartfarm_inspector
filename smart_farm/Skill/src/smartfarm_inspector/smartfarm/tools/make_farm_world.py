"""Generate a farm test world for the Smart Farm Asset Inspector.

Takes the simulator's existing world and adds crop plants along the rows
defined in config/field_map.yaml, with a known health state for each plant.

    python -m smartfarm.tools.make_farm_world \
        --base ~/eagle_robotics_ws/src/robot_simulation/go2/go2_config/worlds/smaller_map.world \
        --out  ~/eagle_robotics_ws/src/robot_simulation/go2/go2_config/worlds/farm.world

Writes two files:
    farm.world             the world to launch
    farm_ground_truth.json plant id -> row, position, true health state

## Why the plants have no collision geometry

`go2_config/launch/simulation.launch.py` runs pointcloud_to_laserscan with
`min_height: -1.0, max_height: 2.0`, so anything solid between the ground and
roughly 2.4 m is flattened into `/scan`. Adding solid plants would therefore:

  * corrupt `/scan` against the `car_tree` map, degrading AMCL;
  * fill the local costmap with obstacles along every row;
  * force a fresh SLAM run to rebuild the map.

Gazebo's ray sensors trace **collision** geometry, and cameras render
**visual** geometry. Giving the plants visuals only means the lidar sees
straight through them while the RealSense sees them normally. The existing
`car_tree` map, AMCL and the costmap all keep working untouched.

The trade-off is that the robot can walk through a plant. For an inspection
mission that is acceptable -- the robot drives the aisle centre and the plants
sit off to one side, in camera view but off the path. Say so in the demo
rather than letting a viewer notice it first.

## Ground truth

Health states are assigned from a fixed seed, so a given `--seed` always
produces the same world. `farm_ground_truth.json` records the true state of
every plant, which is what makes detection false positives and false
negatives countable rather than estimated. Do not regenerate the world
between a run and its scoring.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from ..planning.field_registry import FieldMap

# Colours chosen so they separate under the Excess Green Index
# (ExG = 2G - R - B), the standard RGB vegetation index. Healthy foliage is
# strongly green-dominant; chlorotic tissue shifts toward yellow as the red
# channel rises; diseased tissue goes brown.
HEALTH_STATES = {
    "healthy":  {"ambient": "0.10 0.30 0.08", "diffuse": "0.15 0.48 0.12"},
    "stressed": {"ambient": "0.30 0.32 0.08", "diffuse": "0.52 0.55 0.14"},
    "diseased": {"ambient": "0.34 0.26 0.09", "diffuse": "0.58 0.44 0.16"},
}

STEM = {"ambient": "0.18 0.14 0.06", "diffuse": "0.30 0.24 0.10"}


def plant_sdf(name: str, x: float, y: float, state: str, rng: random.Random) -> str:
    """One visual-only plant: a short stem and a cluster of foliage blobs.

    Deliberately built from primitives rather than a mesh. No external assets
    means no third-party licence question in a corporate deliverable, and
    nothing to install before the world will open.
    """
    c = HEALTH_STATES[state]
    yaw = rng.uniform(0, 6.28)
    scale = rng.uniform(0.85, 1.15)

    blobs = []
    # A ring of blobs plus a crown, jittered so no two plants look identical.
    for i in range(5):
        angle = i * 1.257 + rng.uniform(-0.3, 0.3)
        r = rng.uniform(0.10, 0.20) * scale
        bx = r * 1.4 * (1 if i % 2 else -1) * rng.uniform(0.4, 1.0)
        by = r * 1.4 * rng.uniform(-1.0, 1.0)
        bz = rng.uniform(0.22, 0.40) * scale
        blobs.append(f"""
        <visual name='leaf{i}'>
          <pose>{bx:.3f} {by:.3f} {bz:.3f} 0 0 {angle:.2f}</pose>
          <geometry><sphere><radius>{r:.3f}</radius></sphere></geometry>
          <material>
            <ambient>{c['ambient']} 1</ambient>
            <diffuse>{c['diffuse']} 1</diffuse>
            <specular>0.05 0.05 0.05 1</specular>
          </material>
          <cast_shadows>0</cast_shadows>
        </visual>""")

    return f"""
    <model name='{name}'>
      <static>1</static>
      <pose>{x:.3f} {y:.3f} 0 0 0 {yaw:.3f}</pose>
      <link name='link'>
        <!-- NO <collision> block. See module docstring: collision geometry
             would appear in /scan and break AMCL against the car_tree map. -->
        <visual name='stem'>
          <pose>0 0 {0.14 * scale:.3f} 0 0 0</pose>
          <geometry>
            <cylinder><radius>{0.025 * scale:.3f}</radius>
                      <length>{0.28 * scale:.3f}</length></cylinder>
          </geometry>
          <material>
            <ambient>{STEM['ambient']} 1</ambient>
            <diffuse>{STEM['diffuse']} 1</diffuse>
          </material>
          <cast_shadows>0</cast_shadows>
        </visual>{''.join(blobs)}
      </link>
    </model>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path,
                    help="world to build on, normally smaller_map.world")
    ap.add_argument("--out", required=True, type=Path,
                    help="output .world path")
    ap.add_argument("--field-map", type=Path, default=None)
    ap.add_argument("--spacing", type=float, default=2.5,
                    help="metres between plants along a row")
    ap.add_argument("--offset", type=float, default=1.3,
                    help="lateral offset from the row centreline, metres. "
                         "Keeps plants in camera view but off the robot path.")
    ap.add_argument("--diseased-frac", type=float, default=0.12)
    ap.add_argument("--stressed-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=7,
                    help="fixed seed: the same seed always gives the same "
                         "world and the same ground truth")
    args = ap.parse_args(argv)

    fm = FieldMap(args.field_map) if args.field_map else FieldMap()
    base = args.base.read_text()
    rng = random.Random(args.seed)

    models: list[str] = []
    truth: dict = {}
    counts = {"healthy": 0, "stressed": 0, "diseased": 0}

    for row_id in fm.rows:
        entry, exit_ = fm.row_endpoints(row_id)
        field_id = fm.rows[row_id]["field"]
        length = fm.row_length(row_id)
        n = max(2, int(length // args.spacing) + 1)

        # Rows run along +x here, so the lateral offset is in y. Computed from
        # the row direction rather than assumed, so a north-south row would
        # still get its plants to the side.
        dx, dy = exit_[0] - entry[0], exit_[1] - entry[1]
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        px, py = -dy / norm, dx / norm      # unit normal

        for i in range(n):
            t = i / (n - 1)
            cx = entry[0] + t * dx + px * args.offset
            cy = entry[1] + t * dy + py * args.offset

            if not fm.in_geofence((cx, cy)) or fm.violated_no_go((cx, cy)):
                continue

            r = rng.random()
            state = ("diseased" if r < args.diseased_frac
                     else "stressed" if r < args.diseased_frac + args.stressed_frac
                     else "healthy")
            counts[state] += 1

            plant_id = f"{row_id}_p{i:02d}"
            models.append(plant_sdf(f"crop_{plant_id}", cx, cy, state, rng))
            truth[plant_id] = {
                "row_id": row_id,
                "field_id": field_id,
                "crop": fm.fields[field_id]["crop"],
                "x": round(cx, 3),
                "y": round(cy, 3),
                "health": state,
                "anomaly": state != "healthy",
            }

    # Splice the models in just before </world>.
    if "</world>" not in base:
        print("ERROR: base world has no </world> tag")
        return 1
    banner = ("\n    <!-- ===== crop plants added by make_farm_world.py =====\n"
              "         Visual geometry only, deliberately. Collision geometry\n"
              "         here would appear in /scan and break AMCL against the\n"
              "         car_tree map. ================================== -->\n")
    world = base.replace("</world>", banner + "".join(models) + "\n  </world>")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(world)

    truth_path = args.out.with_name(args.out.stem + "_ground_truth.json")
    truth_path.write_text(json.dumps({
        "world": args.out.name,
        "seed": args.seed,
        "spacing_m": args.spacing,
        "offset_m": args.offset,
        "counts": counts,
        "total": len(truth),
        "plants": truth,
    }, indent=2))

    print(f"wrote {args.out}")
    print(f"  {len(truth)} plants: "
          f"{counts['healthy']} healthy, {counts['stressed']} stressed, "
          f"{counts['diseased']} diseased")
    print(f"  anomaly rate {sum(1 for v in truth.values() if v['anomaly']) / len(truth):.0%}")
    print(f"wrote {truth_path}")
    print("\nLaunch with:")
    print(f"  export GAZEBO_WORLD={args.out}")
    print("  ros2 launch sim_bringup go2_sim.launch.py "
          f"world:={args.out}")
    print("\nThe car_tree map stays valid: no collision geometry was added.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
