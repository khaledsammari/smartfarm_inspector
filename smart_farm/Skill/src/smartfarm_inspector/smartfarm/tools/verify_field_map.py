"""Verify the field map against the simulator's occupancy grid.

Coordinates that look fine in YAML can sit on top of an obstacle or in
unknown space. The global planner then answers CalculatePathToPose with
GOAL_OCCUPIED_OR_UNKNOWN and the mission dies at load time with a message
that does not point at the real cause. This catches it beforehand.

    python -m depot_planner.tools.verify_field_map \
        --map ~/eagle_robotics_ws/src/robot_navigation/nav_launch/maps/car_tree.yaml

Requires pillow, numpy and scipy (all already in the project's pip list).
Exit code is non-zero if any point fails, so it can gate CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from ..planning.field_registry import FieldMap


def load_grid(map_yaml: Path):
    import numpy as np
    from PIL import Image

    meta = yaml.safe_load(map_yaml.read_text())
    img = map_yaml.parent / meta["image"]
    grid = np.array(Image.open(img))
    return grid, float(meta["resolution"]), meta["origin"][:2]


def build_clearance(grid, radius_px: int):
    import numpy as np
    from scipy.ndimage import binary_erosion

    free = grid > 250          # 254 free, 205 unknown, 0 occupied
    k = 2 * radius_px + 1
    return binary_erosion(free, np.ones((k, k)))


def check(clear, res: float, origin, x: float, y: float) -> bool:
    h, w = clear.shape
    c = int(round((x - origin[0]) / res))
    r = int(h - 1 - round((y - origin[1]) / res))
    return 0 <= r < h and 0 <= c < w and bool(clear[r, c])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, type=Path,
                    help="path to the map .yaml used by go2_sim.launch.py")
    ap.add_argument("--field-map", type=Path, default=None)
    ap.add_argument("--radius", type=float, default=0.45,
                    help="robot clearance radius in metres")
    ap.add_argument("--samples", type=int, default=12,
                    help="points sampled along each row")
    args = ap.parse_args(argv)

    fm = FieldMap(args.field_map) if args.field_map else FieldMap()
    grid, res, origin = load_grid(args.map)
    clear = build_clearance(grid, int(round(args.radius / res)))

    h, w = grid.shape
    print(f"map {args.map.name}: {w}x{h} px @ {res} m, origin {origin}")
    print(f"world extent x [{origin[0]:.1f}, {origin[0] + w * res:.1f}] "
          f"y [{origin[1]:.1f}, {origin[1] + h * res:.1f}]")
    print(f"clearance radius {args.radius} m\n")

    failures = []

    def report(label, x, y):
        ok = check(clear, res, origin, x, y)
        print(f"  {'PASS' if ok else 'FAIL'}  {label:34s} ({x:7.2f}, {y:7.2f})")
        if not ok:
            failures.append(label)

    print("base station")
    report("charging_dock", *fm.base_pose)

    print("\nlandmarks")
    for name, (x, y) in fm.landmarks.items():
        report(name, x, y)

    print("\ncrop rows (sampled along each corridor)")
    for rid in fm.rows:
        entry, exit_ = fm.row_endpoints(rid)
        bad = 0
        for i in range(args.samples):
            t = i / (args.samples - 1)
            x = entry[0] + t * (exit_[0] - entry[0])
            y = entry[1] + t * (exit_[1] - entry[1])
            if not check(clear, res, origin, x, y):
                bad += 1
        ok = bad == 0
        print(f"  {'PASS' if ok else 'FAIL'}  {rid:30s} "
              f"{args.samples - bad}/{args.samples} points clear")
        if not ok:
            failures.append(rid)

    print("\nno-go zones must not sit on a crop row")
    for zone in fm.no_go_zones:
        hits = []
        for rid in fm.rows:
            entry, exit_ = fm.row_endpoints(rid)
            for i in range(args.samples * 2):
                t = i / (args.samples * 2 - 1)
                pt = (entry[0] + t * (exit_[0] - entry[0]),
                      entry[1] + t * (exit_[1] - entry[1]))
                if fm.violated_no_go(pt) == zone["name"]:
                    hits.append(rid)
                    break
        ok = not hits
        print(f"  {'PASS' if ok else 'FAIL'}  {zone['name']:30s} "
              f"{'clear of all rows' if ok else 'intersects ' + str(hits)}")
        if not ok:
            failures.append(f"{zone['name']} intersects {hits}")

    print("\ngeofence corners (must be inside the map, may be non-navigable)")
    for i, (x, y) in enumerate(fm.geofence):
        inside = origin[0] <= x <= origin[0] + w * res and \
            origin[1] <= y <= origin[1] + h * res
        print(f"  {'PASS' if inside else 'FAIL'}  corner {i:<27} "
              f"({x:7.2f}, {y:7.2f})")
        if not inside:
            failures.append(f"geofence corner {i}")

    if failures:
        print(f"\n{len(failures)} problem(s): {failures}")
        print("Fix these in config/field_map.yaml before running a mission.")
        return 1
    print("\nField map is consistent with this occupancy grid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
