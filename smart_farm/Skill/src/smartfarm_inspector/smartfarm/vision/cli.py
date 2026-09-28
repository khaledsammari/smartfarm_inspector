"""Command line entry point.

    # analyse a mission's captures and write a report
    python3 -m smartfarm.vision.cli analyse ~/.ros/media/realsense/

    # ... plus row labels and an FP/FN score
    python3 -m smartfarm.vision.cli analyse ~/.ros/media/realsense/ \
        --mission-log mission.json \
        --ground-truth farm_ground_truth.json

    # check what the scorer makes of the ground-truth file, without images
    python3 -m smartfarm.vision.cli inspect farm_ground_truth.json

    # verify the detector works at all, no mission needed
    python3 -m smartfarm.vision.cli selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python3 cli.py` from inside the dir
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "smartfarm.vision"

from .crop_health import (SENSITIVITY, ImagesDirMissing, analyse_directory,
                          attach_waypoints, summarize)
from .report import build_report
from .score import GroundTruthError, load_ground_truth, score

DEFAULT_IMAGES = Path.home() / ".ros" / "media" / "realsense"


def cmd_analyse(args: argparse.Namespace) -> int:
    images_dir = Path(args.images_dir).expanduser()
    try:
        detections = analyse_directory(images_dir, args.sensitivity,
                                       args.limit)
    except ImagesDirMissing as exc:
        print(exc, file=sys.stderr)
        return 2
    if not detections:
        print(f"No images found in {images_dir}", file=sys.stderr)
        print("Nothing to analyse. If a mission just ran, check that capture "
              "is writing here and not to another path.", file=sys.stderr)
        return 2

    associated = False
    if args.mission_log:
        log = json.loads(Path(args.mission_log).expanduser().read_text())
        associated = attach_waypoints(detections, log)
        if not associated:
            print("WARNING: mission log waypoint count does not match the "
                  "image count -- row labels left empty rather than guessed. "
                  "Scoring will fall back to aggregate mode.",
                  file=sys.stderr)

    summary = summarize(detections)
    scoring = gt_meta = None
    if args.ground_truth:
        try:
            plants, gt_meta = load_ground_truth(
                Path(args.ground_truth).expanduser(),
                args.health_key, args.row_key)
            scoring = score(detections, plants, args.row_threshold)
        except GroundTruthError as exc:
            print(f"Ground truth not usable: {exc}", file=sys.stderr)
            print("Continuing without a score; FP/FN will be reported as "
                  "not measured.", file=sys.stderr)

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    payload = {
        "images_dir": str(images_dir),
        "sensitivity": args.sensitivity,
        "waypoints_associated": associated,
        "summary": summary,
        "scoring": scoring,
        "ground_truth": gt_meta,
        "detections": [d.to_dict() for d in detections],
    }
    (outdir / "detections.json").write_text(json.dumps(payload, indent=2))
    report = build_report(detections, summary, scoring, gt_meta,
                          args.sensitivity)
    (outdir / "crop_health_report.md").write_text(report)

    print(f"{summary['total_images']} images | "
          f"{summary['judged']} judged | "
          f"{summary['anomalies']} flagged | "
          f"rate {summary['anomaly_rate']}")
    if scoring and scoring.get("mode") == "row":
        print(f"row-level: TP={scoring['true_positives']} "
              f"FP={scoring['false_positives']} "
              f"FN={scoring['false_negatives']} "
              f"TN={scoring['true_negatives']} "
              f"precision={scoring['precision']} "
              f"recall={scoring['recall']} f1={scoring['f1']}")
    elif scoring:
        print(f"aggregate only: predicted rate "
              f"{scoring['predicted_anomaly_rate']} vs true "
              f"{scoring['true_diseased_rate']} -- no confusion matrix")
    else:
        print("FP/FN: not measured (no usable ground truth)")
    print(f"wrote {outdir/'detections.json'}")
    print(f"wrote {outdir/'crop_health_report.md'}")
    return 0


def cmd_plants(args: argparse.Namespace) -> int:
    """Hue-based per-plant detector -- the one for the Gazebo farm world,
    where green models are normal and yellow/tan models are anomalies."""
    from . import plant_health as ph
    from .report import build_plant_report

    images_dir = Path(args.images_dir).expanduser()
    try:
        kw = {}
        if args.sat_min is not None:
            kw["sat_min"] = args.sat_min
        if args.hue_green_min is not None:
            kw["hue_green_min"] = args.hue_green_min
        if args.min_cluster_px is not None:
            kw["min_cluster_px"] = args.min_cluster_px
        detections = ph.analyse_directory(images_dir, args.limit, **kw)
    except ImagesDirMissing as exc:
        print(exc, file=sys.stderr)
        return 2
    if not detections:
        print(f"No images found in {images_dir}", file=sys.stderr)
        return 2

    associated = False
    if args.mission_log:
        log = json.loads(Path(args.mission_log).expanduser().read_text())
        associated = attach_waypoints(detections, log)
        if not associated:
            print("WARNING: mission log waypoint count does not match the "
                  "image count -- row labels left empty rather than guessed. "
                  "Scoring will fall back to aggregate mode.",
                  file=sys.stderr)

    summary = ph.summarize(detections)
    scoring = gt_meta = None
    if args.ground_truth:
        try:
            plants, gt_meta = load_ground_truth(
                Path(args.ground_truth).expanduser(),
                args.health_key, args.row_key)
            scoring = score(detections, plants, args.row_threshold)
        except GroundTruthError as exc:
            print(f"Ground truth not usable: {exc}", file=sys.stderr)

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    params = {"sat_min": kw.get("sat_min", ph.SAT_MIN),
              "hue_green_min": kw.get("hue_green_min", ph.HUE_GREEN_MIN),
              "min_cluster_px": kw.get("min_cluster_px", ph.MIN_CLUSTER_PX),
              "open_kernel": ph.OPEN_KERNEL,
              "hue_min": ph.HUE_MIN, "hue_max": ph.HUE_MAX}
    (outdir / "plant_detections.json").write_text(json.dumps({
        "images_dir": str(images_dir),
        "params": params,
        "waypoints_associated": associated,
        "summary": summary,
        "scoring": scoring,
        "ground_truth": gt_meta,
        "detections": [d.to_dict() for d in detections],
    }, indent=2))
    (outdir / "plant_health_report.md").write_text(
        build_plant_report(detections, summary, scoring, gt_meta, params))

    if not args.no_overlays:
        ovdir = outdir / "overlays"
        ovdir.mkdir(exist_ok=True)
        for d in detections:
            src = Path(d.image)
            ph.overlay(src, d, ovdir / f"{src.stem}_overlay.png")

    print(f"{summary['total_images']} images | "
          f"{summary['judged']} judged | "
          f"{summary['frames_with_anomaly']} with anomaly | "
          f"rate {summary['anomaly_rate']}")
    print(f"clusters: {summary['clusters_total']} total, "
          f"{summary['clusters_anomalous']} anomalous | "
          f"mean anomalous area {summary['mean_anomalous_area_fraction']}")
    if scoring and scoring.get("mode") == "row":
        print(f"row-level: TP={scoring['true_positives']} "
              f"FP={scoring['false_positives']} "
              f"FN={scoring['false_negatives']} "
              f"TN={scoring['true_negatives']} "
              f"precision={scoring['precision']} "
              f"recall={scoring['recall']} f1={scoring['f1']}")
    elif scoring:
        print(f"aggregate only: predicted {scoring['predicted_anomaly_rate']}"
              f" vs true {scoring['true_diseased_rate']} "
              "-- no confusion matrix")
    else:
        print("FP/FN: not measured (no usable ground truth)")
    print(f"wrote {outdir/'plant_detections.json'}")
    print(f"wrote {outdir/'plant_health_report.md'}")
    if not args.no_overlays:
        print(f"wrote {outdir/'overlays'}/  <-- LOOK AT THESE before "
              "trusting any number")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    try:
        _, meta = load_ground_truth(Path(args.ground_truth).expanduser(),
                                    args.health_key, args.row_key)
    except GroundTruthError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(json.dumps(meta, indent=2))
    if not meta["row_key"]:
        print("\nNo row field found. Without one, scoring can only report "
              "an aggregate rate comparison, not FP/FN. If the records carry "
              "a row under another name, pass --row-key.", file=sys.stderr)
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from .selftest import run
    return run(verbose=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="smartfarm.vision",
        description="Crop health detection from mission imagery.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyse", aliases=["analyze"],
                       help="analyse a directory of captures")
    a.add_argument("images_dir", nargs="?", default=str(DEFAULT_IMAGES),
                   help=f"default: {DEFAULT_IMAGES}")
    a.add_argument("--sensitivity", choices=sorted(SENSITIVITY),
                   default="normal")
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--mission-log", default=None,
                   help="mission JSON, to attach row_id to each detection")
    a.add_argument("--ground-truth", default=None,
                   help="farm_ground_truth.json, to compute FP/FN")
    a.add_argument("--health-key", default=None,
                   help="override the ground-truth health field name")
    a.add_argument("--row-key", default=None,
                   help="override the ground-truth row field name")
    a.add_argument("--row-threshold", type=int, default=1,
                   help="diseased plants needed for a row to count as "
                        "anomalous (default 1)")
    a.add_argument("--outdir", default="vision_out")
    a.set_defaults(func=cmd_analyse)

    pl = sub.add_parser("plants",
                        help="per-plant anomaly detection by hue -- use this "
                             "for the Gazebo farm world (green = normal, "
                             "yellow/tan = anomaly)")
    pl.add_argument("images_dir", nargs="?", default=str(DEFAULT_IMAGES))
    pl.add_argument("--limit", type=int, default=None)
    pl.add_argument("--mission-log", default=None)
    pl.add_argument("--ground-truth", default=None)
    pl.add_argument("--health-key", default=None)
    pl.add_argument("--row-key", default=None)
    pl.add_argument("--row-threshold", type=int, default=1)
    pl.add_argument("--sat-min", type=float, default=None,
                    help="saturation gate separating plants from scenery "
                         "(default 0.45; plants 0.74, conifers 0.28)")
    pl.add_argument("--hue-green-min", type=float, default=None,
                    help="hue degrees above which a plant is healthy "
                         "(default 85)")
    pl.add_argument("--min-cluster-px", type=int, default=None,
                    help="discard connected components below this (default "
                         "150)")
    pl.add_argument("--no-overlays", action="store_true",
                    help="skip writing the annotated images")
    pl.add_argument("--outdir", default="vision_out")
    pl.set_defaults(func=cmd_plants)

    i = sub.add_parser("inspect",
                       help="show how the ground-truth file is parsed")
    i.add_argument("ground_truth")
    i.add_argument("--health-key", default=None)
    i.add_argument("--row-key", default=None)
    i.set_defaults(func=cmd_inspect)

    s = sub.add_parser("selftest",
                       help="run the detector against synthetic scenes")
    s.set_defaults(func=cmd_selftest)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
