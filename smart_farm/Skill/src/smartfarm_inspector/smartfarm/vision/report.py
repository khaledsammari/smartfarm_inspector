"""Markdown report generation -- the thing that goes into D5.

Written so the output can be pasted into the deliverable without editing:
it states the method, the numbers, the unit those numbers are in, and the
limitations, in that order. Where a number is unavailable it says why rather
than leaving a blank.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .crop_health import KNOWN_LIMITATIONS, CHLOROSIS_THRESHOLD, \
    SENSITIVITY, VEG_THRESHOLD


def _table(rows: list[tuple[str, str]]) -> str:
    out = ["| Metric | Value |", "| --- | --- |"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(out)


def _fmt(v) -> str:
    if v is None:
        return "not available"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def build_report(detections, summary: dict, scoring: dict | None = None,
                 gt_meta: dict | None = None,
                 sensitivity: str = "normal") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cfg = SENSITIVITY[sensitivity]

    parts = [
        "# Crop Health Detection Report",
        "",
        f"Generated {ts}",
        "",
        "## Method",
        "",
        "Per-frame crop health from the Excess Green Index "
        "(`ExG = 2g - r - b` on normalised RGB), a standard vegetation index "
        "in precision agriculture. Canopy pixels are segmented at "
        f"`ExG > {VEG_THRESHOLD}`; within the canopy, two features drive the "
        f"verdict: mean ExG, and the fraction of canopy below "
        f"`ExG = {CHLOROSIS_THRESHOLD}` (chlorotic). No trained model is used "
        "-- see Limitations for why, and for what that costs.",
        "",
        f"Sensitivity preset: **{sensitivity}** "
        f"(DISEASED below mean {cfg['diseased_exg']} or above chlorotic "
        f"fraction {cfg['diseased_frac']}; STRESSED below mean "
        f"{cfg['stressed_exg']} or above chlorotic fraction "
        f"{cfg['stressed_frac']}).",
        "",
        "## Detection results",
        "",
        _table([
            ("Images analysed", _fmt(summary["total_images"])),
            ("Frames judged (vegetation present)", _fmt(summary["judged"])),
            ("Frames flagged as anomalous", _fmt(summary["anomalies"])),
            ("Anomaly rate", _fmt(summary["anomaly_rate"])),
        ]),
        "",
        "Verdict breakdown:",
        "",
        "| Verdict | Frames |",
        "| --- | --- |",
    ]
    for verdict in ("HEALTHY", "STRESSED", "DISEASED", "NO_VEGETATION"):
        parts.append(f"| {verdict} | {summary['counts'].get(verdict, 0)} |")
    parts.append("")

    # ---------------------------------------------------------------- KPI
    parts += ["## Detection accuracy (FP / FN)", ""]

    if scoring is None:
        parts += [
            "**Not measured.** No ground-truth file was supplied to the "
            "scorer, so false positives and false negatives are undetermined "
            "for this run. The detector emits a verdict per frame; scoring "
            "requires an answer key matched to the world that was actually "
            "loaded.",
            "",
        ]
    elif scoring.get("mode") == "row":
        parts += [
            f"Scored at **row level** over {scoring['n']} rows "
            f"(`{', '.join(scoring['rows_scored'])}`). A row counts as truly "
            f"anomalous when it contains at least "
            f"{scoring['row_threshold']} diseased plant(s); it counts as "
            "predicted anomalous when any frame captured in that row was "
            "flagged STRESSED or DISEASED.",
            "",
            _table([
                ("True positives", _fmt(scoring["true_positives"])),
                ("False positives", _fmt(scoring["false_positives"])),
                ("False negatives", _fmt(scoring["false_negatives"])),
                ("True negatives", _fmt(scoring["true_negatives"])),
                ("Precision", _fmt(scoring["precision"])),
                ("Recall", _fmt(scoring["recall"])),
                ("F1", _fmt(scoring["f1"])),
                ("Accuracy", _fmt(scoring["accuracy"])),
                ("False positive rate", _fmt(scoring["false_positive_rate"])),
                ("False negative rate", _fmt(scoring["false_negative_rate"])),
            ]),
            "",
        ]
        if scoring["rows_in_truth_only"]:
            parts += [
                "Rows present in the ground truth but never imaged: "
                f"`{', '.join(scoring['rows_in_truth_only'])}`. These are "
                "excluded from the matrix above and represent unsurveyed "
                "coverage, not detection error.",
                "",
            ]
        if scoring["rows_in_detections_only"]:
            parts += [
                "Rows imaged but absent from the ground truth: "
                f"`{', '.join(scoring['rows_in_detections_only'])}`. "
                "Excluded -- likely a world/ground-truth mismatch.",
                "",
            ]
        parts += ["Per-row detail:", "",
                  "| Row | Truth | Prediction | Outcome |",
                  "| --- | --- | --- | --- |"]
        for row, v in scoring["per_row"].items():
            t, p = v["truth_anomalous"], v["predicted_anomalous"]
            outcome = ("TP" if t and p else "FN" if t else
                       "FP" if p else "TN")
            parts.append(
                f"| {row} | {'anomalous' if t else 'healthy'} | "
                f"{'anomalous' if p else 'healthy'} | {outcome} |")
        parts.append("")
    else:
        parts += [
            "**No confusion matrix available.** Detections could not be "
            "associated with rows (the mission log did not line up with the "
            "capture count), so truth and prediction share no common unit. "
            "What can be reported is a rate comparison:",
            "",
            _table([
                ("Frames judged", _fmt(scoring["frames_judged"])),
                ("Frames flagged", _fmt(scoring["frames_flagged"])),
                ("Predicted anomaly rate",
                 _fmt(scoring["predicted_anomaly_rate"])),
                ("Plants in ground truth", _fmt(scoring["plants_in_truth"])),
                ("Plants diseased", _fmt(scoring["plants_diseased"])),
                ("True diseased rate", _fmt(scoring["true_diseased_rate"])),
                ("Rate error", _fmt(scoring["rate_error"])),
            ]),
            "",
            f"> {scoring['caveat']}",
            "",
        ]

    if gt_meta:
        parts += [
            "### Ground truth used",
            "",
            _table([
                ("File", f"`{gt_meta['source']}`"),
                ("Health field", f"`{gt_meta['health_key']}`"),
                ("Row field", f"`{gt_meta['row_key']}`"
                 if gt_meta["row_key"] else "none found"),
                ("Plant records found", _fmt(gt_meta["records_found"])),
                ("Records usable", _fmt(gt_meta["records_usable"])),
                ("Records with unrecognised state",
                 _fmt(gt_meta["records_unrecognised"])),
                ("Diseased / healthy",
                 f"{gt_meta['diseased']} / {gt_meta['healthy']}"),
            ]),
            "",
        ]
        if gt_meta["records_unrecognised"]:
            parts += [
                f"> {gt_meta['records_unrecognised']} record(s) had a health "
                "value the scorer did not recognise and were dropped. "
                f"Values seen: `{gt_meta['distinct_values']}`. Extend "
                "`HEALTHY_VALUES` / `ANOMALY_VALUES` in `score.py` if this "
                "count is non-trivial.",
                "",
            ]

    # -------------------------------------------------------- limitations
    parts += ["## Limitations", ""]
    parts += [f"{i}. {lim}" for i, lim in enumerate(KNOWN_LIMITATIONS, 1)]
    parts += ["", "## Per-frame detections", "",
              "| # | Image | Verdict | Mean ExG | Chlorotic | Canopy | Row |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for i, d in enumerate(detections, 1):
        name = d.image.rsplit("/", 1)[-1]
        parts.append(
            f"| {i} | `{name}` | {d.verdict} | {d.mean_exg:.3f} | "
            f"{d.chlorotic_fraction:.3f} | {d.canopy_fraction:.3f} | "
            f"{d.row_id or '-'} |")
    parts.append("")
    return "\n".join(parts)


def build_plant_report(detections, summary: dict, scoring: dict | None = None,
                       gt_meta: dict | None = None,
                       params: dict | None = None) -> str:
    """Report for the hue-based per-plant detector (`plant_health.py`)."""
    from .plant_health import KNOWN_LIMITATIONS as PLANT_LIMITS
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    p = params or {}

    parts = [
        "# Plant Anomaly Detection Report",
        "",
        f"Generated {ts}",
        "",
        "## Method",
        "",
        "Plant health in this world is encoded as model colour: green plants "
        "are normal, yellow and tan plants are anomalies. Detection is "
        "therefore a hue classification, gated on saturation.",
        "",
        f"1. **Plant mask** -- pixels with saturation > {p.get('sat_min')} "
        f"and hue in [{p.get('hue_min')}, {p.get('hue_max')}] degrees. "
        "Plants measure 0.74 saturation, background conifer scenery 0.28, "
        "and sky/ground exactly 0.00, so this gate has wide margins on both "
        "sides.",
        f"2. **Cleanup** -- {p.get('open_kernel')} px morphological opening "
        "removes brown stems and isolated scenery speckles; connected "
        f"components below {p.get('min_cluster_px')} px are discarded.",
        f"3. **Classification** -- each cluster is labelled by majority hue: "
        f"below {p.get('hue_green_min')} degrees is anomalous, above is "
        "healthy. The hue histogram is nearly empty between 65 and 105 "
        "degrees, so the boundary sits in a real valley rather than being "
        "fitted to it. Majority rather than mean, because brown stems fall "
        "in the anomaly band and would drag a healthy plant's average.",
        "",
        "## Detection results",
        "",
        _table([
            ("Images analysed", _fmt(summary["total_images"])),
            ("Frames judged (plants present)", _fmt(summary["judged"])),
            ("Frames containing an anomaly",
             _fmt(summary["frames_with_anomaly"])),
            ("Frame anomaly rate", _fmt(summary["anomaly_rate"])),
            ("Plant clusters found", _fmt(summary["clusters_total"])),
            ("Clusters classified anomalous",
             _fmt(summary["clusters_anomalous"])),
            ("Mean anomalous canopy area fraction",
             _fmt(summary["mean_anomalous_area_fraction"])),
        ]),
        "",
        "> Clusters are connected components, not plants. Touching bushes "
        "merge, so cluster counts undercount plants and a merged cluster is "
        "labelled by whichever colour dominates it. The area fraction needs "
        "no instance separation and is the more reliable quantity.",
        "",
    ]

    parts += ["## Detection accuracy (FP / FN)", ""]
    if scoring is None:
        parts += [
            "**Not measured.** No ground-truth file was supplied, so false "
            "positives and negatives are undetermined for this run.",
            "",
        ]
    elif scoring.get("mode") == "row":
        parts += [
            f"Scored at **row level** over {scoring['n']} rows. A row is "
            f"truly anomalous when it holds at least "
            f"{scoring['row_threshold']} diseased plant(s), and predicted "
            "anomalous when any frame from that row contained an anomalous "
            "cluster.",
            "",
            _table([
                ("True positives", _fmt(scoring["true_positives"])),
                ("False positives", _fmt(scoring["false_positives"])),
                ("False negatives", _fmt(scoring["false_negatives"])),
                ("True negatives", _fmt(scoring["true_negatives"])),
                ("Precision", _fmt(scoring["precision"])),
                ("Recall", _fmt(scoring["recall"])),
                ("F1", _fmt(scoring["f1"])),
                ("Accuracy", _fmt(scoring["accuracy"])),
                ("False positive rate", _fmt(scoring["false_positive_rate"])),
                ("False negative rate", _fmt(scoring["false_negative_rate"])),
            ]),
            "",
            "| Row | Truth | Prediction | Outcome |",
            "| --- | --- | --- | --- |",
        ]
        for row, v in scoring["per_row"].items():
            t, pr = v["truth_anomalous"], v["predicted_anomalous"]
            parts.append(
                f"| {row} | {'anomalous' if t else 'healthy'} | "
                f"{'anomalous' if pr else 'healthy'} | "
                f"{'TP' if t and pr else 'FN' if t else 'FP' if pr else 'TN'} |")
        parts.append("")
    else:
        parts += [
            "**No confusion matrix available.** Frames could not be "
            "associated with rows, so truth and prediction share no common "
            "unit.",
            "",
            _table([
                ("Frames judged", _fmt(scoring["frames_judged"])),
                ("Frames flagged", _fmt(scoring["frames_flagged"])),
                ("Predicted anomaly rate",
                 _fmt(scoring["predicted_anomaly_rate"])),
                ("Plants in ground truth", _fmt(scoring["plants_in_truth"])),
                ("Plants diseased", _fmt(scoring["plants_diseased"])),
                ("True diseased rate", _fmt(scoring["true_diseased_rate"])),
            ]),
            "",
            f"> {scoring['caveat']}",
            "",
        ]

    if gt_meta:
        parts += [
            "### Ground truth used",
            "",
            _table([
                ("File", f"`{gt_meta['source']}`"),
                ("Health field", f"`{gt_meta['health_key']}`"),
                ("Row field", f"`{gt_meta['row_key']}`"
                 if gt_meta["row_key"] else "none found"),
                ("Plant records", _fmt(gt_meta["records_found"])),
                ("Diseased / healthy",
                 f"{gt_meta['diseased']} / {gt_meta['healthy']}"),
            ]),
            "",
        ]

    parts += ["## Limitations", ""]
    parts += [f"{i}. {lim}" for i, lim in enumerate(PLANT_LIMITS, 1)]
    parts += ["", "## Per-frame detections", "",
              "| # | Image | Verdict | Anom area | Clusters | Anom | "
              "Plant % | Scenery px | Row |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i, d in enumerate(detections, 1):
        name = d.image.rsplit("/", 1)[-1]
        parts.append(
            f"| {i} | `{name}` | {d.verdict} | "
            f"{d.anomalous_area_fraction:.3f} | {d.n_clusters} | "
            f"{d.n_anomalous_clusters} | {d.plant_fraction*100:.1f}% | "
            f"{d.reject_scenery_px} | {d.row_id or '-'} |")
    parts += ["",
              "`Scenery px` is the count of hue-plausible but low-saturation "
              "pixels the plant gate rejected -- background conifers, mostly. "
              "If it drops to zero while scenery is still in shot, the "
              "saturation gate has stopped doing its job and the numbers "
              "above should not be trusted.",
              ""]
    return "\n".join(parts)
