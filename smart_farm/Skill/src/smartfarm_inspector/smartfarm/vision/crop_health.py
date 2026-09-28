"""Crop health detection using RGB vegetation indices.

No trained model, no download, no GPU. Health is judged from colour physics
using the Excess Green Index:

    ExG = 2g - r - b        (r, g, b normalised so r+g+b = 1)

ExG is a standard index in precision agriculture. Healthy foliage is strongly
green-dominant and scores high; chlorosis -- the yellowing that accompanies
most plant stress and disease -- raises the red channel and drops the score;
necrotic tissue goes brown and drops it further.

## Why not a trained classifier

A model trained on real leaf photographs (PlantVillage and similar) would have
to run on Gazebo-rendered plants seen from 2-3 m. That domain gap is a
research problem, not a weekend task. ExG measures colour, so it behaves the
same on renders and on real photographs -- there is nothing to transfer.

The honest limit: this detects *chlorosis*, not disease. A plant yellowing
from drought, nitrogen deficiency or infection all score alike. That is
stated in KNOWN_LIMITATIONS and belongs in the report rather than buried.

## Two features, not one

`mean_exg` alone misses focal damage: a compact severe lesion covering a
quarter of the canopy leaves the frame average close to healthy.
`chlorotic_fraction` -- the share of canopy pixels below the chlorosis
threshold -- catches it, because it counts affected area instead of
averaging it away.

Measured, not assumed. `selftest.py` runs an ablation on focal-necrosis
frames (22-45% of canopy affected) and the current numbers are: mean ExG
alone 4/6, chlorotic fraction alone 6/6, both 6/6. On the broad nine-scene
set either feature separates the classes on its own, so that set does not
justify the second feature -- the focal sweep is what does. Re-run
`selftest` after changing any threshold and quote what it prints, not these
numbers.

## Detection floor

The same sweep puts the smallest reliably-detected lesion at roughly 20% of
canopy area at default sensitivity. Below that, both features stay inside
the healthy band and the frame is reported HEALTHY. Early-stage disease on
a single leaf is therefore out of reach for this method, which is a property
of frame-level colour statistics rather than a tuning problem.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

KNOWN_LIMITATIONS = [
    "Detects chlorosis (yellowing), not disease. Drought stress, nitrogen "
    "deficiency and infection produce similar signatures.",
    "Requires vegetation in frame. Images that are mostly soil or sky are "
    "reported NO_VEGETATION rather than healthy.",
    "Fixed simulator lighting. Outdoors, time of day and cloud cover would "
    "shift the thresholds and need per-site calibration.",
    "Judges the whole frame, not individual plants. A frame containing one "
    "diseased plant among four healthy ones reads as mildly stressed.",
    "Thresholds were calibrated on synthetic scenes, not on field data.",
    "Detection floor is roughly 20% of canopy area affected at default "
    "sensitivity. Early-stage disease on a single leaf reads as healthy.",
]

# Segmentation: pixels above this ExG are treated as vegetation rather than
# soil, sky or structure.
VEG_THRESHOLD = 0.05

# Denoise radius, in pixels, applied before the index is computed.
#
# This is not cosmetic. Sensor noise on bare soil scatters individual pixels
# across the vegetation threshold: on a noisy soil-only frame, 13.5% of
# pixels read as vegetation and the frame comes back DISEASED instead of
# NO_VEGETATION. A 1.5 px blur drops that to 0.0% while leaving real canopy
# untouched (healthy canopy 0.52 -> 0.46 of frame, mean ExG 0.674 -> 0.688).
# Set to 0 to disable if the camera output is already filtered.
DENOISE_RADIUS = 1.5

# Chlorosis threshold. Canopy pixels below this are yellowing.
CHLOROSIS_THRESHOLD = 0.30

# Classification bands on (mean_exg, chlorotic_fraction).
SENSITIVITY = {
    "low":    {"diseased_exg": 0.20, "diseased_frac": 0.70,
               "stressed_exg": 0.42, "stressed_frac": 0.35},
    "normal": {"diseased_exg": 0.25, "diseased_frac": 0.55,
               "stressed_exg": 0.50, "stressed_frac": 0.20},
    "high":   {"diseased_exg": 0.30, "diseased_frac": 0.45,
               "stressed_exg": 0.56, "stressed_frac": 0.12},
}

# A frame needs this many canopy pixels AND this share of the frame before a
# verdict is issued. The absolute floor catches tiny images; the fraction
# catches large ones, where 500 stray pixels is noise rather than a plant.
MIN_VEG_PIXELS = 500
MIN_CANOPY_FRACTION = 0.01

VERDICTS = ("HEALTHY", "STRESSED", "DISEASED", "NO_VEGETATION")

# Places this stack has been observed to write captures to, in the order
# worth checking. Used only to make the "directory missing" error useful.
CANDIDATE_DIRS = (
    "~/.ros/media/realsense",
    "~/.ros/media",
    "~/.ros",
    "~/.gazebo/pictures",
    "~/media/realsense",
    "/tmp/realsense",
)


class ImagesDirMissing(FileNotFoundError):
    """Raised when the capture directory does not exist.

    Carries the search it already did, because "no such directory" on its own
    sends you looking in the wrong place: the usual cause is not a typo but
    that capture never wrote anything, or wrote somewhere else.
    """

    def __init__(self, requested: Path):
        self.requested = Path(requested)
        self.found = self._scan()
        super().__init__(self._message())

    def _scan(self) -> list[tuple[Path, int]]:
        out = []
        for cand in CANDIDATE_DIRS:
            p = Path(cand).expanduser()
            if p.is_dir():
                n = sum(1 for f in p.rglob("*")
                        if f.suffix.lower() in (".png", ".jpg", ".jpeg"))
                out.append((p, n))
        return out

    def _message(self) -> str:
        lines = [f"capture directory does not exist: {self.requested}", ""]
        with_images = [(p, n) for p, n in self.found if n]
        if with_images:
            lines.append("Images were found elsewhere -- try one of these:")
            for p, n in with_images:
                lines.append(f"    {p}   ({n} image(s))")
        elif self.found:
            lines.append("These directories exist but hold no images:")
            for p, _ in self.found:
                lines.append(f"    {p}")
            lines.append("")
            lines.append("So capture is not writing anything. Run a mission "
                         "and check again before analysing.")
        else:
            lines.append("None of the usual capture locations exist either:")
            for c in CANDIDATE_DIRS:
                lines.append(f"    {c}")
            lines.append("")
            lines.append("Widen the search with:")
            lines.append("    find ~ -name '*.png' -newermt '-2 hours' "
                         "2>/dev/null | head -20")
        return "\n".join(lines)


@dataclass
class Detection:
    image: str
    verdict: str            # HEALTHY | STRESSED | DISEASED | NO_VEGETATION
    mean_exg: float
    chlorotic_fraction: float
    canopy_fraction: float
    sensitivity: str
    row_id: str | None = None
    field_id: str | None = None
    seq: int | None = None

    @property
    def anomaly(self) -> bool:
        return self.verdict in ("STRESSED", "DISEASED")

    @property
    def judged(self) -> bool:
        return self.verdict != "NO_VEGETATION"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["anomaly"] = self.anomaly
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Detection":
        fields = {k: v for k, v in d.items()
                  if k in cls.__dataclass_fields__}
        return cls(**fields)


def exg(img: np.ndarray) -> np.ndarray:
    """Excess Green Index over normalised RGB."""
    f = img.astype(np.float32) / 255.0
    s = f.sum(axis=2) + 1e-6
    r, g, b = f[:, :, 0] / s, f[:, :, 1] / s, f[:, :, 2] / s
    return 2 * g - r - b


def analyse_image(path: Path, sensitivity: str = "normal") -> Detection:
    if sensitivity not in SENSITIVITY:
        raise ValueError(
            f"unknown sensitivity {sensitivity!r}; "
            f"expected one of {sorted(SENSITIVITY)}")
    cfg = SENSITIVITY[sensitivity]
    im = Image.open(path).convert("RGB")
    if DENOISE_RADIUS:
        im = im.filter(ImageFilter.GaussianBlur(radius=DENOISE_RADIUS))
    e = exg(np.asarray(im))

    veg = e > VEG_THRESHOLD
    canopy_fraction = float(veg.mean())

    if veg.sum() < MIN_VEG_PIXELS or canopy_fraction < MIN_CANOPY_FRACTION:
        # Not enough plant in frame to judge. Reporting this as healthy would
        # be the most dangerous behaviour available -- a camera pointed at
        # soil would silently pass every plant behind it.
        return Detection(str(path), "NO_VEGETATION", 0.0, 0.0,
                         round(canopy_fraction, 4), sensitivity)

    values = e[veg]
    mean_exg = float(values.mean())
    chlorotic = float((values < CHLOROSIS_THRESHOLD).mean())

    if mean_exg < cfg["diseased_exg"] or chlorotic > cfg["diseased_frac"]:
        verdict = "DISEASED"
    elif mean_exg < cfg["stressed_exg"] or chlorotic > cfg["stressed_frac"]:
        verdict = "STRESSED"
    else:
        verdict = "HEALTHY"

    return Detection(str(path), verdict, round(mean_exg, 4),
                     round(chlorotic, 4), round(canopy_fraction, 4),
                     sensitivity)


def analyse_directory(images_dir: Path, sensitivity: str = "normal",
                      limit: int | None = None) -> list[Detection]:
    """Analyse every image in a directory, oldest first.

    Sorted by modification time because that is the order the mission
    captured them, which is how they get associated with waypoints.
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise ImagesDirMissing(images_dir)
    paths = sorted(
        [p for p in images_dir.iterdir()
         if p.suffix.lower() in (".png", ".jpg", ".jpeg")],
        key=lambda p: p.stat().st_mtime)
    if limit:
        paths = paths[:limit]
    return [analyse_image(p, sensitivity) for p in paths]


def attach_waypoints(detections: list[Detection], mission_log: dict) -> bool:
    """Associate detections with the waypoints that produced them.

    The stack's `devices_manager` writes timestamped filenames and does not
    preserve the `config_json` we attach to each take_picture action, so the
    row and field have to be recovered by matching capture order to waypoint
    order. That holds as long as every waypoint captured exactly once and
    nothing was skipped -- both true for a mission that completed cleanly.

    If the counts do not match, the association is left empty rather than
    guessed: a wrong row label is worse than no row label. Returns True when
    the association was applied, False when it was declined.
    """
    waypoints = (mission_log.get("task", {}).get("waypoints")
                 or mission_log.get("waypoints") or [])
    if not waypoints or len(waypoints) != len(detections):
        return False
    for det, wp in zip(detections, waypoints):
        det.row_id = wp.get("row_id")
        det.field_id = wp.get("field_id")
        det.seq = wp.get("waypoint_id")
    return True


def summarize(detections: list[Detection]) -> dict:
    counts: dict[str, int] = {}
    for d in detections:
        counts[d.verdict] = counts.get(d.verdict, 0) + 1
    judged = [d for d in detections if d.judged]
    anomalies = sum(1 for d in judged if d.anomaly)
    return {
        "total_images": len(detections),
        "judged": len(judged),
        "counts": counts,
        "anomalies": anomalies,
        "anomaly_rate": (round(anomalies / len(judged), 3) if judged else None),
    }
