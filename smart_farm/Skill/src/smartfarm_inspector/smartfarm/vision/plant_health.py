"""Per-plant anomaly detection for the Gazebo farm world.

This is the detector for *this* world, where plant health is encoded as model
colour: green bushes are normal, yellow/tan bushes are anomalies. It is a
different job from `crop_health.py`, which scores a whole frame by greenness,
and it is the right tool here for a specific reason.

## Why hue, not ExG

The Excess Green Index cannot do this job. ExG asks "how green-dominant is
this pixel", and yellow scores middling -- close enough to stressed foliage
that separating the two means threshold-hunting. Worse, the conifers in the
background scenery are a desaturated olive that ExG reads as severely
diseased: run `crop_health` on a frame of this world and it returns DISEASED
with a 0.65 chlorotic fraction, entirely from scenery.

Hue separates the three populations outright. Measured on a real capture from
this world:

    population            median hue   median saturation   pixels
    green plants             112 deg          0.74           6557
    yellow plants             62 deg          0.74           1975
    tan/orange plants         37 deg          0.74           4113
    conifer scenery           71 deg          0.28           8637
    sky / ground / hills        --            0.00         272000

Two gates fall out of that table, both with wide margins:

  Saturation > 0.45 keeps plants and drops everything else. Plants sit at
  0.74, conifers at 0.28, and the achromatic background at exactly 0.00
  (grey and white have no hue at all). Nothing lives near the threshold.

  Hue >= 85 deg is healthy, below is anomalous. The histogram between 65 and
  105 deg is nearly empty -- 483 pixels out of 13293 -- so the boundary sits
  in a real valley rather than being fitted. Both anomaly colours, yellow at
  62 and tan at 37, land on the same side.

The conifers deserve note: their hue is 71 deg, which is *inside* the anomaly
band. Saturation is the only thing separating them from a diseased plant, so
if a world adds saturated-olive scenery this detector will need a shape or
height gate too. That is why `reject_scenery_px` is reported per frame --
watch it, and if it collapses to zero while conifers are still in shot,
something changed.

## What a cluster is, and is not

Connected components on the plant mask give *clusters*, not plants. Adjacent
bushes touch in image space and merge: the real capture yields 7 clusters for
what is visibly a dozen or more plants. So cluster counts are reported and
labelled as clusters, and the primary quantity is `anomalous_area_fraction`
-- the share of plant canopy that is anomalous -- which needs no instance
separation to be meaningful.

Each cluster is classified by *majority* hue rather than mean, because bush
stems are brown and fall in the anomaly band. Majority voting means a green
bush with a brown stem still reads healthy; averaging would drag it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

KNOWN_LIMITATIONS = [
    "Classifies by model colour, not by plant pathology. It reports what the "
    "world encodes (green vs yellow model), which is exactly the ground "
    "truth in simulation and would need replacing on real imagery.",
    "Clusters are not plants. Touching bushes merge into one connected "
    "component, so cluster counts undercount plants and no per-plant "
    "confusion matrix is available from this data.",
    "Background scenery is rejected by saturation alone. Conifer hue (71 "
    "deg) sits inside the anomaly band; only its low saturation (0.28) "
    "excludes it. Saturated non-crop vegetation would be misread.",
    "Fixed simulator lighting and flat shading. Real imagery has specular "
    "highlights and shadow that move both hue and saturation.",
    "Thin structures are removed by a 5 px opening. A genuinely thin "
    "anomalous plant, or a seedling, would be filtered out with the stems.",
]

# --- Gates, all measured on a capture from this world; see module docstring.

# Plants sit at saturation 0.74, conifer scenery at 0.28, sky/ground at 0.00.
SAT_MIN = 0.45
# Excludes near-black pixels, where hue is numerically unstable.
VAL_MIN = 0.04
# Plausible vegetation hue range; outside this is not a plant model.
HUE_MIN, HUE_MAX = 15.0, 175.0
# The valley between yellow plants (62 deg) and green plants (112 deg).
HUE_GREEN_MIN = 85.0

# Opening kernel, in pixels. Removes brown stems and isolated scenery
# speckles without eating into bush bodies.
OPEN_KERNEL = 5
# A connected component smaller than this is speckle, not a plant.
MIN_CLUSTER_PX = 150
# Below this share of the frame there is no plant worth judging.
MIN_PLANT_FRACTION = 0.002

VERDICTS = ("HEALTHY", "ANOMALY", "NO_PLANTS")


@dataclass
class Cluster:
    x: int
    y: int
    w: int
    h: int
    area: int
    median_hue: float
    anomalous: bool


@dataclass
class PlantDetection:
    """Deliberately duck-type compatible with `score.py` and
    `crop_health.attach_waypoints`: both need only `row_id`, `judged` and
    `anomaly`, so row-level FP/FN scoring works on these unchanged."""
    image: str
    verdict: str                    # HEALTHY | ANOMALY | NO_PLANTS
    anomalous_area_fraction: float
    plant_px: int
    healthy_px: int
    anomalous_px: int
    reject_scenery_px: int
    plant_fraction: float
    n_clusters: int
    n_anomalous_clusters: int
    clusters: list[Cluster] = field(default_factory=list)
    row_id: str | None = None
    field_id: str | None = None
    seq: int | None = None

    @property
    def anomaly(self) -> bool:
        return self.verdict == "ANOMALY"

    @property
    def judged(self) -> bool:
        return self.verdict != "NO_PLANTS"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["anomaly"] = self.anomaly
        return d


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected-component labelling, 4-connectivity.

    Uses scipy when present and falls back to a two-pass union-find so the
    module has no hard scipy dependency -- a ROS 2 install is not guaranteed
    to have it and a missing import at 2am is not a useful failure.
    """
    try:
        from scipy import ndimage
        return ndimage.label(mask)
    except ImportError:
        pass

    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent: list[int] = [0]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nxt = 1
    for y in range(h):
        row = mask[y]
        for x in np.flatnonzero(row):
            up = labels[y - 1, x] if y else 0
            left = labels[y, x - 1] if x else 0
            if up and left:
                labels[y, x] = min(up, left)
                union(up, left)
            elif up or left:
                labels[y, x] = up or left
            else:
                labels[y, x] = nxt
                parent.append(nxt)
                nxt += 1

    remap = np.zeros(nxt, dtype=np.int32)
    seen: dict[int, int] = {}
    for i in range(1, nxt):
        r = find(i)
        if r not in seen:
            seen[r] = len(seen) + 1
        remap[i] = seen[r]
    return remap[labels], len(seen)


def _open(mask: np.ndarray, k: int) -> np.ndarray:
    """Morphological opening via PIL rank filters (no scipy needed)."""
    if k <= 1:
        return mask
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img = img.filter(ImageFilter.MinFilter(k)).filter(ImageFilter.MaxFilter(k))
    return np.asarray(img) > 127


def analyse_image(path: Path, sat_min: float = SAT_MIN,
                  hue_green_min: float = HUE_GREEN_MIN,
                  min_cluster_px: int = MIN_CLUSTER_PX) -> PlantDetection:
    im = Image.open(path).convert("RGB")
    hsv = np.asarray(im.convert("HSV")).astype(np.float32)
    hue = hsv[:, :, 0] * 360.0 / 255.0
    sat = hsv[:, :, 1] / 255.0
    val = hsv[:, :, 2] / 255.0

    in_hue = (hue >= HUE_MIN) & (hue <= HUE_MAX) & (val > VAL_MIN)
    plants = in_hue & (sat > sat_min)
    # Everything hue-plausible but too washed out to be a plant model. Watch
    # this number: it is the scenery the saturation gate is throwing away.
    scenery = int((in_hue & (sat > 0.15) & (sat <= sat_min)).sum())

    plants = _open(plants, OPEN_KERNEL)
    labels, n = _label(plants)

    clusters: list[Cluster] = []
    kept = np.zeros_like(plants)
    for i in range(1, n + 1):
        m = labels == i
        area = int(m.sum())
        if area < min_cluster_px:
            continue
        ys, xs = np.where(m)
        hues = hue[m]
        # Majority vote, not mean: brown stems sit in the anomaly band and
        # would drag a green bush's average.
        anomalous = bool((hues < hue_green_min).mean() > 0.5)
        clusters.append(Cluster(
            x=int(xs.min()), y=int(ys.min()),
            w=int(xs.max() - xs.min() + 1), h=int(ys.max() - ys.min() + 1),
            area=area, median_hue=round(float(np.median(hues)), 1),
            anomalous=anomalous))
        kept |= m

    plant_px = int(kept.sum())
    plant_fraction = plant_px / kept.size
    if plant_px == 0 or plant_fraction < MIN_PLANT_FRACTION:
        # No plant in frame. Reporting HEALTHY here would be the dangerous
        # answer -- a camera pointed at sky silently passing the whole row.
        return PlantDetection(
            str(path), "NO_PLANTS", 0.0, plant_px, 0, 0, scenery,
            round(plant_fraction, 5), 0, 0, [])

    anomalous_px = int((kept & (hue < hue_green_min)).sum())
    healthy_px = plant_px - anomalous_px
    n_anom = sum(1 for c in clusters if c.anomalous)

    return PlantDetection(
        image=str(path),
        verdict="ANOMALY" if n_anom else "HEALTHY",
        anomalous_area_fraction=round(anomalous_px / plant_px, 4),
        plant_px=plant_px,
        healthy_px=healthy_px,
        anomalous_px=anomalous_px,
        reject_scenery_px=scenery,
        plant_fraction=round(plant_fraction, 5),
        n_clusters=len(clusters),
        n_anomalous_clusters=n_anom,
        clusters=clusters,
    )


def analyse_directory(images_dir: Path, limit: int | None = None,
                      **kw) -> list[PlantDetection]:
    """Oldest first -- that is capture order, which is how frames map to
    waypoints."""
    from .crop_health import ImagesDirMissing
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise ImagesDirMissing(images_dir)
    paths = sorted(
        [p for p in images_dir.iterdir()
         if p.suffix.lower() in (".png", ".jpg", ".jpeg")],
        key=lambda p: p.stat().st_mtime)
    if limit:
        paths = paths[:limit]
    return [analyse_image(p, **kw) for p in paths]


def summarize(detections: list[PlantDetection]) -> dict:
    counts: dict[str, int] = {}
    for d in detections:
        counts[d.verdict] = counts.get(d.verdict, 0) + 1
    judged = [d for d in detections if d.judged]
    return {
        "total_images": len(detections),
        "judged": len(judged),
        "counts": counts,
        "frames_with_anomaly": sum(1 for d in judged if d.anomaly),
        "anomaly_rate": (round(sum(1 for d in judged if d.anomaly)
                               / len(judged), 3) if judged else None),
        "clusters_total": sum(d.n_clusters for d in judged),
        "clusters_anomalous": sum(d.n_anomalous_clusters for d in judged),
        "mean_anomalous_area_fraction": (
            round(float(np.mean([d.anomalous_area_fraction for d in judged])), 4)
            if judged else None),
    }


def overlay(path: Path, detection: PlantDetection, out: Path) -> Path:
    """Write a visual check: green plants boxed blue, anomalies boxed red.

    Look at this before trusting any number. It is the fastest way to catch
    the failure that matters -- scenery being counted as a plant.
    """
    from PIL import ImageDraw
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    for c in detection.clusters:
        colour = (220, 30, 30) if c.anomalous else (30, 80, 220)
        d.rectangle([c.x, c.y, c.x + c.w, c.y + c.h], outline=colour, width=2)
        d.text((c.x + 2, max(0, c.y - 11)),
               f"{'ANOM' if c.anomalous else 'OK'} h{c.median_hue:.0f}",
               fill=colour)
    d.text((4, 4), f"{detection.verdict}  "
                   f"anom_area={detection.anomalous_area_fraction:.2f}  "
                   f"clusters={detection.n_clusters}"
                   f"({detection.n_anomalous_clusters} anom)",
           fill=(220, 30, 30) if detection.anomaly else (30, 80, 220))
    out = Path(out)
    im.save(out)
    return out
