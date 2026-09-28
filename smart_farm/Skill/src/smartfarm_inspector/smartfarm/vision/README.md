# smartfarm/vision

Crop health detection from the RGB imagery captured during a mission. Turns a
directory of captures into per-frame verdicts, and — given a ground-truth
file — into an FP/FN number.

No trained model, no weights to download, no GPU. Dependencies are `numpy`
and `Pillow`, both of which a ROS 2 install already has.

## Install

Drop the directory in place and confirm it imports:

```bash
python3 -m smartfarm.vision.cli selftest
```

That generates synthetic scenes and classifies them. It needs no simulator,
no captures and no ground truth, so it is the first thing to run — if it
fails, nothing downstream is worth looking at.

## Use

```bash
# 1. verify the module works
python3 -m smartfarm.vision.cli selftest

# 2. analyse a mission's captures
python3 -m smartfarm.vision.cli analyse ~/.ros/media/realsense/

# 3. add row labels and an actual FP/FN score
python3 -m smartfarm.vision.cli analyse ~/.ros/media/realsense/ \
    --mission-log mission.json \
    --ground-truth farm_ground_truth.json \
    --outdir vision_out
```

Outputs land in `--outdir` (default `vision_out/`):

- `detections.json` — machine-readable: every frame, both feature values, the
  verdict, the row it came from, and the scoring block.
- `crop_health_report.md` — the human-readable report, written to be pasted
  into D5 without editing.

If the ground-truth file will not parse, check what the scorer makes of it
before touching any code:

```bash
python3 -m smartfarm.vision.cli inspect farm_ground_truth.json
```

That prints which field it picked as the health state, which as the row, and
how many records it could use. If it guessed wrong, `--health-key` and
`--row-key` override it. That is a flag, not an edit.

## Method

`ExG = 2g - r - b` on normalised RGB — the Excess Green Index, a standard
vegetation index in precision agriculture. Healthy foliage is strongly
green-dominant and scores high. Chlorosis (the yellowing that accompanies
most plant stress) raises red and drops the score; necrosis drops it
further.

Two features per frame, not one:

- `mean_exg` — average index over canopy pixels.
- `chlorotic_fraction` — share of canopy below the chlorosis threshold.

The second one is load-bearing. A compact severe lesion over a quarter of the
canopy barely moves the average but clearly moves the fraction. The self-test
ablation measures this on focal-necrosis frames: mean alone 4/6, fraction
alone 6/6, both 6/6.

Frames are denoised (1.5 px Gaussian) before the index is computed. This is
not cosmetic: on a noisy soil-only frame, 13.5% of raw pixels cross the
vegetation threshold and the frame comes back `DISEASED`. After the blur,
0.0% do, and it correctly reports `NO_VEGETATION`.

Three sensitivity presets (`--sensitivity low|normal|high`) move the
thresholds. `normal` is the calibrated default; `high` trades false positives
for recall.

## Scoring, and what it can honestly claim

A detection judges a *frame*; a frame contains several plants. So a per-plant
confusion matrix is not available from this data and the scorer does not
pretend otherwise. It has two modes:

**Row mode** (preferred). Needs `row_id` on each detection, which comes from
`--mission-log`. A row is truly anomalous if it holds at least
`--row-threshold` diseased plants (default 1); it is predicted anomalous if
any frame from that row was flagged. Yields a real TP/FP/FN/TN over rows.

**Aggregate mode** (fallback). Used when rows are unavailable. Compares the
predicted anomaly rate against the true diseased rate. This is a calibration
check, *not* a confusion matrix — it cannot separate a false positive from a
false negative, and both the JSON and the report say so.

Always quote the unit: "recall 0.83 over 6 rows" is defensible; "recall 0.83"
is not.

### Row association is order-based

`devices_manager` writes timestamped filenames and does not preserve the
`config_json` attached to each `take_picture` action, so the row has to be
recovered by matching capture order to waypoint order. That holds when every
waypoint captured exactly once and nothing was skipped. If the counts do not
match, `attach_waypoints` refuses and leaves the labels empty rather than
guessing — a wrong row label is worse than no row label — and scoring drops
to aggregate mode with a warning on stderr.

## Known limitations

These are in `crop_health.KNOWN_LIMITATIONS` and are reproduced in every
generated report. They belong in the deliverable, not buried in a docstring.

1. Detects chlorosis, not disease. Drought, nitrogen deficiency and infection
   produce similar signatures.
2. Requires vegetation in frame. Mostly-soil or mostly-sky frames are
   reported `NO_VEGETATION` rather than healthy.
3. Fixed simulator lighting. Outdoors, time of day and cloud cover shift the
   thresholds and would need per-site calibration.
4. Judges the whole frame, not individual plants.
5. Thresholds calibrated on synthetic scenes, not field data.
6. Detection floor is roughly 20% of canopy area affected at default
   sensitivity. Early-stage disease on a single leaf reads as healthy.

The self-test's 9/9 is a logic check on flat-coloured synthetic scenes with
no specular highlights, shadow or motion blur. It is not detection accuracy
and must not be reported as such. The accuracy number comes from scoring real
captures against the ground truth of the world that was actually loaded.

## Files

| File | Purpose |
| --- | --- |
| `crop_health.py` | The detector: index, segmentation, thresholds, verdicts |
| `score.py` | Ground-truth parsing and the confusion matrix |
| `report.py` | Markdown report generation |
| `selftest.py` | Synthetic scenes, no-vegetation guard, feature ablation |
| `cli.py` | `analyse` / `inspect` / `selftest` entry points |

---

# Which detector to use

There are two, and for your Gazebo world you want the second one.

| | `analyse` (crop_health) | `plants` (plant_health) |
| --- | --- | --- |
| Question it asks | how green is this canopy | which plants are green, which are yellow |
| Feature | Excess Green Index | HSV hue, gated on saturation |
| Unit | whole frame | per plant cluster |
| Right when | health shows as greenness, real or realistic imagery | health is encoded as model colour |
| On the tree world | returns DISEASED from conifer scenery alone | rejects scenery, splits green from yellow |

`crop_health` fails on this world for a concrete reason: conifer needles are a
desaturated olive that ExG reads as severely chlorotic, so a frame of pure
background scenery comes back `DISEASED` with a 0.65 chlorotic fraction. Hue
separates the populations instead, and saturation throws the scenery out.

## plants -- the detector for this world

```bash
# look at the pictures first
python3 -m smartfarm.vision.cli plants /path/to/media/realsense --outdir ~/vision_out

# then add rows and score it
python3 -m smartfarm.vision.cli plants /path/to/media/realsense \
    --mission-log /abs/path/mission.json \
    --ground-truth /abs/path/farm_ground_truth.json \
    --outdir ~/vision_out
```

Outputs: `plant_detections.json`, `plant_health_report.md`, and
`overlays/` -- one annotated image per frame, blue box for a healthy plant,
red for an anomaly, with the median hue printed on each. **Open the overlays
before quoting any number.** They are the fastest way to catch the failure
that actually matters: scenery being counted as a plant.

### The gates, and where they came from

Measured on the capture from your world, not guessed:

| population | median hue | median saturation | pixels |
| --- | --- | --- | --- |
| green plants | 112 deg | 0.74 | 6557 |
| yellow plants | 62 deg | 0.74 | 1975 |
| tan/orange plants | 37 deg | 0.74 | 4113 |
| conifer scenery | 71 deg | 0.28 | 8637 |
| sky / ground / hills | -- | 0.00 | 272000 |

- **`--sat-min` (0.45)** separates plants from everything else. Plants sit at
  0.74, conifers at 0.28, achromatic background at exactly 0.00. Nothing
  lives near the threshold.
- **`--hue-green-min` (85)** splits healthy from anomalous. The hue histogram
  holds 483 pixels out of 13293 between 65 and 105 degrees, so the boundary
  sits in a real valley. Both anomaly colours, yellow at 62 and tan at 37,
  fall on the same side.

Note the trap: conifer hue is 71 degrees, *inside* the anomaly band.
Saturation is the only thing keeping them out. That is why every report
prints `Scenery px` -- the count of hue-plausible, low-saturation pixels the
gate rejected. If that number collapses to zero while conifers are still in
shot, the gate has stopped working and the numbers are not trustworthy.

### Clusters are not plants

Connected components merge touching bushes: the real capture gives 7 clusters
for what is visibly a dozen or more plants. So cluster counts undercount, and
a merged cluster takes the label of whichever colour dominates it. The
primary quantity is `anomalous_area_fraction`, the share of plant canopy that
is anomalous, which needs no instance separation. Each cluster is classified
by *majority* hue rather than mean, because brown stems fall in the anomaly
band and would drag a healthy plant's average.

### Verified

On your frame: 7 clusters, 4 anomalous, `anomalous_area_fraction` 0.47,
8637 scenery pixels rejected, zero boxes on any conifer. Row-level scoring
was checked with a deliberately wrong ground-truth row and correctly
returned one false negative (TP=4 FP=0 FN=1 TN=1), so the confusion matrix
is not merely printing zeros.
