"""Synthetic self-test -- proves the detector works without a mission run.

Three things are checked:

1. Nine scenes, four healthy and five affected, must be classified correctly
   as anomaly / not-anomaly. Current result: 9/9.
2. A frame of bare noisy soil must come back NO_VEGETATION. This is the
   dangerous failure mode -- a camera pointed at dirt silently passing every
   plant behind it -- and it is why the detector denoises before segmenting.
   Without the blur, soil noise puts 13.5% of pixels over the vegetation
   threshold and the frame is reported DISEASED.
3. An ablation on focal-necrosis frames, to check the chlorotic fraction is
   load-bearing rather than decorative. Current result: mean ExG alone 4/6,
   chlorotic fraction alone 6/6, both 6/6.

Note what this does and does not establish. 9/9 says the classification
logic is sound and the thresholds are self-consistent. It says nothing about
Gazebo renders or field imagery: these are flat-coloured synthetic scenes
with none of the specular highlights, shadow or motion blur a real capture
has. Cite it in the report as a sanity check, never as detection accuracy.
The accuracy number has to come from scoring real captures against
farm_ground_truth.json.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .crop_health import SENSITIVITY, analyse_image

SOIL = (110, 75, 50)
HEALTHY_LEAF = (60, 140, 45)
CHLOROTIC_LEAF = (170, 165, 60)
NECROTIC_LEAF = (120, 90, 45)

# (name, affected_fraction, necrotic_share_of_affected, expect_anomaly)
SCENES = [
    ("healthy_1",        0.00, 0.0, False),
    ("healthy_2",        0.05, 0.0, False),
    ("healthy_3",        0.00, 0.0, False),
    ("healthy_4",        0.08, 0.2, False),
    ("mild_chlorosis",   0.30, 0.0, True),
    ("focal_necrosis",   0.22, 1.0, True),
    ("blotchy_disease",  0.60, 0.3, True),
    ("severe_disease",   0.85, 0.5, True),
    ("necrotic",         0.95, 0.8, True),
]

# Focal-necrosis sweep used by the ablation: share of canopy taken by one
# compact severe lesion.
FOCAL_SWEEP = (0.22, 0.26, 0.30, 0.35, 0.40, 0.45)

SIZE = (240, 320)  # h, w
CANOPY_FRACTION = 0.55
RNG = np.random.default_rng(7)


def _smooth_field(h: int, w: int, radius: float = 12.0) -> np.ndarray:
    """A low-frequency random field in [0, 1].

    Used instead of per-pixel randomness because real chlorosis appears in
    contiguous patches. Per-pixel salt-and-pepper is not just unrealistic --
    the detector's denoise step averages it back into the surrounding green,
    so a scene built that way tests the blur rather than the detector.
    """
    base = (RNG.random((h, w)) * 255).astype(np.uint8)
    blurred = Image.fromarray(base).filter(
        ImageFilter.GaussianBlur(radius=radius))
    f = np.asarray(blurred).astype(np.float32) / 255.0
    lo, hi = float(f.min()), float(f.max())
    return (f - lo) / (hi - lo + 1e-6)


def _scene(affected: float, necrotic_share: float) -> np.ndarray:
    h, w = SIZE
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, :] = SOIL

    # Canopy mask: a few overlapping elliptical blobs, deterministic per call
    # via the module RNG so runs are comparable.
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), dtype=bool)
    for _ in range(6):
        cy, cx = RNG.integers(40, h - 40), RNG.integers(40, w - 40)
        ry, rx = RNG.integers(35, 70), RNG.integers(35, 80)
        mask |= (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) < 1.0
    # Trim toward the target canopy fraction so canopy size is not itself the
    # signal being measured.
    if mask.mean() > CANOPY_FRACTION:
        keep = _smooth_field(h, w, radius=6.0) < (CANOPY_FRACTION / mask.mean())
        mask &= keep

    # Assign leaf condition within the canopy, in patches.
    if affected > 0:
        field = _smooth_field(h, w)
        cut = float(np.quantile(field[mask], affected))
        affected_mask = mask & (field <= cut)
    else:
        affected_mask = np.zeros((h, w), dtype=bool)

    if necrotic_share > 0 and affected_mask.any():
        nfield = _smooth_field(h, w, radius=8.0)
        ncut = float(np.quantile(nfield[affected_mask], necrotic_share))
        necrotic_mask = affected_mask & (nfield <= ncut)
    else:
        necrotic_mask = np.zeros((h, w), dtype=bool)

    chlorotic_mask = affected_mask & ~necrotic_mask
    healthy_mask = mask & ~affected_mask

    img[healthy_mask] = HEALTHY_LEAF
    img[chlorotic_mask] = CHLOROTIC_LEAF
    img[necrotic_mask] = NECROTIC_LEAF

    img += RNG.normal(0, 6, img.shape)   # sensor noise
    return np.clip(img, 0, 255).astype(np.uint8)


def _soil_only() -> np.ndarray:
    h, w = SIZE
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, :] = SOIL
    img += RNG.normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def write_scenes(outdir: Path) -> list[tuple[Path, bool]]:
    """Write the synthetic scenes; returns (path, expect_anomaly) pairs."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, (name, affected, necrotic, expect) in enumerate(SCENES):
        path = outdir / f"{i:02d}_{name}.png"
        Image.fromarray(_scene(affected, necrotic)).save(path)
        written.append((path, expect))
    return written


def run(verbose: bool = False) -> int:
    """Returns 0 on pass, 1 on failure -- usable as an exit code."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        scenes = write_scenes(tmp)

        correct = 0
        if verbose:
            print(f"{'scene':<20} {'verdict':<14} {'meanExG':>8} "
                  f"{'chlor':>7} {'expect':>9}  ok")
        for path, expect in scenes:
            det = analyse_image(path)
            ok = det.anomaly == expect
            correct += ok
            if verbose:
                print(f"{path.stem:<20} {det.verdict:<14} "
                      f"{det.mean_exg:>8.3f} {det.chlorotic_fraction:>7.3f} "
                      f"{'anomaly' if expect else 'healthy':>9}  "
                      f"{'PASS' if ok else 'FAIL'}")

        soil = tmp / "soil_only.png"
        Image.fromarray(_soil_only()).save(soil)
        soil_det = analyse_image(soil)
        soil_ok = soil_det.verdict == "NO_VEGETATION"
        if verbose:
            print(f"{'soil_only':<20} {soil_det.verdict:<14} "
                  f"{'-':>8} {'-':>7} {'no veg':>9}  "
                  f"{'PASS' if soil_ok else 'FAIL'}")

        # Ablation. Does the chlorotic fraction earn its place, or would the
        # mean do on its own? The broad scene set above does not answer that
        # -- both features separate it -- so the ablation runs on focal
        # necrosis, where a compact severe lesion leaves the frame average
        # close to healthy. Each feature is scored alone against the same
        # frames, using its own threshold.
        cfg = SENSITIVITY["normal"]
        focal = []
        for i, a in enumerate(FOCAL_SWEEP):
            p = tmp / f"focal_{i}.png"
            Image.fromarray(_scene(a, 1.0)).save(p)
            focal.append((a, analyse_image(p)))

        mean_only = sum(1 for _, d in focal
                        if d.mean_exg < cfg["stressed_exg"])
        frac_only = sum(1 for _, d in focal
                        if d.chlorotic_fraction > cfg["stressed_frac"])
        both = sum(1 for _, d in focal if d.anomaly)
        if verbose:
            n = len(focal)
            print(f"\nablation on {n} focal-necrosis frames "
                  f"({int(min(FOCAL_SWEEP)*100)}-{int(max(FOCAL_SWEEP)*100)}%"
                  " of canopy affected), anomalies caught:")
            print(f"  mean ExG alone .......... {mean_only}/{n}")
            print(f"  chlorotic fraction alone  {frac_only}/{n}")
            print(f"  both (shipped detector) .. {both}/{n}")

    total = len(scenes)
    if verbose:
        print(f"\nbinary anomaly detection: {correct}/{total}")
        print(f"no-vegetation guard: {'PASS' if soil_ok else 'FAIL'}")
    failed = (correct != total) or not soil_ok
    if verbose and failed:
        print("\nSELFTEST FAILED -- do not cite detector numbers until this "
              "passes.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run(verbose=True))
