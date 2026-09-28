"""Field registry: the bridge between farmer vocabulary and map coordinates.

The parser produces field_ids as a farmer would say them -- "the north
field", "sector 3", "2". None of those can be navigated to. This module
resolves them against the field map, and says so plainly when it cannot.

Resolution is deliberately conservative. An unresolvable field is an error,
not a guess: silently inspecting the wrong field is worse than refusing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

def _default_config() -> Path:
    from ..config_paths import config_file
    return config_file("field_map.yaml")


CONFIG = None   # resolved lazily so the installed share dir is found

Point = tuple[float, float]


def point_in_polygon(pt: Point, polygon: list[list[float]]) -> bool:
    x, y = pt
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
    return inside


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and filler words a farmer would use."""
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\b(the|a|an|please|all|of)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class Resolution:
    """Outcome of resolving one field_id."""

    raw: str
    field_id: str | None
    matched_alias: str | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.field_id is not None


class FieldMap:
    def __init__(self, path: Path | None = None):
        path = Path(path) if path else _default_config()
        self.raw = yaml.safe_load(path.read_text())
        self.fields: dict = self.raw["fields"]
        self.rows: dict = self.raw["rows"]
        self.landmarks: dict = self.raw["landmarks"]
        self.geofence = self.raw["geofence"]
        self.no_go_zones = self.raw.get("no_go_zones", [])
        self.base_pose: Point = tuple(self.raw["base_station"]["pose"][:2])
        self.robot_cfg = self.raw["robot"]
        self.limits = self.raw.get("limits", {})
        self.devices = self.raw.get("devices", {})
        self.map_name = self.raw.get("map_name", "")
        self.frame_id = self.raw.get("frame_id", "map")
        self.farm_name = self.raw.get("farm_name", self.raw.get("farm_id", ""))
        self._alias_index = self._build_alias_index()
        self._check_consistency()

    def _build_alias_index(self) -> dict[str, str]:
        idx: dict[str, str] = {}
        for fid, f in self.fields.items():
            idx[_normalize(fid)] = fid
            for alias in f.get("aliases", []):
                idx[_normalize(alias)] = fid
        return idx

    def _check_consistency(self) -> None:
        """Catch field-map mistakes at load, not mid-mission."""
        for fid, f in self.fields.items():
            for rid in f["rows"]:
                if rid not in self.rows:
                    raise ValueError(f"field '{fid}' lists unknown row '{rid}'")
        for rid, r in self.rows.items():
            if r["field"] not in self.fields:
                raise ValueError(f"row '{rid}' belongs to unknown field '{r['field']}'")

    # --- resolution ----------------------------------------------------
    def resolve_field(self, raw: str) -> Resolution:
        """Map one farmer-supplied field reference onto a mapped field."""
        key = _normalize(raw)
        if not key:
            return Resolution(raw, None, reason="empty field reference")

        if key in self._alias_index:
            fid = self._alias_index[key]
            return Resolution(raw, fid, matched_alias=key)

        # A bare number: "2" -> whichever field lists "2" as an alias.
        digits = re.findall(r"\d+", key)
        if digits and digits[0] in self._alias_index:
            fid = self._alias_index[digits[0]]
            return Resolution(raw, fid, matched_alias=digits[0])

        # Containment, e.g. "north field near the gate" -> north_field.
        # Longest alias first so "north field" beats "north".
        for alias in sorted(self._alias_index, key=len, reverse=True):
            if alias and alias in key:
                return Resolution(raw, self._alias_index[alias], matched_alias=alias)

        return Resolution(
            raw, None,
            reason=f"'{raw}' is not a field on this farm. Known fields: "
                   f"{sorted(self.fields)}")

    def resolve_crop(self, crop: str) -> str | None:
        for fid, f in self.fields.items():
            if f["crop"] == crop:
                return fid
        return None

    def fields_growing(self, crop: str) -> list[str]:
        return [fid for fid, f in self.fields.items() if f["crop"] == crop]

    # --- geometry ------------------------------------------------------
    def field_rows(self, field_id: str) -> list[str]:
        return list(self.fields[field_id]["rows"])

    def row_endpoints(self, row_id: str) -> tuple[Point, Point]:
        r = self.rows[row_id]
        return tuple(r["entry"]), tuple(r["exit"])

    def row_length(self, row_id: str) -> float:
        return float(self.rows[row_id]["length_m"])

    def in_geofence(self, pt: Point) -> bool:
        return point_in_polygon(pt, self.geofence)

    def violated_no_go(self, pt: Point) -> str | None:
        for z in self.no_go_zones:
            if point_in_polygon(pt, z["polygon"]):
                return z["name"]
        return None

    # --- prompt context ------------------------------------------------
    def context_prompt(self) -> str:
        """What the LLM is told about the farm, so it stops inventing fields."""
        lines = [f"Farm: {self.farm_name}", "", "Fields on this farm:"]
        for fid, f in self.fields.items():
            lines.append(
                f"  - {fid}: crop={f['crop']}, rows={f['rows']}, "
                f"also called {f.get('aliases', [])}")
        lines.append("\nIf the farmer names a field that is not listed above, "
                     "set clarification_needed rather than guessing.")
        return "\n".join(lines)


@dataclass
class RobotState:
    """Live snapshot pushed from ROS 2. The task can be structurally perfect
    and still be unsafe to start right now."""

    battery_soc: float = 0.95
    pose: Point = (0.0, 0.0)
    pose_covariance: float = 0.05
    camera_stream_active: bool = True
    free_storage_mb: float = 4096.0
    faults: list[str] = field(default_factory=list)

    def remaining_energy_wh(self, capacity_wh: float) -> float:
        return self.battery_soc * capacity_wh
