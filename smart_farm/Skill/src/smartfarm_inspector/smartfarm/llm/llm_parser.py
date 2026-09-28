"""Parse farmer requests using Claude.

Changes from the original, each for a reason:

  * The few-shot examples were loaded from config and then never used. They
    are now in the prompt, which is the whole point of loading them.
  * The model is told which fields actually exist on this farm, so it stops
    inventing them. The original prompt described crops and inspection types
    but not fields, which left field_ids as unconstrained free text.
  * Field ids are resolved against the field map and unresolvable ones are
    reported, rather than passed downstream as "unspecified".
  * Malformed JSON is retried once with the parse error fed back, instead of
    returning an error to the farmer on the first stumble.
  * Dropping invalid crops silently is now recorded, so the farmer is told
    what was ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config_paths import config_file
from ..planning.field_registry import FieldMap
from .llm_provider import LLMResponse, extract_json, get_llm_provider

VALID_INSPECTION_TYPES = [
    "disease_detection", "weed_assessment", "irrigation_check",
    "crop_stress", "quick_scan",
]


@dataclass
class LLMParsedRequest:
    """LLM-parsed inspection request."""

    raw_request: str
    field_ids: list          # resolved, mapped field ids
    crops: list
    inspection_types: list
    urgency: str
    confidence: float
    clarification_needed: str | None = None
    error: str | None = None
    unresolved_fields: list = field(default_factory=list)
    dropped_crops: list = field(default_factory=list)
    attempts: int = 1
    usage: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.clarification_needed is None


class LLMRequestParser:
    """Parse farmer requests using Claude, grounded in the field map."""

    def __init__(self, crops_config_path: str | Path | None = None,
                 few_shot_examples_path: str | Path | None = None,
                 field_map: FieldMap | None = None,
                 provider=None,
                 max_retries: int = 1,
                 config_dir: str | Path | None = None):
        self.llm = provider or get_llm_provider()
        self.map = field_map or FieldMap()
        self.max_retries = max_retries

        cdir = Path(config_dir) if config_dir else None
        crops_path = Path(crops_config_path or
                          (cdir / "crops.json" if cdir
                           else config_file("crops.json")))
        few_shot_path = Path(few_shot_examples_path or
                             (cdir / "few_shot_examples.json" if cdir
                              else config_file("few_shot_examples.json")))

        self.crops_db = self._load(crops_path, {"crops": {}})
        self.few_shot_data = self._load(few_shot_path, {"few_shot_examples": []})

        self.valid_crops = list(self.crops_db.get("crops", {}))
        self.valid_inspection_types = list(VALID_INSPECTION_TYPES)

    @staticmethod
    def _load(path: Path, default: dict) -> dict:
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            print(f"Warning: {path} not found, using defaults")
            return default

    # ------------------------------------------------------------------
    def _build_system_prompt(self) -> str:
        crops_list = ", ".join(self.valid_crops) or "tomato, lettuce, corn, cucumber"
        types_list = ", ".join(self.valid_inspection_types)

        # The examples exist in config for a reason; put them in the prompt.
        examples = ""
        for ex in self.few_shot_data.get("few_shot_examples", []):
            examples += (f"\nFarmer: {ex['input']}\n"
                         f"JSON: {json.dumps(ex['output'])}\n")

        return f"""You are an agricultural inspection coordinator for a real \
robot working a real farm. Parse the farmer's request into JSON.

{self.map.context_prompt()}

Return ONLY valid JSON with these keys:
1. field_ids: fields to inspect. Use the exact field id from the list above \
(e.g. "north_field"). If the farmer names a field that is not on this farm, \
leave field_ids empty and set clarification_needed.
2. crops: crop types mentioned. Must be one of: {crops_list}
3. inspection_types: must be one of: {types_list}
4. urgency: routine, medium, or urgent
5. confidence: 0.0-1.0
6. clarification_needed: null, or a question for the farmer

Rules:
- Never invent a field id. Only ids from the farm listing above are valid.
- If the farmer names a crop, you may infer the field that grows it.
- If the request is too vague to act on, ask via clarification_needed rather \
than guessing.

Worked examples:{examples}
Return ONLY JSON, no other text, no markdown fences."""

    # ------------------------------------------------------------------
    def parse(self, request: str) -> LLMParsedRequest:
        system_prompt = self._build_system_prompt()
        prompt = request
        last_error = ""
        usage: dict = {}

        for attempt in range(self.max_retries + 1):
            try:
                response: LLMResponse = self.llm.call(prompt=prompt,
                                                      system_prompt=system_prompt)
                usage = response.usage
                data = extract_json(response.content)

                if data is None:
                    last_error = "response was not valid JSON"
                    prompt = (f"{request}\n\nYour previous reply could not be "
                              f"parsed as JSON. Return ONLY a JSON object.")
                    continue

                return self._validate(request, data, attempt + 1, usage)

            except Exception as exc:  # noqa: BLE001 - surface any provider fault
                return LLMParsedRequest(
                    raw_request=request, field_ids=[], crops=[],
                    inspection_types=[], urgency="routine", confidence=0.0,
                    error=str(exc), attempts=attempt + 1, usage=usage)

        return LLMParsedRequest(
            raw_request=request, field_ids=[], crops=[], inspection_types=[],
            urgency="routine", confidence=0.0,
            error=f"Failed to parse LLM response after "
                  f"{self.max_retries + 1} attempts: {last_error}",
            attempts=self.max_retries + 1, usage=usage)

    # ------------------------------------------------------------------
    def _validate(self, request: str, data: dict, attempts: int,
                  usage: dict) -> LLMParsedRequest:
        def as_list(v):
            if isinstance(v, list):
                return v
            return [v] if v else []

        # --- fields: resolve against the map -------------------------
        resolved, unresolved = [], []
        for raw in as_list(data.get("field_ids")):
            r = self.map.resolve_field(str(raw))
            if r.ok:
                if r.field_id not in resolved:
                    resolved.append(r.field_id)
            else:
                unresolved.append(r.reason)

        crops = as_list(data.get("crops"))
        valid_crops = [c for c in crops if c in self.valid_crops]
        dropped = [c for c in crops if c not in self.valid_crops]

        # If no field resolved but a crop did, infer the field from the crop.
        # "check the tomatoes" is a normal way for a farmer to speak.
        if not resolved:
            for c in valid_crops:
                resolved += [f for f in self.map.fields_growing(c)
                             if f not in resolved]

        types = as_list(data.get("inspection_types"))
        valid_types = [t for t in types if t in self.valid_inspection_types]

        urgency = str(data.get("urgency", "routine")).lower()
        if urgency not in ("routine", "medium", "urgent"):
            urgency = "routine"

        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5

        clarification = data.get("clarification_needed")

        # An unresolvable field is a clarification, not a silent default. The
        # original fell back to ['unspecified'], which produced a task the
        # robot could not execute and no explanation of why.
        if unresolved and not resolved:
            clarification = clarification or "; ".join(unresolved)
        elif not resolved and not clarification:
            clarification = (
                "Which field should I inspect? This farm has: "
                f"{sorted(self.map.fields)}")

        return LLMParsedRequest(
            raw_request=request,
            field_ids=resolved,
            crops=valid_crops,
            inspection_types=valid_types or ["quick_scan"],
            urgency=urgency,
            confidence=confidence,
            clarification_needed=clarification,
            unresolved_fields=unresolved,
            dropped_crops=dropped,
            attempts=attempts,
            usage=usage,
        )
