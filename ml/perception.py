"""Real Mk1.5 perception. Backend imports these four names.

Use Perceptron Mk1.5, not Isaac. Isaac drives the arm. Mk1.5 boxes are on a
0–1000 grid; convert them to pixels before returning the contract shapes.

The server imports this module when USE_MOCK_PERCEPTION=0. Keep the signatures
and the return shapes in shared/schemas.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from perceptron import Client, image, parse_annotations
from PIL import Image as PILImage

from shared.constants import MAX_SHOTS, MAX_TOTAL_S, SHOTS
from shared.schemas import Intent, LocateResult, ScoreResult
from shared.shots import (
    disabled_note,
    enabled_tasks,
    fallback_shots,
    guess_object,
    normalize_shots,
    parse_total,
)

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MODEL = "perceptron-mk1.5"
PARSE_TIMEOUT_S = 18
LOCATE_TIMEOUT_S = 18
SCORE_TIMEOUT_S = 45

_client: Client | None = None

def _intent_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "object": {"type": "string"},
            "style": {"type": "string"},
            "total_s": {"type": "number", "minimum": 1, "maximum": MAX_TOTAL_S},
            "shots": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_SHOTS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "task": {"type": "string", "enum": enabled_tasks()},
                        "duration_s": {"type": "number", "minimum": 1, "maximum": MAX_TOTAL_S},
                    },
                    "required": ["task", "duration_s"],
                },
            },
        },
        "required": ["object", "style", "total_s", "shots"],
    }


_SCORE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "best_idx": {"type": "integer"},
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "idx": {"type": "integer"},
                    "lighting": {"type": "number", "minimum": 0, "maximum": 10},
                    "composition": {"type": "number", "minimum": 0, "maximum": 10},
                    "reason": {"type": "string"},
                },
                "required": ["idx", "lighting", "composition", "reason"],
            },
        },
    },
    "required": ["best_idx", "scores"],
}

def _intent_prompt() -> str:
    menu = "; ".join(f"{name}: {SHOTS[name]['look']}" for name in enabled_tasks())
    return (
        "Read this filming request and plan the shots. "
        "object is the thing they named. If they name nothing, use apple. "
        "total_s is the finished length in seconds from 1 to 30. If they give none, use 15. "
        f"shots is an ordered list drawn only from this menu: {menu}. "
        "If they name a move, use exactly that move. "
        "If the request is general, such as an ad, a showcase, or a cinematic video, "
        "pick the one to three shots that suit it, with the establishing shot first and the hero shot last. "
        "Never repeat a task. The shot durations should add up to total_s. "
        "style is a short look. If they name none, use cinematic."
    )


_PRESENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"present": {"type": "boolean"}},
    "required": ["present"],
}

_PRESENT_PROMPT = (
    "Is a real physical {obj} visible in this photo? "
    "A plain colored square, a blank grey frame, or text does not count. "
    "present is true only when the {obj} itself is visible."
)

_LOCATE_PROMPT = (
    "Locate the single {obj} in this photo. Return one point_box around only that "
    "object, with mention={obj} and asset_idx=0. Use normalized 0-1000 coordinates, "
    "top-left then bottom-right, tight on the object's edges. "
    "A plain colored square, a blank grey frame, or text is not the {obj}. "
    "If the real {obj} is not visible, say it is missing and do not return a point_box."
)

_SCORE_PROMPT = (
    "There are {count} stills, numbered 0 through {last}. Each shows one {obj}. "
    "Return exactly {count} scores, one for every idx, including the last. "
    "lighting and composition are integers from 0 to 10. "
    "10 means the {obj} is large in the frame, near the center, evenly lit, and alone on a clear table. "
    "0 to 3 means the {obj} is tiny, stuck in a corner, covered by a dark shadow, or sharing the frame "
    "with arms, cables, or other clutter. A sharp {obj} in the corner of a shadowed table is still a bad still. "
    "A recognizable {obj} in a bad frame still scores low. "
    "When several stills are well lit, best_idx is the one where the {obj} is largest and most centered. "
    "reason names the size, the light on the {obj}, or the clutter in that still."
)


def _sdk() -> Client:
    global _client
    if _client is None:
        key = os.getenv("PERCEPTRON_API_KEY", "").strip()
        if not key:
            raise RuntimeError("PERCEPTRON_API_KEY is missing")
        _client = Client(provider="perceptron", model=MODEL, api_key=key)
    return _client


def _answer(response, *, allow_length: bool = False) -> str:
    choice = response.choices[0]
    if choice.finish_reason != "stop" and not (allow_length and choice.finish_reason == "length"):
        raise RuntimeError(f"Mk1.5 stopped early: {choice.finish_reason}")
    text = (choice.message.content or "").strip()
    if not text:
        raise RuntimeError("Mk1.5 returned an empty answer")
    return text


def _json_call(
    messages: list,
    schema: dict,
    name: str,
    timeout: float,
    *,
    reasoning_effort: str = "low",
    max_completion_tokens: int = 700,
) -> dict:
    response = _sdk().chat.completions.create(
        model=MODEL,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": schema},
        },
    )
    try:
        payload = json.loads(_answer(response))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Mk1.5 returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Mk1.5 returned JSON that is not an object")
    return payload


def _axis(value: float, length: int) -> float:
    pixel = float(value) / 1000.0 * length
    return min(max(pixel, 0.0), float(length))


def _to_pixels(box, width: int, height: int) -> tuple[list[float], list[float]]:
    x1 = _axis(box.top_left.x, width)
    y1 = _axis(box.top_left.y, height)
    x2 = _axis(box.bottom_right.x, width)
    y2 = _axis(box.bottom_right.y, height)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    bbox = [left, top, right, bottom]
    center = [(left + right) / 2, (top + bottom) / 2]
    return bbox, center


def _matches(mention: str | None, obj: str) -> bool:
    label = (mention or "").strip().lower()
    target = obj.strip().lower()
    if not label or not target:
        return False
    return label == target or label in target or target in label


def parse_intent(text: str) -> dict:
    """Return Intent: object, style, shots [{task, duration_s}], note."""
    try:
        payload = _json_call(
            [{"role": "user", "content": f"{_intent_prompt()}\n\nRequest: {text.strip()}"}],
            _intent_schema(),
            "intent",
            PARSE_TIMEOUT_S,
        )
    except Exception:
        payload = {}
    total = payload.get("total_s") or parse_total(text)
    shots = normalize_shots(payload.get("shots"), total) or fallback_shots(text, total)
    obj = str(payload.get("object", "")).strip() or guess_object(text)
    style = str(payload.get("style", "")).strip() or "cinematic"
    return Intent(object=obj, style=style, shots=shots, note=disabled_note(text)).model_dump()


def locate(image_path: str, obj: str) -> dict:
    """Return LocateResult: found, bbox [x1,y1,x2,y2], center [x,y]."""
    with PILImage.open(image_path) as frame:
        width, height = frame.size
    present = _json_call(
        [
            {
                "role": "user",
                "content": [
                    image(image_path),
                    {"type": "text", "text": _PRESENT_PROMPT.format(obj=obj)},
                ],
            }
        ],
        _PRESENT_SCHEMA,
        "present",
        LOCATE_TIMEOUT_S,
    )
    if present.get("present") is not True:
        return LocateResult(found=False, bbox=[], center=[]).model_dump()
    response = _sdk().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    image(image_path),
                    {"type": "text", "text": _LOCATE_PROMPT.format(obj=obj)},
                ],
            }
        ],
        max_completion_tokens=512,
        timeout=LOCATE_TIMEOUT_S,
        reasoning_effort="none",
        vision_config={"annotation_format": "box"},
    )
    parsed = parse_annotations(_answer(response, allow_length=True), expects="box")
    boxes = [segment["value"] for segment in parsed.segments if segment["kind"] == "box"]
    chosen = next((box for box in boxes if _matches(box.mention, obj)), None)
    if chosen is None and len(boxes) == 1:
        chosen = boxes[0]
    if chosen is None:
        return LocateResult(found=False, bbox=[], center=[]).model_dump()
    bbox, center = _to_pixels(chosen, width, height)
    if bbox[2] - bbox[0] < 1 or bbox[3] - bbox[1] < 1:
        return LocateResult(found=False, bbox=[], center=[]).model_dump()
    return LocateResult(found=True, bbox=bbox, center=center).model_dump()


def score_angles(stills: list[str], obj: str) -> dict:
    """Return ScoreResult: best_idx and per-still lighting, composition, reason."""
    if not stills:
        raise ValueError("score_angles needs at least one still")
    content: list = []
    for index, path in enumerate(stills):
        content.append(image(path))
        content.append({"type": "text", "text": f"That was still {index}."})
    content.append(
        {
            "type": "text",
            "text": _SCORE_PROMPT.format(obj=obj, count=len(stills), last=len(stills) - 1),
        }
    )
    payload = _json_call(
        [{"role": "user", "content": content}],
        _SCORE_SCHEMA,
        "angle_scores",
        SCORE_TIMEOUT_S,
        reasoning_effort="high",
        max_completion_tokens=4500,
    )
    by_idx = {}
    for score in payload.get("scores") or []:
        if not isinstance(score, dict):
            continue
        try:
            idx = int(score["idx"])
        except (KeyError, TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(stills) or idx in by_idx:
            continue
        by_idx[idx] = {
            "idx": idx,
            "lighting": float(score["lighting"]),
            "composition": float(score["composition"]),
            "reason": str(score.get("reason") or "no reason"),
        }
    values = [score["lighting"] for score in by_idx.values()] + [
        score["composition"] for score in by_idx.values()
    ]
    if values and max(values) <= 1:
        for score in by_idx.values():
            score["lighting"] *= 10
            score["composition"] *= 10
    for score in by_idx.values():
        score["lighting"] = min(max(score["lighting"], 0.0), 10.0)
        score["composition"] = min(max(score["composition"], 0.0), 10.0)
    for idx in range(len(stills)):
        by_idx.setdefault(
            idx,
            {"idx": idx, "lighting": 0.0, "composition": 0.0, "reason": "no score returned"},
        )
    scores = [by_idx[idx] for idx in range(len(stills))]
    try:
        hinted = int(payload.get("best_idx", 0))
    except (TypeError, ValueError):
        hinted = 0
    hinted = min(max(hinted, 0), len(stills) - 1)

    def rank_key(score: dict) -> tuple[float, int]:
        return (score["lighting"] + score["composition"], 1 if score["idx"] == hinted else 0)

    best_idx = max(scores, key=rank_key)["idx"]
    return ScoreResult(best_idx=best_idx, scores=scores).model_dump()


def best_window(video_path: str, duration_s: float) -> dict:
    """Optional. Return WindowResult {"start_s": ...}. Fallback is start_s=0.5."""
    raise NotImplementedError
