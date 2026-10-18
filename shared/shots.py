"""Shot planning helpers. Pure Python, no model call.

`normalize_shots` makes any planner output safe to run. `fallback_shots` plans
from keywords when Mk1.5 is mocked, down, or returns nothing usable.
"""

import re
from typing import Any

from shared.constants import DEFAULT_TOTAL_S, MAX_SHOTS, MAX_TOTAL_S, SHOTS

Registry = dict[str, dict]


def enabled_tasks(registry: Registry | None = None) -> list[str]:
    reg = SHOTS if registry is None else registry
    return [name for name, spec in reg.items() if spec.get("enabled")]


def clamp_total(total_s: float | None) -> float:
    try:
        value = float(total_s) if total_s is not None else DEFAULT_TOTAL_S
    except (TypeError, ValueError):
        value = DEFAULT_TOTAL_S
    return min(max(value, 1.0), MAX_TOTAL_S)


def normalize_shots(shots: Any, total_s: float | None, registry: Registry | None = None) -> list[dict]:
    """Return [{"task", "duration_s"}] using only enabled tasks.

    Drops unknown or disabled tasks, caps the count, clamps each shot to its
    min/max, and rescales so the durations sum to the total. Returns [] when
    nothing usable is left. The caller decides the fallback.
    """
    reg = SHOTS if registry is None else registry
    allowed = set(enabled_tasks(reg))
    kept: list[tuple[str, float]] = []
    for item in shots if isinstance(shots, list) else []:
        if not isinstance(item, dict):
            continue
        task = item.get("task")
        if task not in allowed:
            continue
        try:
            weight = float(item.get("duration_s"))
        except (TypeError, ValueError):
            weight = 0.0
        if weight <= 0:
            weight = float(reg[task]["default_s"])
        kept.append((task, weight))
    kept = kept[:MAX_SHOTS]
    if not kept:
        return []

    total = clamp_total(total_s)
    # Drop trailing shots until every one can have at least its minimum.
    while len(kept) > 1 and sum(reg[t]["min_s"] for t, _ in kept) > total:
        kept.pop()
    # A single shot cannot run longer than its max.
    if len(kept) == 1:
        total = min(total, float(reg[kept[0][0]]["max_s"]))
    total = max(total, sum(float(reg[t]["min_s"]) for t, _ in kept))
    total = min(total, sum(float(reg[t]["max_s"]) for t, _ in kept))

    lows = [float(reg[t]["min_s"]) for t, _ in kept]
    highs = [float(reg[t]["max_s"]) for t, _ in kept]
    weights = [w for _, w in kept]
    # Water-fill: scale by weight, pin any shot that hits a bound, share the rest.
    durations: list[float | None] = [None] * len(kept)
    free = set(range(len(kept)))
    remaining = total
    while free:
        weight_sum = sum(weights[i] for i in free)
        pinned = False
        for i in sorted(free):
            share = remaining * weights[i] / weight_sum
            if share < lows[i] or share > highs[i]:
                durations[i] = lows[i] if share < lows[i] else highs[i]
                remaining -= durations[i]
                free.discard(i)
                pinned = True
                break
        if not pinned:
            for i in free:
                durations[i] = remaining * weights[i] / weight_sum
            break
    rounded = [round(d or 0.0, 2) for d in durations]
    # Put the rounding drift on the last shot that has room, so the sum is exact.
    drift = round(total - sum(rounded), 2)
    for i in reversed(range(len(rounded))):
        if lows[i] <= rounded[i] + drift <= highs[i]:
            rounded[i] = round(rounded[i] + drift, 2)
            break
    return [{"task": t, "duration_s": d} for (t, _), d in zip(kept, rounded, strict=True)]


def parse_total(text: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:s\b|sec|second)", text.lower())
    return clamp_total(float(match.group(1))) if match else DEFAULT_TOTAL_S


def requested_tasks(text: str, registry: Registry | None = None) -> list[str]:
    """Every task the text names, enabled or not, in the order they appear."""
    reg = SHOTS if registry is None else registry
    lowered = text.lower()
    found: list[tuple[int, str]] = []
    for name, spec in reg.items():
        positions = [lowered.find(k) for k in spec["keywords"] if k in lowered]
        if positions:
            found.append((min(positions), name))
    return [name for _, name in sorted(found)]


def fallback_shots(text: str, total_s: float | None = None, registry: Registry | None = None) -> list[dict]:
    """Keyword plan. Named enabled moves run in the order said. A vague query
    gets the enabled tasks in story order, establishing first, hero last."""
    reg = SHOTS if registry is None else registry
    allowed = enabled_tasks(reg)
    named = [t for t in requested_tasks(text, reg) if t in allowed]
    tasks = named or sorted(allowed, key=lambda t: reg[t]["order"])[:MAX_SHOTS]
    total = parse_total(text) if total_s is None else total_s
    return normalize_shots([{"task": t, "duration_s": reg[t]["default_s"]} for t in tasks], total, reg)


def disabled_note(text: str, registry: Registry | None = None) -> str | None:
    """Say so when the query asks for a move we have not trained yet."""
    reg = SHOTS if registry is None else registry
    missing = [t for t in requested_tasks(text, reg) if not reg[t].get("enabled")]
    if not missing:
        return None
    allowed = ", ".join(enabled_tasks(reg))
    return f"{', '.join(missing)} not available yet, using {allowed}"


def guess_object(text: str) -> str:
    match = re.search(
        r"\b(?:of|at|on|around)\s+(?:this|the|a|an|my|that)?\s*([a-z][a-z \-]*?)"
        r"(?=\s+(?:for|in|with|from|using)\b|\s*[.,!?]|\s*$)",
        text.lower().strip(),
    )
    obj = match.group(1).strip() if match else ""
    return obj or "apple"
