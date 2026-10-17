"""Hard-coded perception responses with the same shapes as ml.perception.

Backend imports this module while USE_MOCKS=1.
"""

from shared.schemas import Intent, LocateResult, ScoreResult, WindowResult


def parse_intent(text: str) -> dict:
    del text
    return Intent(object="apple", duration_s=15, move="orbit", style="cinematic").model_dump()


def locate(image_path: str, obj: str) -> dict:
    del image_path, obj
    return LocateResult(found=True, bbox=[120, 80, 360, 400], center=[240, 240]).model_dump()


def score_angles(stills: list[str], obj: str) -> dict:
    del obj
    scores = []
    for idx, _path in enumerate(stills):
        scores.append(
            {
                "idx": idx,
                "lighting": 6,
                "composition": 6,
                "reason": "mock score",
            }
        )
    if not scores:
        scores.append({"idx": 0, "lighting": 6, "composition": 6, "reason": "mock score"})
    best_idx = min(2, len(scores) - 1)
    scores[best_idx]["lighting"] = 8
    scores[best_idx]["composition"] = 8
    scores[best_idx]["reason"] = "mock best angle"
    return ScoreResult(best_idx=best_idx, scores=scores).model_dump()


def best_window(video_path: str, duration_s: float) -> dict:
    del video_path, duration_s
    return WindowResult(start_s=1.2).model_dump()
