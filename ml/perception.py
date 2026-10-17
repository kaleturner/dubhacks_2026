"""Real Mk1.5 perception. Backend imports these four names.

Use Perceptron Mk1.5, not Isaac. Isaac drives the arm. Mk1.5 boxes are on a
0–1000 grid; convert them to pixels before returning the contract shapes.

Until the ML handoff, the server should call ml.mock_perception instead
(USE_MOCKS=1). Keep the signatures and the return shapes in shared/schemas.py.
"""


def parse_intent(text: str) -> dict:
    """Return Intent: object, duration_s, move, style."""
    raise NotImplementedError


def locate(image_path: str, obj: str) -> dict:
    """Return LocateResult: found, bbox [x1,y1,x2,y2], center [x,y]."""
    raise NotImplementedError


def score_angles(stills: list[str], obj: str) -> dict:
    """Return ScoreResult: best_idx and per-still lighting, composition, reason."""
    raise NotImplementedError


def best_window(video_path: str, duration_s: float) -> dict:
    """Optional. Return WindowResult {"start_s": ...}. Fallback is start_s=0.5."""
    raise NotImplementedError
