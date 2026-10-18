"""Pydantic models for the phone WebSocket, perception, and arm-facing payloads.

The phone is a remote: prompt in, status and result out. It never films.

Examples live in shared/examples/. If a shape changes, update the model, the
example JSON, and CONTRACTS.md in the same change.
"""

from typing import Literal

from pydantic import BaseModel, Field

Stage = Literal["PARSE", "LOCATE", "SCOUT", "PLAN", "RECORD", "POST", "DONE", "ERROR"]
IntentMove = Literal["orbit", "push_in", "hero_rise"]


class PromptMessage(BaseModel):
    type: Literal["prompt"]
    text: str


class StatusMessage(BaseModel):
    type: Literal["status"]
    stage: Stage
    detail: str
    stills: list[str] | None = None
    best_idx: int | None = None


class ResultMessage(BaseModel):
    type: Literal["result"]
    job_id: str
    url: str
    duration_s: float


class Intent(BaseModel):
    object: str
    duration_s: float = Field(gt=0, le=30)
    move: IntentMove
    style: str


class LocateResult(BaseModel):
    found: bool
    bbox: list[float]
    center: list[float]


class AngleScore(BaseModel):
    idx: int
    lighting: float
    composition: float
    reason: str


class ScoreResult(BaseModel):
    best_idx: int
    scores: list[AngleScore]


class WindowResult(BaseModel):
    start_s: float


class TranscribeResult(BaseModel):
    text: str
