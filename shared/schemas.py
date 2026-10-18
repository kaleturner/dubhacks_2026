"""Pydantic models for the phone WebSocket, perception, and arm-facing payloads.

The phone is a remote: prompt in, status and result out. It never films.

Examples live in shared/examples/. If a shape changes, update the model, the
example JSON, and CONTRACTS.md in the same change.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Stage = Literal["PARSE", "LOCATE", "SCOUT", "PLAN", "RECORD", "POST", "DONE", "ERROR"]


class PromptMessage(BaseModel):
    type: Literal["prompt"]
    text: str


class StatusMessage(BaseModel):
    type: Literal["status"]
    stage: Stage
    detail: str
    stills: list[str] | None = None
    best_idx: int | None = None


class Shot(BaseModel):
    task: str
    duration_s: float = Field(gt=0, le=30)


class ResultMessage(BaseModel):
    type: Literal["result"]
    job_id: str
    url: str
    duration_s: float
    shots: list[Shot] | None = None
    edit_url: str | None = None


class Intent(BaseModel):
    object: str
    style: str
    shots: list[Shot] = Field(min_length=1, max_length=3)
    note: str | None = None

    @property
    def duration_s(self) -> float:
        return round(sum(shot.duration_s for shot in self.shots), 2)

    @model_validator(mode="after")
    def _total_within_limit(self) -> "Intent":
        if self.duration_s > 30.001:
            raise ValueError("shots add up to more than 30 seconds")
        return self


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
