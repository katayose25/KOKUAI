from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PatientCategory(StrEnum):
    cut = "cut"
    bruise = "bruise"
    burn = "burn"
    emergency = "emergency"
    other = "other"


class EncounterStatus(StrEnum):
    intake = "intake"
    recording = "recording"
    transcribed = "transcribed"
    draft = "draft"
    confirmed = "confirmed"


class Patient(BaseModel):
    id: str = Field(default_factory=lambda: new_id("p"))
    name: str
    age: int
    sex: Literal["male", "female", "other", "unknown"] = "unknown"
    visit_type: Literal["first_visit", "follow_up"] = "first_visit"
    category: PatientCategory = PatientCategory.other
    chief_complaint: str = ""
    memo: str = ""
    created_at: str = Field(default_factory=now_iso)


class ImageAsset(BaseModel):
    id: str = Field(default_factory=lambda: new_id("img"))
    filename: str
    path: str
    finding: str = ""
    created_at: str = Field(default_factory=now_iso)


class AudioAsset(BaseModel):
    id: str = Field(default_factory=lambda: new_id("aud"))
    kind: Literal["upload", "recording"]
    filename: str
    path: str
    duration_sec: float | None = None
    status: Literal["uploaded", "transcribed"] = "uploaded"
    created_at: str = Field(default_factory=now_iso)


class TranscriptTurn(BaseModel):
    speaker: Literal["doctor", "patient", "speaker_00", "speaker_01", "unknown"]
    start: float | None = None
    end: float | None = None
    text: str


class ClinicalPrompt(BaseModel):
    kind: Literal["alert", "negation", "need_check", "follow_up"]
    title: str
    detail: str
    priority: int = 3


class ChartDraft(BaseModel):
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""
    handoff: str = ""


class Encounter(BaseModel):
    id: str = Field(default_factory=lambda: new_id("e"))
    patient_id: str
    status: EncounterStatus = EncounterStatus.intake
    images: list[ImageAsset] = Field(default_factory=list)
    audio_sources: list[AudioAsset] = Field(default_factory=list)
    transcript: list[TranscriptTurn] = Field(default_factory=list)
    clinical_prompts: list[ClinicalPrompt] = Field(default_factory=list)
    live_status: str = "idle"
    live_message: str = ""
    live_processed_chunks: int = 0
    live_cancel_requested: bool = False
    chart: ChartDraft = Field(default_factory=ChartDraft)
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)

    @property
    def upload_dir(self) -> Path:
        return Path("hackathon_app/storage/uploads") / self.id
