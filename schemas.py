"""Pydantic data shapes for ROM session — copied/trimmed from minihack4."""
from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

PainLocation = Literal["neck", "shoulder", "back", "wrist", "knee", "eye_head", "other"]
RiskLevel = Literal["low", "moderate", "high"]
Severity = Literal["none", "mild", "moderate", "severe"]


class RedFlags(BaseModel):
    numbness: bool | None = None
    weakness: bool | None = None
    balance_problem: bool | None = None
    severe_headache: bool | None = None
    after_trauma: bool | None = None
    incontinence: bool | None = None
    fever_unwell: bool | None = None

    def any_true(self) -> bool:
        return any(getattr(self, f) is True for f in self.model_fields)

    def list_triggered(self) -> list[str]:
        return [f for f in self.model_fields if getattr(self, f) is True]


class AssessmentState(BaseModel):
    """Subset compatible with `tools.check_red_flags` / scoring helpers."""

    pain_location: list[PainLocation] = Field(default_factory=list)
    pain_score: int | None = None
    duration_days: int | None = None
    sitting_hours_per_session: float | None = None
    break_frequency_minutes: int | None = None
    uses_laptop_only: bool | None = None
    monitor_at_eye_level: bool | None = None
    external_keyboard: bool | None = None
    pain_radiating: bool | None = None
    red_flags: RedFlags = Field(default_factory=RedFlags)
    vision_factors: Any = None  # not used in ROM analyzer flow

    REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "pain_location",
        "pain_score",
        "duration_days",
        "sitting_hours_per_session",
        "break_frequency_minutes",
        "uses_laptop_only",
    )

    def is_complete(self) -> bool:
        for f in self.REQUIRED_FIELDS:
            v = getattr(self, f)
            if v is None or (isinstance(v, list) and not v):
                return False
        return True

    def missing_fields(self) -> list[str]:
        return [
            f for f in self.REQUIRED_FIELDS
            if getattr(self, f) is None or (isinstance(getattr(self, f), list) and not getattr(self, f))
        ]


class RiskBreakdown(BaseModel):
    pain_intensity: int = 0
    duration: int = 0
    sitting_duration: int = 0
    break_frequency: int = 0
    workstation: int = 0
    symptoms: int = 0

    def total(self) -> int:
        return (
            self.pain_intensity + self.duration + self.sitting_duration
            + self.break_frequency + self.workstation + self.symptoms
        )


class RiskResult(BaseModel):
    score: int
    breakdown: RiskBreakdown
    risk_level: RiskLevel
    risk_factors: list[str]
    contributing_categories: list[str]
    vision_uplift_applied: bool = False


class RedFlagResult(BaseModel):
    has_red_flag: bool
    red_flags: list[str]
    message: str


class ErgonomicPlan(BaseModel):
    daily_plan: list[dict[str, Any]] = Field(default_factory=list)
    chair: list[str] = Field(default_factory=list)
    monitor: list[str] = Field(default_factory=list)
    keyboard_mouse: list[str] = Field(default_factory=list)
    break_routine: list[str] = Field(default_factory=list)
    stretches: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# ROM session
# ---------------------------------------------------------------------------


class PoseResult(BaseModel):
    body_part: str
    max_angle: float
    normal_max: float
    percent_of_normal: float
    peak_timestamp: float
    peak_frame_path: str
    pain_score: int | None = None
    pain_note: str = ""
    capture_reason: str = ""        # "user_signal", "auto_plateau", "pain_threshold"
    threshold_percent: float = 70.0  # functional-ROM cutoff (AAOS-style)
    passed: bool = False             # percent_of_normal >= threshold_percent
    completed: bool = True


class RomSessionState(BaseModel):
    pain_location: list[str] = Field(default_factory=list)
    pose_queue: list[str] = Field(default_factory=list)
    current_pose: str | None = None
    current_pain_score: int | None = None
    results: list[PoseResult] = Field(default_factory=list)
    red_flag: RedFlagResult | None = None
    started_at: float
    ended_at: float | None = None
    session_dir: str

    def is_complete(self) -> bool:
        return (
            bool(self.results)
            and len(self.results) >= len(self.pose_queue)
            and self.current_pose is None
        )


class PerPoseFinding(BaseModel):
    body_part: str
    max_angle: float
    percent_of_normal: float
    pain_score: int | None
    commentary: str
    image_path: str


class RomFinalReport(BaseModel):
    overall_risk_level: RiskLevel
    summary: str
    per_pose_findings: list[PerPoseFinding] = Field(default_factory=list)
    seven_day_plan: ErgonomicPlan | None = None
    see_clinician: bool = False
    rationale: str = ""
    generated_at: float = 0.0
    session_dir: str = ""
