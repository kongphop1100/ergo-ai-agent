"""Final ROM-session analyzer.

Takes a finished `RomSessionState` (with N PoseResults + saved peak frames)
and produces a `RomFinalReport` via a multimodal LLM call: each peak JPEG
is attached to the prompt so the model can comment on form, plus the numeric
angle/pain/percent values are summarized in text.

Output is a structured Thai-language report with risk level + per-pose
commentary + 7-day plan + clinician flag.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

from llm import call_openrouter, get_default_vision_model, parse_model_json
from schemas import (
    AssessmentState,
    ErgonomicPlan,
    PerPoseFinding,
    RomFinalReport,
    RomSessionState,
)
from tools import generate_ergonomic_plan

ANALYZER_SYSTEM = (
    "You are a physiotherapy report generator. You receive measured ROM (range-of-motion) "
    "values + photos at peak from a guided self-test session. Pain scores may be missing. "
    "You produce a structured Thai-language report with a risk level (low/moderate/high), "
    "per-pose commentary, and a clinician-referral flag. You are NOT a doctor — never "
    "diagnose. Communicate in Thai (ค่ะ tone). Output STRICT JSON only — no prose, "
    "no markdown fences."
)


def _build_user_prompt(session: RomSessionState) -> str:
    rows: list[str] = []
    for i, r in enumerate(session.results, 1):
        pain = "unknown" if r.pain_score is None else str(r.pain_score)
        note = f" (note: {r.pain_note})" if r.pain_note else ""
        reason = f", capture_reason={r.capture_reason}" if r.capture_reason else ""
        rows.append(
            f"{i}. {r.body_part}: max_angle={r.max_angle:.1f}° "
            f"({r.percent_of_normal:.0f}% of normal {r.normal_max:.0f}°), "
            f"pain={pain}{'/10' if r.pain_score is not None else ''}{reason}{note}"
        )
    listing = "\n".join(rows) if rows else "(no completed poses)"

    return (
        "Below is a tele-rehab ROM self-test session. The user reported pain in: "
        f"{', '.join(session.pain_location) or 'unspecified'}.\n\n"
        f"Measurements:\n{listing}\n\n"
        f"Each measurement is paired with a JPEG photo of the user at the peak of the motion.\n\n"
        "Produce a structured Thai-language report. Return JSON in EXACTLY this shape — "
        "no extra keys, no markdown:\n"
        "{\n"
        '  "overall_risk_level": "low" | "moderate" | "high",\n'
        '  "summary": "1-2 sentences in Thai",\n'
        '  "per_pose_findings": [\n'
        '    {"body_part": "...", "max_angle": 0.0, "percent_of_normal": 0.0, '
        '"pain_score": 0 or null, "commentary": "1 sentence in Thai about this pose", '
        '"image_path": "...path verbatim from input..."}\n'
        "  ],\n"
        '  "see_clinician": false,\n'
        '  "rationale": "1-2 sentences in Thai explaining the risk classification"\n'
        "}\n\n"
        "Risk-level rules:\n"
        "- high: any pose <30% of normal, OR any known pain >=7, OR see_clinician=true\n"
        "- moderate: any pose 30-70% of normal, OR any known pain 4-6\n"
        "- low: all poses >=70% of normal AND no known pain above 3\n"
        "When pain is unknown, classify from ROM percent and visible compensation only. "
        "Set see_clinician=true if any pose <30% with concerning visible compensation, or if you observe "
        "obvious compensation/asymmetry in the photos.\n"
        "Per-pose commentary should reference the visible posture in the photo "
        "(e.g. ห่อไหล่, ก้มลำตัวชดเชย, etc.) — be specific, cautious phrasing."
    )


def _encode_image(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        b = path.read_bytes()
    except Exception:
        return None
    return f"data:image/jpeg;base64,{base64.b64encode(b).decode('ascii')}"


def analyze(session: RomSessionState) -> RomFinalReport:
    session_dir = Path(session.session_dir)

    # If no poses completed, short-circuit with a graceful low-info report
    if not session.results:
        return RomFinalReport(
            overall_risk_level="low",
            summary="ยังไม่มีท่าใดเสร็จสิ้นในการประเมินครั้งนี้ครับ",
            per_pose_findings=[],
            seven_day_plan=None,
            see_clinician=False,
            rationale="ไม่มีข้อมูลพอสำหรับประเมินความเสี่ยง",
            generated_at=time.time(),
            session_dir=str(session_dir),
        )

    # Build multimodal user content: text + each peak frame inline
    user_text = _build_user_prompt(session)
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for r in session.results:
        full = session_dir / r.peak_frame_path
        url = _encode_image(full)
        if url is not None:
            content.append({"type": "image_url", "image_url": {"url": url}})

    messages = [
        {"role": "system", "content": ANALYZER_SYSTEM},
        {"role": "user", "content": content},
    ]

    try:
        msg = call_openrouter(
            messages,
            model=get_default_vision_model(),
            response_format={"type": "json_object"},
            timeout=120.0,
        )
        raw = msg.get("content") or ""
        parsed = parse_model_json(raw) or {}
    except Exception as exc:
        # LLM unreachable — fall back to a deterministic rule-based report
        return _fallback_report(session, error=str(exc))

    # Parse per-pose findings
    findings_in = parsed.get("per_pose_findings") or []
    findings: list[PerPoseFinding] = []
    by_part = {r.body_part: r for r in session.results}
    for f in findings_in:
        bp = str(f.get("body_part") or "")
        match = by_part.get(bp)
        if match is None:
            continue
        findings.append(
            PerPoseFinding(
                body_part=bp,
                max_angle=float(f.get("max_angle") or match.max_angle),
                percent_of_normal=float(f.get("percent_of_normal") or match.percent_of_normal),
                pain_score=match.pain_score,
                commentary=str(f.get("commentary") or ""),
                image_path=match.peak_frame_path,
            )
        )
    # If LLM dropped any poses, fill them in deterministically so the report is complete
    seen_parts = {f.body_part for f in findings}
    for r in session.results:
        if r.body_part not in seen_parts:
            findings.append(
                PerPoseFinding(
                    body_part=r.body_part,
                    max_angle=r.max_angle,
                    percent_of_normal=r.percent_of_normal,
                    pain_score=r.pain_score,
                    commentary="(ไม่มีคำอธิบายจาก LLM)",
                    image_path=r.peak_frame_path,
                )
            )

    # 7-day plan: reuse the existing generator with a synthetic AssessmentState
    pseudo_state = AssessmentState(
        pain_location=list(session.pain_location),
        pain_score=max((r.pain_score or 0) for r in session.results),
        duration_days=0,
        sitting_hours_per_session=0,
        break_frequency_minutes=0,
        uses_laptop_only=None,
    )
    # Build a minimal RiskResult-compatible object via calculate_risk_score path
    try:
        from tools import calculate_risk_score
        rr = calculate_risk_score(pseudo_state)
        plan: ErgonomicPlan | None = generate_ergonomic_plan(pseudo_state, rr)
    except Exception:
        plan = None

    risk = parsed.get("overall_risk_level") or _derive_risk(session)
    if risk not in ("low", "moderate", "high"):
        risk = _derive_risk(session)

    return RomFinalReport(
        overall_risk_level=risk,
        summary=str(parsed.get("summary") or _derive_summary(session)),
        per_pose_findings=findings,
        seven_day_plan=plan,
        see_clinician=bool(parsed.get("see_clinician") or _derive_see_clinician(session)),
        rationale=str(parsed.get("rationale") or ""),
        generated_at=time.time(),
        session_dir=str(session_dir),
    )


# ---------------------------------------------------------------------------
# Deterministic fallbacks (no LLM)
# ---------------------------------------------------------------------------

def _derive_risk(session: RomSessionState) -> str:
    if not session.results:
        return "low"
    worst_pct = min(r.percent_of_normal for r in session.results)
    worst_pain = max((r.pain_score or 0) for r in session.results)
    if worst_pct < 30 or worst_pain >= 7:
        return "high"
    if worst_pct < 70 or worst_pain >= 4:
        return "moderate"
    return "low"


def _derive_see_clinician(session: RomSessionState) -> bool:
    for r in session.results:
        if r.percent_of_normal < 30 and (r.pain_score or 0) >= 5:
            return True
    return False


def _derive_summary(session: RomSessionState) -> str:
    n = len(session.results)
    if n == 0:
        return "ยังไม่มีข้อมูลครับ"
    pcts = [f"{r.body_part} {r.percent_of_normal:.0f}%" for r in session.results]
    return f"ทำได้ {n} ท่า: " + ", ".join(pcts)


def _fallback_report(session: RomSessionState, *, error: str) -> RomFinalReport:
    risk = _derive_risk(session)
    findings = [
        PerPoseFinding(
            body_part=r.body_part,
            max_angle=r.max_angle,
            percent_of_normal=r.percent_of_normal,
            pain_score=r.pain_score,
            commentary=f"({r.percent_of_normal:.0f}% ของมุมปกติ — ข้อมูลจากการวัดเท่านั้น)",
            image_path=r.peak_frame_path,
        )
        for r in session.results
    ]
    pseudo_state = AssessmentState(
        pain_location=list(session.pain_location),
        pain_score=max((r.pain_score or 0) for r in session.results),
    )
    try:
        from tools import calculate_risk_score
        rr = calculate_risk_score(pseudo_state)
        plan: ErgonomicPlan | None = generate_ergonomic_plan(pseudo_state, rr)
    except Exception:
        plan = None
    return RomFinalReport(
        overall_risk_level=risk,
        summary=_derive_summary(session) + " (LLM ไม่ตอบกลับ ใช้ผลคำนวณตรง)",
        per_pose_findings=findings,
        seven_day_plan=plan,
        see_clinician=_derive_see_clinician(session),
        rationale=f"fallback (LLM error: {error})",
        generated_at=time.time(),
        session_dir=session.session_dir,
    )
