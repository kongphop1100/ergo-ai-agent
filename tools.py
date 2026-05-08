"""Pure-Python tools the agent calls to score the assessment.

No LLM, no I/O. The agent passes `state` in, gets a structured result out.
This makes the scoring auditable and unit-testable independent of any model.
"""
from __future__ import annotations

import json
from pathlib import Path

from schemas import (
    AssessmentState,
    ErgonomicPlan,
    RedFlagResult,
    RiskBreakdown,
    RiskResult,
)

GUIDELINES_PATH = Path(__file__).parent / "data" / "ergonomic_guidelines.json"

RED_FLAG_LABELS_TH = {
    "numbness": "อาการชา/เหน็บ",
    "weakness": "กล้ามเนื้ออ่อนแรง",
    "balance_problem": "เดิน/ทรงตัวลำบาก",
    "severe_headache": "ปวดศีรษะรุนแรงร่วมกับปวดคอ",
    "after_trauma": "ปวดหลังเกิดอุบัติเหตุ",
    "incontinence": "กลั้นปัสสาวะ/อุจจาระไม่อยู่",
    "fever_unwell": "มีไข้/รู้สึกป่วยจริงจัง",
}

RED_FLAG_MESSAGE = (
    "ตรวจพบสัญญาณเตือนที่ควรปรึกษาแพทย์ก่อนเริ่มท่ายืดหรือปรับพฤติกรรม "
    "ขอแนะนำให้ติดต่อแพทย์/นักกายภาพเพื่อประเมินอย่างละเอียดก่อน "
    "(หมายเหตุ: ระบบนี้เป็นการคัดกรองเบื้องต้น ไม่ใช่การวินิจฉัยทางการแพทย์)"
)


def check_red_flags(state: AssessmentState) -> RedFlagResult:
    triggered = state.red_flags.list_triggered()
    if state.pain_radiating is True and "numbness" not in triggered:
        triggered.append("ปวดร้าวลงแขน")
    if not triggered:
        return RedFlagResult(
            has_red_flag=False,
            red_flags=[],
            message="ไม่พบสัญญาณเตือนที่ต้องพบแพทย์เร่งด่วน",
        )
    labeled = [RED_FLAG_LABELS_TH.get(f, f) for f in triggered]
    return RedFlagResult(has_red_flag=True, red_flags=labeled, message=RED_FLAG_MESSAGE)


def _score_pain_intensity(pain: int | None) -> int:
    if pain is None:
        return 0
    if pain <= 2:
        return 0
    if pain <= 5:
        return 1
    return 2


def _score_duration(days: int | None) -> int:
    if days is None:
        return 0
    if days < 3:
        return 0
    if days <= 14:
        return 1
    return 2


def _score_sitting(hours: float | None) -> int:
    if hours is None:
        return 0
    if hours < 2:
        return 0
    if hours <= 4:
        return 1
    return 2


def _score_break(minutes: int | None) -> int:
    if minutes is None:
        return 0
    if minutes <= 60:
        return 0
    if minutes <= 120:
        return 1
    return 2


def _score_workstation(state: AssessmentState) -> tuple[int, list[str]]:
    """Returns (points, list of contributing factor descriptions)."""
    issues: list[str] = []
    if state.uses_laptop_only is True:
        issues.append("ใช้ laptop อย่างเดียว ไม่มี keyboard/mouse แยก")
    if state.monitor_at_eye_level is False:
        issues.append("จออยู่ต่ำกว่าระดับสายตา ทำให้ก้มคอ")
    if state.external_keyboard is False and state.uses_laptop_only is not True:
        issues.append("ไม่มี external keyboard")
    pts = 0 if not issues else (1 if len(issues) == 1 else 2)
    return pts, issues


def _score_symptoms(state: AssessmentState, red_flag: RedFlagResult) -> tuple[int, list[str]]:
    items: list[str] = []
    pts = 0
    if red_flag.has_red_flag or state.pain_radiating is True:
        items.append("มีอาการชา/ร้าว/อ่อนแรง")
        pts = 2
    elif (state.pain_score or 0) >= 6 or (state.duration_days or 0) > 14:
        items.append("ปวดเรื้อรังหรือระดับความปวดสูง")
        pts = 1
    return pts, items


def calculate_risk_score(state: AssessmentState) -> RiskResult:
    red_flag = check_red_flags(state)

    pi = _score_pain_intensity(state.pain_score)
    du = _score_duration(state.duration_days)
    si = _score_sitting(state.sitting_hours_per_session)
    br = _score_break(state.break_frequency_minutes)
    ws_pts, ws_issues = _score_workstation(state)
    sy_pts, sy_issues = _score_symptoms(state, red_flag)

    breakdown = RiskBreakdown(
        pain_intensity=pi,
        duration=du,
        sitting_duration=si,
        break_frequency=br,
        workstation=ws_pts,
        symptoms=sy_pts,
    )

    vision_uplift = False
    if state.vision_factors and state.vision_factors.has_severe():
        # Bump workstation to 2 (the photo confirmed setup is bad) and
        # +1 to symptoms (capped) — see plan section "combined risk-score uplift rule".
        if breakdown.workstation < 2:
            breakdown.workstation = 2
            vision_uplift = True
        if breakdown.symptoms < 2:
            breakdown.symptoms = min(2, breakdown.symptoms + 1)
            vision_uplift = True

    factors: list[str] = []
    if pi == 2:
        factors.append(f"ปวดระดับสูง ({state.pain_score}/10)")
    if du == 2:
        factors.append(f"ปวดเรื้อรัง ({state.duration_days} วัน)")
    if si == 2:
        factors.append(f"นั่งต่อเนื่องนานเกิน 4 ชม. ({state.sitting_hours_per_session} ชม.)")
    if br == 2:
        factors.append("แทบไม่ได้ลุกพัก")
    factors.extend(ws_issues)
    factors.extend(sy_issues)
    if state.vision_factors and state.vision_factors.has_moderate_or_severe():
        for name, f in state.vision_factors.factors.items():
            if f.severity in ("moderate", "severe"):
                factors.append(f"กล้องเห็น: {name} ({f.severity})")

    contributing = []
    if pi:
        contributing.append("Pain intensity")
    if du:
        contributing.append("Duration")
    if si:
        contributing.append("Sitting duration")
    if br:
        contributing.append("Break frequency")
    if breakdown.workstation:
        contributing.append("Workstation")
    if breakdown.symptoms:
        contributing.append("Symptoms")

    score = breakdown.total()
    if score <= 4:
        level = "low"
    elif score <= 8:
        level = "moderate"
    else:
        level = "high"

    return RiskResult(
        score=score,
        breakdown=breakdown,
        risk_level=level,
        risk_factors=factors,
        contributing_categories=contributing,
        vision_uplift_applied=vision_uplift,
    )


def _load_guidelines() -> dict:
    if GUIDELINES_PATH.exists():
        return json.loads(GUIDELINES_PATH.read_text(encoding="utf-8"))
    return {}


def generate_ergonomic_plan(state: AssessmentState, risk: RiskResult) -> ErgonomicPlan:
    g = _load_guidelines()

    chair = list(g.get("chair_default", []))
    monitor = list(g.get("monitor_default", []))
    keyboard_mouse = list(g.get("keyboard_mouse_default", []))
    break_routine = list(g.get("break_default", []))
    stretches = list(g.get("stretches_default", []))

    if state.uses_laptop_only:
        keyboard_mouse += g.get("if_laptop_only", [])
        monitor += g.get("if_laptop_only_monitor", [])
    if state.monitor_at_eye_level is False:
        monitor += g.get("if_monitor_low", [])
    if (state.break_frequency_minutes or 0) > 60:
        break_routine += g.get("if_no_breaks", [])
    if (state.sitting_hours_per_session or 0) > 4:
        break_routine += g.get("if_long_sitting", [])
    if "neck" in state.pain_location:
        stretches += g.get("stretches_neck", [])
    if "shoulder" in state.pain_location:
        stretches += g.get("stretches_shoulder", [])
    if "back" in state.pain_location:
        stretches += g.get("stretches_back", [])
    if "wrist" in state.pain_location:
        stretches += g.get("stretches_wrist", [])
    if "eye_head" in state.pain_location:
        break_routine += g.get("if_eye_strain", [])

    seen = set()
    def dedupe(items):
        out = []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    chair = dedupe(chair)
    monitor = dedupe(monitor)
    keyboard_mouse = dedupe(keyboard_mouse)
    break_routine = dedupe(break_routine)
    stretches = dedupe(stretches)

    high_priority = []
    if state.uses_laptop_only:
        high_priority.append("ยก laptop ให้จอใกล้ระดับสายตา + ใช้ keyboard/mouse แยก")
    if state.monitor_at_eye_level is False:
        high_priority.append("ปรับจอให้อยู่ระดับสายตา")
    if (state.break_frequency_minutes or 0) > 60:
        high_priority.append("ตั้ง reminder ลุกเปลี่ยนท่าทุก 30–60 นาที")
    if not high_priority:
        high_priority = ["ตรวจสอบระยะห่างของจอและท่านั่งให้ตรงและสบาย"]

    daily = []
    daily.append({"day": 1, "actions": ["บันทึก pain score เช้า/เย็น", high_priority[0]]})
    daily.append({"day": 2, "actions": ["บันทึก pain score เช้า/เย็น"] + (high_priority[1:2] or ["ทำท่ายืดคอ/บ่า 5 นาที"])})
    daily.append({"day": 3, "actions": ["ตรวจสอบความสูงเก้าอี้และการรองรับหลัง", "ทำท่ายืด 10 นาที"]})
    daily.append({"day": 4, "actions": ["บันทึก pain score และเทียบกับวันที่ 1"]})
    daily.append({"day": 5, "actions": ["จัด workstation ตามคำแนะนำที่ยังไม่ทำ", "ลุกเดิน 5 นาทีทุกชั่วโมง"]})
    daily.append({"day": 6, "actions": ["ทำท่ายืดที่แนะนำครบทุกชุด"]})
    daily.append({"day": 7, "actions": ["สรุปอาการสัปดาห์นี้", "ถ้าอาการไม่ดีขึ้น/แย่ลง → พบแพทย์/นักกายภาพ"]})

    return ErgonomicPlan(
        daily_plan=daily,
        chair=chair,
        monitor=monitor,
        keyboard_mouse=keyboard_mouse,
        break_routine=break_routine,
        stretches=stretches,
    )
