"""ROM session data tables + helpers.

Loads `data/aaos_normal_rom.json` once and exposes:
- POSE_SEQUENCE_BY_LOCATION  — pain_location → ordered list of poses
- ANGLE_TRIPLES              — pose → (p1, p2, p3) keypoint names
- POSE_INSTRUCTIONS_TH       — pose → Thai instruction sentence
- NORMAL_MAX                 — pose → AAOS reference angle in degrees
- select_pose_sequence()     — combine multiple pain locations into one queue
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "aaos_normal_rom.json"
_DATA = json.loads(DATA_PATH.read_text(encoding="utf-8"))

POSE_SEQUENCE_BY_LOCATION: dict[str, list[str]] = _DATA["pose_sequence_by_location"]
ANGLE_TRIPLES: dict[str, list[str]] = _DATA["angle_triples"]
POSE_INSTRUCTIONS_TH: dict[str, str] = _DATA["pose_instructions_th"]
NORMAL_MAX: dict[str, float] = {
    k: float(v)
    for k, v in _DATA.items()
    if not k.startswith("_") and isinstance(v, (int, float))
}


def select_pose_sequence(pain_location: list[str]) -> list[str]:
    """MOCK MODE — only 2 shoulder poses for demo, regardless of pain_location.

    Original logic preserved below for when we ship more poses; comment-toggle
    to switch back to the full per-location sequencing.
    """
    return ["shoulder_flexion", "shoulder_abduction"]

    # # Real version (off for demo):
    # seen: set[str] = set()
    # out: list[str] = []
    # for loc in pain_location or []:
    #     for pose in POSE_SEQUENCE_BY_LOCATION.get(loc, []):
    #         if pose not in seen:
    #             seen.add(pose)
    #             out.append(pose)
    # if not out:
    #     out = list(POSE_SEQUENCE_BY_LOCATION.get("other", ["shoulder_flexion"]))
    # return out
