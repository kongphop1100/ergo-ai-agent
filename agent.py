"""ERGO AI Voice-Guided ROM Session Agent.

Vision-Agents agent backed by:
  - Stream WebRTC edge (browser ↔ agent audio/video)
  - OpenAI Realtime API (gpt-realtime-2025-08-28) for voice + tool calling
  - YOLOv8-Pose (via our RomSessionProcessor) for live joint-angle tracking

Run:
    uv run agent.py

This launches the Vision-Agents CLI runner. Set call_id env or use the
default "ergo-rom" so the frontend (Next.js) can target the same call.

Required env (in .env at project root):
    STREAM_API_KEY
    STREAM_API_SECRET
    OPENAI_API_KEY        # for Realtime
    OPENROUTER_API_KEY    # for the post-session analyzer
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.edge.events import TrackAddedEvent, TrackRemovedEvent
from vision_agents.plugins import getstream, openai

from prompts import ROM_SYSTEM_PROMPT
from rom_processor import RomSessionProcessor

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

REALTIME_MODEL = "gpt-realtime-2"
REALTIME_VOICE = "marin"  # alloy | marin | echo | sage | shimmer


def setup_llm(rom: RomSessionProcessor) -> openai.Realtime:
    """Create the Realtime LLM and register the 7 ROM tools on it."""
    # Latency tuning:
    # - send_video=False — YOLO does the visual work; sending frames to OpenAI just
    #   eats bandwidth and slows the round-trip
    # - The Realtime model still hears + speaks; vision is handled server-side by YOLO tools
    llm = openai.Realtime(
        model=REALTIME_MODEL,
        voice=REALTIME_VOICE,
        send_video=False,
    )

    @llm.register_function(
        description=(
            "Persist one fact from the user. Currently used for "
            "field='pain_location' with a list of strings."
        ),
    )
    async def record_answer(field: str, value: Any) -> Dict[str, Any]:
        return rom.record_answer(field, value)

    @llm.register_function(
        description=(
            "Build the ROM pose queue from the user's pain locations. "
            "Returns {pose_queue, n_poses}. Call immediately after pain_location is known."
        ),
    )
    async def select_pose_sequence(pain_location: List[str]) -> Dict[str, Any]:
        return rom.select_pose_sequence(pain_location)

    @llm.register_function(
        description=(
            "Run the red-flag screen. Pass answers as a dict, e.g. "
            "{numbness: true, weakness: false, balance_problem: false}. Returns "
            "has_red_flag, list of triggered flags, guidance message. If has_red_flag, "
            "STOP the session and recommend a clinician."
        ),
    )
    async def check_red_flags(red_flags: Dict[str, bool]) -> Dict[str, Any]:
        return rom.check_red_flags(red_flags)

    @llm.register_function(
        description=(
            "Begin a specific ROM pose. Tells the camera tracker to focus on the "
            "joint angle for this pose. MUST be called BEFORE the user starts moving. "
            "body_part is one of: shoulder_flexion, shoulder_abduction, elbow_flexion, "
            "knee_flexion, trunk_lateral_left, trunk_lateral_right."
        ),
    )
    async def start_pose(body_part: str) -> Dict[str, Any]:
        return rom.start_pose(body_part)

    @llm.register_function(
        description=(
            "Save the peak frame for the currently-active pose. Call AFTER the user "
            "verbally confirms they've reached their max (พอ / ไม่ไหว / เจ็บแล้ว). "
            "Returns max_angle, percent_of_normal, peak_frame_path, passed (bool), "
            "threshold_percent, and a verdict string in Thai."
        ),
    )
    async def capture_peak_now() -> Dict[str, Any]:
        return rom.capture_peak_now()

    @llm.register_function(
        description=(
            "Read the LIVE current joint angle while the user is mid-motion. "
            "Returns {angle, percent_of_normal, passed, coach_hint}. Call this "
            "every ~2-3 seconds during an active pose to give live encouragement "
            "(e.g. 'ตอนนี้ 110 องศา ใกล้เกณฑ์แล้ว'). Do NOT call this before "
            "start_pose or after capture_peak_now. Do NOT call it more than once "
            "per ~2 seconds."
        ),
    )
    async def get_current_angle() -> Dict[str, Any]:
        return rom.get_current_angle()

    @llm.register_function(
        description=(
            "Attach a 0-10 pain score to the most recently captured pose. Call "
            "RIGHT AFTER capture_peak_now once you've asked the user."
        ),
    )
    async def record_pain_score(score: int, note: str = "") -> Dict[str, Any]:
        return rom.record_pain_score(score, note)

    @llm.register_function(
        description=(
            "Finalize the session — runs the multimodal analyzer over all captured "
            "peak frames + numbers + pain scores and writes final_report.json. "
            "May take 5-15 seconds. Call exactly once after all poses are done."
        ),
    )
    async def end_session() -> Dict[str, Any]:
        return rom.end_session()

    return llm


async def create_agent(**kwargs) -> Agent:
    # Latency tuning: 10 fps + 320 imgsz keeps ROM-grade accuracy at ~3x lower CPU load
    # If the user has CUDA, override device="cuda" via env or here for ~5-10x speedup.
    rom = RomSessionProcessor(target_fps=10, imgsz=320)
    llm = setup_llm(rom)

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="ERGO AI Physio", id="ergo-physio"),
        instructions=ROM_SYSTEM_PROMPT,
        # Single processor — RomSessionProcessor now both runs YOLO and publishes
        # the annotated track back (skeleton overlay + live angle text). One
        # YOLO inference per frame, correct color format, no upstream blue-screen bug.
        processors=[rom],
        llm=llm,
    )
    logger.info("[agent] created. session dir: %s", rom.session_dir)

    # --- Track-flow diagnostics ---------------------------------------
    # Subscribe to the edge's track events so we can see in the log whether
    # the user's webcam + mic actually reach the agent. If nothing shows
    # up here when the user joins, the issue is browser-side (permissions
    # / mute), NOT the agent code.
    @agent.edge.events.subscribe
    async def _diag_track_added(event: TrackAddedEvent):
        if event.participant is None:
            return
        if event.participant.user_id == agent.agent_user.id:
            return  # ignore the agent's own published tracks
        logger.info(
            "✅ DIAG: track ADDED  type=%s  user=%s  track_id=%s",
            getattr(event.track_type, "name", event.track_type),
            event.participant.user_id,
            (event.track_id or "")[:12],
        )

    @agent.edge.events.subscribe
    async def _diag_track_removed(event: TrackRemovedEvent):
        if event.participant is None:
            return
        if event.participant.user_id == agent.agent_user.id:
            return
        logger.info(
            "❌ DIAG: track REMOVED type=%s  user=%s",
            getattr(event.track_type, "name", event.track_type),
            event.participant.user_id,
        )

    # Watchdog: if we don't see any user track within 15s of agent join,
    # print a big warning so the user knows to check browser permissions.
    async def _track_watchdog():
        await asyncio.sleep(15)
        user_tracks = [
            t for t in agent._active_video_tracks.values() if not t.processor
        ]
        if not user_tracks:
            logger.warning("=" * 60)
            logger.warning(
                "⚠️  DIAG: 15 sec passed and NO user video track received."
            )
            logger.warning(
                "    → Browser likely didn't grant camera. In the demo URL,"
            )
            logger.warning(
                "    → click the lock icon in the address bar and Allow Camera + Mic,"
            )
            logger.warning(
                "    → then refresh the tab. If audio also fails, same fix."
            )
            logger.warning("=" * 60)

    asyncio.create_task(_track_watchdog())
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)
    async with agent.join(call):
        # Let the model deliver its disclaimer + first question on connection
        await agent.simple_response(
            "ทักทายผู้ใช้ในภาษาไทย แจ้ง disclaimer และถามว่าปวดบริเวณไหน"
        )
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
