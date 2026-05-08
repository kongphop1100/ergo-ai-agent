"""ROM Session Vision-Agents processor.

Subscribes to the user's WebRTC video track, runs YOLOv8-Pose at ~15 fps,
computes the joint angle for the currently-active pose, and maintains a
5-second circular buffer of (timestamp, angle, frame) so capture_peak_now()
can lift the actual peak frame even if the user signals "พอ" slightly late.

Owns the `RomSessionState` and exposes the 7 ROM tools the LLM calls.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import aiortc
import av
import cv2
import numpy as np
from vision_agents.core.processors.base_processor import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack

from rom_session import (
    ANGLE_TRIPLES,
    NORMAL_MAX,
    POSE_INSTRUCTIONS_TH,
    select_pose_sequence as compute_sequence,
)
from schemas import (
    AssessmentState,
    PoseResult,
    RedFlagResult,
    RedFlags,
    RomSessionState,
)
from tools import check_red_flags as run_red_flag_check

logger = logging.getLogger(__name__)

ROLLING_SECONDS = 5.0
DEFAULT_TARGET_FPS = 15
VISIBILITY_THRESHOLD = 0.4
FRAME_BUFFER_LONG_EDGE = 480

# Pass threshold for functional ROM — % of AAOS normal max.
# 70% is a common cutoff used in physiotherapy ROM-screening literature.
PASS_THRESHOLD_PERCENT = 70.0

# COCO-17 keypoint names (ultralytics convention)
KP_INDEX = {
    "NOSE": 0,
    "LEFT_EYE": 1, "RIGHT_EYE": 2,
    "LEFT_EAR": 3, "RIGHT_EAR": 4,
    "LEFT_SHOULDER": 5, "RIGHT_SHOULDER": 6,
    "LEFT_ELBOW": 7, "RIGHT_ELBOW": 8,
    "LEFT_WRIST": 9, "RIGHT_WRIST": 10,
    "LEFT_HIP": 11, "RIGHT_HIP": 12,
    "LEFT_KNEE": 13, "RIGHT_KNEE": 14,
    "LEFT_ANKLE": 15, "RIGHT_ANKLE": 16,
}

# COCO-17 skeleton edges (pairs of indices)
SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (5, 11), (6, 12), (11, 12),                     # torso
    (11, 13), (13, 15), (12, 14), (14, 16),         # legs
    (0, 1), (0, 2), (1, 3), (2, 4),                 # head
]


def _compute_angle(kpts: np.ndarray, triple_names: list[str]) -> float | None:
    try:
        idxs = [KP_INDEX[n] for n in triple_names]
    except KeyError:
        return None
    if any(float(kpts[i][2]) < VISIBILITY_THRESHOLD for i in idxs):
        return None
    p1, p2, p3 = (np.asarray(kpts[i][:2], dtype=float) for i in idxs)
    a = p1 - p2
    b = p3 - p2
    cross = float(a[0] * b[1] - a[1] * b[0])
    dot = float(np.dot(a, b))
    return math.degrees(math.atan2(abs(cross), dot))


def _resize_for_buffer(frame_bgr: np.ndarray, max_long_edge: int = FRAME_BUFFER_LONG_EDGE) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    scale = max_long_edge / max(h, w)
    if scale < 1.0:
        return cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame_bgr.copy()


class RomSessionProcessor(VideoProcessorPublisher):
    """Single-inference YOLO processor — both subscribes (for the angle buffer)
    and publishes a NEW video track back to the call so the user sees themselves
    with skeleton overlay + live angle text in their browser.

    Why a publisher: the upstream `ultralytics.YOLOPoseProcessor` has a bug where
    its published frames render all-blue (`av.VideoFrame.from_ndarray(...)` is
    called without `format="rgb24"` → RGB data interpreted as YUV420p). We do
    the same job in-house with the format kwarg fixed, AND share the YOLO
    inference with the angle-math path so we don't pay for it twice.
    """

    name = "rom_session"

    def __init__(
        self,
        session_dir: Path | str | None = None,
        model_path: str = "yolov8n-pose.pt",
        target_fps: int = DEFAULT_TARGET_FPS,
        device: str = "cpu",
        imgsz: int = 480,
    ) -> None:
        if session_dir is None:
            session_dir = (
                Path(__file__).parent
                / "sessions"
                / time.strftime("rom_%Y%m%d_%H%M%S", time.localtime())
            )
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "peaks").mkdir(exist_ok=True)

        self.state = RomSessionState(
            started_at=time.time(),
            session_dir=str(self.session_dir),
        )

        self.target_fps = target_fps
        self.imgsz = imgsz
        self.device = device
        self.model_path = model_path

        # Lazy YOLO load — first frame triggers actual model load on the worker thread
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rom_yolo")
        self._shutdown = False
        self._forwarder: Optional[VideoForwarder] = None

        self._buffer: deque[tuple[float, float, np.ndarray]] = deque(
            maxlen=int(ROLLING_SECONDS * target_fps) + 5
        )
        self._latest_angle: Optional[float] = None

        # Diagnostic counters — exposed in capture_peak_now error so the user can
        # see exactly where the pipeline breaks (no track / no YOLO / no person /
        # low visibility).
        self._frames_received = 0
        self._yolo_calls = 0
        self._yolo_detections = 0    # YOLO returned keypoints
        self._angles_computed = 0    # angle math succeeded (visibility passed)

        # Outbound video track — published back to the Stream call so the
        # browser shows the user's webcam with skeleton overlay.
        self._video_track = QueuedVideoTrack()
        self._save()

    # ------------------------------------------------------------------
    # VideoPublisher contract
    # ------------------------------------------------------------------

    def publish_video_track(self) -> aiortc.VideoStreamTrack:
        return self._video_track

    # ------------------------------------------------------------------
    # Vision-Agents lifecycle
    # ------------------------------------------------------------------

    async def process_video(
        self,
        track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        if self._forwarder is not None:
            try:
                await self._forwarder.remove_frame_handler(self._on_frame)
            except Exception:
                logger.debug("removing previous handler failed", exc_info=True)
            self._forwarder = None

        self._forwarder = (
            shared_forwarder
            if shared_forwarder is not None
            else VideoForwarder(
                track,
                max_buffer=self.target_fps,
                fps=self.target_fps,
                name="rom_forwarder",
            )
        )
        self._forwarder.add_frame_handler(
            self._on_frame, fps=float(self.target_fps), name="rom_session"
        )
        logger.info(
            "🎯 rom_session processor started @ %d fps | participant=%s",
            self.target_fps, participant_id,
        )

    async def stop_processing(self) -> None:
        self._shutdown = True
        if self._forwarder is not None:
            try:
                await self._forwarder.remove_frame_handler(self._on_frame)
            except Exception:
                pass
            self._forwarder = None

    async def close(self) -> None:
        self._shutdown = True
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Per-frame: YOLO + angle + buffer
    # ------------------------------------------------------------------

    async def _on_frame(self, frame: av.VideoFrame) -> None:
        if self._shutdown:
            return
        try:
            arr_rgb = frame.to_ndarray(format="rgb24")
        except Exception:
            logger.exception("frame.to_ndarray failed")
            return

        # Run YOLO every frame — we need keypoints both for the angle buffer
        # and for drawing the skeleton overlay we publish back.
        try:
            loop = asyncio.get_event_loop()
            self._yolo_calls += 1
            kpts = await loop.run_in_executor(self._executor, self._yolo_keypoints, arr_rgb)
        except Exception:
            logger.exception("YOLO inference failed")
            kpts = None

        if kpts is not None:
            self._yolo_detections += 1

        # If a pose is active, count this frame and try to compute the angle
        active_angle: Optional[float] = None
        if self.state.current_pose is not None:
            self._frames_received += 1
            if self._frames_received == 1:
                logger.info(
                    "🎬 rom_session received first frame for pose=%s",
                    self.state.current_pose,
                )
            if kpts is not None:
                triple = ANGLE_TRIPLES.get(self.state.current_pose)
                if triple:
                    angle = _compute_angle(kpts, triple)
                    if angle is not None:
                        self._angles_computed += 1
                        self._latest_angle = angle
                        active_angle = angle
                        bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
                        self._buffer.append(
                            (time.time(), angle, _resize_for_buffer(bgr))
                        )

        # Publish annotated frame back to the call (always, even if no pose active)
        try:
            annotated = self._draw_overlay(arr_rgb.copy(), kpts, active_angle)
            # CRITICAL: pass format="rgb24" — upstream YOLOPoseProcessor omits this
            # which is why their published track renders blue.
            out_frame = av.VideoFrame.from_ndarray(annotated, format="rgb24")
            await self._video_track.add_frame(out_frame)
        except Exception:
            logger.exception("publish frame failed")

    # ------------------------------------------------------------------
    # Skeleton + angle overlay (drawn on a copy of the RGB array)
    # ------------------------------------------------------------------

    def _draw_overlay(
        self,
        arr_rgb: np.ndarray,
        kpts: Optional[np.ndarray],
        active_angle: Optional[float],
    ) -> np.ndarray:
        h, w = arr_rgb.shape[:2]

        if kpts is not None:
            active_triple_idxs: set[int] = set()
            if self.state.current_pose:
                triple_names = ANGLE_TRIPLES.get(self.state.current_pose, [])
                active_triple_idxs = {KP_INDEX[n] for n in triple_names if n in KP_INDEX}

            # skeleton edges (gray-blue) for context
            for a, b in SKELETON_EDGES:
                if a < len(kpts) and b < len(kpts):
                    if float(kpts[a][2]) > 0.3 and float(kpts[b][2]) > 0.3:
                        cv2.line(
                            arr_rgb,
                            (int(kpts[a][0]), int(kpts[a][1])),
                            (int(kpts[b][0]), int(kpts[b][1])),
                            (0, 200, 255),  # in RGB this reads as cyan-ish
                            2,
                        )
            # all keypoints (small green dots)
            for i in range(len(kpts)):
                if float(kpts[i][2]) > 0.3:
                    cv2.circle(
                        arr_rgb,
                        (int(kpts[i][0]), int(kpts[i][1])),
                        4,
                        (50, 230, 50),
                        -1,
                    )
            # active triple — emphasized
            if active_triple_idxs:
                pts = [
                    (int(kpts[i][0]), int(kpts[i][1]))
                    for i in active_triple_idxs
                    if i < len(kpts) and float(kpts[i][2]) > 0.3
                ]
                # connect the triple
                if self.state.current_pose:
                    triple_names = ANGLE_TRIPLES.get(self.state.current_pose, [])
                    if len(triple_names) == 3 and all(n in KP_INDEX for n in triple_names):
                        idxs = [KP_INDEX[n] for n in triple_names]
                        if all(i < len(kpts) and float(kpts[i][2]) > 0.3 for i in idxs):
                            cv2.line(
                                arr_rgb,
                                (int(kpts[idxs[0]][0]), int(kpts[idxs[0]][1])),
                                (int(kpts[idxs[1]][0]), int(kpts[idxs[1]][1])),
                                (255, 220, 0),  # yellow
                                3,
                            )
                            cv2.line(
                                arr_rgb,
                                (int(kpts[idxs[1]][0]), int(kpts[idxs[1]][1])),
                                (int(kpts[idxs[2]][0]), int(kpts[idxs[2]][1])),
                                (255, 220, 0),
                                3,
                            )
                # red dot on each triple keypoint
                for x, y in pts:
                    cv2.circle(arr_rgb, (x, y), 7, (255, 50, 50), -1)

        # Top-left status banner
        banner_h = 36
        cv2.rectangle(arr_rgb, (0, 0), (w, banner_h), (0, 0, 0), -1)
        if self.state.current_pose:
            angle_text = f"{active_angle:.0f}°" if active_angle is not None else "..."
            label = f"{self.state.current_pose}  {angle_text}"
        else:
            label = "ERGO AI — waiting for pose"
        cv2.putText(
            arr_rgb,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return arr_rgb

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO  # heavy import — lazy
        self._model = YOLO(self.model_path)
        try:
            self._model.to(self.device)
        except Exception:
            pass

    def _yolo_keypoints(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        self._ensure_model()
        results = self._model(frame_rgb, verbose=False, imgsz=self.imgsz)
        if not results or results[0].keypoints is None:
            return None
        data = results[0].keypoints.data
        if hasattr(data, "cpu"):
            data = data.cpu().numpy()
        if len(data) == 0:
            return None
        return data[0]

    # ------------------------------------------------------------------
    # Peak retrieval
    # ------------------------------------------------------------------

    def peak_in_buffer(self) -> Optional[tuple[float, float, np.ndarray]]:
        """Return (max_angle_smoothed, timestamp, frame_bgr) or None."""
        if not self._buffer:
            return None
        entries = list(self._buffer)
        if len(entries) < 3:
            best = max(entries, key=lambda e: e[1])
            return best[1], best[0], best[2]
        smoothed: list[tuple[float, float, np.ndarray]] = []
        for i in range(1, len(entries) - 1):
            window = sorted(entries[i - 1 : i + 2], key=lambda e: e[1])
            ts, _, fr = entries[i]
            smoothed.append((ts, window[1][1], fr))
        best = max(smoothed, key=lambda e: e[1])
        return best[1], best[0], best[2]

    # ------------------------------------------------------------------
    # Tool implementations (sync — wrapped in async by agent.py registry)
    # ------------------------------------------------------------------

    def record_answer(self, field: str, value: Any) -> dict[str, Any]:
        if field == "pain_location":
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                return {"error": f"pain_location must be a list, got {type(value).__name__}"}
            self.state.pain_location = [str(x).strip().lower() for x in value]
            self._save()
            return {"updated": "pain_location", "value": list(self.state.pain_location)}
        return {"warning": f"field {field!r} not used by ROM session"}

    def select_pose_sequence(self, pain_location: list[str]) -> dict[str, Any]:
        locs = pain_location or list(self.state.pain_location)
        locs = [str(x).strip().lower() for x in locs if x]
        sequence = compute_sequence(locs)
        self.state.pain_location = locs
        self.state.pose_queue = sequence
        self._save()
        return {"pose_queue": sequence, "n_poses": len(sequence)}

    def check_red_flags(self, red_flags: dict[str, bool] | None = None) -> dict[str, Any]:
        flags = RedFlags(**(red_flags or {}))
        proxy = AssessmentState(red_flags=flags)
        rf = run_red_flag_check(proxy)
        self.state.red_flag = rf
        self._save()
        return rf.model_dump()

    def start_pose(self, body_part: str) -> dict[str, Any]:
        if body_part not in ANGLE_TRIPLES:
            return {"error": f"unsupported body_part: {body_part}"}
        self.state.current_pose = body_part
        self.state.current_pain_score = None
        self._buffer.clear()
        self._latest_angle = None
        # Reset per-pose diagnostics so capture_peak_now's diag dict reflects
        # only THIS pose's frame flow, not earlier ones.
        self._frames_received = 0
        self._yolo_calls = 0
        self._yolo_detections = 0
        self._angles_computed = 0
        self._save()
        return {
            "body_part": body_part,
            "normal_max_deg": NORMAL_MAX.get(body_part, 0.0),
            "instruction_th": POSE_INSTRUCTIONS_TH.get(body_part, ""),
        }

    def capture_peak_now(self) -> dict[str, Any]:
        if not self.state.current_pose:
            return {"error": "no current pose — call start_pose first"}
        peak = self.peak_in_buffer()
        if peak is None:
            # Diagnostic: tell the LLM which stage the pipeline broke at so it can
            # give the user actionable advice rather than a generic "try again".
            diag = {
                "frames_received": self._frames_received,
                "yolo_calls": self._yolo_calls,
                "yolo_detections": self._yolo_detections,
                "angles_computed": self._angles_computed,
                "buffer_size": len(self._buffer),
                "current_pose": self.state.current_pose,
            }
            if self._frames_received == 0:
                hint = "no video frames received — check that the user has enabled their webcam"
            elif self._yolo_detections == 0:
                hint = "YOLO ran but did not detect a person — user may be out of frame or lighting too dim"
            elif self._angles_computed == 0:
                hint = (
                    "person detected but the relevant joint keypoints had low visibility "
                    "— ask user to face the camera and ensure the moving limb is visible"
                )
            else:
                hint = "buffer was cleared too quickly — user signalled before any post-smoothing entries accumulated"
            logger.warning("capture_peak_now: empty buffer | %s | hint: %s", diag, hint)
            return {"error": "no valid angle in buffer", "diagnostics": diag, "hint": hint}
        max_angle, peak_ts, frame_bgr = peak
        body_part = self.state.current_pose
        normal_max = NORMAL_MAX.get(body_part, 180.0)
        percent = round((max_angle / normal_max) * 100, 1) if normal_max else 0.0

        rel_path = f"peaks/{body_part}_{int(round(max_angle))}deg.jpg"
        full_path = self.session_dir / rel_path
        try:
            cv2.imwrite(str(full_path), frame_bgr)
        except Exception as exc:
            return {"error": f"failed to write frame: {exc}"}

        passed = percent >= PASS_THRESHOLD_PERCENT
        if passed:
            verdict = f"ผ่านเกณฑ์ {percent:.0f}% (เกณฑ์ {PASS_THRESHOLD_PERCENT:.0f}%)"
        else:
            verdict = f"ยังไม่ผ่านเกณฑ์ {percent:.0f}% (เกณฑ์ {PASS_THRESHOLD_PERCENT:.0f}%)"

        result = PoseResult(
            body_part=body_part,
            max_angle=round(float(max_angle), 1),
            normal_max=normal_max,
            percent_of_normal=percent,
            peak_timestamp=float(peak_ts),
            peak_frame_path=rel_path,
            pain_score=self.state.current_pain_score,
            threshold_percent=PASS_THRESHOLD_PERCENT,
            passed=passed,
        )
        self.state.results.append(result)
        self._save()
        out = result.model_dump()
        # Convenience field for the LLM to read aloud directly.
        out["verdict"] = verdict
        return out

    def get_current_angle(self) -> dict[str, Any]:
        """Live-coaching tool: agent calls this every few seconds during a pose
        to give running feedback like 'ตอนนี้ 110 องศา ลองอีกนิด'."""
        if not self.state.current_pose:
            return {"error": "no current pose — call start_pose first"}
        normal_max = NORMAL_MAX.get(self.state.current_pose, 180.0)
        if self._latest_angle is None:
            return {
                "angle": None,
                "percent_of_normal": None,
                "passed": False,
                "person_visible": self._yolo_detections > 0,
                "frames_received": self._frames_received,
                "hint": (
                    "still warming up — say 'ค่อยๆเริ่มได้เลยครับ' if angle is None "
                    "and frames are flowing"
                ),
            }
        percent = round((self._latest_angle / normal_max) * 100, 1) if normal_max else 0.0
        passed = percent >= PASS_THRESHOLD_PERCENT
        # Coaching hint based on how close to threshold we are
        if percent < 30:
            coach = "ยังยกได้อีก"
        elif percent < 50:
            coach = "เกินครึ่งทางแล้ว"
        elif percent < PASS_THRESHOLD_PERCENT:
            coach = "ใกล้เกณฑ์แล้ว ลองอีกนิด"
        elif percent < 90:
            coach = "ผ่านเกณฑ์แล้ว ดีมาก"
        else:
            coach = "เยี่ยม ค้างไว้"
        return {
            "body_part": self.state.current_pose,
            "angle": round(float(self._latest_angle), 1),
            "percent_of_normal": percent,
            "normal_max": normal_max,
            "threshold_percent": PASS_THRESHOLD_PERCENT,
            "passed": passed,
            "coach_hint": coach,
        }

    def record_pain_score(self, score: int | float, note: str = "") -> dict[str, Any]:
        try:
            score_int = int(round(float(score)))
        except (TypeError, ValueError):
            return {"error": f"could not parse score: {score!r}"}
        score_int = max(0, min(10, score_int))
        target = next(
            (p for p in reversed(self.state.results) if p.pain_score is None), None
        )
        if target is None:
            return {"error": "no recent pose to attach pain score to"}
        target.pain_score = score_int
        target.pain_note = note or ""
        self.state.current_pain_score = score_int
        self.state.current_pose = None
        self._save()
        return {
            "body_part": target.body_part,
            "pain_score": score_int,
            "remaining_poses": [
                p for p in self.state.pose_queue
                if p not in {r.body_part for r in self.state.results}
            ],
        }

    def end_session(self) -> dict[str, Any]:
        from rom_analyzer import analyze  # lazy
        self.state.ended_at = time.time()
        self.state.current_pose = None
        self._save()
        try:
            report = analyze(self.state)
        except Exception as exc:
            err = {"error": f"analyzer failed: {exc}"}
            (self.session_dir / "final_report_error.json").write_text(
                json.dumps(err, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return err
        report.session_dir = str(self.session_dir)
        report.generated_at = time.time()
        (self.session_dir / "final_report.json").write_text(
            json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report.model_dump()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            (self.session_dir / "rom_session.json").write_text(
                json.dumps(self.state.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("state save failed")
