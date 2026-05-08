"""Local ROM Session Agent — no Stream, no Vision-Agents framework.

Standalone Python script that:
- Opens the laptop webcam in a cv2 window with live skeleton + angle banner
- Streams mic audio to OpenAI Realtime API via WebSocket directly
- Plays the assistant's audio reply through the speakers
- Runs YOLO + the same 7 ROM tools as agent.py

Run:
    uv run local_agent.py

Required env:
    OPENAI_API_KEY        (Realtime + Whisper)
    OPENROUTER_API_KEY    (post-session analyzer, same as agent.py)

Press 'q' in the camera window or Ctrl+C in terminal to stop.

Why this exists:
    The Vision-Agents path has ~500ms latency in TH because of the
    Browser→Stream→Agent→OpenAI hop chain. This local script cuts every
    hop except `mic → OpenAI` so latency is just the user→OpenAI RTT
    (≈ same as ChatGPT voice mode in the browser).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from prompts import ROM_SYSTEM_PROMPT
from rom_processor import (
    KP_INDEX,
    PASS_THRESHOLD_PERCENT,
    SKELETON_EDGES,
    _compute_angle,
    _resize_for_buffer,
)
from rom_session import (
    ANGLE_TRIPLES,
    NORMAL_MAX,
    POSE_INSTRUCTIONS_TH,
    select_pose_sequence as compute_sequence,
)
from schemas import (
    AssessmentState,
    PoseResult,
    RedFlags,
    RomSessionState,
)
from tools import check_red_flags as run_red_flag_check

logger = logging.getLogger("local_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_MS = 50
CHUNK_FRAMES = SAMPLE_RATE * CHUNK_MS // 1000
TARGET_FPS = 10
ROLLING_SECONDS = 5.0
DEFAULT_MODEL = "gpt-realtime-2025-08-28"
DEFAULT_VOICE = "marin"


# ---------------------------------------------------------------------------
# Tools (mirror of rom_processor — but standalone so no Vision-Agents needed)
# ---------------------------------------------------------------------------

class LocalRomState:
    def __init__(self, record_video: bool = True) -> None:
        ts = time.strftime("local_%Y%m%d_%H%M%S")
        self.session_dir = Path(__file__).parent / "sessions" / ts
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "peaks").mkdir(exist_ok=True)

        self.state = RomSessionState(
            started_at=time.time(),
            session_dir=str(self.session_dir),
        )
        self._buffer: deque[tuple[float, float, np.ndarray]] = deque(
            maxlen=int(ROLLING_SECONDS * TARGET_FPS) + 5
        )
        self._latest_angle: Optional[float] = None
        self._frames_received = 0
        self._yolo_calls = 0
        self._yolo_detections = 0
        self._angles_computed = 0

        # Video + audio recording. We write 3 raw files during the session:
        #   recording.mp4  — annotated video frames (no audio, cv2 limitation)
        #   mic.wav        — what the mic captured (user voice)
        #   ai.wav         — what the AI spoke (output we sent to speaker)
        # On close we ffmpeg-merge them into final.mp4 (video + mixed audio).
        self.record_video = record_video
        self._video_path = self.session_dir / "recording.mp4"
        self._mic_path = self.session_dir / "mic.wav"
        self._ai_path = self.session_dir / "ai.wav"
        self._final_path = self.session_dir / "final.mp4"
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._mic_wav: Optional[wave.Wave_write] = None
        self._ai_wav: Optional[wave.Wave_write] = None
        self._video_lock = threading.Lock()
        self._audio_lock = threading.Lock()
        self._frames_written = 0
        self._mic_bytes_written = 0
        self._ai_bytes_written = 0
        # Wall-clock anchor for sync. ai.wav gets silence padding between
        # AI utterances so its timeline matches real elapsed time. Video at
        # close gets re-fps'd via ffmpeg so 707 frames over 76 sec → 9.3 fps
        # (instead of 10 fps which made it play 7% fast).
        self._recording_start: Optional[float] = None
        self._video_first_frame_time: Optional[float] = None
        self._video_last_frame_time: Optional[float] = None
        if record_video:
            try:
                self._mic_wav = wave.open(str(self._mic_path), "wb")
                self._mic_wav.setnchannels(CHANNELS)
                self._mic_wav.setsampwidth(2)  # int16
                self._mic_wav.setframerate(SAMPLE_RATE)
                self._ai_wav = wave.open(str(self._ai_path), "wb")
                self._ai_wav.setnchannels(CHANNELS)
                self._ai_wav.setsampwidth(2)
                self._ai_wav.setframerate(SAMPLE_RATE)
            except Exception:
                logger.exception("audio wav open failed — recording disabled")
                self._mic_wav = None
                self._ai_wav = None

        from ultralytics import YOLO
        self._model = YOLO("yolov8n-pose.pt")
        try:
            self._model.to("cpu")
        except Exception:
            pass

    # --- vision pipeline (called from webcam thread) ---

    def yolo_keypoints(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        results = self._model(frame_rgb, verbose=False, imgsz=320)
        if not results or results[0].keypoints is None:
            return None
        data = results[0].keypoints.data
        if hasattr(data, "cpu"):
            data = data.cpu().numpy()
        if len(data) == 0:
            return None
        return data[0]

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        arr_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        kpts = self.yolo_keypoints(arr_rgb)
        self._yolo_calls += 1
        if kpts is not None:
            self._yolo_detections += 1

        active_angle: Optional[float] = None
        if self.state.current_pose is not None:
            self._frames_received += 1
            if kpts is not None:
                triple = ANGLE_TRIPLES.get(self.state.current_pose)
                if triple:
                    angle = _compute_angle(kpts, triple)
                    if angle is not None:
                        self._angles_computed += 1
                        self._latest_angle = angle
                        active_angle = angle
                        self._buffer.append((time.time(), angle, _resize_for_buffer(frame_bgr)))

        annotated_rgb = self._draw_overlay(arr_rgb.copy(), kpts, active_angle)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        # Append to recording (lazy-init on first frame).
        if self.record_video:
            self._write_video_frame(annotated_bgr)

        # Drop the latest annotated frame to disk so the Streamlit viewer can
        # show it as a quasi-live preview (st.image polls this file). Throttled
        # to ~5 fps to limit disk I/O.
        try:
            now = time.time()
            if not hasattr(self, "_last_preview_time"):
                self._last_preview_time = 0.0
            if now - self._last_preview_time >= 0.2:
                cv2.imwrite(str(self.session_dir / "latest_frame.jpg"), annotated_bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 70])
                self._last_preview_time = now
        except Exception:
            pass

        return annotated_bgr

    # ------------------------------------------------------------------
    # Video recording — annotated frames → MP4 in session dir
    # ------------------------------------------------------------------

    def _write_video_frame(self, frame_bgr: np.ndarray) -> None:
        now = time.time()
        with self._video_lock:
            if self._video_writer is None:
                h, w = frame_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(self._video_path),
                    fourcc,
                    float(TARGET_FPS),
                    (w, h),
                )
                if not writer.isOpened():
                    logger.warning(
                        "could not open video writer for %s — recording disabled",
                        self._video_path,
                    )
                    self.record_video = False
                    return
                self._video_writer = writer
                self._video_first_frame_time = now
                if self._recording_start is None:
                    self._recording_start = now
                logger.info("📼 recording → %s (%dx%d @ %d fps target)", self._video_path, w, h, TARGET_FPS)
            self._video_last_frame_time = now
            try:
                self._video_writer.write(frame_bgr)
                self._frames_written += 1
            except Exception:
                logger.exception("video write failed")

    def write_mic_chunk(self, chunk: bytes) -> None:
        if self._mic_wav is None:
            return
        with self._audio_lock:
            self._anchor_start_locked()
            self._pad_silence_locked(self._mic_wav, "mic")
            try:
                self._mic_wav.writeframes(chunk)
                self._mic_bytes_written += len(chunk)
            except Exception:
                logger.exception("mic.wav write failed")

    def write_ai_chunk(self, chunk: bytes) -> None:
        if self._ai_wav is None:
            return
        with self._audio_lock:
            self._anchor_start_locked()
            self._pad_silence_locked(self._ai_wav, "ai")
            try:
                self._ai_wav.writeframes(chunk)
                self._ai_bytes_written += len(chunk)
            except Exception:
                logger.exception("ai.wav write failed")

    def _anchor_start_locked(self) -> None:
        """Set the wall-clock anchor on first audio/video write so all 3 files
        align to the same t=0. Caller must hold _audio_lock or _video_lock."""
        if self._recording_start is None:
            self._recording_start = time.time()

    def _pad_silence_locked(self, wav: wave.Wave_write, which: str) -> None:
        """Insert silence into the wav so its current end matches wall-clock.

        Without this, ai.wav would have AI utterances back-to-back with no
        gaps between them, even though in real time the AI was silent for
        seconds. The result: audio plays out of sync with the video.

        No per-call cap — if the AI is silent for the whole session we may
        need to pad multiple minutes at close. Write in 30-sec chunks so a
        single huge buffer doesn't strain memory.
        """
        if self._recording_start is None:
            return
        elapsed = time.time() - self._recording_start
        target_bytes = int(elapsed * SAMPLE_RATE * 2)  # int16 = 2 bytes/sample
        target_bytes -= target_bytes % 2  # align to sample boundary
        current = self._mic_bytes_written if which == "mic" else self._ai_bytes_written
        gap = target_bytes - current
        if gap <= 0:
            return
        chunk_size = SAMPLE_RATE * 2 * 30  # 30 sec chunks
        silence_chunk = b"\x00" * chunk_size
        try:
            remaining = gap
            while remaining > 0:
                write = min(remaining, chunk_size)
                wav.writeframes(silence_chunk[:write])
                if which == "mic":
                    self._mic_bytes_written += write
                else:
                    self._ai_bytes_written += write
                remaining -= write
        except Exception:
            logger.exception("%s.wav silence pad failed", which)

    def close_recording(self) -> None:
        """Finalize all 3 raw files, then merge to final.mp4. Safe to call twice."""
        with self._video_lock:
            if self._video_writer is not None:
                try:
                    self._video_writer.release()
                except Exception:
                    pass
                logger.info(
                    "📼 recording saved: %s (%d frames, ~%.1f sec)",
                    self._video_path,
                    self._frames_written,
                    self._frames_written / TARGET_FPS,
                )
                self._video_writer = None
        with self._audio_lock:
            # Final tail-pad so both wav files reach the same wall-clock end —
            # otherwise ai.wav cuts off after the last AI utterance and the
            # merged video has the AI voice ending before the video ends.
            if self._recording_start is not None:
                if self._mic_wav is not None:
                    self._pad_silence_locked(self._mic_wav, "mic")
                if self._ai_wav is not None:
                    self._pad_silence_locked(self._ai_wav, "ai")
            for wav, name, n in (
                (self._mic_wav, "mic.wav", self._mic_bytes_written),
                (self._ai_wav, "ai.wav", self._ai_bytes_written),
            ):
                if wav is not None:
                    try:
                        wav.close()
                        secs = n / (SAMPLE_RATE * 2)  # int16 = 2 bytes/sample
                        logger.info("🎙️  %s saved (~%.1f sec audio)", name, secs)
                    except Exception:
                        pass
            self._mic_wav = None
            self._ai_wav = None
        # Now ffmpeg-merge the 3 files into final.mp4 (video + mixed audio).
        self._merge_to_final_mp4()

    def _merge_to_final_mp4(self) -> None:
        if not (self._video_path.exists() and self._mic_path.exists() and self._ai_path.exists()):
            logger.info("🎬 skipping merge — one or more raw files missing")
            return
        if self._frames_written == 0 or self._mic_bytes_written == 0:
            logger.info("🎬 skipping merge — no frames or audio recorded")
            return
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            logger.warning(
                "🎬 imageio-ffmpeg not installed — leaving raw files. "
                "Install with: uv pip install imageio-ffmpeg"
            )
            return

        # Compute the ACTUAL video fps from the wall-clock duration of the
        # capture so the video plays back at the right speed. cv2.VideoWriter
        # used the declared TARGET_FPS but real frame intervals were typically
        # longer (YOLO inference jitter). Without this fix, video plays ~7%
        # faster than the audio.
        actual_fps = float(TARGET_FPS)
        if (
            self._video_first_frame_time is not None
            and self._video_last_frame_time is not None
            and self._frames_written > 1
        ):
            duration = self._video_last_frame_time - self._video_first_frame_time
            if duration > 0:
                actual_fps = (self._frames_written - 1) / duration
        actual_fps = max(1.0, min(actual_fps, 60.0))

        # Re-interpret the input video's frame rate via -r before -i, then
        # re-encode the video stream so the timestamps actually update.
        cmd = [
            ffmpeg_exe, "-y",
            "-r", f"{actual_fps:.3f}",
            "-i", str(self._video_path),
            "-i", str(self._mic_path),
            "-i", str(self._ai_path),
            "-filter_complex", "[1:a][2:a]amix=inputs=2:duration=longest:dropout_transition=0[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            str(self._final_path),
        ]
        logger.info("🎬 merging into %s @ %.2f fps ...", self._final_path, actual_fps)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except Exception as exc:
            logger.warning("🎬 ffmpeg failed: %s", exc)
            return
        if result.returncode == 0 and self._final_path.exists():
            size_mb = self._final_path.stat().st_size / (1024 * 1024)
            logger.info(
                "🎬 final.mp4 ready: %s (%.1f MB) — video re-fps'd to %.2f, audio padded to wall-clock",
                self._final_path, size_mb, actual_fps,
            )
        else:
            logger.warning(
                "🎬 ffmpeg exit %d — raw files kept.\nstderr tail: %s",
                result.returncode, (result.stderr or "")[-400:],
            )

    def _draw_overlay(self, arr_rgb, kpts, active_angle):
        h, w = arr_rgb.shape[:2]
        if kpts is not None:
            active_idxs: set[int] = set()
            if self.state.current_pose:
                triple = ANGLE_TRIPLES.get(self.state.current_pose, [])
                active_idxs = {KP_INDEX[n] for n in triple if n in KP_INDEX}
            for a, b in SKELETON_EDGES:
                if a < len(kpts) and b < len(kpts):
                    if float(kpts[a][2]) > 0.3 and float(kpts[b][2]) > 0.3:
                        cv2.line(
                            arr_rgb,
                            (int(kpts[a][0]), int(kpts[a][1])),
                            (int(kpts[b][0]), int(kpts[b][1])),
                            (0, 200, 255),
                            2,
                        )
            for i in range(len(kpts)):
                if float(kpts[i][2]) > 0.3:
                    cv2.circle(arr_rgb, (int(kpts[i][0]), int(kpts[i][1])), 4, (50, 230, 50), -1)
            if active_idxs and self.state.current_pose:
                triple = ANGLE_TRIPLES.get(self.state.current_pose, [])
                if len(triple) == 3 and all(n in KP_INDEX for n in triple):
                    idxs = [KP_INDEX[n] for n in triple]
                    if all(i < len(kpts) and float(kpts[i][2]) > 0.3 for i in idxs):
                        cv2.line(
                            arr_rgb,
                            (int(kpts[idxs[0]][0]), int(kpts[idxs[0]][1])),
                            (int(kpts[idxs[1]][0]), int(kpts[idxs[1]][1])),
                            (255, 220, 0), 3,
                        )
                        cv2.line(
                            arr_rgb,
                            (int(kpts[idxs[1]][0]), int(kpts[idxs[1]][1])),
                            (int(kpts[idxs[2]][0]), int(kpts[idxs[2]][1])),
                            (255, 220, 0), 3,
                        )
                for i in active_idxs:
                    if i < len(kpts) and float(kpts[i][2]) > 0.3:
                        cv2.circle(
                            arr_rgb,
                            (int(kpts[i][0]), int(kpts[i][1])),
                            7, (255, 50, 50), -1,
                        )

        cv2.rectangle(arr_rgb, (0, 0), (w, 36), (0, 0, 0), -1)
        if self.state.current_pose:
            angle_text = f"{active_angle:.0f}°" if active_angle is not None else "..."
            label = f"{self.state.current_pose}  {angle_text}"
        else:
            label = "ERGO AI local — waiting for pose"
        cv2.putText(arr_rgb, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return arr_rgb

    # --- peak retrieval ---

    def peak_in_buffer(self):
        if not self._buffer:
            return None
        entries = list(self._buffer)
        if len(entries) < 3:
            best = max(entries, key=lambda e: e[1])
            return best[1], best[0], best[2]
        smoothed = []
        for i in range(1, len(entries) - 1):
            window = sorted(entries[i - 1 : i + 2], key=lambda e: e[1])
            smoothed.append((entries[i][0], window[1][1], entries[i][2]))
        best = max(smoothed, key=lambda e: e[1])
        return best[1], best[0], best[2]

    # --- tool implementations ---

    def record_answer(self, field, value):
        if field == "pain_location":
            if isinstance(value, str):
                value = [value]
            self.state.pain_location = [str(x).strip().lower() for x in value]
            self._save()
            return {"updated": "pain_location", "value": list(self.state.pain_location)}
        return {"warning": f"field {field} not used"}

    def select_pose_sequence(self, pain_location):
        locs = [str(x).strip().lower() for x in (pain_location or self.state.pain_location)]
        seq = compute_sequence(locs)
        self.state.pain_location = locs
        self.state.pose_queue = seq
        self._save()
        return {"pose_queue": seq, "n_poses": len(seq)}

    def check_red_flags(self, red_flags=None):
        flags = RedFlags(**(red_flags or {}))
        proxy = AssessmentState(red_flags=flags)
        rf = run_red_flag_check(proxy)
        self.state.red_flag = rf
        self._save()
        return rf.model_dump()

    def start_pose(self, body_part):
        if body_part not in ANGLE_TRIPLES:
            return {"error": f"unsupported body_part: {body_part}"}
        self.state.current_pose = body_part
        self.state.current_pain_score = None
        self._buffer.clear()
        self._latest_angle = None
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

    def get_current_angle(self):
        if not self.state.current_pose:
            return {"error": "no current pose — call start_pose first"}
        normal_max = NORMAL_MAX.get(self.state.current_pose, 180.0)
        if self._latest_angle is None:
            return {
                "angle": None,
                "person_visible": self._yolo_detections > 0,
                "frames_received": self._frames_received,
                "hint": "still warming up",
            }
        percent = round((self._latest_angle / normal_max) * 100, 1) if normal_max else 0.0
        passed = percent >= PASS_THRESHOLD_PERCENT
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

    def capture_peak_now(self):
        if not self.state.current_pose:
            return {"error": "no current pose"}
        peak = self.peak_in_buffer()
        if peak is None:
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
                hint = "person detected but joint keypoints had low visibility"
            else:
                hint = "buffer was cleared too quickly"
            return {"error": "no valid angle in buffer", "diagnostics": diag, "hint": hint}
        max_angle, peak_ts, frame_bgr = peak
        body_part = self.state.current_pose
        normal_max = NORMAL_MAX.get(body_part, 180.0)
        percent = round((max_angle / normal_max) * 100, 1) if normal_max else 0.0
        passed = percent >= PASS_THRESHOLD_PERCENT
        rel_path = f"peaks/{body_part}_{int(round(max_angle))}deg.jpg"
        cv2.imwrite(str(self.session_dir / rel_path), frame_bgr)
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
        out["verdict"] = verdict
        return out

    def record_pain_score(self, score, note=""):
        try:
            score_int = max(0, min(10, int(round(float(score)))))
        except (TypeError, ValueError):
            return {"error": f"could not parse score: {score!r}"}
        target = next((p for p in reversed(self.state.results) if p.pain_score is None), None)
        if target is None:
            return {"error": "no recent pose"}
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

    def end_session(self):
        from rom_analyzer import analyze
        self.state.ended_at = time.time()
        self.state.current_pose = None
        self._save()
        try:
            report = analyze(self.state)
        except Exception as exc:
            return {"error": f"analyzer failed: {exc}"}
        report.session_dir = str(self.session_dir)
        report.generated_at = time.time()
        (self.session_dir / "final_report.json").write_text(
            json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report.model_dump()

    def _save(self):
        try:
            (self.session_dir / "rom_session.json").write_text(
                json.dumps(self.state.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Realtime API + audio + webcam orchestration
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {"type": "function", "name": "record_answer",
     "description": "Persist one fact. field='pain_location' takes a list.",
     "parameters": {"type": "object", "properties": {"field": {"type": "string"}, "value": {}}, "required": ["field", "value"]}},
    {"type": "function", "name": "select_pose_sequence",
     "description": "Build the ROM pose queue from pain locations.",
     "parameters": {"type": "object", "properties": {"pain_location": {"type": "array", "items": {"type": "string"}}}, "required": ["pain_location"]}},
    {"type": "function", "name": "check_red_flags",
     "description": "Run red-flag screen, e.g. {numbness: true}.",
     "parameters": {"type": "object", "properties": {"red_flags": {"type": "object"}}}},
    {"type": "function", "name": "start_pose",
     "description": "Begin a specific ROM pose. body_part: shoulder_flexion / shoulder_abduction / elbow_flexion / knee_flexion / trunk_lateral_left / trunk_lateral_right.",
     "parameters": {"type": "object", "properties": {"body_part": {"type": "string"}}, "required": ["body_part"]}},
    {"type": "function", "name": "get_current_angle",
     "description": "Read live joint angle while user is mid-motion.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "capture_peak_now",
     "description": "Save peak frame for current pose. Returns max_angle, percent_of_normal, passed, verdict.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "record_pain_score",
     "description": "Attach 0-10 pain score to most recent pose.",
     "parameters": {"type": "object", "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 10}, "note": {"type": "string"}}, "required": ["score"]}},
    {"type": "function", "name": "end_session",
     "description": "Finalize, run analyzer, write final_report.json.",
     "parameters": {"type": "object", "properties": {}}},
]


class LocalAgent:
    def __init__(
        self,
        model: str,
        voice: str,
        camera_index: int = 0,
        no_window: bool = False,
        record_video: bool = True,
    ):
        self.rom = LocalRomState(record_video=record_video)
        self.model = model
        self.voice = voice
        self.camera_index = camera_index
        self.no_window = no_window
        self.audio_in_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self._stop = threading.Event()
        # Echo guard: when AI is speaking, drop mic chunks so the AI's own
        # speaker output doesn't loop back through the mic and hallucinate
        # transcripts (Whisper template hallucinations like "Thank you for
        # watching" / "โปรดไลค์ครับ" come from this).
        self._ai_speaking = False
        self._ai_speaking_until = 0.0  # add a small tail after audio.done

    # --- webcam thread ---
    def webcam_thread(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error("could not open webcam")
            return
        logger.info("📹 webcam loop started")
        try:
            target_dt = 1.0 / TARGET_FPS
            last_t = 0.0
            while not self._stop.is_set():
                now = time.monotonic()
                if now - last_t < target_dt:
                    time.sleep(max(0.0, target_dt - (now - last_t)))
                    continue
                last_t = now
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                annotated = self.rom.process_frame(frame)
                if not self.no_window:
                    cv2.imshow("ERGO AI — local", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self._stop.set()
        finally:
            cap.release()
            if not self.no_window:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass

    # --- realtime audio + LLM loop ---
    async def run(self):
        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("[local] OPENAI_API_KEY missing — set in .env")
            sys.exit(1)
        client = AsyncOpenAI(api_key=api_key)

        loop = asyncio.get_running_loop()
        speaker = sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16", blocksize=0)
        speaker.start()

        def mic_callback(indata, frames, time_info, status):
            if self._stop.is_set():
                raise sd.CallbackStop
            try:
                asyncio.run_coroutine_threadsafe(self.audio_in_queue.put(bytes(indata)), loop)
            except RuntimeError:
                pass

        mic = sd.RawInputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
            blocksize=CHUNK_FRAMES, callback=mic_callback,
        )
        mic.start()

        cam_thread = threading.Thread(target=self.webcam_thread, daemon=True)
        cam_thread.start()

        try:
            print(f"[local] connecting to {self.model} (voice={self.voice})")
            print(f"[local] session dir: {self.rom.session_dir}")
            async with client.beta.realtime.connect(model=self.model) as conn:
                await conn.session.update(session={
                    "instructions": ROM_SYSTEM_PROMPT,
                    "voice": self.voice,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": {
                        "type": "server_vad",
                        # Higher threshold = less trigger-happy on background
                        # noise / faint speaker echo. Increase silence so we
                        # don't pick up between-word pauses as turn ends.
                        "threshold": 0.65,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 800,
                    },
                    "tools": TOOL_SCHEMAS,
                    "tool_choice": "auto",
                })
                await conn.response.create()

                send_task = asyncio.create_task(self._send_audio_loop(conn))
                try:
                    await self._receive_loop(conn, speaker)
                finally:
                    send_task.cancel()
                    try:
                        await send_task
                    except (asyncio.CancelledError, Exception):
                        pass
        finally:
            self._stop.set()
            try:
                mic.stop(); mic.close()
            except Exception:
                pass
            try:
                speaker.stop(); speaker.close()
            except Exception:
                pass
            # Flush + close the MP4 file so it's playable.
            self.rom.close_recording()
            print(f"\n[local] saved to {self.rom.session_dir}")

    async def _send_audio_loop(self, conn):
        while not self._stop.is_set():
            try:
                chunk = await asyncio.wait_for(self.audio_in_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if not chunk:
                continue
            # Always record the mic to mic.wav — even chunks dropped by the
            # echo guard so the playback shows what actually went into the mic.
            self.rom.write_mic_chunk(chunk)
            # Echo guard: drop mic chunks while AI is speaking (and 300ms after).
            # Without AEC, the speaker → mic loop would feed the AI's own voice
            # back as fake user transcripts.
            if self._ai_speaking or time.time() < self._ai_speaking_until:
                continue
            b64 = base64.b64encode(chunk).decode("ascii")
            try:
                await conn.input_audio_buffer.append(audio=b64)
            except Exception as exc:
                print(f"[local] send error: {exc}")
                break

    async def _receive_loop(self, conn, speaker):
        async for event in conn:
            et = getattr(event, "type", "")
            if et == "response.audio.delta":
                self._ai_speaking = True
                audio_b64 = getattr(event, "delta", "") or ""
                if audio_b64:
                    try:
                        pcm = base64.b64decode(audio_b64)
                        speaker.write(pcm)
                        # Mirror AI audio to ai.wav so we can mux it into final.mp4
                        self.rom.write_ai_chunk(pcm)
                    except Exception as exc:
                        print(f"[speaker] {exc}")
            elif et == "response.audio.done":
                # Hold the echo guard for 300ms after AI finishes — the
                # speaker buffer keeps emitting briefly after the API
                # signals done.
                self._ai_speaking = False
                self._ai_speaking_until = time.time() + 0.3
            elif et == "response.audio_transcript.delta":
                sys.stdout.write(event.delta); sys.stdout.flush()
            elif et == "response.audio_transcript.done":
                print()
                self._ai_speaking = False
                self._ai_speaking_until = time.time() + 0.3
            elif et == "conversation.item.input_audio_transcription.completed":
                tx = getattr(event, "transcript", "")
                # Filter common Whisper hallucinations from silence
                if tx:
                    tlow = tx.strip().lower()
                    bogus = (
                        "thank you for watching" in tlow
                        or "subscribe" in tlow
                        or "โปรดไลค์" in tx
                        or "กดติดตาม" in tx
                        or len(tlow) < 4
                    )
                    if bogus:
                        print(f"\n[you] [filtered hallucination: {tx!r}]")
                    else:
                        print(f"\n[you] {tx}")
            elif et == "response.function_call_arguments.done":
                name = getattr(event, "name", "")
                call_id = getattr(event, "call_id", "")
                raw_args = getattr(event, "arguments", "") or "{}"
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args = {}
                print(f"\n[tool] {name}({json.dumps(args, ensure_ascii=False)[:120]})")
                result = self._dispatch_tool(name, args)
                try:
                    await conn.conversation.item.create(item={
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result, ensure_ascii=False, default=str),
                    })
                    await conn.response.create()
                except Exception as exc:
                    print(f"[tool-reply error] {exc}")
            elif et == "error":
                err = getattr(event, "error", None)
                print(f"\n[ERROR] {err}")
            if self._stop.is_set():
                break

    def _dispatch_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        rom = self.rom
        if name == "record_answer":
            return rom.record_answer(args.get("field", ""), args.get("value"))
        if name == "select_pose_sequence":
            return rom.select_pose_sequence(args.get("pain_location") or [])
        if name == "check_red_flags":
            return rom.check_red_flags(args.get("red_flags") or {})
        if name == "start_pose":
            return rom.start_pose(str(args.get("body_part") or ""))
        if name == "get_current_angle":
            return rom.get_current_angle()
        if name == "capture_peak_now":
            return rom.capture_peak_now()
        if name == "record_pain_score":
            return rom.record_pain_score(args.get("score", 0), str(args.get("note") or ""))
        if name == "end_session":
            self._stop.set()
            return rom.end_session()
        return {"error": f"unknown tool: {name}"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.getenv("OPENAI_REALTIME_MODEL", DEFAULT_MODEL))
    p.add_argument("--voice", default=os.getenv("OPENAI_REALTIME_VOICE", DEFAULT_VOICE))
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--no-window", action="store_true")
    p.add_argument(
        "--list-devices", action="store_true",
        help="List all audio devices and exit (use to find Bluetooth Hands-Free indices)",
    )
    p.add_argument("--mic-index", type=int, default=None, help="Audio input device index")
    p.add_argument("--speaker-index", type=int, default=None, help="Audio output device index")
    p.add_argument(
        "--no-recording", action="store_true",
        help="Disable saving the annotated video to recording.mp4",
    )
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        print(f"\ndefault input : {sd.default.device[0]}")
        print(f"default output: {sd.default.device[1]}")
        print(
            "\nFor Bluetooth headphones, pick the 'Hands-Free' device"
            " (Windows: it has a 'Headset' name and supports both in+out)."
        )
        sys.exit(0)

    # Pin chosen audio devices BEFORE creating streams.
    if args.mic_index is not None or args.speaker_index is not None:
        mic = args.mic_index if args.mic_index is not None else sd.default.device[0]
        spk = args.speaker_index if args.speaker_index is not None else sd.default.device[1]
        sd.default.device = (mic, spk)
        try:
            mic_name = sd.query_devices(mic)["name"]
            spk_name = sd.query_devices(spk)["name"]
            print(f"[audio] mic     #{mic} → {mic_name}")
            print(f"[audio] speaker #{spk} → {spk_name}")
        except Exception:
            pass
    else:
        try:
            mi, so = sd.default.device
            print(f"[audio] mic     #{mi} → {sd.query_devices(mi)['name']}  (default — use --list-devices to verify)")
            print(f"[audio] speaker #{so} → {sd.query_devices(so)['name']}  (default — use --list-devices to verify)")
        except Exception:
            pass

    load_dotenv()
    agent = LocalAgent(
        model=args.model, voice=args.voice,
        camera_index=args.camera_index, no_window=args.no_window,
        record_video=not args.no_recording,
    )
    print("[local] starting. mic+speaker live + cv2 window. press Ctrl+C or 'q' to stop.")
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
