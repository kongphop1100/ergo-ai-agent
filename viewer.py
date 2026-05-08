"""ERGO AI viewer — standalone video-call style UI for local_agent.

Two states:
  • PRE-CALL: device picker + big "📞 Start call" button + last session preview
  • IN-CALL: full-width live frame (skeleton overlay) + status bar +
             "End call" button. Auto-refreshes every ~0.5s so the frame
             behaves like a low-fps video stream.

Run from the project dir:
    cd ergo-ai-agent
    uv run streamlit run viewer.py

NOTE: this is the viewer for the LOCAL agent (Path A). It runs on your own
machine because it spawns local_agent.py which needs your webcam + mic +
speaker. There's nothing to deploy to a cloud — share the repo and have
the user run `uv run streamlit run viewer.py` locally.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

HERE = Path(__file__).parent
SESSIONS = HERE / "sessions"
load_dotenv(HERE / ".env")

st.set_page_config(page_title="ERGO AI — ROM Call", page_icon="📞", layout="wide")


# ---------------------------------------------------------------------------
# Session state + helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "proc" not in st.session_state:
        st.session_state.proc = None
    if "active_dir" not in st.session_state:
        st.session_state.active_dir = None
    if "launch_time" not in st.session_state:
        st.session_state.launch_time = 0.0
    if "log_path" not in st.session_state:
        st.session_state.log_path = None


def _proc_running() -> bool:
    p = st.session_state.proc
    return p is not None and p.poll() is None


def _stop_proc() -> None:
    p = st.session_state.proc
    if p is not None:
        try:
            p.terminate()
        except Exception:
            pass
    st.session_state.proc = None


def _list_audio_devices():
    try:
        import sounddevice as sd
    except Exception:
        return [], [], 0, 0
    try:
        devices = sd.query_devices()
    except Exception:
        return [], [], 0, 0
    inputs = [(i, d["name"]) for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]
    outputs = [(i, d["name"]) for i, d in enumerate(devices) if d.get("max_output_channels", 0) > 0]
    try:
        default_in, default_out = sd.default.device
    except Exception:
        default_in, default_out = -1, -1
    in_idx = next((pos for pos, (i, _) in enumerate(inputs) if i == default_in), 0)
    out_idx = next((pos for pos, (i, _) in enumerate(outputs) if i == default_out), 0)
    return inputs, outputs, in_idx, out_idx


def _show_log_tail() -> None:
    """Render the last lines of local_agent's stdout/stderr log if available."""
    lp = st.session_state.get("log_path")
    if not lp:
        return
    try:
        text = Path(lp).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    if not text.strip():
        return
    tail = "\n".join(text.splitlines()[-25:])
    with st.expander("📜 local_agent log (25 บรรทัดสุดท้าย)", expanded=True):
        st.code(tail)


def _latest_session_dir() -> Path | None:
    if not SESSIONS.exists():
        return None
    dirs = [d for d in SESSIONS.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# IN-CALL view (active session)
# ---------------------------------------------------------------------------

def render_in_call() -> None:
    # Top bar
    cols = st.columns([5, 1])
    with cols[0]:
        st.markdown("### 🎥 ERGO AI — In Call")
    with cols[1]:
        if st.button("📞 End call", type="primary", use_container_width=True):
            _stop_proc()
            st.rerun()

    # Detect a freshly-created session dir (mtime newer than launch_time)
    if st.session_state.active_dir is None:
        candidate = _latest_session_dir()
        if (
            candidate is not None
            and candidate.stat().st_mtime >= st.session_state.launch_time - 1.0
        ):
            st.session_state.active_dir = candidate

    sd_path = st.session_state.active_dir
    if sd_path is None:
        # Subprocess may have died before creating its session dir
        if not _proc_running():
            st.error("❌ local_agent ปิดตัวก่อนเปิดกล้อง")
            _show_log_tail()
            if st.button("กลับหน้าเลือกอุปกรณ์"):
                st.session_state.proc = None
                st.rerun()
            return
        st.info("⏳ กำลังเริ่ม local_agent... (Python imports + YOLO โหลดครั้งแรก ~5-10 วินาที)")
        time.sleep(0.7)
        st.rerun()
        return

    # Live video tile
    latest_frame = sd_path / "latest_frame.jpg"
    if latest_frame.exists():
        st.image(str(latest_frame), use_container_width=True)
    elif not _proc_running():
        st.error("❌ local_agent หยุดทำงานก่อนเปิดกล้องได้")
        _show_log_tail()
        if st.button("กลับหน้าเลือกอุปกรณ์"):
            st.session_state.proc = None
            st.session_state.active_dir = None
            st.rerun()
        return
    else:
        st.info("⏳ กำลังเปิดกล้อง... รอให้ YOLO โหลดโมเดล (~6 MB ครั้งแรก)")

    # Status bar
    state_file = sd_path / "rom_session.json"
    state = {}
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    cols = st.columns(4)
    cols[0].metric("Pose ตอนนี้", state.get("current_pose") or "—")
    cols[1].metric("ทำเสร็จ", str(len(state.get("results", []))))
    cols[2].metric("คิว", str(len(state.get("pose_queue", []))))
    cols[3].metric("Pain ล่าสุด", str(state.get("current_pain_score") or "—"))

    # Captures so far (compact)
    results = state.get("results", [])
    if results:
        st.markdown("**📋 Captures:**")
        line = " | ".join(
            f"{'✅' if r.get('passed') else '⚠️'} {r['body_part']} {r['max_angle']:.0f}° "
            f"({r.get('percent_of_normal', 0):.0f}%)"
            f" pain={r.get('pain_score', '?')}"
            for r in results
        )
        st.markdown(line)

    # Red-flag
    if state.get("red_flag", {}).get("has_red_flag"):
        rf = state["red_flag"]
        st.error("🚨 " + rf.get("message", "") + " — " + ", ".join(rf.get("red_flags", [])))

    # Auto-refresh — fast enough to feel like video, slow enough to not hammer streamlit
    time.sleep(0.5)
    st.rerun()


# ---------------------------------------------------------------------------
# PRE-CALL view (device picker + start)
# ---------------------------------------------------------------------------

def render_pre_call() -> None:
    st.title("📞 ERGO AI — ROM Self-Test")
    st.caption(
        "⚠️ การประเมินนี้เป็นการคัดกรองเบื้องต้น **ไม่ใช่การวินิจฉัยทางการแพทย์**"
    )

    # Quick env check
    cols = st.columns(2)
    with cols[0]:
        if os.getenv("OPENAI_API_KEY", "").strip():
            st.success("✅ OPENAI_API_KEY")
        else:
            st.error("❌ OPENAI_API_KEY ยังไม่ได้ตั้งใน `.env`")
    with cols[1]:
        if os.getenv("OPENROUTER_API_KEY", "").strip():
            st.success("✅ OPENROUTER_API_KEY (analyzer)")
        else:
            st.warning("⚠️ OPENROUTER_API_KEY ไม่มี — fallback report")

    st.markdown("---")

    # Device picker
    inputs, outputs, default_in_pos, default_out_pos = _list_audio_devices()
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**🎤 ไมโครโฟน**")
        if inputs:
            mic = st.selectbox(
                "Mic",
                inputs,
                format_func=lambda x: f"#{x[0]} — {x[1][:50]}",
                index=default_in_pos,
                label_visibility="collapsed",
                key="mic_pick",
            )
            mic_idx = mic[0]
        else:
            st.warning("ไม่พบไมโครโฟน")
            mic_idx = None
    with cols[1]:
        st.markdown("**🔊 ลำโพง / หูฟัง**")
        if outputs:
            spk = st.selectbox(
                "Speaker",
                outputs,
                format_func=lambda x: f"#{x[0]} — {x[1][:50]}",
                index=default_out_pos,
                label_visibility="collapsed",
                key="spk_pick",
            )
            spk_idx = spk[0]
        else:
            st.warning("ไม่พบลำโพง")
            spk_idx = None

    cols = st.columns(3)
    with cols[0]:
        voice = st.selectbox("🗣️ AI voice", ["marin", "alloy", "echo", "sage", "shimmer"])
    with cols[1]:
        camera_index = st.number_input("📹 Camera", min_value=0, max_value=10, value=0, step=1)
    with cols[2]:
        no_recording = st.checkbox("🚫 ไม่บันทึก", value=False)

    st.markdown("&nbsp;")

    # Big start button
    disabled = mic_idx is None or spk_idx is None or not os.getenv("OPENAI_API_KEY", "").strip()
    if st.button(
        "📞  Start call  📞",
        type="primary",
        use_container_width=True,
        disabled=disabled,
    ):
        cmd = [
            sys.executable, "-u",
            str(HERE / "local_agent.py"),
            "--mic-index", str(mic_idx),
            "--speaker-index", str(spk_idx),
            "--voice", voice,
            "--camera-index", str(camera_index),
            "--no-window",  # cv2 window not needed — Streamlit shows the frame
        ]
        if no_recording:
            cmd.append("--no-recording")
        # Capture the subprocess's stdout/stderr so we can surface errors in
        # the Streamlit UI when launch fails (e.g. invalid camera index).
        log_path = HERE / "sessions" / f"_launch_{int(time.time())}.log"
        log_path.parent.mkdir(exist_ok=True)
        try:
            log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
            launch_time = time.time()
            proc = subprocess.Popen(
                cmd, cwd=str(HERE),
                stdout=log_fh, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_CONSOLE
                if hasattr(subprocess, "CREATE_NEW_CONSOLE") else 0,
            )
        except Exception as exc:
            st.error(f"launch failed: {exc}")
            return
        st.session_state.proc = proc
        st.session_state.launch_time = launch_time
        st.session_state.log_path = str(log_path)
        st.session_state.active_dir = None  # let render_in_call detect the new dir
        st.rerun()

    # Show last session if any
    last = _latest_session_dir()
    if last is not None:
        st.markdown("---")
        st.markdown("### 📜 Last session")
        report_file = last / "final_report.json"
        if report_file.exists():
            try:
                report = json.loads(report_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                report = {}
            risk = report.get("overall_risk_level", "?")
            badges = {"low": ("🟢", "#0a7d0a"), "moderate": ("🟡", "#b8860b"),
                      "high": ("🔴", "#b00020"), "?": ("⚪", "#666")}
            badge, color = badges.get(risk, badges["?"])
            st.markdown(
                f"<div style='padding:0.8rem;border-radius:8px;background:{color}1a;"
                f"border-left:6px solid {color};'>"
                f"<b>{badge} {last.name}</b> — Risk: <b>{risk.upper()}</b><br>"
                f"<small>{report.get('summary', '')}</small></div>",
                unsafe_allow_html=True,
            )
            video = last / "final.mp4"
            if not video.exists():
                video = last / "recording.mp4"
            if video.exists():
                with st.expander("🎬 ดูวีดีโอ session"):
                    st.video(str(video))
            findings = report.get("per_pose_findings", [])
            if findings:
                with st.expander("📋 ผลรายท่า"):
                    for f in findings:
                        st.markdown(
                            f"**{f['body_part']}** — {f['max_angle']:.0f}° "
                            f"({f.get('percent_of_normal', 0):.0f}%) pain={f.get('pain_score', '?')}"
                        )
                        if f.get("commentary"):
                            st.caption(f.get("commentary"))
            plan = report.get("seven_day_plan")
            if plan and plan.get("daily_plan"):
                with st.expander("🗓️ แผน 7 วัน"):
                    for day in plan["daily_plan"]:
                        st.markdown(f"**Day {day['day']}**")
                        for action in day.get("actions", []):
                            st.markdown(f"- {action}")
        else:
            st.info(f"Session ล่าสุด `{last.name}` ยังไม่จบ — ไม่มี final report")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()
    if _proc_running():
        # If user closed the cv2 / agent process from outside, fall through
        render_in_call()
    else:
        # If the process died but we still have an active_dir, the post-call
        # reload will show the final report below the device picker.
        render_pre_call()


if __name__ == "__main__":
    main()
