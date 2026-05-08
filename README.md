# ERGO AI — Voice-Guided ROM Session

Tele-rehab Range-of-Motion screening agent built on **Vision-Agents** (GetStream). The AI physiotherapist greets in Thai, asks where it hurts, picks a pose sequence, and walks the user through each motion: "ลองยกแขนขวาขึ้นช้าๆ … ไหวไหมครับ … OK ทำได้ 142 องศา 79% ของช่วงปกติ"

For every pose the agent listens for the user's "พอ / ไม่ไหว / เจ็บแล้ว" verbal stop signal, then captures the peak frame from a 5-second YOLO angle buffer. After all poses, a multimodal LLM analyses the captured photos + numbers + pain scores into a Thai-language report.

> ⚠️ **คำเตือน:** ระบบนี้เป็นการ **คัดกรองเบื้องต้น** ไม่ใช่การวินิจฉัยทางการแพทย์

## Architecture

```
[browser] ─WebRTC─► Stream Edge ─► Vision-Agents Agent (this repo)
                                       │
                                       ├ openai.Realtime (gpt-realtime-2025-08-28) — voice + tools
                                       └ RomSessionProcessor — YOLO + angle buffer + 7 ROM tools
                                                │
                                       sessions/<ts>/
                                          rom_session.json
                                          peaks/<body_part>_<deg>deg.jpg
                                          final_report.json   ← rom_analyzer (vision LLM via OpenRouter)
```

## Setup

1. **Install** (Python 3.11+ recommended)
   ```powershell
   uv sync
   ```

2. **Sign up at https://getstream.io** — create a "Video & Audio" app, copy the API key + secret.

3. **Create `.env`** (copy from `.env.example`):
   ```ini
   STREAM_API_KEY=...
   STREAM_API_SECRET=...
   OPENAI_API_KEY=sk-proj-...        # for openai.Realtime
   OPENROUTER_API_KEY=sk-or-v1-...   # for the post-session analyzer
   ```

4. **Run**:
   ```powershell
   uv run agent.py
   ```
   The Vision-Agents CLI will print a join URL. Open it (or use the matching call_id from a Next.js frontend) — the agent joins the call and starts the session.

## ROM tools the LLM can call

| Tool | Purpose |
|---|---|
| `record_answer(field, value)` | persist pain_location |
| `select_pose_sequence(pain_location)` | build the pose queue |
| `check_red_flags(red_flags)` | numbness / weakness / etc. → STOP if any |
| `start_pose(body_part)` | tell tracker which joint to focus on |
| `capture_peak_now()` | save peak frame for current pose |
| `record_pain_score(score, note)` | attach 0-10 to the most recent pose |
| `end_session()` | run analyzer, write `final_report.json` |

## Pose set (MVP, COCO-17 keypoints only)

- `shoulder_flexion` — RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW (normal max 180°)
- `shoulder_abduction` — same triple, different camera angle (normal 180°)
- `elbow_flexion` — RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST (normal 150°)
- `knee_flexion` — RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE (normal 135°)
- `trunk_lateral_left/right` — placeholder triples (normal 35°)

Pain location → pose sequence:
- `shoulder` → flexion + abduction + elbow_flexion
- `back` → shoulder_flexion + trunk_lateral_left/right
- `knee` → knee_flexion
- `wrist` → elbow_flexion (wrist needs hand keypoints — Phase 2)
- `neck` / `eye_head` → shoulder_flexion (neck angles need face landmarks — Phase 2)

## Files

| File | Purpose |
|---|---|
| `agent.py` | main entry — Vision-Agents `Agent` + LLM + tool registration |
| `rom_processor.py` | `RomSessionProcessor` — `VideoProcessor` that runs YOLO + buffer + state |
| `rom_session.py` | data lookups (POSE_SEQUENCE_BY_LOCATION, ANGLE_TRIPLES, etc.) |
| `rom_analyzer.py` | multimodal LLM final report (uses OpenRouter) |
| `prompts.py` | physio-guide system prompt |
| `schemas.py` | Pydantic models (PoseResult, RomSessionState, RomFinalReport) |
| `tools.py` | pure-Python red-flag / scoring / 7-day plan helpers |
| `llm.py` | OpenRouter wrapper used by analyzer |
| `data/aaos_normal_rom.json` | clinical data tables (AAOS normal ROM, triples, instructions) |

## Frontend (`frontend/`)

Next.js 14 App Router, deployed on Vercel.

```
frontend/
├── app/
│   ├── page.tsx                 — landing + "เริ่ม Session" button
│   ├── api/session/route.ts     — server-only: mints Stream JWT, asks Python
│   │                              agent (Render) to spawn an agent on the call
│   └── session/[callId]/page.tsx — joins Stream call via @stream-io/video-react-sdk
└── .env.example                 — NEXT_PUBLIC_STREAM_API_KEY, STREAM_API_SECRET,
                                   AGENT_BACKEND_URL
```

**Local dev:**
```powershell
# Terminal 1 — Python agent
uv run agent.py serve --host 0.0.0.0 --port 8000

# Terminal 2 — Next.js frontend
cd frontend
cp .env.example .env.local   # fill in keys
npm install
npm run dev
# → http://localhost:3000
```

The browser, the frontend's API route, and the Python agent all coordinate via
**call_id**: the API route generates a fresh UUID, asks the Python agent to
join `POST /calls/{call_id}/sessions`, and returns the call_id + a freshly
minted user JWT to the browser. The browser then joins the same call via
`@stream-io/video-react-sdk`.

## Deploy

- **Python agent → Render.com** (Docker). `render.yaml` + `Dockerfile` at repo
  root. Set env: `STREAM_API_KEY`, `STREAM_API_SECRET`, `OPENAI_API_KEY`,
  `OPENROUTER_API_KEY`, `CORS_ORIGINS`.
- **Frontend → Vercel.** Import repo, set "Root Directory" to `frontend/`. Set
  env: `NEXT_PUBLIC_STREAM_API_KEY`, `STREAM_API_SECRET`, `AGENT_BACKEND_URL`
  (= the Render URL).
- After both are up, set Render's `CORS_ORIGINS` to your Vercel domain.

## Verification (smoke test, no LLM key needed)

```powershell
uv run python -c "
import time, numpy as np
from rom_processor import RomSessionProcessor
rom = RomSessionProcessor()
rom.select_pose_sequence(['shoulder'])
rom.check_red_flags({'numbness': False})
for bp, peaks, pain in [
    ('shoulder_flexion',  [110, 130, 142, 142, 141, 130], 5),
    ('shoulder_abduction',[110, 138, 138, 137, 130], 4),
    ('elbow_flexion',     [100, 130, 148, 147, 130], 2),
]:
    rom.start_pose(bp)
    for a in peaks:
        rom._buffer.append((time.time(), float(a), np.zeros((300,400,3), dtype=np.uint8)))
    print(bp, rom.capture_peak_now()['max_angle'])
    rom.record_pain_score(pain)
report = rom.end_session()
print('risk:', report['overall_risk_level'])
"
```

Expected: 3 angles captured, fallback report (no key) prints `risk: moderate`.

## Origin

Architecture cloned from [GetStream/Vision-Agents](https://github.com/GetStream/Vision-Agents). Reused tools/scoring/prompts from the earlier Streamlit MVP at `minihack4/streamlit_app/`.
