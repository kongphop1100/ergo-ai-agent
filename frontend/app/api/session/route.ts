import { NextResponse } from "next/server";
import { StreamClient } from "@stream-io/node-sdk";
import { randomUUID } from "node:crypto";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const STREAM_API_KEY = process.env.NEXT_PUBLIC_STREAM_API_KEY;
const STREAM_API_SECRET = process.env.STREAM_API_SECRET;
const AGENT_BACKEND_URL = process.env.AGENT_BACKEND_URL;
const CALL_TYPE = process.env.STREAM_CALL_TYPE ?? "default";

export async function POST() {
  if (!STREAM_API_KEY || !STREAM_API_SECRET) {
    return NextResponse.json(
      { error: "Server missing NEXT_PUBLIC_STREAM_API_KEY or STREAM_API_SECRET" },
      { status: 500 }
    );
  }
  if (!AGENT_BACKEND_URL) {
    return NextResponse.json(
      { error: "Server missing AGENT_BACKEND_URL" },
      { status: 500 }
    );
  }

  const callId = randomUUID();
  const userId = `ergo-user-${randomUUID().slice(0, 8)}`;

  // Mint a Stream user JWT (server-side only — secret never leaves Vercel)
  const stream = new StreamClient(STREAM_API_KEY, STREAM_API_SECRET);
  const token = stream.generateUserToken({
    user_id: userId,
    validity_in_seconds: 60 * 60, // 1 hour
  });

  // Tell the Python agent to spawn an agent and join this call
  const agentRes = await fetch(
    `${AGENT_BACKEND_URL.replace(/\/$/, "")}/calls/${encodeURIComponent(callId)}/sessions`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ call_type: CALL_TYPE }),
    }
  );

  if (!agentRes.ok) {
    const detail = await agentRes.text().catch(() => "");
    return NextResponse.json(
      { error: `Agent backend rejected session: ${agentRes.status} ${detail}` },
      { status: 502 }
    );
  }

  return NextResponse.json({
    call_id: callId,
    call_type: CALL_TYPE,
    token,
    api_key: STREAM_API_KEY,
    user_id: userId,
  });
}
