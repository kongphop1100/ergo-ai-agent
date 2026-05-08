"use client";

import "@stream-io/video-react-sdk/dist/css/styles.css";

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import {
  CallControls,
  SpeakerLayout,
  StreamCall,
  StreamTheme,
  StreamVideo,
  StreamVideoClient,
  type Call,
} from "@stream-io/video-react-sdk";

export default function SessionPage() {
  const router = useRouter();
  const params = useParams<{ callId: string }>();
  const search = useSearchParams();

  const callId = params.callId;
  const callType = search.get("call_type") ?? "default";
  const token = search.get("token");
  const apiKey = search.get("api_key");
  const userId = search.get("user_id");

  const missing = !callId || !token || !apiKey || !userId;

  const client = useMemo(() => {
    if (missing) return null;
    return new StreamVideoClient({
      apiKey: apiKey!,
      user: { id: userId!, name: "ERGO User" },
      token: token!,
    });
  }, [missing, apiKey, userId, token]);

  const [call, setCall] = useState<Call | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!client || !callId) return;
    const c = client.call(callType, callId);
    setCall(c);
    c.join({ create: true }).catch((err) => {
      console.error("call.join failed", err);
      setError(err instanceof Error ? err.message : String(err));
    });
    return () => {
      c.leave().catch(() => {});
    };
  }, [client, callId, callType]);

  useEffect(() => {
    return () => {
      client?.disconnectUser().catch(() => {});
    };
  }, [client]);

  if (missing) {
    return (
      <main className="flex flex-1 items-center justify-center px-6 py-12">
        <div className="max-w-md space-y-3">
          <h2 className="text-xl font-semibold">ลิงก์ไม่ถูกต้อง</h2>
          <p className="text-sm text-neutral-400">
            ขาดข้อมูลการเชื่อมต่อ — กรุณากลับไปเริ่ม session ใหม่
          </p>
          <button
            onClick={() => router.push("/")}
            className="rounded-md bg-rose-500 px-4 py-2 text-sm font-medium text-white hover:bg-rose-400"
          >
            กลับหน้าแรก
          </button>
        </div>
      </main>
    );
  }

  if (!client || !call) {
    return (
      <main className="flex flex-1 items-center justify-center">
        <p className="text-neutral-400">กำลังเชื่อมต่อ Stream...</p>
      </main>
    );
  }

  return (
    <StreamVideo client={client}>
      <StreamCall call={call}>
        <StreamTheme className="flex flex-1 flex-col">
          <header className="flex items-center justify-between px-4 py-3 text-sm text-neutral-300">
            <div>🎥 ERGO AI — In Call</div>
            <div className="text-xs text-neutral-500">call_id: {callId.slice(0, 8)}</div>
          </header>

          {error && (
            <div className="mx-4 mb-2 rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">
              ❌ {error}
            </div>
          )}

          <div className="flex-1 px-2 pb-2">
            <SpeakerLayout participantsBarPosition="bottom" />
          </div>

          <div className="border-t border-neutral-800 bg-neutral-900/50 px-4 py-3">
            <CallControls
              onLeave={() => {
                router.push("/");
              }}
            />
          </div>
        </StreamTheme>
      </StreamCall>
    </StreamVideo>
  );
}
