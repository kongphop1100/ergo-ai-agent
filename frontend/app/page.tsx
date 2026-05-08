"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function startSession() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/session", { method: "POST" });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      const { call_id, call_type, token, api_key, user_id } = await res.json();
      const params = new URLSearchParams({ call_type, token, api_key, user_id });
      router.push(`/session/${call_id}?${params.toString()}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setLoading(false);
    }
  }

  return (
    <main className="flex flex-1 items-center justify-center px-6 py-12">
      <div className="w-full max-w-xl space-y-8">
        <div className="space-y-3">
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
            ERGO AI
          </h1>
          <p className="text-neutral-300">
            ทดสอบช่วงการเคลื่อนไหว (Range of Motion) ด้วยเสียงและกล้อง —
            ผู้ช่วย AI จะแนะนำคุณทีละท่าและประเมินผลให้
          </p>
          <p className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-200">
            ⚠️ การประเมินนี้เป็นการ <b>คัดกรองเบื้องต้น</b> ไม่ใช่การวินิจฉัยทางการแพทย์
          </p>
        </div>

        <button
          onClick={startSession}
          disabled={loading}
          className="w-full rounded-xl bg-rose-500 px-6 py-4 text-lg font-semibold text-white shadow-lg shadow-rose-500/20 transition hover:bg-rose-400 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? "กำลังเชื่อมต่อ..." : "📞  เริ่ม Session"}
        </button>

        {error && (
          <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">
            ❌ {error}
          </div>
        )}

        <p className="text-xs text-neutral-500">
          ระบบจะขออนุญาตเข้าถึงกล้องและไมโครโฟนของคุณ — เปิดใช้งานเฉพาะระหว่าง session เท่านั้น
        </p>
      </div>
    </main>
  );
}
