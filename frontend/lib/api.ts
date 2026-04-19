const API_BASE = "http://localhost:8000";

export async function startSession(topic: string, assignmentText?: string) {
  const res = await fetch(`${API_BASE}/api/session/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic, assignment_text: assignmentText }),
  });
  if (!res.ok) throw new Error("Failed to start session");
  return res.json();
}

export async function sendMessage(sessionId: string, message: string) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!res.ok) throw new Error("Failed to send message");
  return res.json();
}

export async function getSessionSummary(sessionId: string) {
  const res = await fetch(`${API_BASE}/api/session/${sessionId}/summary`);
  if (!res.ok) throw new Error("Failed to get summary");
  return res.json();
}

export async function transcribeAudio(blob: Blob): Promise<string> {
  const form = new FormData();
  form.append("audio", blob, "recording.webm");
  const res = await fetch(`${API_BASE}/api/stt`, { method: "POST", body: form });
  if (!res.ok) throw new Error("STT failed");
  const data = await res.json();
  return data.text as string;
}

export async function fetchSpeechWithTiming(text: string): Promise<{
  audio_base64: string;
  alignment: {
    chars: string[];
    char_start_times_seconds: number[];
    char_durations_seconds: number[];
  };
} | null> {
  try {
    const res = await fetch(`${API_BASE}/api/tts/with-timing`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, voice: "af_bella" }),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    console.error("TTS with timing failed:", e);
    return null;
  }
}

export async function fetchSpeechBlob(text: string): Promise<Blob | null> {
  try {
    const res = await fetch(`${API_BASE}/api/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, voice: "af_heart" }),
    });
    if (!res.ok) return null;
    return await res.blob();
  } catch (e) {
    console.error("TTS fetch failed:", e);
    return null;
  }
}

export type StreamEvent =
  | { type: "text"; sentence: string }
  | { type: "audio"; wav: string; sentence: string; alignment: { chars: string[]; char_start_times_seconds: number[]; char_durations_seconds: number[] } }
  | { type: "eval"; evaluation: any; current_objective: any; session_complete: boolean }
  | { type: "done" };

export async function* sendMessageStream(
  sessionId: string,
  message: string
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!res.ok) throw new Error("Stream failed");

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try { yield JSON.parse(line.slice(6)) as StreamEvent; } catch {}
      }
    }
  }
}

// ── Pipelined stream (LLM and TTS run concurrently) ──────────────────────────

export type PipelineEvent =
  | { type: "text"; content: string }
  | { type: "audio"; content: string; text: string; alignment: { chars: string[]; char_start_times_seconds: number[]; char_durations_seconds: number[] } }
  | { type: "done" };

export async function* streamResponse(
  sessionId: string,
  message: string
): AsyncGenerator<PipelineEvent> {
  const url = `${API_BASE}/api/stream?session_id=${encodeURIComponent(sessionId)}&message=${encodeURIComponent(message)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Stream failed");

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try { yield JSON.parse(line.slice(6)) as PipelineEvent; } catch {}
      }
    }
  }
}

export async function ingestPDF(file: File): Promise<{ message: string; chunks_indexed: number; filename: string }> {
  const form = new FormData();
  form.append("file", file, file.name);
  const res = await fetch(`${API_BASE}/api/rag/ingest`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(err.detail ?? "Upload failed");
  }
  return res.json();
}

export async function evaluateSession(sessionId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/api/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error("Evaluate failed");
  return res.json();
}