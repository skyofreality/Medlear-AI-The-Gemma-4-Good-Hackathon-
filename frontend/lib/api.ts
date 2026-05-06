const API_BASE = "";

export type RetrievalMode = "uploaded_pdf" | "knowledge_base" | "general_medical";

export async function startSession(
  topic: string,
  retrievalMode: RetrievalMode,
  docId?: string,
  assignmentText?: string
) {
  const res = await fetch(`${API_BASE}/api/session/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      topic,
      retrieval_mode: retrievalMode,
      doc_id: docId,
      assignment_text: assignmentText,
    }),
  });
  if (!res.ok) {
    let message = "Failed to start session";
    const text = await res.text().catch(() => "");
    if (text) {
      try {
        const data = JSON.parse(text);
        if (data?.detail) {
          message = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
        } else {
          message = text;
        }
      } catch {
        message = text;
      }
    }
    throw new Error(message);
  }
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

export type WordAlignment = {
  words: string[];
  wtimes: number[];
  wdurations: number[];
};

export type StreamEvent =
  | { type: "text"; sentence: string }
  | { type: "audio"; wav: string; sentence: string; alignment?: WordAlignment }
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

export async function ingestPDF(file: File): Promise<{ message: string; chunks_indexed: number; filename: string; doc_id: string }> {
  const form = new FormData();
  form.append("file", file, file.name);
  const res = await fetch(`${API_BASE}/api/rag/ingest`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(err.detail ?? "Upload failed");
  }
  return res.json();
}

