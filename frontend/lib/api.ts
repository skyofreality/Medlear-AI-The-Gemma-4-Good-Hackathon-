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

export type StreamEvent =
  | { type: "text"; sentence: string }
  | { type: "audio"; wav: string }
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