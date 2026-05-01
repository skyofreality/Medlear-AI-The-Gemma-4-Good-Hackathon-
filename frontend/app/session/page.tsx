"use client";
import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { sendMessageStream, transcribeAudio } from "@/lib/api";
import TalkingHeadAvatar, { AvatarHandle } from "@/components/TalkingHeadAvatar";

interface Objective {
  id: number;
  verb: string;
  objective: string;
  completed: boolean;
  comprehension_score: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function SessionPage() {
  const router = useRouter();
  const [session, setSession] = useState<any>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [currentObjective, setCurrentObjective] = useState<Objective | null>(null);
  const [objectives, setObjectives] = useState<Objective[]>([]);
  const [sessionComplete, setSessionComplete] = useState(false);
  const [sessionSummary, setSessionSummary] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const greetingFired = useRef(false);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const avatarRef = useRef<AvatarHandle>(null);
  const moodTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inactivityTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const audioQueueEndAtRef = useRef<number>(0);

  function setMoodTemporary(mood: string, durationMs = 6000) {
    if (moodTimerRef.current) clearTimeout(moodTimerRef.current);
    avatarRef.current?.setMood(mood);
    moodTimerRef.current = setTimeout(() => {
      avatarRef.current?.setMood("neutral");
    }, durationMs);
  }

  function detectMoodFromText(text: string): string | null {
    const t = text.toLowerCase();
    if (/\b(exactly|brilliant|perfect|great job|well done|yes!|correct|nice work|good|right)\b/.test(t)) return "happy";
    if (/\b(wrong|no,|not quite|that's not|vague|incorrect|nope|hmm, no)\b/.test(t)) return "sad";
    if (/\b(seriously\?|come on|really\?|you're not|wake up|pay attention)\b/.test(t)) return "angry";
    return null;
  }

  function resetInactivityTimer() {
    if (inactivityTimerRef.current) clearTimeout(inactivityTimerRef.current);
    inactivityTimerRef.current = setTimeout(() => {
      avatarRef.current?.setMood("angry");
    }, 90000);
  }

  function estimateWavDurationMs(b64: string): number {
    try {
      const binary = atob(b64);
      const ch = binary.charCodeAt(22) | (binary.charCodeAt(23) << 8);
      const sr = binary.charCodeAt(24) | (binary.charCodeAt(25) << 8) | (binary.charCodeAt(26) << 16) | (binary.charCodeAt(27) << 24);
      const bps = binary.charCodeAt(34) | (binary.charCodeAt(35) << 8);
      const size = binary.charCodeAt(40) | (binary.charCodeAt(41) << 8) | (binary.charCodeAt(42) << 16) | (binary.charCodeAt(43) << 24);
      const bytesPerSec = sr * ch * (bps / 8);
      if (!bytesPerSec) return 3000;
      return (size / bytesPerSec) * 1000;
    } catch { return 3000; }
  }

  function scheduleSpeak(b64: string, sentence: string, alignment?: { words: string[]; wtimes: number[]; wdurations: number[] }) {
    const dur = estimateWavDurationMs(b64);
    const now = performance.now();
    const startAt = Math.max(now, audioQueueEndAtRef.current);
    audioQueueEndAtRef.current = startAt + dur;
    avatarRef.current?.speak(b64, sentence, alignment);
  }

  async function waitForAudioQueueDrained(bufferMs = 150) {
    const remaining = audioQueueEndAtRef.current - performance.now();
    if (remaining > 0) {
      await new Promise(r => setTimeout(r, remaining + bufferMs));
    }
  }

  useEffect(() => {
    const stored = localStorage.getItem("medlearn_session");
    if (!stored) { router.push("/"); return; }
    const s = JSON.parse(stored);
    setSession(s);
    setObjectives(s.objectives);
    setCurrentObjective(s.current_objective);
    if (greetingFired.current) return;
    greetingFired.current = true;
    triggerGreeting(s.session_id);
    resetInactivityTimer();
    return () => {
      if (moodTimerRef.current) clearTimeout(moodTimerRef.current);
      if (inactivityTimerRef.current) clearTimeout(inactivityTimerRef.current);
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function runStream(sessionId: string, userMessage: string, isGreeting = false) {
    setLoading(true);
    avatarRef.current?.setMood("neutral");
    audioQueueEndAtRef.current = performance.now();
    let fullText = "";
    let firstTextEvent = true;
    let pendingLocal: {
      evaluation: any,
      nextObjective: any,
      sessionComplete: boolean,
      sessionSummary: string,
      prevObjectiveId: number | undefined,
    } | null = null;

    try {
      for await (const event of sendMessageStream(sessionId, userMessage)) {
        if (event.type === "text") {
          const sentence = (event as any).sentence as string;
          fullText += (fullText ? " " : "") + sentence;
          if (firstTextEvent) {
            if (isGreeting) {
              setMessages([{ role: "assistant", content: fullText }]);
            } else {
              setMessages(prev => [...prev, { role: "assistant", content: fullText }]);
            }
            setLoading(false);
            firstTextEvent = false;
          } else {
            setMessages(prev => {
              const updated = [...prev];
              updated[updated.length - 1] = { role: "assistant", content: fullText };
              return updated;
            });
          }
          const sentenceMood = detectMoodFromText(sentence);
          if (sentenceMood) setMoodTemporary(sentenceMood, 5000);
        } else if (event.type === "audio") {
          const audioEvent = event as any;
          scheduleSpeak(audioEvent.wav, audioEvent.sentence, audioEvent.alignment);
        } else if (event.type === "eval") {
          const ev = event as any;
          const evalData = ev.evaluation || {};
          if (evalData.advanced) {
            pendingLocal = {
              evaluation: evalData,
              nextObjective: ev.current_objective,
              sessionComplete: !!ev.session_complete,
              sessionSummary: ev.session_summary || "",
              prevObjectiveId: currentObjective?.id,
            };
          } else {
            if (evalData.score < 0.35) {
              setMoodTemporary("angry", 6000);
            } else if (evalData.score < 0.55) {
              setMoodTemporary("sad", 5000);
            }
            if (ev.current_objective) setCurrentObjective(ev.current_objective);
            if (ev.session_complete) {
              setSessionComplete(true);
              setSessionSummary(ev.session_summary || "");
            }
          }
        } else if (event.type === "done") {
          if (pendingLocal) {
            const p = pendingLocal;
            // Wait for Response A to finish speaking, then quietly update state.
            // No transition bubble or speech — next student turn will naturally
            // generate the response on the new objective.
            await waitForAudioQueueDrained();
            setObjectives(prev => prev.map((o: any) =>
              o.id === p.prevObjectiveId
                ? { ...o, completed: true, comprehension_score: p.evaluation.score }
                : o
            ));
            setMoodTemporary("happy", 7000);
            if (p.nextObjective) setCurrentObjective(p.nextObjective);
            if (p.sessionComplete) {
              setSessionComplete(true);
              setSessionSummary(p.sessionSummary || "");
            }
            fullText = "";
            firstTextEvent = true;
            pendingLocal = null;
          }
        }
      }
    } catch (e) {
      console.error(e);
      setMessages(prev => [...prev, { role: "assistant", content: "Sorry, something went wrong. Please try again." }]);
    } finally {
      setLoading(false);
    }
  }

  async function toggleRecording() {
    if (isRecording) {
      mediaRecorder.current?.stop();
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "audio/ogg";
    const recorder = new MediaRecorder(stream, { mimeType });
    audioChunks.current = [];
    recorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.current.push(e.data); };
    recorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      setIsRecording(false);
      if (!session || loading) return;
      const blob = new Blob(audioChunks.current, { type: mimeType });
      try {
        const transcript = await transcribeAudio(blob);
        if (!transcript.trim()) return;
        resetInactivityTimer();
        setMessages(prev => [...prev, { role: "user", content: transcript }]);
        await runStream(session.session_id, transcript);
      } catch {
        console.error("Transcription failed");
      }
    };
    recorder.start();
    mediaRecorder.current = recorder;
    setIsRecording(true);
  }

  async function triggerGreeting(sessionId: string) {
    await runStream(sessionId, "Hello, I am ready to start learning.", true);
  }

  async function handleSend() {
    if (!input.trim() || loading || !session) return;
    const userMessage = input.trim();
    setInput("");
    resetInactivityTimer();
    setMessages(prev => [...prev, { role: "user", content: userMessage }]);
    await runStream(session.session_id, userMessage);
  }

  if (!session) return null;

  return (
    <div className="h-screen flex flex-col bg-gray-50 overflow-hidden">

      {/* Top bar */}
      <div className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between shrink-0">
        <div>
          <h1 className="text-sm font-semibold text-gray-900">{session.topic}</h1>
          <p className="text-xs text-gray-400 mt-0.5">
            {objectives.filter(o => o.completed).length} of {objectives.length} objectives complete
          </p>
        </div>
        <button
          onClick={() => router.push("/")}
          className="text-xs text-gray-400 hover:text-gray-600"
        >
          End session
        </button>
      </div>

      {/* Main content: avatar (60%) + chat panel (40%) */}
      <div className="flex flex-1 overflow-hidden">

        {/* Avatar panel */}
        <div className="w-[60%] shrink-0">
          <TalkingHeadAvatar ref={avatarRef} />
        </div>

        {/* Right panel */}
        <div className="w-[40%] flex flex-col border-l border-gray-200 bg-white overflow-hidden">

          {/* Objectives tracker */}
          <div className="border-b border-gray-100 p-3 overflow-y-auto max-h-44 shrink-0">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">
              Objectives
            </p>
            <div className="space-y-1.5">
              {objectives.map((obj, i) => {
                const isCurrent = currentObjective?.id === obj.id && !obj.completed;
                const isDone = obj.completed;
                return (
                  <div
                    key={obj.id}
                    className={`rounded-lg px-2.5 py-2 text-xs border transition-all ${
                      isDone
                        ? "bg-teal-50 border-teal-200 text-teal-800"
                        : isCurrent
                        ? "bg-purple-50 border-purple-300 text-purple-900"
                        : "bg-gray-50 border-gray-100 text-gray-400"
                    }`}
                  >
                    <div className="flex items-center gap-1.5">
                      <span className={`w-4 h-4 rounded-full flex items-center justify-center text-xs font-bold shrink-0
                        ${isDone ? "bg-teal-500 text-white" : isCurrent ? "bg-purple-500 text-white" : "bg-gray-200 text-gray-400"}`}>
                        {isDone ? "✓" : i + 1}
                      </span>
                      <span className="font-medium truncate">{obj.verb} — {obj.objective}</span>
                      {isDone && (
                        <span className="ml-auto shrink-0 text-teal-600 font-medium">
                          {Math.round(obj.comprehension_score * 100)}%
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Current objective banner */}
          {currentObjective && !sessionComplete && (
            <div className="bg-purple-50 border-b border-purple-100 px-4 py-1.5 shrink-0">
              <p className="text-xs text-purple-700 truncate">
                <span className="font-semibold">Now: </span>
                {currentObjective.verb} {currentObjective.objective}
              </p>
            </div>
          )}

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                {msg.role === "assistant" && (
                  <div className="w-6 h-6 rounded-full bg-teal-600 flex items-center justify-center
                                  text-white text-xs font-bold mr-2 mt-1 shrink-0">
                    M
                  </div>
                )}
                <div className={`max-w-xs rounded-2xl px-3 py-2 text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "bg-purple-600 text-white rounded-tr-sm"
                    : "bg-gray-100 text-gray-800 rounded-tl-sm"
                }`}>
                  {msg.content}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex justify-start">
                <div className="w-6 h-6 rounded-full bg-teal-600 flex items-center justify-center
                                text-white text-xs font-bold mr-2 shrink-0">
                  M
                </div>
                <div className="bg-gray-100 rounded-2xl rounded-tl-sm px-3 py-2">
                  <div className="flex gap-1">
                    <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{animationDelay:"0ms"}}/>
                    <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{animationDelay:"150ms"}}/>
                    <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{animationDelay:"300ms"}}/>
                  </div>
                </div>
              </div>
            )}

            {sessionComplete && (
              <div className="w-full max-w-lg mx-auto rounded-2xl border border-gray-200 bg-white p-6">
                {/* Header */}
                <p className="text-sm uppercase tracking-wide text-gray-500 mb-1">Session complete</p>
                <h2 className="text-xl font-semibold text-gray-900">{session.topic}</h2>

                <div className="border-t border-gray-100 my-4" />

                {/* Summary */}
                <p className="text-sm font-medium text-gray-700 mb-2">How you did</p>
                <p className="text-sm text-gray-600 leading-relaxed">
                  {sessionSummary || "You worked through all objectives for this session."}
                </p>

                <div className="border-t border-gray-100 my-4" />

                {/* Objectives review */}
                <p className="text-sm font-medium text-gray-700 mb-2">Objectives</p>
                <div className="space-y-1.5">
                  {objectives.map((obj) => {
                    const mastered = obj.completed && obj.comprehension_score >= 0.75;
                    return (
                      <div key={obj.id} className="flex items-center gap-2">
                        <span className={`text-sm shrink-0 ${mastered ? "text-green-500" : "text-gray-300"}`}>
                          {mastered ? "✓" : "○"}
                        </span>
                        <span className="text-sm text-gray-700 flex-1">{obj.verb} {obj.objective}</span>
                        <span className="text-xs text-gray-400 shrink-0">
                          {obj.completed ? `${Math.round(obj.comprehension_score * 100)}%` : "—"}
                        </span>
                      </div>
                    );
                  })}
                </div>

                <div className="border-t border-gray-100 my-4" />

                <button
                  onClick={() => router.push("/")}
                  className="w-full bg-violet-600 text-white py-2 rounded-xl text-sm font-medium hover:bg-violet-700 transition-colors"
                >
                  Start a new session
                </button>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input bar */}
          {!sessionComplete && (
            <div className="bg-white border-t border-gray-200 px-4 py-3 shrink-0">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSend()}
                  placeholder="Type your answer..."
                  disabled={loading}
                  className="flex-1 border border-gray-300 rounded-xl px-3 py-2 text-sm
                             focus:outline-none focus:ring-2 focus:ring-teal-500
                             focus:border-transparent disabled:bg-gray-50"
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || loading}
                  className="bg-teal-600 hover:bg-teal-700 disabled:bg-gray-300
                             text-white px-4 py-2 rounded-xl text-sm font-medium
                             transition-colors disabled:cursor-not-allowed"
                >
                  Send
                </button>
              </div>
              <div className="flex justify-center mt-2">
                <button
                  onClick={toggleRecording}
                  disabled={loading}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-colors disabled:cursor-not-allowed ${
                    isRecording
                      ? "bg-red-100 text-red-600 hover:bg-red-200"
                      : "bg-gray-100 text-gray-500 hover:bg-gray-200 disabled:bg-gray-50 disabled:text-gray-300"
                  }`}
                >
                  <span className={`w-2 h-2 rounded-full ${isRecording ? "bg-red-500 animate-pulse" : "bg-gray-400"}`} />
                  {isRecording ? "Recording — click to send" : "Click to speak"}
                </button>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
