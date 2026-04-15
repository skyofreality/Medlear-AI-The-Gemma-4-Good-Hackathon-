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
  const [isRecording, setIsRecording] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const greetingFired = useRef(false);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const avatarRef = useRef<AvatarHandle>(null);

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
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function runStream(sessionId: string, userMessage: string, isGreeting = false) {
    setLoading(true);
    let fullText = "";
    let firstSentence = true;

    try {
      for await (const event of sendMessageStream(sessionId, userMessage)) {
        if (event.type === "text") {
          fullText += (fullText ? " " : "") + event.sentence;
          if (firstSentence) {
            if (isGreeting) {
              setMessages([{ role: "assistant", content: fullText }]);
            } else {
              setMessages(prev => [...prev, { role: "assistant", content: fullText }]);
            }
            setLoading(false);
            firstSentence = false;
          } else {
            setMessages(prev => {
              const updated = [...prev];
              updated[updated.length - 1] = { role: "assistant", content: fullText };
              return updated;
            });
          }
        } else if (event.type === "audio") {
          avatarRef.current?.speak(event.wav, event.sentence);
        } else if (event.type === "eval") {
          if (event.evaluation?.advanced) {
            setObjectives(prev => prev.map(o =>
              o.id === currentObjective?.id
                ? { ...o, completed: true, comprehension_score: event.evaluation.score }
                : o
            ));
          }
          if (event.current_objective) setCurrentObjective(event.current_objective);
          if (event.session_complete) setSessionComplete(true);
        }
      }
    } catch {
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
              <div className="text-center py-6">
                <div className="text-3xl mb-2">🎉</div>
                <h2 className="text-base font-semibold text-gray-900 mb-1">Session complete!</h2>
                <p className="text-xs text-gray-500 mb-3">You've worked through all objectives.</p>
                <button
                  onClick={() => router.push("/")}
                  className="bg-teal-600 text-white px-5 py-2 rounded-xl text-sm font-medium hover:bg-teal-700"
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
