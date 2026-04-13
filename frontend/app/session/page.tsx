"use client";
import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { sendMessageStream } from "@/lib/api";

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
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const greetingFired = useRef(false);
  const audioQueue = useRef<HTMLAudioElement[]>([]);
  const audioPlaying = useRef(false);

  function enqueueAudio(wavBase64: string) {
    const binary = atob(wavBase64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const url = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
    const audio = new Audio(url);
    audio.onended = () => { URL.revokeObjectURL(url); playNextAudio(); };
    audioQueue.current.push(audio);
    if (!audioPlaying.current) playNextAudio();
  }

  function playNextAudio() {
    const next = audioQueue.current.shift();
    if (!next) { audioPlaying.current = false; return; }
    audioPlaying.current = true;
    next.play().catch(() => playNextAudio());
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
            // Add the assistant bubble on the first sentence
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
          enqueueAudio(event.wav);
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
    <div className="min-h-screen bg-gray-50 flex flex-col">

      {/* Top bar */}
      <div className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
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

      <div className="flex flex-1 overflow-hidden">

        {/* Sidebar — objective tracker */}
        <div className="w-72 bg-white border-r border-gray-200 p-4 overflow-y-auto hidden md:block">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-3">
            Session objectives
          </p>
          <div className="space-y-2">
            {objectives.map((obj, i) => {
              const isCurrent = currentObjective?.id === obj.id && !obj.completed;
              const isDone = obj.completed;
              return (
                <div
                  key={obj.id}
                  className={`rounded-xl p-3 text-xs border transition-all ${
                    isDone
                      ? "bg-teal-50 border-teal-200 text-teal-800"
                      : isCurrent
                      ? "bg-purple-50 border-purple-300 text-purple-900"
                      : "bg-gray-50 border-gray-100 text-gray-400"
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`w-4 h-4 rounded-full flex items-center justify-center text-xs font-bold
                      ${isDone ? "bg-teal-500 text-white" : isCurrent ? "bg-purple-500 text-white" : "bg-gray-200 text-gray-400"}`}>
                      {isDone ? "✓" : i + 1}
                    </span>
                    <span className="font-semibold">{obj.verb}</span>
                  </div>
                  <p className="leading-relaxed pl-6">{obj.objective}</p>
                  {isDone && (
                    <p className="pl-6 mt-1 text-teal-600 font-medium">
                      Score: {Math.round(obj.comprehension_score * 100)}%
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Chat area */}
        <div className="flex-1 flex flex-col">

          {/* Current objective banner */}
          {currentObjective && !sessionComplete && (
            <div className="bg-purple-50 border-b border-purple-100 px-6 py-2">
              <p className="text-xs text-purple-700">
                <span className="font-semibold">Now learning: </span>
                {currentObjective.verb} {currentObjective.objective}
              </p>
            </div>
          )}

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                {msg.role === "assistant" && (
                  <div className="w-7 h-7 rounded-full bg-teal-600 flex items-center justify-center 
                                  text-white text-xs font-bold mr-2 mt-1 shrink-0">
                    M
                  </div>
                )}
                <div className={`max-w-lg rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "bg-purple-600 text-white rounded-tr-sm"
                    : "bg-white border border-gray-200 text-gray-800 rounded-tl-sm"
                }`}>
                  {msg.content}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex justify-start">
                <div className="w-7 h-7 rounded-full bg-teal-600 flex items-center justify-center 
                                text-white text-xs font-bold mr-2 shrink-0">
                  M
                </div>
                <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-sm px-4 py-3">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{animationDelay:"0ms"}}/>
                    <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{animationDelay:"150ms"}}/>
                    <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{animationDelay:"300ms"}}/>
                  </div>
                </div>
              </div>
            )}

            {sessionComplete && (
              <div className="text-center py-8">
                <div className="text-4xl mb-3">🎉</div>
                <h2 className="text-lg font-semibold text-gray-900 mb-1">Session complete!</h2>
                <p className="text-sm text-gray-500 mb-4">You've worked through all objectives.</p>
                <button
                  onClick={() => router.push("/")}
                  className="bg-teal-600 text-white px-6 py-2 rounded-xl text-sm font-medium hover:bg-teal-700"
                >
                  Start a new session
                </button>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input bar */}
          {!sessionComplete && (
            <div className="bg-white border-t border-gray-200 px-6 py-4">
              <div className="flex gap-3">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSend()}
                  placeholder="Type your answer..."
                  disabled={loading}
                  className="flex-1 border border-gray-300 rounded-xl px-4 py-2.5 text-sm
                             focus:outline-none focus:ring-2 focus:ring-teal-500 
                             focus:border-transparent disabled:bg-gray-50"
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || loading}
                  className="bg-teal-600 hover:bg-teal-700 disabled:bg-gray-300 
                             text-white px-5 py-2.5 rounded-xl text-sm font-medium 
                             transition-colors disabled:cursor-not-allowed"
                >
                  Send
                </button>
              </div>
              <p className="text-xs text-gray-400 mt-2 text-center">
                Voice input coming soon · Powered by Gemma 4 · Fully offline
              </p>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}