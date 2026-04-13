"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { startSession } from "@/lib/api";

export default function Home() {
  const router = useRouter();
  const [topic, setTopic] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleStart() {
    if (!topic.trim()) return;
    setLoading(true);
    setError("");
    try {
      const session = await startSession(topic);
      // Store session in localStorage for the chat page
      localStorage.setItem("medlearn_session", JSON.stringify(session));
      router.push("/session");
    } catch (e) {
      setError("Failed to start session. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gray-50 flex items-center justify-center p-6">
      <div className="w-full max-w-lg">

        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-4xl font-semibold text-gray-900 mb-2">
            MedLearn AI
          </h1>
          <p className="text-gray-500 text-base">
            Your offline medical tutor. Speak, learn, understand.
          </p>
        </div>

        {/* Card */}
        <div className="bg-white rounded-2xl border border-gray-200 p-8 shadow-sm">

          <label className="block text-sm font-medium text-gray-700 mb-2">
            What do you want to learn today?
          </label>
          <input
            type="text"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleStart()}
            placeholder="e.g. Myocardial infarction pathophysiology"
            className="w-full border border-gray-300 rounded-xl px-4 py-3 text-gray-900 
                       placeholder-gray-400 focus:outline-none focus:ring-2 
                       focus:ring-teal-500 focus:border-transparent text-sm mb-6"
          />

          {error && (
            <p className="text-red-500 text-sm mb-4">{error}</p>
          )}

          <button
            onClick={handleStart}
            disabled={!topic.trim() || loading}
            className="w-full bg-teal-600 hover:bg-teal-700 disabled:bg-gray-300 
                       text-white font-medium py-3 rounded-xl transition-colors 
                       text-sm disabled:cursor-not-allowed"
          >
            {loading ? "Building your session..." : "Start learning"}
          </button>

          <div className="flex items-center gap-3 my-6">
            <div className="flex-1 h-px bg-gray-200" />
            <span className="text-xs text-gray-400">or</span>
            <div className="flex-1 h-px bg-gray-200" />
          </div>

          <div className="border-2 border-dashed border-gray-200 rounded-xl p-6 
                          text-center text-gray-400 text-sm cursor-not-allowed">
            Upload assignment PDF
            <p className="text-xs mt-1 text-gray-300">Coming in Phase 2</p>
          </div>

        </div>

        {/* Footer */}
        <p className="text-center text-xs text-gray-400 mt-6">
          Fully offline · Powered by Gemma 4 via Ollama
        </p>
      </div>
    </main>
  );
}