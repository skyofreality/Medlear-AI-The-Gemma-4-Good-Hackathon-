"use client";
import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { startSession, ingestPDF } from "@/lib/api";

export default function Home() {
  const router = useRouter();
  const [topic, setTopic] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{ ok: boolean; msg: string } | null>(null);
  const [docId, setDocId] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleUpload() {
    if (!uploadFile) return;
    setUploading(true);
    setUploadStatus(null);
    try {
      const result = await ingestPDF(uploadFile);
      setDocId(result.doc_id);
      setUploadStatus({ ok: true, msg: `Indexed ${result.chunks_indexed} chunks from ${result.filename}` });
    } catch (e: any) {
      setDocId("");
      setUploadStatus({ ok: false, msg: e.message ?? "Upload failed" });
    } finally {
      setUploading(false);
    }
  }

  async function handleStart() {
    if (!topic.trim()) return;
    if (uploadFile && !docId) {
      setError("Index the selected PDF before starting, or remove it to use the knowledge base.");
      return;
    }
    const retrievalMode = docId ? "uploaded_pdf" : "knowledge_base";
    setLoading(true);
    setError("");
    try {
      const session = await startSession(topic, retrievalMode, docId || undefined);
      // Store session in localStorage for the chat page
      localStorage.setItem("medlearn_session", JSON.stringify(session));
      router.push("/session");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start session");
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
            Your AI medical tutor. Speak, learn, understand.
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
            {loading ? (docId ? "Analysing your PDF and finding relevant curriculum sections..." : "Preparing objectives from your curriculum...") : "Start learning"}
          </button>

          <div className="flex items-center gap-3 my-6">
            <div className="flex-1 h-px bg-gray-200" />
            <span className="text-xs text-gray-400">or</span>
            <div className="flex-1 h-px bg-gray-200" />
          </div>

          <div
            className="border-2 border-dashed border-gray-200 hover:border-teal-400 rounded-xl p-6
                        text-center text-sm transition-colors cursor-pointer"
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files[0];
              if (f?.type === "application/pdf") { setUploadFile(f); setUploadStatus(null); setDocId(""); }
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0] ?? null;
                setUploadFile(f);
                setUploadStatus(null);
                setDocId("");
              }}
            />
            {uploadFile ? (
              <span className="text-gray-700 font-medium">{uploadFile.name}</span>
            ) : (
              <>
                <span className="text-gray-400">Upload assignment PDF</span>
                <p className="text-xs mt-1 text-gray-300">Click or drag and drop</p>
              </>
            )}
          </div>

          {uploadFile && !uploadStatus && (
            <button
              onClick={handleUpload}
              disabled={uploading}
              className="w-full mt-3 bg-gray-100 hover:bg-gray-200 disabled:bg-gray-50
                         text-gray-700 font-medium py-2.5 rounded-xl transition-colors
                         text-sm disabled:cursor-not-allowed"
            >
              {uploading ? "Indexing with Gemma Vision..." : "Index PDF"}
            </button>
          )}

          {uploadStatus && (
            <p className={`text-xs mt-2 ${uploadStatus.ok ? "text-teal-600" : "text-red-500"}`}>
              {uploadStatus.msg}
            </p>
          )}

        </div>

        {/* Footer */}
        <p className="text-center text-xs text-gray-400 mt-6">
          Powered by Gemma 4 via Ollama
        </p>
      </div>
    </main>
  );
}
