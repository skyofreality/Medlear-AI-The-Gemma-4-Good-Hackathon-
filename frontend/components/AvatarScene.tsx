"use client";
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";

export interface AvatarSceneHandle {
  speakText: (audioBlob: Blob, text: string) => Promise<void>;
  setExpression: (mood: string) => void;
  stopSpeaking: () => void;
}

const MOOD_MAP: Record<string, string> = {
  thinking: "neutral",
  listening: "happy",
  encouraging: "happy",
  waiting: "neutral",
  step_cleared: "happy",
  error: "sad",
};

const AvatarScene = forwardRef<AvatarSceneHandle>((_, ref) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const headRef = useRef<any>(null);
  const [ready, setReady] = useState(false);
  const [fallback, setFallback] = useState(false);
  const [speaking, setSpeaking] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;
    let head: any = null;
    let cancelled = false;

    async function init() {
      try {
        // Use Function constructor to bypass TypeScript static analysis and
        // webpack bundling — talkinghead.mjs must be a native browser ES module import.
        const importFn = new Function('return import("/talkinghead.mjs")');
        const { TalkingHead } = await importFn();

        if (cancelled) return;

        head = new TalkingHead(containerRef.current, {
          ttsEndpoint: null,
          cameraView: "upper",
          cameraRotateEnable: false,
          modelPixelRatio: 1,
          lightAmbientColor: 0xffffff,
          lightAmbientIntensity: 2,
          lightDirectColor: 0xffffff,
          lightDirectIntensity: 1,
          lightDirectPhi: 0.1,
          lightDirectTheta: 0.5,
        });

        await head.showAvatar({
          url: "/dr-2.glb",
          body: "F",
          avatarMood: "neutral",
          ttsLang: "en-US",
          ttsVoice: "en-US-Standard-F",
          lipsyncLang: "en",
        });

        if (cancelled) {
          try { head.stopSpeaking?.(); head.close?.(); } catch {}
          return;
        }

        headRef.current = head;
        setReady(true);
      } catch (e) {
        console.error("TalkingHead failed to load:", e);
        if (!cancelled) setFallback(true);
      }
    }

    init();

    return () => {
      cancelled = true;
      if (head) {
        try { head.stopSpeaking?.(); } catch {}
        try { head.close?.(); } catch {}
      }
    };
  }, []);

  useImperativeHandle(ref, () => ({
    speakText: async (audioBlob: Blob, text: string) => {
      if (!headRef.current) return;
      try {
        setSpeaking(true);
        const buffer = await audioBlob.arrayBuffer();
        await headRef.current.speakAudio(buffer, { text, lipsyncLang: "en" });
      } catch (e) {
        console.error("speakAudio failed:", e);
      } finally {
        setSpeaking(false);
      }
    },
    setExpression: (mood: string) => {
      if (!headRef.current) return;
      const mapped = MOOD_MAP[mood] ?? mood;
      try { headRef.current.setMood?.(mapped); } catch {}
    },
    stopSpeaking: () => {
      if (!headRef.current) return;
      try { headRef.current.stopSpeaking?.(); } catch {}
    },
  }));

  // Fallback: WebGL unavailable or model failed to load
  if (fallback) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-900">
        <div className={`w-32 h-32 rounded-full bg-[#0F6E56] flex items-center justify-center ${speaking ? "animate-pulse" : ""}`}>
          <span className="text-white font-semibold text-sm">Dr. Mira</span>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full relative bg-gray-900">
      {/* TalkingHead appends its canvas into this div */}
      <div ref={containerRef} className="w-full h-full" />
      {/* Loading state — shown until avatar is fully initialised */}
      {!ready && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-32 h-32 rounded-full bg-[#0F6E56] flex items-center justify-center animate-pulse">
            <span className="text-white font-semibold text-sm">Dr. Mira</span>
          </div>
        </div>
      )}
    </div>
  );
});

AvatarScene.displayName = "AvatarScene";
export default AvatarScene;
