"use client";
import { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';

export interface AvatarHandle {
  speak: (audioBase64: string, sentence: string) => Promise<void>;
  stop: () => void;
}

interface Props {
  modelUrl?: string;
  onStart?: () => void;
  onComplete?: () => void;
}

const TalkingHeadAvatar = forwardRef<AvatarHandle, Props>((props, ref) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const headRef = useRef<any>(null);
  const modelUrl = props.modelUrl ?? "/dani.glb";

  useImperativeHandle(ref, () => ({
    speak: async (audioBase64: string, sentence: string) => {
      if (!headRef.current) return;
      props.onStart?.();
      try {
        // Decode base64 WAV → ArrayBuffer
        const binary = window.atob(audioBase64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

        // Resume AudioContext if browser suspended it (autoplay policy)
        if (headRef.current.audioCtx.state === 'suspended') {
          await headRef.current.audioCtx.resume();
        }

        // Decode through TalkingHead's own AudioContext so timing is in sync
        const audioBuffer = await headRef.current.audioCtx.decodeAudioData(bytes.buffer);

        // Distribute audio duration evenly across words
        const durationMs = audioBuffer.duration * 1000;
        const words = sentence.split(/\s+/).filter(w => w.length > 0);
        const wordDuration = words.length > 0 ? durationMs / words.length : durationMs;
        const wtimes = words.map((_, i) => i * wordDuration);
        const wdurations = words.map(() => wordDuration);

        // speakAudio is synchronous (pushes to internal queue) — do NOT await
        // Visemes are omitted: TalkingHead computes them from words internally
        headRef.current.speakAudio(
          { audio: audioBuffer, words, wtimes, wdurations },
          { lipsyncLang: "en" }
        );

        props.onComplete?.();
      } catch (error) {
        console.error("Avatar speech error:", error);
      }
    },
    stop: () => {
      if (headRef.current) {
        headRef.current.stop();
        props.onComplete?.();
      }
    },
  }));

  useEffect(() => {
    if (!containerRef.current) return;
    let isMounted = true;

    async function init() {
      try {
        const { TalkingHead } = await (new Function('return import("/talkinghead.mjs")')() as Promise<any>);
        if (!isMounted) return;

        const head = new TalkingHead(containerRef.current, {
          ttsEndpoint: "N/A",
          cameraView: "upper",
          lipsyncModules: ["en"],
          lightAmbientIntensity: 1,
          lightDirectIntensity: 0,
          lightDirectColor: 0xffffff,
          lightSpotColor: 0x3388ff,
          lightSpotIntensity: 40,
          lightDirectPhi: 3.14,
          lightDirectTheta: 3.14,
          lightSpotDispersion: 1.5,
          avatarIdleEyeContact: 0.8,
          avatarMood: "neutral",
        });

        await head.showAvatar({ url: modelUrl, animationsUrl: "/animations.glb" });
        if (!isMounted) return;

        headRef.current = head;
        head.stop();
        head.start();
      } catch (e) {
        console.error("TalkingHead failed to load:", e);
      }
    }

    init();

    return () => {
      isMounted = false;
      if (headRef.current) {
        headRef.current.stop();
        headRef.current = null;
      }
    };
  }, [modelUrl]);

  return <div ref={containerRef} className="w-full h-full" />;
});

TalkingHeadAvatar.displayName = "TalkingHeadAvatar";
export default TalkingHeadAvatar;
