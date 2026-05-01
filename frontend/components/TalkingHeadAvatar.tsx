"use client";
import { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';

export interface WordAlignment {
  words: string[];
  wtimes: number[];
  wdurations: number[];
}

export interface AvatarHandle {
  speak: (audioBase64: string, sentence: string, alignment?: WordAlignment) => Promise<void>;
  stop: () => void;
  setMood: (mood: string) => void;
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
    speak: async (audioBase64: string, sentence: string, alignment?: WordAlignment) => {
      if (!headRef.current) return;
      props.onStart?.();
      try {
        const binary = window.atob(audioBase64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

        if (headRef.current.audioCtx.state === 'suspended') {
          await headRef.current.audioCtx.resume();
        }

        const audioBuffer = await headRef.current.audioCtx.decodeAudioData(bytes.buffer);

        let words: string[];
        let wtimes: number[];
        let wdurations: number[];

        if (alignment && alignment.words.length > 0) {
          // Real per-word timing from Kokoro pred_dur — drives accurate visemes
          words = alignment.words.map(w => w.replace(/[^a-zA-Z0-9']/g, '')).filter(w => w.length > 0);
          wtimes = [...alignment.wtimes];
          wdurations = [...alignment.wdurations];
          // Trim back to whatever survived the strip
          if (words.length !== alignment.words.length) {
            const kept: { w: string; t: number; d: number }[] = [];
            for (let i = 0; i < alignment.words.length; i++) {
              const w = alignment.words[i].replace(/[^a-zA-Z0-9']/g, '');
              if (w.length > 0) kept.push({ w, t: alignment.wtimes[i], d: alignment.wdurations[i] });
            }
            words = kept.map(k => k.w);
            wtimes = kept.map(k => k.t);
            wdurations = kept.map(k => k.d);
          }
        } else {
          // Fallback: even-distribution synthesis from sentence text
          const durationMs = audioBuffer.duration * 1000;
          words = sentence.split(/\s+/)
            .map(w => w.replace(/[^a-zA-Z0-9']/g, ''))
            .filter(w => w.length > 0);
          wtimes = [];
          wdurations = [];
          if (words.length === 1) {
            wtimes.push(0);
            wdurations.push(durationMs);
          } else if (words.length > 1) {
            const MIN_WORD_MS = 130;
            const totalChars = words.reduce((sum, w) => sum + w.length, 0) || 1;
            const flexMs = durationMs - MIN_WORD_MS * words.length;
            let t = 0;
            for (const w of words) {
              const dur = MIN_WORD_MS + (w.length / totalChars) * flexMs;
              wtimes.push(t);
              wdurations.push(dur);
              t += dur;
            }
          }
        }

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
    setMood: (mood: string) => {
      try { headRef.current?.setMood(mood); } catch {}
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
