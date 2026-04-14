"use client";
import { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';

export interface AlignmentData {
  chars: string[];
  char_start_times_seconds: number[];
  char_durations_seconds: number[];
}

export interface AvatarHandle {
  speak: (audioBase64: string, alignment: AlignmentData) => Promise<void>;
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
  const lipsyncRef = useRef<any>(null);
  const speechTimerRef = useRef<any>(null);
  const modelUrl = props.modelUrl ?? "/dani.glb";

  useImperativeHandle(ref, () => ({
    speak: async (audioBase64: string, alignment: AlignmentData) => {
      if (!headRef.current) return;
      props.onStart?.();
      try {
        // 1. Convert Base64 to ArrayBuffer and decode via avatar's AudioContext
        const binary = window.atob(audioBase64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const audioBuffer = await headRef.current.audioCtx.decodeAudioData(bytes.buffer);

        // 2. Process character alignment into word-level timing
        const { words, wtimes, wdurations } = processAlignmentToWords(alignment);

        // 3. Use LipsyncEn to convert words to visemes with precise timing
        const allVisemes: any[] = [];
        const allVTimes: number[] = [];
        const allVDurations: number[] = [];

        if (lipsyncRef.current) {
          words.forEach((word, index) => {
            const wordStartTime = wtimes[index];
            const wordDuration = wdurations[index];
            const wordVisData = lipsyncRef.current.wordsToVisemes(word);
            const totalUnits = wordVisData.durations.reduce((a: any, b: any) => a + b, 0);
            const unitInMs = totalUnits > 0 ? wordDuration / totalUnits : 0;
            let currentOffset = 0;
            wordVisData.visemes.forEach((vis: any, i: number) => {
              allVisemes.push(vis);
              allVTimes.push(wordStartTime + currentOffset * unitInMs);
              const visDur = wordVisData.durations[i] * unitInMs;
              allVDurations.push(visDur);
              currentOffset += wordVisData.durations[i];
            });
          });
        }

        // 4. Combine into the final syncData object
        const syncData = {
          audio: audioBuffer,
          words,
          wtimes,
          wdurations,
          visemes: allVisemes,
          vtimes: allVTimes,
          vdurations: allVDurations,
          audioEncoding: "wav",
        };

        // 5. Resume context if suspended, then play
        if (headRef.current.audioCtx.state === 'suspended') {
          await headRef.current.audioCtx.resume();
        }
        const totalDuration = allVDurations.reduce((acc, d) => acc + d, 0);

        await headRef.current.speakAudio(syncData);

        if (speechTimerRef.current) clearTimeout(speechTimerRef.current);
        speechTimerRef.current = setTimeout(() => {
          if (props.onComplete) props.onComplete();
          speechTimerRef.current = null;
        }, totalDuration);
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
        // Load LipsyncEn as a native browser ES module
        const lipsyncMod = await (new Function('return import("/lipsync-en.mjs")')() as Promise<any>);
        if (!isMounted) return;
        lipsyncRef.current = new lipsyncMod.LipsyncEn();

        // Load TalkingHead as a native browser ES module
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
      if (speechTimerRef.current) clearTimeout(speechTimerRef.current);
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

// Convert character-level alignment from backend into word-level timing (ms)
const processAlignmentToWords = (alignment: AlignmentData) => {
  const words: string[] = [];
  const wtimes: number[] = [];
  const wdurations: number[] = [];

  const chars = alignment.chars;
  const starts = alignment.char_start_times_seconds;

  let currentWord = "";
  let wordStartTime = 0;

  chars.forEach((char: string, i: number) => {
    const startTimeMs = starts[i] * 1000;

    if (currentWord === "" && char !== " ") {
      wordStartTime = startTimeMs;
    }

    const isSpace = char === " ";
    const isLastChar = i === chars.length - 1;

    if (isSpace || isLastChar) {
      if (isLastChar && !isSpace) currentWord += char;
      if (currentWord !== "") {
        words.push(currentWord);
        wtimes.push(wordStartTime);
        wdurations.push(startTimeMs - wordStartTime + (isLastChar ? 100 : 0));
        currentWord = "";
      }
    } else {
      currentWord += char;
    }
  });

  return { words, wtimes, wdurations };
};
