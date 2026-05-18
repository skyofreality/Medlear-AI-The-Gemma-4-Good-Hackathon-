# MedLearn AI — Socratic Medical Tutor Powered by Gemma 4

> **"Understanding cannot be transferred — only guided into existence."**
> *Dr. Mira, AI Medical Tutor*

The first AI Socratic tutor to pair a real-time lip-synced 3D avatar with a full pedagogical pipeline — teaching medical students the way the best doctors teach: by asking questions, not giving answers. Students learn through live spoken dialogue with Dr. Mira, a talking avatar whose mouth moves to the actual rhythm of speech, guided by Bloom's taxonomy objectives generated directly from their own curriculum material.

**Built for the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon)** · [Watch Demo](https://youtu.be/hSbmAuMDsl4?si=Y-8_4Q4cnWsTafOa)

| Model | Role |
|---|---|
| `gemma4:e4b` via Ollama | LLM — teaching, evaluation, objectives, vision OCR, span judging, session summary |
| `all-MiniLM-L6-v2` | Embeddings — curriculum chunk retrieval via ChromaDB |
| Kokoro `af_bella` | TTS — 24 kHz speech synthesis with word-level lip-sync timing |
| faster-whisper `base.en` | STT — student voice transcription |

---

## The Problem

There is a global shortage of 10 million health workers, concentrated in low- and middle-income countries. Medical schools are under-resourced, faculty-to-student ratios are crushing, and the students who need the most support — those studying from scanned PDFs on aging laptops — get the least of it.

Passive reading doesn't build clinical reasoning. Flashcards don't teach differential diagnosis. What actually works is a skilled teacher who asks you the right question at the right moment — who knows when to push, when to scaffold, and when to simply give you one foothold and let you climb.

Most students never get that teacher.

---

## The Solution

MedLearn AI gives every student a Socratic tutor that:

- Reads their actual curriculum (uploaded PDF, scanned textbooks included)
- Generates pedagogically-grounded learning objectives from that specific material
- Teaches through dialogue — asking questions, not giving answers
- Evaluates genuine understanding, not keyword matching
- Runs on consumer hardware with a quantised edge model (`gemma4:e4b`)

**It is not a chatbot. It is a pedagogical pipeline.**

---

## Demo

| Home — Topic + PDF Upload | Session — Socratic Dialogue with Dr. Mira |
|---|---|
| Student enters a topic and optionally uploads their curriculum PDF | Avatar speaks, listens, evaluates, adapts — one objective at a time |

```
Topic: "Myocardial infarction pathophysiology"
↓
[PDF ingested — 847 chunks indexed in 23s]
↓
Objectives generated from YOUR curriculum:
  1. Identify the coronary artery territories at risk in STEMI
  2. Explain the ischaemic cascade leading to cardiomyocyte death
  3. Differentiate Type 1 from Type 2 MI by mechanism
  4. Analyze how reperfusion injury paradoxically worsens outcomes
↓
Dr. Mira: "Before we talk about what happens during an MI,
           can you tell me what the heart muscle actually needs
           to survive, and where it gets it from?"
↓
[Student answers → evaluated → gaps targeted → advances when ready]
```

---

## How Gemma 4 Powers Everything

This project uses Gemma 4 in **seven distinct roles**, exploiting its multimodal, function-calling, structured-output, and streaming capabilities across a multi-agent pipeline.

| # | Role | File | Gemma 4 Capability Used | Temperature |
|---|---|---|---|---|
| 1 | **Vision OCR** | `rag.py` | Multimodal — reads scanned PDF pages as images, extracts text + describes diagrams | 0.1 |
| 2 | **Topic Span Judge** | `rag.py` | Structured JSON output — selects which of 20 candidate RAG chunks are relevant to the topic | 0.1 |
| 3 | **Curriculum Orchestrator** | `orchestrator.py` | Structured JSON (schema-enforced) — generates 4–8 Bloom's taxonomy objectives grounded in source material | 0.3 |
| 4 | **Socratic Teacher (Dr. Mira)** | `teaching.py` | Streaming chat — real-time sentence-by-sentence dialogue, Socratic questioning, gap targeting | 0.75 |
| 5 | **Comprehension Evaluator** | `evaluator.py` | Native function calling (`submit_evaluation` tool) — 5-dimension scoring, advance decision | 0.1 |
| 6 | **Session Summariser** | `session.py` | Non-streaming chat — personalised performance narrative naming specific objectives | 0.4 |
| 7 | **Thinking Extractor** | `orchestrator.py` | Tool call fallback — re-extracts structured objectives if model returns reasoning instead of schema | 0.1 |

### Gemma 4 Feature Usage at a Glance

```
Multimodal (vision)      ████████████████████  ✓  scanned PDF OCR
Native function calling  ████████████████████  ✓  evaluation tool, extraction fallback
Structured JSON output   ████████████████████  ✓  objectives + span selection (Ollama format param)
Streaming tokens         ████████████████████  ✓  real-time Socratic dialogue
Edge quantisation (e4b)  ████████████████████  ✓  runs on consumer GPU / CPU via Ollama
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STUDENT'S DEVICE                         │
│                                                                 │
│  ┌──────────────────┐          ┌──────────────────────────────┐ │
│  │   Next.js 16     │◄────────►│      FastAPI (Python 3.12)   │ │
│  │   React 19       │  SSE     │                              │ │
│  │   Tailwind 4     │  stream  │  ┌────────────────────────┐  │ │
│  │                  │          │  │   Gemma 4 (gemma4:e4b) │  │ │
│  │  ┌────────────┐  │          │  │   via Ollama           │  │ │
│  │  │TalkingHead │  │          │  │                        │  │ │
│  │  │ Avatar     │  │          │  │  orchestrator.py ──────┼──┼─┤ objectives
│  │  │(Three.js)  │  │          │  │  teaching.py ──────────┼──┼─┤ stream
│  │  │ lip-sync   │  │          │  │  evaluator.py ─────────┼──┼─┤ tool calls
│  │  │ word align │  │          │  │  rag.py (vision+judge) ┼──┼─┤ multimodal
│  │  └────────────┘  │          │  │  session.py (summary)  ┼──┼─┤ summary
│  │                  │          │  └────────────────────────┘  │ │
│  │  Kokoro TTS ─────┼──────────┼── 24kHz WAV + word timings   │ │
│  │  Whisper STT ────┼──────────┼── faster-whisper base.en     │ │
│  │                  │          │  ChromaDB ─── all-MiniLM-L6  │ │
│  └──────────────────┘          └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
             Gemma 4 (e4b) runs via Ollama — no external API required.
```

---

## The Teaching Pipeline

```
Student uploads PDF
        │
        ▼
┌───────────────────────────────────────┐
│  INGEST (once)                        │
│  PyMuPDF text extraction              │
│  + Gemma 4 Vision for sparse pages    │  ← scanned textbooks work
│  → 1000-char chunks → ChromaDB        │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  SESSION START                        │
│  1. RAG: retrieve 20 candidate spans  │
│  2. Gemma 4 Judge: accept ≤8 relevant │
│  3. Gemma 4 Orchestrator: generate    │
│     4–8 Bloom's objectives from       │
│     ONLY the accepted spans           │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  PER-TURN LOOP  (repeats for each objective)          │
│                                                       │
│  Student speaks/types                                 │
│       │                                               │
│       ▼                                               │
│  Gemma 4 Teacher (streaming)                          │
│  → sentence 1 → Kokoro TTS → avatar speaks + lip-syncs│
│  → sentence 2 → Kokoro TTS → queued                  │
│  → ...                                                │
│       │                                               │
│       ▼                                               │
│  Gemma 4 Evaluator (tool call, silent)                │
│  scores: source_alignment  topic_alignment            │
│          objective_alignment  year_level              │
│          answer_correctness                           │
│       │                                               │
│  advance? ──yes──► next objective (silent transition) │
│       │                                               │
│  advance? ──no───► pending_feedback injected into     │
│                    next teacher prompt (gap targeting) │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  SESSION COMPLETE                     │
│  Gemma 4 writes personalised summary  │
│  naming specific objectives mastered  │
│  vs needing review                    │
└───────────────────────────────────────┘
```

---

## Key Technical Highlights

### Ollama-Native Architecture
All inference runs through Ollama's `/api/chat` endpoint. The project exploits three distinct Ollama features:

| Ollama Feature | Where Used | Why |
|---|---|---|
| `format:` JSON schema enforcement | Objectives + span judge | Guarantees parseable Bloom's objectives — no retry loops |
| `tools:` function calling | Evaluation + fallback extraction | Structured 5-dimension scoring without prompt-hacking |
| `stream: true` token streaming | Socratic teaching | First sentence reaches TTS in < 1s; avatar starts speaking before generation completes |
| `gemma4:e4b` edge quantisation | All inference | 4-bit — runs on consumer GPU or CPU via Ollama |

### Bloom's Taxonomy Grounding
Objectives are not hallucinated. Every objective is:
1. Generated **only** from spans the LLM judge accepted as topic-relevant
2. Required to cite a `source_hint` (3–5 word phrase from the source material)
3. Assigned a `confidence` score (0.0–1.0) indicating how well the source grounds it
4. Labelled with exactly one Bloom's verb: **Identify, Explain, Differentiate, Apply, Analyze**

### Lip-Synced Avatar with Real Word Timing
Kokoro TTS returns per-token `start_ts`/`end_ts` timestamps. These drive the TalkingHead viseme engine with genuine word-level alignment — not interpolated estimates. The avatar mouth moves to the actual rhythm of speech.

### The Pending Feedback Loop
When a student doesn't advance an objective, the evaluator's `feedback_focus` and `missing_elements` are injected into the *next* teacher prompt — silently. Dr. Mira's next question is automatically targeted at the exact gap, without the student ever seeing the evaluation output.

```
Student answer → Evaluator → "missing: role of troponin release"
                                    ↓
                         pending_feedback stored
                                    ↓
                    Next teacher prompt includes:
                    "Focus on: troponin release mechanism
                     Missing elements: cardiac troponin specificity"
                                    ↓
               Dr. Mira: "You've explained the ischaemia well.
                          What happens to the cardiomyocyte membrane
                          as ATP runs out — and what does that
                          release into the bloodstream?"
```

---

## Stack

| Layer | Technology |
|---|---|
| LLM | `gemma4:e4b` via Ollama |
| Backend | FastAPI 0.135 · Python 3.12 |
| Frontend | Next.js 16.2 · React 19 · Tailwind 4 |
| Vector DB | ChromaDB 1.5 · all-MiniLM-L6-v2 embeddings |
| TTS | Kokoro 0.9.4 · ONNX · `af_bella` · 24 kHz |
| STT | faster-whisper 1.2 · `base.en` · CPU int8 |
| Avatar | TalkingHead (Three.js) · GLB model · lip-sync visemes |
| PDF Extraction | PyMuPDF + Gemma 4 Vision (sparse/scanned pages) |

---

## The "For Good" Case

| Problem | How MedLearn AI addresses it |
|---|---|
| Faculty shortage in LMICs | One Gemma 4 instance can tutor unlimited students simultaneously |
| Scanned / low-quality textbooks | Vision pipeline extracts text from image-only PDFs |
| No dedicated tutor for every student | One Gemma 4 instance tutors through natural spoken conversation |
| Passive reading doesn't build reasoning | Socratic method forces the student to construct understanding |
| Students don't know what they don't know | Bloom's objectives make the learning target explicit and trackable |
| Generic AI tutors ignore the curriculum | Objectives are grounded in the student's own uploaded material |
| Data privacy concerns block institutional adoption | Architecture is designed to deploy on a university's own servers — Ollama + ChromaDB + all models run on-premise, no student data or curriculum content ever leaves the institution |

---

## Setup

### Prerequisites
- [Ollama](https://ollama.com) with `gemma4:e4b` pulled: `ollama pull gemma4:e4b`
- Python 3.12
- Node.js 18+

### Backend

```bash
cd backend
python -m venv venv312
venv312\Scripts\activate        # Windows
# source venv312/bin/activate   # Mac/Linux
pip install -r requirements.txt

cp .env.example .env
# Edit .env if your Ollama is not on localhost:11434

uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

### Optional — Pre-ingest a knowledge base
Drop curriculum PDFs into any directory and ingest via the `/api/rag/ingest` endpoint, or upload through the UI. The `knowledge_base` retrieval mode queries all ingested documents.

---

## Project Structure

```
medlearn-ai/
├── backend/
│   └── app/
│       ├── main.py          # FastAPI routes
│       ├── config.py        # Ollama URL, model name
│       ├── orchestrator.py  # Bloom's objective generation
│       ├── session.py       # Session state + summary
│       ├── teaching.py      # Dr. Mira Socratic streamer
│       ├── evaluator.py     # 5-dimension comprehension eval
│       ├── rag.py           # PDF ingest, ChromaDB, span judge
│       ├── tts.py           # Kokoro synthesis + word alignment
│       └── stt.py           # faster-whisper transcription
└── frontend/
    ├── app/
    │   ├── page.tsx         # Topic input + PDF upload
    │   └── session/
    │       └── page.tsx     # Avatar + chat + objectives
    ├── components/
    │   └── TalkingHeadAvatar.tsx  # Three.js TalkingHead wrapper
    └── lib/
        └── api.ts           # API client + SSE stream reader
```

---

## What Makes This Different

Most LLM-based education tools are wrappers around a chat interface. MedLearn AI is a pedagogical system:

- **Gemma 4 is not one agent — it is four**: an Orchestrator, a Judge, a Teacher, and an Evaluator, each with a distinct role, temperature, and output contract
- **It does not answer questions — it asks them.** The student must construct the understanding themselves
- **Evaluation is multi-dimensional**, not binary. Five independent axes distinguish a student who parroted a definition from one who genuinely reasoned through it
- **The feedback loop is invisible to the student** — gaps are targeted through the teacher's questions, not through correction
- **It runs on edge hardware** — the `e4b` quantisation means a laptop GPU (or even CPU) is sufficient

---

*MedLearn AI — because every student deserves a tutor who actually listens.*
