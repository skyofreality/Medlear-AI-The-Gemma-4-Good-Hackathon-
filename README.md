# MedLearn AI

AI-powered medical education platform with an interactive avatar tutor.

## Prerequisites

- Node.js 18+
- Python 3.12

---

## Setup

### Terminal 1 — Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs at http://localhost:3000

### Terminal 2 — Backend

```bash
cd backend
./venv312/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Runs at http://localhost:8000

> **Windows cmd:** use `venv312\Scripts\activate` (backslash)

---

## Environment Variables

Copy and fill in secrets:

```bash
cp backend/.env.example backend/.env
```

If no `.env.example` exists, create `backend/.env` with required keys (API keys, DB config, etc.).

---

## Project Structure

```
medlearn-ai/
├── frontend/   # Next.js app
└── backend/    # FastAPI app
    ├── app/
    │   └── main.py
    ├── requirements.txt
    └── venv312/
```
