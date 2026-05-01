import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = OLLAMA_BASE_URL + "/api/chat"
OLLAMA_TAGS_URL = OLLAMA_BASE_URL + "/api/tags"

MODEL = os.getenv("MEDLEARN_MODEL", "gemma4:e4b")

CORS_ORIGIN = os.getenv("CORS_ORIGIN", "http://localhost:3000")
