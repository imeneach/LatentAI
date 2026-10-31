import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
ORIGINALS_DIR = DATA_DIR / "originals"
ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDER = os.getenv("EMBEDDER", "tiny")                 # "tiny" (no deps) or "clip"
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.80"))  # cosine similarity
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
CORS_ORIGINS = [o for o in os.getenv("CORS_ORIGINS", "*").split(",") if o]
