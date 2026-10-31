"""Vector index of known originals. Exact cosine search; uses FAISS when installed."""
import json
import threading
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None


class VectorIndex:
    def __init__(self, directory: Path, dim: int, name: str):
        self.dir, self.dim = Path(directory), dim
        self.vec_path = self.dir / f"{name}.vectors.npy"
        self.meta_path = self.dir / f"{name}.meta.json"
        self.lock = threading.Lock()
        self.vecs = np.zeros((0, dim), np.float32)
        self.meta: list[dict] = []
        if self.vec_path.exists() and self.meta_path.exists():
            self.vecs = np.load(self.vec_path)
            self.meta = json.loads(self.meta_path.read_text())
        self._faiss = None
        self._rebuild()

    def _rebuild(self):
        if faiss is not None and len(self.vecs):
            self._faiss = faiss.IndexFlatIP(self.dim)
            self._faiss.add(self.vecs)
        else:
            self._faiss = None

    def __len__(self):
        return len(self.meta)

    def has(self, key: str) -> dict | None:
        return next((m for m in self.meta if m["id"] == key), None)

    def add(self, vec: np.ndarray, meta: dict):
        with self.lock:
            if self.has(meta["id"]):
                return
            self.vecs = np.vstack([self.vecs, vec.astype(np.float32)[None]])
            self.meta.append(meta)
            np.save(self.vec_path, self.vecs)
            self.meta_path.write_text(json.dumps(self.meta))
            self._rebuild()

    def search(self, vec: np.ndarray, k: int = 5) -> list[dict]:
        if not len(self):
            return []
        k = min(k, len(self))
        q = vec.astype(np.float32)[None]
        if self._faiss is not None:
            scores, ids = self._faiss.search(q, k)
            scores, ids = scores[0], ids[0]
        else:
            s = self.vecs @ q[0]
            ids = np.argsort(-s)[:k]
            scores = s[ids]
        return [{**self.meta[i], "score": float(sc)} for i, sc in zip(ids, scores)]
