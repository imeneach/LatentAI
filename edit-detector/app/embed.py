"""Fingerprinting. 'tiny' needs only OpenCV; 'clip' is semantic and crop-robust."""
from functools import lru_cache

import cv2
import numpy as np

from . import config


class TinyEmbedder:
    """Equalised 32x32 grayscale thumbnail. Robust to tone/colour edits, NOT to crops."""
    name, dim = "tiny", 32 * 32

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.equalizeHist(cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA))
        g = g.astype(np.float32)
        g -= g.mean()
        return (g / (np.linalg.norm(g) + 1e-8)).ravel()


class ClipEmbedder:
    name, dim = "clip", 512

    def __init__(self):
        import open_clip
        import torch
        self.torch = torch
        self.model, _, self.pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.model.eval()

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        from PIL import Image
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with self.torch.no_grad():
            v = self.model.encode_image(self.pre(im).unsqueeze(0))[0]
        return (v / v.norm()).numpy().astype(np.float32)


@lru_cache(maxsize=1)
def get_embedder():
    return ClipEmbedder() if config.EMBEDDER == "clip" else TinyEmbedder()
