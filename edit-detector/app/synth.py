"""Synthetic scenes and degradations, shared by tests and benchmarks."""
import cv2
import numpy as np


def make_scene(seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    img = np.zeros((480, 640, 3), np.float32)
    yy, xx = np.mgrid[0:480, 0:640]
    for c in range(3):
        img[..., c] = 60 + 120 * (xx / 640) * r.random() + 80 * (yy / 480) * r.random()
    for _ in range(40):
        col = tuple(int(v) for v in r.integers(30, 230, 3))
        cv2.circle(img, (int(r.integers(0, 640)), int(r.integers(0, 480))), int(r.integers(10, 60)), col, -1)
        cv2.rectangle(img, tuple(int(v) for v in r.integers(0, 600, 2)),
                      tuple(int(v) for v in r.integers(40, 640, 2)), col, int(r.integers(1, 4)))
    img = cv2.GaussianBlur(img, (0, 0), 1.2) + r.normal(0, 2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def jpeg(img: np.ndarray, q: int = 90) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def crop_resize(img: np.ndarray, frac: float = 0.9, width: int = 480) -> np.ndarray:
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y, x = (h - ch) // 2, (w - cw) // 2
    c = img[y:y + ch, x:x + cw]
    return cv2.resize(c, (width, int(width * ch / cw)), interpolation=cv2.INTER_AREA)
