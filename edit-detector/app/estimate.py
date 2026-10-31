"""Layer 3 placeholder: heuristic estimate when no original exists.
Replace with a trained regressor (see README, 'Next: ML estimator')."""
import cv2
import numpy as np


def estimate(bgr: np.ndarray) -> dict:
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (256, max(1, round(256 * h / w))), interpolation=cv2.INTER_AREA)
    rgb = small[..., ::-1].astype(np.float32) / 255
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    p5, p50, p95 = np.percentile(luma, [5, 50, 95])
    clip = float(((luma < 0.012) | (luma > 0.988)).mean())
    sat = float((cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1] / 255).mean())
    R, G, B = rgb.reshape(-1, 3).mean(0)
    cl = lambda v: max(-1.0, min(1.0, float(v)))
    return {
        "params": {
            "Exposure2012": round(cl(np.log2(max(0.02, p50) / 0.45) / 3) * 3, 2),
            "Contrast2012": round(cl(((p95 - p5) - 0.72) * 3) * 100),
            "Saturation": round(cl((sat - 0.32) * 3) * 100),
            "IncrementalTemperature": round(cl((R - B) / (R + B + 1e-6) * 5) * 100),
            "IncrementalTint": round(cl((G - (R + B) / 2) / (G + 1e-6) * -5) * 100),
        },
        "confidence": round(max(0.08, 0.3 * (1 - min(0.7, clip * 6))), 2),
    }
