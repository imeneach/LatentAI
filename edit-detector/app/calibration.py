"""Per-slider linear correction (scale, offset) learned from real Lightroom exports."""
import json

from . import config


def _path():
    return config.DATA_DIR / "calibration.json"


def load() -> dict:
    p = _path()
    return json.loads(p.read_text()) if p.exists() else {}


def apply(params: dict) -> tuple[dict, bool]:
    out, used = dict(params), False
    for k, c in load().get("params", {}).items():
        if k in out and c.get("n", 0) >= 5 and c.get("r2", 0) >= 0.5:
            out[k] = c["scale"] * out[k] + c["offset"]
            used = True
    return out, used


def summary() -> dict:
    cal = load().get("params", {})
    return {"calibrated_sliders": sorted(k for k, c in cal.items() if c.get("n", 0) >= 5 and c.get("r2", 0) >= 0.5),
            "samples": max([c.get("n", 0) for c in cal.values()] or [0])}
