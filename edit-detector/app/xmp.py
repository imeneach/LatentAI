"""Layer 1: read exact Lightroom / Camera Raw settings from XMP (crs:*)."""
import json
import re
import shutil
import subprocess

_ATTR = re.compile(rb'crs:(\w+)="([^"]*)"')
_ELEM = re.compile(rb"<crs:(\w+)>([^<]*)</crs:\1>")

SLIDERS = [
    "Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012",
    "Blacks2012", "Temperature", "Tint", "Vibrance", "Saturation", "Clarity2012",
    "Texture", "Dehaze", "Sharpness", "GrainAmount", "PostCropVignetteAmount",
]


def _val(s: str):
    try:
        return float(s)
    except ValueError:
        return s


def read_crs(data: bytes) -> dict:
    """Regex fallback: works on any container that embeds the XMP packet."""
    a, b = data.find(b"<x:xmpmeta"), data.find(b"</x:xmpmeta>")
    if a < 0 or b < 0:
        return {}
    packet = data[a : b + 12]
    out = {}
    for rx in (_ATTR, _ELEM):
        for m in rx.finditer(packet):
            out[m.group(1).decode()] = _val(m.group(2).decode("latin1"))
    return out


def read_crs_exiftool(data: bytes) -> dict:
    if not shutil.which("exiftool"):
        return {}
    try:
        p = subprocess.run(["exiftool", "-j", "-XMP-crs:all", "-"], input=data,
                           capture_output=True, timeout=20)
        row = json.loads(p.stdout or b"[{}]")[0]
    except Exception:
        return {}
    row.pop("SourceFile", None)
    return row


def read_settings(data: bytes) -> dict:
    """Return only slider settings that are actually present."""
    raw = read_crs_exiftool(data) or read_crs(data)
    return {k: raw[k] for k in SLIDERS if k in raw}
