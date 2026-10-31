"""Learn a per-slider correction from DNG files only:  python -m scripts.calibrate_dng ./folder

For each NAME.dng in the folder:
  settings = the Lightroom XMP embedded in the DNG (or NAME.xmp next to it, if present)
  original = the raw sensor data developed with neutral settings (rawpy)
  edited   = the edited preview JPEG that Lightroom embeds in the DNG
Needs >= 5 samples per slider. Writes DATA_DIR/calibration.json, same format as scripts.calibrate.

Install first:  pip install rawpy opencv-python numpy

Caveat: "original" is NOT Lightroom's own rendering of the unedited photo (rawpy uses a
different baseline tone curve and color profile), so the learned scale/offset will be noisier
than with real Lightroom JPEG exports. Check the r2 values in the output.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from app import config, model
from app.align import align
from app.fit import fit
from app.imageio import decode
from app.xmp import read_settings

MAX_SIDE = 1024


def embedded_xmp(data: bytes):
    """Return the XMP packet inside the file that holds Lightroom (crs:) settings."""
    pos = 0
    while True:
        start = data.find(b"<x:xmpmeta", pos)
        if start < 0:
            return None
        end_tag = b"</x:xmpmeta>"
        end = data.find(end_tag, start)
        if end < 0:
            return None
        packet = data[start:end + len(end_tag)]
        if b"crs:" in packet:
            return packet
        pos = end + len(end_tag)


def shrink(bgr):
    h, w = bgr.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return bgr


def to_bytes(bgr) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()


def load_images(dng: Path):
    import rawpy

    with rawpy.imread(str(dng)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
        thumb = raw.extract_thumb()
    original = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if thumb.format == rawpy.ThumbFormat.JPEG:
        edited = cv2.imdecode(np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_COLOR)
    else:
        edited = cv2.cvtColor(np.asarray(thumb.data), cv2.COLOR_RGB2BGR)
    if edited is None:
        raise ValueError("could not decode the embedded preview")
    return shrink(original), shrink(edited)


def main(folder: str):
    pairs = {k: [] for k in model.KEYS}
    used = skipped = 0
    for dng in sorted(Path(folder).glob("*.dng")):
        sidecar = dng.with_suffix(".xmp")
        xmp = sidecar.read_bytes() if sidecar.exists() else embedded_xmp(dng.read_bytes())
        if not xmp:
            print("skipped", dng.stem, "- no Lightroom settings (crs:) found in XMP")
            skipped += 1
            continue
        try:
            truth = read_settings(xmp)
        except Exception as ex:  # noqa: BLE001
            print("skipped", dng.stem, "- cannot read settings:", ex)
            skipped += 1
            continue
        try:
            original, edited = load_images(dng)
        except Exception as ex:  # noqa: BLE001
            print("skipped", dng.stem, "- cannot read image data:", ex)
            skipped += 1
            continue

        diff = float(np.abs(cv2.resize(original, edited.shape[1::-1]).astype(float) - edited).mean())
        if diff < 2:
            print(f"warning {dng.stem}: original and edited look almost identical (mean diff {diff:.1f})")

        # the preview may be rotated differently from the raw; try the three rotations too
        al = None
        for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
            cand = edited if rot is None else cv2.rotate(edited, rot)
            al = align(decode(to_bytes(original)), decode(to_bytes(cand)))
            if al.ok:
                break
        if not al.ok:
            print("skipped", dng.stem, "- alignment failed:", al.reason)
            skipped += 1
            continue

        r = fit(al.original, al.edited, al.valid, calibrate=False)
        for k in model.KEYS:
            if k in truth:
                pairs[k].append((r["params"][k], float(truth[k])))
        used += 1
        print("ok", dng.stem)

    print(f"\n{used} photos used, {skipped} skipped")
    out = {}
    for k, v in pairs.items():
        if len(v) < 5:
            print(f"{k:20s} only {len(v)} samples (need 5) - not calibrated")
            continue
        x, y = np.array(v).T
        scale, offset = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (scale * x + offset)) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
        out[k] = {"scale": round(float(scale), 4), "offset": round(float(offset), 4),
                  "r2": round(float(r2), 3), "n": len(v)}
        print(f"{k:20s} scale {scale:6.3f} offset {offset:7.2f} r2 {r2:5.2f} n {len(v)}")
    (config.DATA_DIR / "calibration.json").write_text(json.dumps({"params": out}, indent=1))
    print(f"wrote {len(out)} calibrated sliders")


if __name__ == "__main__":
    main(sys.argv[1])
