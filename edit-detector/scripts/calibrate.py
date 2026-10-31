"""Learn a per-slider correction from real Lightroom exports:  python -m scripts.calibrate ./pairs

For each sample NAME put three files in the folder:
  NAME.jpg       the unedited original
  NAME_edit.jpg  the export after editing in Lightroom
  NAME.xmp       the settings used (XMP sidecar or preset)
Needs >= 5 samples per slider. Writes DATA_DIR/calibration.json, which the API then applies.
Temperature/tint cannot be calibrated this way (Lightroom stores Kelvin, we fit a relative shift)."""
import json
import sys
from pathlib import Path

import numpy as np

from app import config, model
from app.align import align
from app.fit import fit
from app.imageio import decode
from app.xmp import read_settings


def main(folder: str):
    pairs = {k: [] for k in model.KEYS}
    for xmp in sorted(Path(folder).glob("*.xmp")):
        o, e = Path(folder) / f"{xmp.stem}.jpg", Path(folder) / f"{xmp.stem}_edit.jpg"
        if not (o.exists() and e.exists()):
            continue
        truth = read_settings(xmp.read_bytes())
        al = align(decode(o.read_bytes()), decode(e.read_bytes()))
        if not al.ok:
            print("skipped", xmp.stem, al.reason)
            continue
        r = fit(al.original, al.edited, al.valid, calibrate=False)
        for k in model.KEYS:
            if k in truth:
                pairs[k].append((r["params"][k], float(truth[k])))
    out = {}
    for k, v in pairs.items():
        if len(v) < 5:
            continue
        x, y = np.array(v).T
        scale, offset = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (scale * x + offset)) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
        out[k] = {"scale": round(float(scale), 4), "offset": round(float(offset), 4), "r2": round(float(r2), 3), "n": len(v)}
        print(f"{k:20s} scale {scale:6.3f} offset {offset:7.2f} r2 {r2:5.2f} n {len(v)}")
    (config.DATA_DIR / "calibration.json").write_text(json.dumps({"params": out}, indent=1))
    print(f"wrote {len(out)} calibrated sliders")


if __name__ == "__main__":
    main(sys.argv[1])
