"""Accuracy benchmark on synthetic edits:  python -m scripts.benchmark [N]
Edits are rendered with our own forward model, then cropped, resized and JPEG-compressed,
so this measures the solver and alignment, NOT agreement with real Lightroom."""
import sys

import cv2
import numpy as np

from app import model
from app.align import align
from app.fit import fit
from app.imageio import decode
from app.synth import crop_resize, jpeg, make_scene


def main(n: int = 24):
    rng = np.random.default_rng(1)
    lim = np.array([1.5, 50, 50, 50, 30, 30, 50, 40, 30])
    errs, covered, failed = [], [], 0
    for i in range(n):
        o = make_scene(100 + i)
        truth = rng.uniform(-1, 1, 9) * lim
        e = (model.render(o[..., ::-1] / 255.0, truth)[..., ::-1] * 255).astype(np.uint8)
        e = decode(jpeg(crop_resize(e, 0.92, 480), 88))
        al = align(o, e)
        if not al.ok:
            failed += 1
            continue
        r = fit(al.original, al.edited, al.valid, calibrate=False)
        est = np.array([r["params"][k] for k in model.KEYS], float)
        se = np.array([r["uncertainty"][k] for k in model.KEYS], float)
        errs.append(est - truth)
        covered.append(np.abs(est - truth) <= 2 * np.maximum(se, 1e-6))
    E, C = np.abs(np.array(errs)), np.array(covered)
    print(f"{len(errs)} fitted, {failed} alignment failures")
    print(f"{'slider':26s} {'mean abs err':>12s} {'within 2 sigma':>15s}")
    for j, k in enumerate(model.KEYS):
        print(f"{k:26s} {E[:, j].mean():12.2f} {C[:, j].mean() * 100:14.0f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24)
