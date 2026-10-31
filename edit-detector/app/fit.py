"""Layer 2 fitting: given an aligned original/edited pair, recover global slider values.

Per-channel histogram matching gives a monotone tone curve for R, G, B and luma.
Slider values are read off those curves; the residual after applying the curves
tells us whether a global edit explains the image (low) or local/AI edits do (high).
"""
import cv2
import numpy as np

from . import calibration, model


def _lin(v):
    v = np.asarray(v, np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def cdf_lut(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ha = np.bincount(a, minlength=256).astype(np.float64)
    hb = np.bincount(b, minlength=256).astype(np.float64)
    A, B = np.cumsum(ha) / ha.sum(), np.cumsum(hb) / hb.sum()
    lut = np.searchsorted(B, A, side="left").clip(0, 255)
    return np.maximum.accumulate(lut).astype(np.uint8)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return np.rint(rgb @ np.array([0.2126, 0.7152, 0.0722])).astype(np.uint8)


def _clip(v, lo=-100, hi=100):
    return float(max(lo, min(hi, v)))


def fit(original_bgr: np.ndarray, edited_bgr: np.ndarray, valid: np.ndarray | None = None,
        calibrate: bool = True) -> dict:
    h, w = edited_bgr.shape[:2]
    mask = np.ones((h, w), bool) if valid is None else valid
    if mask.sum() < 500:
        mask = np.ones((h, w), bool)
    o_rgb = original_bgr[..., ::-1]
    e_rgb = edited_bgr[..., ::-1]
    o, e = o_rgb[mask], e_rgb[mask]

    luts = [cdf_lut(o[:, k], e[:, k]) for k in range(3)] + [cdf_lut(_luma(o), _luma(e))]
    Y = luts[3].astype(np.int32)

    # Residual of the global model, and a map of where it fails (local edits).
    pred = np.stack([luts[k][o_rgb[..., k]] for k in range(3)], -1)
    res_map = np.abs(pred.astype(np.int16) - e_rgb.astype(np.int16)).mean(-1) / 255.0
    residual = float(res_map[mask].mean())
    local_fraction = float((cv2.GaussianBlur(res_map.astype(np.float32), (0, 0), 7)[mask] > 0.08).mean())

    # Saturation ratio (HSV S) over the valid area.
    so = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)[..., 1][mask].mean()
    se = cv2.cvtColor(edited_bgr, cv2.COLOR_BGR2HSV)[..., 1][mask].mean()

    # Vignette: luminance error ratio, corners vs centre.
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / np.sqrt(2)
    pl, el = _luma(pred.astype(np.float64)).astype(np.float64) + 1, _luma(e_rgb.astype(np.float64)).astype(np.float64) + 1
    ratio = el / pl
    cm, km = mask & (r < 0.3), mask & (r > 0.8)
    vig = 0.0
    if cm.sum() > 200 and km.sum() > 200:
        vig = _clip(np.log(ratio[km].mean() / ratio[cm].mean()) * 150)

    params = {
        "Exposure2012": round(float(np.log2((_lin(Y[118]) + 1e-4) / (_lin(118) + 1e-4))), 2),
        "Contrast2012": round(_clip(((Y[192] - Y[64]) / 128 - 1) * 150)),
        "Highlights2012": round(_clip((Y[230] - Y[200] - 30) / 30 * 100)),
        "Shadows2012": round(_clip((Y[55] - Y[25] - 30) / 30 * 100)),
        "Whites2012": round(_clip((Y[250] - 250) * 4)),
        "Blacks2012": round(_clip(Y[6] * 4)),
        "Saturation": round(_clip((se / max(so, 1e-3) - 1) * 100)),
        # Relative shifts: valid as Lightroom Incremental* preset fields (not Kelvin).
        "IncrementalTemperature": round(_clip((int(luts[0][128]) - int(luts[2][128])) / 128 * 300)),
        "IncrementalTint": round(_clip((int(luts[1][128]) - (int(luts[0][128]) + int(luts[2][128])) / 2) / 128 * -300)),
        "PostCropVignetteAmount": round(vig),
    }
    # Analysis-by-synthesis fit replaces the heuristic curve read-outs for the core sliders.
    # Fit on a lightly blurred copy and on flat areas only: this removes sub-pixel
    # alignment error, sharpening/clarity, grain and JPEG noise, which are NOT slider
    # edits but otherwise inflate the error and drag confidence down.
    ob = cv2.GaussianBlur(o_rgb, (0, 0), 1.5)
    eb = cv2.GaussianBlur(e_rgb, (0, 0), 1.5)
    g = cv2.GaussianBlur(cv2.cvtColor(np.ascontiguousarray(o_rgb), cv2.COLOR_RGB2GRAY), (0, 0), 1.5)
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    flat = mask & (grad <= np.percentile(grad[mask], 60))
    if flat.sum() < 2000:
        flat = mask
    mf = model.fit_model(ob[flat], eb[flat], r[flat] / r.max())
    raw, calibrated = calibration.apply(mf["params"]) if calibrate else (mf["params"], False)
    uncertainty = {}
    for k in model.ALL_KEYS:
        d = 2 if k == "Exposure2012" else 0
        params[k] = round(float(raw[k]), d) if d else int(round(float(raw[k])))
        uncertainty[k] = round(mf["se"][k], d)
    return {
        "params": params, "uncertainty": uncertainty, "calibrated": calibrated,
        "model_rmse": round(mf["rmse"], 4), "quality": float(np.exp(-mf["rmse"] / 0.06)),
        "residual": round(residual, 4),
        "local_fraction": round(local_fraction, 3),
        "curves": {n: l.tolist() for n, l in zip(["r", "g", "b", "luma"], luts)},
        "luts": luts,
    }