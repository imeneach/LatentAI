"""Forward model of the main Lightroom-style sliders + analysis-by-synthesis fitting.

Instead of reading slider values off a tone curve with fixed formulas, we render the
original with candidate slider values and optimise them until the render matches the
edit. The solver's Jacobian also gives an uncertainty per slider, so correlated sliders
(e.g. Highlights vs Whites) are reported honestly with wide error bars.

NOTE: Lightroom's real maths is proprietary. This is our own approximation, so absolute
values are model-relative until calibrated on real exports (see scripts/calibrate.py).
"""
import numpy as np
from scipy.optimize import least_squares

KEYS = ["Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012",
        "Blacks2012", "Saturation", "IncrementalTemperature", "IncrementalTint"]
# Extra sliders (v0.3): Vibrance (saturation that protects already-saturated colours) and
# the post-crop vignette (needs pixel position). KEYS keeps its old order for compatibility.
ALL_KEYS = KEYS + ["Vibrance", "PostCropVignetteAmount"]
N_CORE, N_ALL = len(KEYS), len(ALL_KEYS)
_SCALE = np.array([1.0] + [50.0] * (N_ALL - 1))
_LO = np.array([-5.0] + [-100.0] * (N_ALL - 1))
_HI = np.array([5.0] + [100.0] * (N_ALL - 1))
PRIOR = 0.1
# Empirical error floor per slider (from scripts/benchmark.py): the Jacobian alone is
# over-confident because crop, JPEG and model mismatch add systematic error.
_SE_FLOOR = np.array([0.04, 4, 14, 7, 17, 14, 6, 4, 7, 8, 6])
_LUMA = np.array([0.2126, 0.7152, 0.0722])


def _ss(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _to_lin(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _to_srgb(v):
    v = np.clip(v, 0, 1)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


def render(rgb: np.ndarray, p, r=None) -> np.ndarray:
    """rgb: float sRGB in [0,1], shape (...,3). p: slider values in ALL_KEYS order
    (the first 9 alone are accepted too). r: normalised radius 0..1 of each pixel from
    the image centre, shape (...), only needed for the vignette."""
    p = np.concatenate([np.asarray(p, float), np.zeros(N_ALL)])[:N_ALL]
    e, c, h, s, w, b, sat, t, ti, vib, vig = p
    gains = np.array([2 ** (t / 100 * 0.35), 2 ** (-ti / 100 * 0.25), 2 ** (-t / 100 * 0.35)])
    lin = _to_lin(rgb) * gains * 2 ** e
    if r is not None and vig:
        # Negative amount darkens the corners (smooth radial falloff, flat inside r<0.3).
        lin = lin * (2 ** (vig / 100 * 1.6 * _ss(np.asarray(r), 0.3, 1.0)))[..., None]
    x = _to_srgb(lin)
    y = np.maximum(x @ _LUMA, 1e-3)
    y2 = (y - 0.5) * (1 + 0.8 * c / 100) + 0.5
    y2 = y2 + 0.25 * h / 100 * _ss(y, 0.5, 1) + 0.25 * s / 100 * (1 - _ss(y, 0, 0.5)) \
        + 0.2 * w / 100 * _ss(y, 0.75, 1) + 0.2 * b / 100 * (1 - _ss(y, 0, 0.25))
    x = np.clip(x * (np.clip(y2, 0, 1) / y)[..., None], 0, 1)
    yl = (x @ _LUMA)[..., None]
    # Vibrance boosts muted colours more than vivid ones; Saturation is uniform.
    chroma = (x.max(-1) - x.min(-1)) / np.maximum(x.max(-1), 1e-3)
    k = (1 + sat / 100) * (1 + vib / 100 * (1 - chroma))[..., None]
    return np.clip(yl + (x - yl) * k, 0, 1)


def _sample(rgb: np.ndarray, max_px: int) -> np.ndarray:
    """Pick pixels evenly across the brightness range (not evenly across the image), so the
    few very dark / very bright pixels that pin down Blacks, Shadows, Highlights and Whites
    are not drowned out by the mid-tones."""
    n = len(rgb)
    if n <= max_px:
        return np.arange(n)
    luma = (rgb.astype(np.float32) / 255) @ _LUMA.astype(np.float32)
    bins = np.minimum((luma * 10).astype(int), 9)
    per, picks = max_px // 10, []
    for b in range(10):
        ids = np.flatnonzero(bins == b)
        if len(ids):
            picks.append(ids[np.linspace(0, len(ids) - 1, min(len(ids), per)).astype(int)])
    idx = np.concatenate(picks)
    if len(idx) < max_px:                      # top up from the rest, evenly
        rest = np.setdiff1d(np.arange(n), idx, assume_unique=False)
        idx = np.concatenate([idx, rest[np.linspace(0, len(rest) - 1, min(len(rest), max_px - len(idx))).astype(int)]])
    return np.sort(idx)


def fit_model(orig_rgb: np.ndarray, edit_rgb: np.ndarray, r=None, max_px: int = 6000) -> dict:
    """orig_rgb, edit_rgb: uint8 (N,3) arrays of corresponding pixels (or float 0..255).
    r: optional (N,) radius of each pixel from the image centre (0..1); enables the vignette fit."""
    idx = _sample(orig_rgb, max_px)
    o = orig_rgb[idx].astype(np.float64) / 255
    e = edit_rgb[idx].astype(np.float64) / 255
    rr = None if r is None else np.asarray(r, np.float64)[idx]
    lo, hi = _LO / _SCALE, _HI / _SCALE
    if rr is None:           # no positions: the vignette slider cannot be observed
        lo, hi = lo.copy(), hi.copy()
        lo[-1], hi[-1] = -1e-9, 1e-9

    def data(x):
        return (render(o, x * _SCALE, rr) - e).ravel()

    def resid(x):  # weak ridge prior pulls poorly-determined sliders toward 0
        return np.concatenate([data(x), PRIOR * x])

    # Multi-start: flat zero start plus a start whose exposure matches the median
    # brightness change. Keeps the better optimum when the first one is a poor local minimum.
    med = lambda a: max(float(np.median(_to_lin(a) @ _LUMA)), 1e-4)
    x1 = np.zeros(N_ALL)
    x1[0] = float(np.clip(np.log2(med(e) / med(o)), -4, 4)) / _SCALE[0]
    best = None
    for x0 in (np.zeros(N_ALL), x1):
        res = least_squares(resid, np.clip(x0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), loss="soft_l1",
                            f_scale=0.03, diff_step=1e-3, max_nfev=80)
        if best is None or res.cost < best.cost:
            best = res
    res = best
    d2 = np.sort(data(res.x) ** 2)
    # Trimmed RMSE: ignore the worst 15% of pixels (local brushes, clipped areas, objects
    # that moved) so a few outliers do not zero the confidence of an otherwise good fit.
    rmse = float(np.sqrt(np.mean(d2[: max(1, int(len(d2) * 0.85))])))
    J = res.jac
    cov = np.linalg.pinv(J.T @ J) * max(rmse, 1e-4) ** 2
    se = np.sqrt(np.clip(np.diag(cov), 0, None) * _SCALE ** 2 + _SE_FLOOR ** 2)
    p = res.x * _SCALE
    return {"params": dict(zip(ALL_KEYS, map(float, p))), "se": dict(zip(ALL_KEYS, map(float, se))), "rmse": rmse}