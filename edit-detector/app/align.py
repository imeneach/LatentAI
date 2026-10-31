"""Register the original onto the edited frame (handles crop, resize, small rotation)."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .imageio import resize_max


@dataclass
class AlignResult:
    ok: bool
    reason: str = ""
    edited: np.ndarray | None = None      # edited at working resolution
    original: np.ndarray | None = None    # original warped into the edited frame
    valid: np.ndarray | None = None       # bool mask of pixels covered by the original
    inliers: int = 0
    matches: int = 0
    crop_fraction: float = 1.0            # share of the original's area visible in the edit
    homography: list = field(default_factory=list)


def align(orig: np.ndarray, edit: np.ndarray, side: int = 1024) -> AlignResult:
    e, o = resize_max(edit, side), resize_max(orig, side)
    clahe = cv2.createCLAHE(2.0, (8, 8))   # equalise so tone edits don't break matching
    ge = clahe.apply(cv2.cvtColor(e, cv2.COLOR_BGR2GRAY))
    go = clahe.apply(cv2.cvtColor(o, cv2.COLOR_BGR2GRAY))
    sift = cv2.SIFT_create(4000)
    ko, do = sift.detectAndCompute(go, None)
    ke, de = sift.detectAndCompute(ge, None)
    if do is None or de is None or len(ko) < 12 or len(ke) < 12:
        return AlignResult(False, "too_few_features")
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(do, de, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 12:
        return AlignResult(False, "too_few_matches", matches=len(good))
    src = np.float32([ko[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([ke[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    inl = int(mask.sum()) if mask is not None else 0
    if H is None or inl < 12:
        return AlignResult(False, "homography_failed", inliers=inl, matches=len(good))
    h, w = e.shape[:2]
    warped = cv2.warpPerspective(o, H, (w, h), flags=cv2.INTER_LINEAR)
    cover = cv2.warpPerspective(np.full(o.shape[:2], 255, np.uint8), H, (w, h))
    valid = cv2.erode(cover, np.ones((5, 5), np.uint8)) > 250
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    back = cv2.perspectiveTransform(corners, np.linalg.inv(H)).reshape(-1, 2)
    crop = float(abs(cv2.contourArea(back.astype(np.float32))) / (o.shape[0] * o.shape[1]))
    return AlignResult(True, "", e, warped, valid, inl, len(good), min(crop, 1.0), H.tolist())
