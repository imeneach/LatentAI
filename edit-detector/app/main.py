"""Edit Detector API.

Pipeline for POST /fit:
  1. exact XMP metadata  ->  2. provided original or index match -> align -> fit
  3. heuristic estimate (placeholder for a trained model)

Privacy: uploaded edited images are processed in memory and never stored.
Only originals that the operator explicitly adds via /index are saved.
"""
import hashlib
import logging
import time
import ipaddress
import socket
import urllib.parse
from pathlib import Path

import httpx
import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import config
from . import calibration
from .align import align
from .embed import get_embedder
from .estimate import estimate
from .export import cube, xmp_preset
from .fit import fit
from .imageio import decode
from .index import VectorIndex
from .schemas import AnalyzeResponse
from .xmp import read_settings

log = logging.getLogger("edit_detector")
app = FastAPI(
    title="Edit Detector API", version="0.2.0",
    description="Find out how a photo was edited. `POST /api/v1/analyze` runs the full pipeline: "
                "exact metadata, then original match and fit, then a heuristic estimate.",
    openapi_tags=[{"name": "analysis", "description": "Detect and fit an edit"},
                  {"name": "library", "description": "Known originals used for matching"},
                  {"name": "utility", "description": "Helpers"}])
router = APIRouter()
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])

_embedder = get_embedder()
index = VectorIndex(config.DATA_DIR, _embedder.dim, _embedder.name)


async def _read(f: UploadFile) -> bytes:
    data = await f.read()
    if len(data) > config.MAX_UPLOAD:
        raise HTTPException(413, "File too large")
    return data


def _img(data: bytes) -> np.ndarray:
    try:
        return decode(data)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.exception_handler(StarletteHTTPException)
async def http_error(request, exc):
    """Every error has the same shape: {"error": {"code", "message"}} (plus legacy "detail")."""
    return JSONResponse({"error": {"code": exc.status_code, "message": str(exc.detail)}, "detail": exc.detail},
                        status_code=exc.status_code)


def _check_url(url: str):
    """Block non-http(s) and internal addresses (SSRF guard)."""
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise HTTPException(400, "Only http(s) URLs are allowed")
    try:
        infos = socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80))
    except socket.gaierror:
        raise HTTPException(400, "Host not found")
    for i in infos:
        ip = ipaddress.ip_address(i[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise HTTPException(400, "Address not allowed")


@router.get("/fetch", tags=["utility"], summary="Load an image from a direct URL")
def fetch_image(url: str):
    """Download an image from a direct URL so the browser UI can analyse it (avoids CORS).
    The image is streamed back and never stored. Note: the DNS check is best-effort;
    run behind an egress firewall in production."""
    cur = url
    with httpx.Client(timeout=10, follow_redirects=False, headers={"User-Agent": "EditDetector/0.1"}) as c:
        for _ in range(4):
            _check_url(cur)
            with c.stream("GET", cur) as r:
                if r.is_redirect:
                    cur = urllib.parse.urljoin(cur, r.headers.get("location", ""))
                    continue
                if r.status_code != 200:
                    raise HTTPException(502, f"Remote server returned {r.status_code}")
                ctype = r.headers.get("content-type", "").split(";")[0].strip()
                if not ctype.startswith("image/"):
                    raise HTTPException(415, "That URL is not an image")
                buf = bytearray()
                for chunk in r.iter_bytes():
                    buf += chunk
                    if len(buf) > config.MAX_UPLOAD:
                        raise HTTPException(413, "Image too large")
                return Response(bytes(buf), media_type=ctype)
    raise HTTPException(400, "Too many redirects")


@router.get("/health", tags=["utility"])
def health():
    return {"ok": True, "embedder": _embedder.name, "indexed": len(index), **calibration.summary()}


@router.post("/metadata", tags=["analysis"], summary="Read exact settings from file metadata")
async def metadata(image: UploadFile = File(...)):
    """Layer 1: exact settings from XMP, if the file still carries them."""
    s = read_settings(await _read(image))
    return {"found": bool(s), "settings": s}


@router.post("/originals", tags=["library"], summary="Add a licensed original to the index")
@router.post("/index", include_in_schema=False)
async def add_original(image: UploadFile = File(...), source: str = Form("")):
    """Add a known/licensed original to the search index."""
    data = await _read(image)
    key = hashlib.sha256(data).hexdigest()[:32]
    if index.has(key):
        return {"id": key, "added": False}
    img = _img(data)
    path = config.ORIGINALS_DIR / f"{key}.jpg"
    path.write_bytes(data)
    index.add(_embedder.embed(img), {"id": key, "path": str(path), "source": source})
    return {"id": key, "added": True, "total": len(index)}


@router.post("/search", tags=["library"], summary="Find candidate originals")
async def search(image: UploadFile = File(...), k: int = Form(5)):
    """Layer 2a: find candidate originals for an upload."""
    hits = index.search(_embedder.embed(_img(await _read(image))), k)
    return {"threshold": config.MATCH_THRESHOLD,
            "matches": [{"id": h["id"], "source": h["source"], "score": round(h["score"], 4)} for h in hits]}


def _estimate_response(img, reason, match=None):
    e = estimate(img)
    return {"layer": "estimate", "exact": False, "reason": reason, "match": match,
            "params": e["params"], "confidence": e["confidence"],
            "warnings": ["No original was found, so these values are rough guesses."],
            "exports": {"xmp": xmp_preset(e["params"])}}


@router.post("/analyze", tags=["analysis"], summary="Detect how a photo was edited",
             response_model=AnalyzeResponse, response_model_exclude_none=True)
@router.post("/fit", include_in_schema=False, response_model=AnalyzeResponse, response_model_exclude_none=True)
async def fit_edit(
    edited: UploadFile = File(...),
    original: UploadFile | None = File(None),
    force_fit: bool = Form(False),
    include_cube: bool = Form(True),
):
    data = await _read(edited)
    edit_img = _img(data)

    exact = read_settings(data)
    if exact and not force_fit:
        return {"layer": "metadata", "exact": True, "params": exact, "confidence": 1.0,
                "exports": {"xmp": xmp_preset(dict(exact))}}

    orig_img, match = None, None
    if original is not None:
        orig_img, match = _img(await _read(original)), {"source": "provided"}
    else:
        hits = index.search(_embedder.embed(edit_img), 3)
        if hits and hits[0]["score"] >= config.MATCH_THRESHOLD:
            from pathlib import Path
            orig_img = decode(Path(hits[0]["path"]).read_bytes())
            match = {"source": "index", "id": hits[0]["id"], "origin": hits[0]["source"],
                     "score": round(hits[0]["score"], 4)}

    if orig_img is None:
        return _estimate_response(edit_img, "no_original_found")

    t0 = time.perf_counter()
    al = align(orig_img, edit_img)
    t1 = time.perf_counter()
    if not al.ok:
        return _estimate_response(edit_img, f"alignment_failed:{al.reason}", match)

    r = fit(al.original, al.edited, al.valid)
    t2 = time.perf_counter()
    # Quality of the model fit x alignment support; uncalibrated values are model-relative.
    conf = r["quality"] * min(1.0, al.inliers / 40) * (1.0 if r["calibrated"] else 0.85)
    conf = round(max(0.1, min(0.95, conf)), 2)
    warns = []
    if r["residual"] > 0.07 or r["local_fraction"] > 0.25:
        warns.append("Local or AI edits are likely, so sliders explain only part of the change.")
    if al.crop_fraction < 0.9:
        warns.append("The photo was cropped, which lowers accuracy.")
    out = {
        "warnings": warns,
        "layer": "fitted", "exact": False, "match": match, "params": r["params"],
        "confidence": conf, "residual": r["residual"],
        "local_fraction": r["local_fraction"],
        "unrecognized_pipeline": bool(r["residual"] > 0.07 or r["local_fraction"] > 0.25),
        "alignment": {"inliers": al.inliers, "matches": al.matches,
                      "crop_fraction": round(al.crop_fraction, 3)},
        "curves": r["curves"], "uncertainty": r["uncertainty"], "calibrated": r["calibrated"],
        "timings_ms": {"align": round((t1 - t0) * 1000), "fit": round((t2 - t1) * 1000)},
        "exports": {"xmp": xmp_preset(r["params"])},
    }
    log.info("analyze layer=fitted conf=%.2f rmse=%.4f inliers=%d align_ms=%d fit_ms=%d",
             conf, r["model_rmse"], al.inliers, out["timings_ms"]["align"], out["timings_ms"]["fit"])
    if include_cube:
        out["exports"]["cube"] = cube(r["luts"])
    return out


# Routes live under /api/v1 (documented) and at the root (legacy, hidden from the docs).
app.include_router(router, prefix="/api/v1")
app.include_router(router, include_in_schema=False)

# Serve the single-file site in app/static (index.html + analyze.html).
# NOTE: the old React build in web/dist used to take priority here, which is why
# edits to app/static/index.html never showed up. It is no longer used.
_root = Path(__file__).resolve().parent
app.mount("/", StaticFiles(directory=_root / "static", html=True), name="ui")
