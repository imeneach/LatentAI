# Edit Detector API

Shazam for photo edits: fingerprint an upload, find its original, and fit the slider settings that turn one into the other.

```bash
pip install -r requirements-dev.txt
pytest                                   # 6 tests on synthetic edits
uvicorn app.main:app --reload            # UI: http://localhost:8000/   API docs: /docs
python -m scripts.index_folder ./originals "my-licensed-set"
```

Optional: `pip install -r requirements-clip.txt` and `EMBEDDER=clip` for semantic, crop-robust search and FAISS. `exiftool` on the PATH is used automatically when present; otherwise a regex XMP reader is used.

## Interface
`web/` is a React (Vite) app. A production build is already in `web/dist`, and the server serves it at `http://localhost:8000/`, so Node is NOT needed to use it. To change the design: `cd web && npm install && npm run dev` (hot reload on http://localhost:5173, proxied to the API), then `npm run build` and restart the server.

## Endpoints (all under `/api/v1`, also available at the root for older clients)
| Route | Purpose |
|---|---|
| `POST /metadata` | Exact `crs:*` settings from XMP, if the file still has them |
| `POST /originals` (alias `/index`) | Add a licensed original to the search index |
| `POST /search` | Candidate originals for an upload, with cosine scores |
| `GET /fetch?url=` | Download an image from a direct URL for the UI (blocks internal addresses, nothing stored) |
| `POST /analyze` (alias `/fit`) | Full pipeline: metadata, then original (provided or found) then align then fit, else heuristic estimate. Returns params, confidence, residual, `unrecognized_pipeline`, tone curves, XMP preset and `.cube` LUT |

## How `/fit` decides
1. XMP present: return exact values (confidence 1.0).
2. Original provided, or index match above `MATCH_THRESHOLD`: SIFT + RANSAC homography aligns the original to the edit (handles crop and resize), then per-channel histogram matching yields tone curves, from which sliders are read. High residual or large unexplained regions set `unrecognized_pipeline`.
3. Otherwise a heuristic estimate with confidence <= 0.3.

## Accuracy
Core sliders are recovered by analysis-by-synthesis (`app/model.py`): render the original with candidate slider values and optimise until it matches the edit, with a weak prior and per-slider uncertainty (`uncertainty` in the response; shown as ± in the UI).

Benchmark (`python -m scripts.benchmark 24`): synthetic edits made with the same model, then cropped 8%, resized and JPEG-compressed at quality 88. This measures the solver and alignment only, not agreement with real Lightroom, and the uncertainty floors were tuned on this same data, so coverage is optimistic.
```
24 fitted, 0 alignment failures
slider                     mean abs err  within 2 sigma
Exposure2012                       0.02            100%
Contrast2012                       2.08            100%
Highlights2012                     7.91             96%
Shadows2012                        4.55            100%
Whites2012                        10.56            100%
Blacks2012                        11.34            100%
Saturation                         4.09            100%
IncrementalTemperature             2.17             96%
IncrementalTint                    5.22             96%
```
Exposure, contrast, saturation and warmth are well determined; highlights, whites and blacks overlap in their effect, so their errors and error bars are larger by nature.

**Making it accurate against real Lightroom:** export 20+ pairs (original, edited export, XMP settings), put them in a folder as `NAME.jpg`, `NAME_edit.jpg`, `NAME.xmp`, and run `python -m scripts.calibrate ./pairs`. The API then applies a learned scale and offset per slider (`/api/v1/health` shows which sliders are calibrated).

## Known limits
- Default `tiny` embedder is not crop-robust; use CLIP for real data.
- Absolute slider values follow our own forward model of Lightroom's sliders. Until you run `scripts.calibrate` on real exports they are close approximations, not Lightroom's exact numbers.
- Fitted values are global approximations; sliders interact, so treat them as a starting point.
- Temperature and tint are relative (`IncrementalTemperature/Tint`), not Kelvin.
- Uploaded edits are never stored. Index only images you have rights to.

## Next
ML estimator trained on synthetic edits (MIT-Adobe FiveK + randomized pipeline), larger licensed preset/LUT library, confidence calibration on degraded images, Celery/Redis job queue, Postgres + pgvector for scale.


Errors always look like `{"error": {"code": 400, "message": "..."}}`. Interactive docs: `/docs`.
