from pydantic import BaseModel


class Exports(BaseModel):
    xmp: str
    cube: str | None = None


class AnalyzeResponse(BaseModel):
    layer: str                       # "metadata" | "fitted" | "estimate"
    exact: bool
    confidence: float                # 0..1
    params: dict[str, float | str]   # Lightroom/Camera Raw crs:* names
    exports: Exports
    warnings: list[str] = []
    reason: str | None = None        # why the pipeline fell back, if it did
    residual: float | None = None
    local_fraction: float | None = None
    unrecognized_pipeline: bool | None = None
    match: dict | None = None
    alignment: dict | None = None
    curves: dict[str, list[int]] | None = None
    uncertainty: dict[str, float] | None = None   # approx. 1-sigma per slider
    calibrated: bool | None = None                # corrected against real Lightroom exports
    timings_ms: dict[str, int] | None = None
