import os, tempfile
os.environ["DATA_DIR"] = tempfile.mkdtemp()

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.align import align
from app.fit import fit
from app.main import app
from app.synth import crop_resize, jpeg, make_scene
from app.xmp import read_crs

rng = np.random.default_rng(7)


def apply_edit(bgr, exposure=0.5, contrast=1.25, sat=1.3, warm=1.08):
    x = bgr.astype(np.float32) / 255
    x = x * 2 ** exposure
    x = (x - 0.5) * contrast + 0.5
    hsv = cv2.cvtColor(np.clip(x, 0, 1), cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 1)
    x = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    x[..., 2] *= warm; x[..., 0] /= warm
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def test_read_crs_regex():
    blob = b'junk<x:xmpmeta><rdf:Description crs:Exposure2012="+0.50" crs:Temperature="5400"/></x:xmpmeta>junk'
    assert read_crs(blob) == {"Exposure2012": 0.5, "Temperature": 5400.0}


def test_align_recovers_crop():
    orig = make_scene(1)
    edited = crop_resize(apply_edit(orig))
    al = align(orig, edited)
    assert al.ok and al.inliers > 30
    assert 0.7 < al.crop_fraction < 0.95


def test_fit_signs_and_residual():
    orig = make_scene(2)
    edited = crop_resize(apply_edit(orig))
    al = align(orig, edited)
    r = fit(al.original, al.edited, al.valid)
    p = r["params"]
    assert p["Exposure2012"] > 0.2
    assert p["Contrast2012"] > 5
    assert p["Saturation"] > 5
    assert p["IncrementalTemperature"] > 0
    assert r["residual"] < 0.06


def test_local_edit_is_flagged():
    orig = make_scene(3)
    edited = apply_edit(orig, 0.2, 1.1, 1.0, 1.0)
    edited[100:380, 150:500] = 255 - edited[100:380, 150:500]  # big non-global change
    al = align(orig, edited)
    r = fit(al.original, al.edited, al.valid)
    assert r["residual"] > 0.07 or r["local_fraction"] > 0.25


def test_api_search_and_fit_via_index():
    client = TestClient(app)
    scenes = [make_scene(s) for s in range(10, 16)]
    for s in scenes:
        assert client.post("/index", files={"image": ("o.jpg", jpeg(s), "image/jpeg")},
                           data={"source": "test"}).json()["added"]
    edited = jpeg(crop_resize(apply_edit(scenes[3], 0.4, 1.2, 1.2, 1.05), frac=1.0, width=480))
    hits = client.post("/search", files={"image": ("e.jpg", edited, "image/jpeg")}).json()["matches"]
    assert hits[0]["score"] > 0.8
    out = client.post("/fit", files={"edited": ("e.jpg", edited, "image/jpeg")}).json()
    assert out["layer"] == "fitted" and out["match"]["source"] == "index"
    assert out["params"]["Exposure2012"] > 0.1
    assert "LUT_3D_SIZE 17" in out["exports"]["cube"] and "<x:xmpmeta" in out["exports"]["xmp"]


def test_api_unknown_image_falls_back_to_estimate():
    client = TestClient(app)
    out = client.post("/fit", files={"edited": ("e.jpg", jpeg(make_scene(99)), "image/jpeg")}).json()
    assert out["layer"] == "estimate" and out["confidence"] <= 0.3


def test_fetch_blocks_internal_and_bad_urls():
    client = TestClient(app)
    for u in ["http://127.0.0.1/x.jpg", "http://localhost:8000/health", "file:///etc/passwd", "http://169.254.169.254/"]:
        assert client.get("/fetch", params={"url": u}).status_code == 400


def test_v1_analyze_schema_errors_and_ui():
    client = TestClient(app)
    out = client.post("/api/v1/analyze", files={"edited": ("e.jpg", jpeg(make_scene(98)), "image/jpeg")}).json()
    assert out["layer"] == "estimate" and out["exports"]["xmp"] and out["warnings"] and "residual" not in out
    bad = client.post("/api/v1/analyze", files={"edited": ("x.jpg", b"not an image", "image/jpeg")})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == 400
    assert client.get("/").status_code == 200
    assert "/api/v1/analyze" in client.get("/openapi.json").json()["paths"]


def _model_edit(scene, truth):
    from app import model
    e = (model.render(scene[..., ::-1] / 255.0, truth)[..., ::-1] * 255).astype(np.uint8)
    return cv2.imdecode(np.frombuffer(jpeg(crop_resize(e, 0.92, 480), 88), np.uint8), 1)


def test_model_recovers_known_sliders():
    from app import model
    truth = np.array([0.8, 30, -20, 25, 10, -10, 25, 20, -10], float)
    o = make_scene(5)
    al = align(o, _model_edit(o, truth))
    r = fit(al.original, al.edited, al.valid, calibrate=False)
    got = np.array([r["params"][k] for k in model.KEYS], float)
    assert abs(got[0] - 0.8) < 0.1 and abs(got[1] - 30) < 8 and abs(got[6] - 25) < 10
    assert abs(got[7] - 20) < 8 and r["model_rmse"] < 0.03
    assert all(r["uncertainty"][k] > 0 for k in model.KEYS)


def test_calibration_learns_scale(tmp_path):
    from app import calibration, config, model
    from scripts import calibrate
    from app.imageio import decode as dec
    rng = np.random.default_rng(3)
    for i in range(6):
        o = make_scene(200 + i)
        truth = rng.uniform(-1, 1, 9) * np.array([1, 50, 40, 40, 30, 30, 50, 30, 30])
        (tmp_path / f"s{i}.jpg").write_bytes(jpeg(o))
        cv2.imwrite(str(tmp_path / f"s{i}_edit.jpg"), _model_edit(o, truth))
        lr = 0.8 * truth[1]  # pretend Lightroom's contrast slider is 0.8x our model's
        (tmp_path / f"s{i}.xmp").write_text(f'<x:xmpmeta crs:Contrast2012="{lr:.2f}"></x:xmpmeta>')
    calibrate.main(str(tmp_path))
    try:
        c = calibration.load()["params"]["Contrast2012"]
        assert 0.65 < c["scale"] < 0.95 and c["r2"] > 0.9
        out, used = calibration.apply({"Contrast2012": 50.0})
        assert used and 33 < out["Contrast2012"] < 47
    finally:
        (config.DATA_DIR / "calibration.json").unlink(missing_ok=True)
