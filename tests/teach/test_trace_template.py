"""Trace template download (design 14.5): stateless, same three files as lessonkit.trace_template."""

from __future__ import annotations

import io
import zipfile

from PIL import Image

from .conftest import IMAGE


def _snapshot(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) if root.exists() else []


def _post(client, data: bytes, name="My Drawing (v2).png"):
    return client.post("/teach/api/trace-template", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_zip_contains_three_files_and_stores_nothing(teach):
    before = _snapshot(teach.storage.root)
    r = _post(teach.client, IMAGE)
    assert r.status_code == 200 and r.mimetype == "application/zip"
    assert 'filename="My-Drawing--v2-trace-template.zip"' in r.headers["Content-Disposition"]
    with zipfile.ZipFile(io.BytesIO(r.data)) as zf:
        names = sorted(zf.namelist())
        assert names == ["My-Drawing--v2-README.txt", "My-Drawing--v2-faded.png", "My-Drawing--v2-template.svg"]
        faded = Image.open(io.BytesIO(zf.read("My-Drawing--v2-faded.png")))
        assert faded.size == Image.open(io.BytesIO(IMAGE)).size
        svg = zf.read("My-Drawing--v2-template.svg").decode("utf-8")
        assert 'viewBox="0 0 256 256"' in svg and "<image" in svg
    assert _snapshot(teach.storage.root) == before


def test_bad_image_is_422(teach):
    r = _post(teach.client, b"not an image")
    assert r.status_code == 422 and r.get_json()["errors"][0]["code"] == "image_format"


def test_missing_image_is_422(teach):
    r = teach.client.post("/teach/api/trace-template", data={}, content_type="multipart/form-data")
    assert r.status_code == 422
