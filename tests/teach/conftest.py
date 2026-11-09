"""Shared fixtures for the Teach tests: a fresh Flask app per test, local storage in tmp_path,
and a fake scoring queue so most tests never start the subprocess worker."""

from __future__ import annotations

import base64
import io
import json
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from flask import Flask

from teach import register_teach
from teach.scoring_worker import Busy
from teach.storage import LocalStorage

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "teach" / "example"
SAMPLES = REPO / "tests" / "pipeline" / "samples"
LESSON = REPO / "tests" / "lessons" / "cross-t2-w8-a35"
IMAGE = (EXAMPLE / "image.png").read_bytes()
REFERENCE = (EXAMPLE / "reference.svg").read_bytes()
META = json.loads((EXAMPLE / "request.json").read_text(encoding="utf-8"))["meta"]
TOKEN = "T" * 40


def png_bytes(w: int = 8, h: int = 8, value: int = 255) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (w, h), value).save(buf, format="PNG")
    return buf.getvalue()


def png_header_only(w: int, h: int) -> bytes:
    """A PNG signature plus IHDR claiming w x h (no pixel data), for header-only checks."""
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    return b"\x89PNG\r\n\x1a\n" + chunk


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


THUMB = png_bytes(16, 16, 128)


class FakeQueue:
    """Stands in for ScoringQueue: returns a canned worker result immediately."""

    def __init__(self, outcome: str = "fail"):
        self.outcome = outcome
        self.busy: str | None = None
        self.calls: list[dict] = []

    def run(self, ip, image_png, reference_json, params, scoring):
        if self.busy:
            raise Busy(self.busy)
        self.calls.append({"ip": ip, "params": params, "scoring": scoring, "paths": len(reference_json["paths"])})
        if self.outcome == "timeout":
            return {"outcome": "timeout", "timed_out": True, "variants": {}, "engine_commit": "test",
                    "elapsed_ms": 30000.0, "overlay_png": None, "thumb_png": None}
        summary = {"passed": self.outcome == "pass", "min_single_coverage": 0.5, "max_p95_px": 3.0, "path_count": 5}
        return {"outcome": self.outcome, "summary": summary, "variant": "prune_off", "engine_commit": "test",
                "elapsed_ms": 12.0, "timed_out": False,
                "variants": {"prune_off": {"summary": summary, "lines": []}, "prune_on": {"summary": summary}},
                "overlay_png": THUMB, "thumb_png": THUMB}


def base_env(tmp_path: Path, **extra) -> dict:
    env = {"TEACH_ENABLED": "1", "TEACH_ADMIN_TOKEN": TOKEN, "TEACH_LOCAL_DIR": str(tmp_path / "store")}
    env.update(extra)
    return env


def multipart(image=IMAGE, reference=REFERENCE, meta=None, **meta_changes) -> dict:
    m = dict(META if meta is None else meta)
    m.update(meta_changes)
    data = {"meta": json.dumps(m)}
    if image is not None:
        data["image"] = (io.BytesIO(image), "image.png")
    if reference is not None:
        data["reference"] = (io.BytesIO(reference), "reference.svg")
    return data


@pytest.fixture
def make_teach(tmp_path):
    def make(env_extra: dict | None = None, queue=None, notifier=None, **overrides):
        app = Flask("teach_test")
        app.config["TESTING"] = True
        storage = overrides.pop("storage", None) or LocalStorage(tmp_path / "store")
        q = queue or FakeQueue()
        ok = register_teach(app, env=base_env(tmp_path, **(env_extra or {})), storage=storage, queue=q,
                            notifier=notifier, **overrides)
        assert ok, "Teach did not register"
        return SimpleNamespace(app=app, client=app.test_client(), storage=storage, queue=q,
                               svc=app.blueprints["teach"].teach_service)
    return make


@pytest.fixture
def teach(make_teach):
    return make_teach()


def post(client, url, **kw):
    return client.post(url, data=multipart(**kw), content_type="multipart/form-data")


def submit(t, **kw):
    r = post(t.client, "/teach/api/submissions", **kw)
    assert r.status_code == 201, r.get_json()
    body = r.get_json()
    sub_id = body["manage_url"].split("/teach/manage/")[1].split("#")[0]
    token = body["manage_url"].split("#k=")[1]
    return body, sub_id, token
