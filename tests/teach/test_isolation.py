"""Isolation (design 3.2, 18): the real app's url_map with Teach off is exactly the pre-Teach map,
and with Teach on the only additions are under /teach. Runs the real app in subprocesses so the
environment is read exactly as in production."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from .conftest import REPO, TOKEN

# The 28 rules of centerline_web_app before Teach (taken from the pre-Teach commit 3a9e74e).
PRE_TEACH_RULES = sorted([
    "/", "/api/test-runs", "/api/test-runs/<run_id>", "/api/test-runs/<run_id>/progress", "/api/test-runs/execute",
    "/auto_detect_min_path_length", "/auto_detect_threshold", "/auto_detect_threshold_progress/<session_id>",
    "/auto_detect_threshold_start", "/auto_tune_extraction", "/auto_tune_extraction_start",
    "/auto_tune_progress/<session_id>", "/auto_tune_sample_region", "/auto_tune_tile_preview",
    "/benchmark_metrics/<session_id>", "/download_svg/<session_id>", "/generate_svg", "/process",
    "/process_immediate", "/progress/<session_id>", "/static/<path:filename>", "/stop_auto_detect",
    "/stop_auto_tune", "/stop_optimization", "/test-ui", "/test_svg/<session_id>", "/upload", "/upload_remote",
])

SCRIPT = r"""
import json, logging, os, sys
logging.basicConfig(level=logging.INFO, format="%(name)s|%(message)s", stream=sys.stderr)
sys.path.insert(0, os.getcwd())
import centerline_web_app as m
c = m.app.test_client()
rules = sorted([r.rule, r.endpoint, sorted(r.methods)] for r in m.app.url_map.iter_rules())
print("@@" + json.dumps({"rules": rules, "teach": c.get("/teach").status_code,
                         "root": c.get("/").status_code}))
"""


def _app_rules(env_changes: dict) -> tuple[dict, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TEACH_", "R2_", "RAILWAY_"))}
    env.update(env_changes)
    proc = subprocess.run([sys.executable, "-c", SCRIPT], cwd=str(REPO), env=env, capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@")][-1]
    return json.loads(line[2:]), proc.stderr


@pytest.fixture(scope="module")
def flag_off():
    return _app_rules({})


def test_flag_off_url_map_unchanged(flag_off):
    data, stderr = flag_off
    rules = [r[0] for r in data["rules"]]
    assert sorted(rules) == PRE_TEACH_RULES
    assert not [r for r in rules if r.startswith("/teach") or r == "/llms.txt"]
    assert data["teach"] == 404 and data["root"] == 200
    assert "teach|" not in stderr  # silent when off


def test_flag_on_only_adds_teach_rules(flag_off, tmp_path):
    off, _ = flag_off
    on, stderr = _app_rules({"TEACH_ENABLED": "1", "TEACH_ADMIN_TOKEN": TOKEN,
                             "TEACH_LOCAL_DIR": str(tmp_path / "store")})
    off_set = {json.dumps(r) for r in off["rules"]}
    on_set = {json.dumps(r) for r in on["rules"]}
    assert off_set <= on_set, "an existing rule changed"
    added = [json.loads(r) for r in on_set - off_set]
    assert added and all(r[0].startswith("/teach/") or r[0] == "/teach" for r in added)
    assert all(r[1].startswith("teach.") for r in added)
    assert on["teach"] in (200, 308) and on["root"] == 200
    assert "teach|Teach enabled (local storage)" in stderr


def test_flag_on_misconfigured_equals_off(flag_off, tmp_path):
    off, _ = flag_off
    bad, stderr = _app_rules({"TEACH_ENABLED": "1", "TEACH_ADMIN_TOKEN": "short",
                              "TEACH_LOCAL_DIR": str(tmp_path / "store")})
    assert bad["rules"] == off["rules"] and bad["teach"] == 404
    lines = [ln for ln in stderr.splitlines() if ln.startswith("teach|")]
    assert lines == ["teach|Teach disabled: TEACH_ADMIN_TOKEN is shorter than 32 characters"]


def test_registration_is_one_line():
    src = (REPO / "centerline_web_app.py").read_text(encoding="utf-8")
    lines = [ln for ln in src.splitlines() if "teach" in ln.lower()]
    assert lines == ["from teach import register_teach; register_teach(app)  "
                     "# Teach (docs/teach/design.md); off unless TEACH_ENABLED"]


def test_teach_never_imports_tests():
    for path in (REPO / "teach").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "from tests" not in text and "import tests" not in text, path
