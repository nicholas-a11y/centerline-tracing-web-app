import io
import json
import zipfile

import pytest

from lessonkit.lesson import load_lesson
from teach.exporting import build_ai_bundle, build_lesson_archive, export_defaults, write_lesson_archive
from teach.storage import submission_key

from .conftest import submit
from .test_admin import AUTH


def test_lesson_export_matches_existing_runner_and_preserves_teach_metadata(teach, tmp_path):
    body, sid, _ = submit(teach, title="Crossing export test")
    meta = teach.svc.load_meta(sid)
    values = export_defaults(meta)
    values["lesson_id"] = "teach-1-crossing-export-test"
    archive, checks = build_lesson_archive(teach.svc, body["number"], values, lessons_dir=tmp_path)
    assert checks["ok"] and checks["registration"]
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        root = values["lesson_id"]
        assert set(zipped.namelist()) == {
            f"{root}/lesson.json", f"{root}/image.png", f"{root}/reference.svg"
        }
        out = tmp_path / root
        out.mkdir()
        zipped.extractall(tmp_path)
    lesson = load_lesson(out)
    assert lesson.id == root
    assert lesson.status == "active"
    assert lesson.params == meta["params"]
    assert lesson.meta["teach"]["case_number"] == body["number"]
    assert lesson.meta["teach"]["settings"] == meta["params"]
    assert lesson.meta["teach"]["consent_flags"]["consent"] == meta["consent"]
    assert lesson.meta["expected_behaviour"] == values["expected_behaviour"]


def test_cli_writer_uses_same_checked_lesson_export(teach, tmp_path):
    body, _, _ = submit(teach, title="CLI export test")
    meta = teach.svc.load_meta(teach.svc.id_for_number(body["number"]))
    values = export_defaults(meta)
    values["lesson_id"] = "teach-cli-export"
    folder = write_lesson_archive(teach.svc, body["number"], values, tmp_path, lessons_dir=tmp_path)
    loaded = load_lesson(folder)
    assert loaded.id == "teach-cli-export"
    assert (folder / "reference.svg").is_file()
    assert (folder / "image.png").is_file()
    assert teach.svc.load_meta(meta["id"])["lesson"]["lesson_id"] == "teach-cli-export"


def test_export_checks_block_invalid_consent_and_duplicate_ids(teach, tmp_path):
    body, sid, _ = submit(teach)
    meta = teach.svc.load_meta(sid)
    values = export_defaults(meta)
    values["lesson_id"] = "existing-folder"
    (tmp_path / "existing-folder").mkdir()
    from teach.exporting import ExportError, check_lesson_export
    with pytest.raises(ExportError):
        check_lesson_export(teach.svc, body["number"], values, lessons_dir=tmp_path)
    values["lesson_id"] = "teach-unique-export"
    meta["consent"]["licence"] = "not-cc0"
    teach.svc._save(meta)
    result = check_lesson_export(teach.svc, body["number"], values, lessons_dir=tmp_path)
    assert result["ok"] is False
    assert any("CC0" in error for error in result["errors"])


def test_ai_bundle_contains_private_review_material_for_any_status(teach):
    body, sid, _ = submit(teach, title="Bundle any status")
    teach.client.patch(f"/teach/api/admin/cases/{body['number']}", headers=AUTH, json={"status": "rejected"})
    archive = build_ai_bundle(teach.svc, body["number"])
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        assert set(zipped.namelist()) == {
            "bitmap.png", "hand_traced.svg", "app_centerline.svg", "overlay.png", "metadata.json", "README.md"
        }
        meta = json.loads(zipped.read("metadata.json"))
        assert meta["case_number"] == body["number"]
        assert meta["teach"]["me_too_count"] == 0
        assert meta["settings"]
        assert zipped.read("hand_traced.svg") == teach.storage.get(submission_key(sid, "reference.raw.svg"))
        assert b"hand_traced.svg" in zipped.read("README.md")


def test_ai_bundle_regenerates_missing_app_centerline(teach):
    body, sid, _ = submit(teach)
    key = submission_key(sid, "app_centerline.svg")
    teach.storage.delete(key)
    before = len(teach.svc.load_meta(sid)["results"])
    archive = build_ai_bundle(teach.svc, body["number"])
    assert teach.storage.get(key)
    assert len(teach.svc.load_meta(sid)["results"]) == before + 1
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        assert zipped.read("app_centerline.svg") == teach.storage.get(key)


def test_admin_export_page_bundle_download_and_merged_flag(teach):
    body, _, _ = submit(teach)
    number = body["number"]
    page = teach.client.get(f"/teach/admin/cases/{number}/export")
    assert page.status_code == 200
    check = teach.client.post(f"/teach/api/admin/cases/{number}/export-check", headers=AUTH,
                              json=export_defaults(teach.svc.load_meta(teach.svc.id_for_number(number))))
    assert check.status_code == 200 and check.get_json()["checks"]["ok"]
    values = export_defaults(teach.svc.load_meta(teach.svc.id_for_number(number)))
    lesson_zip = teach.client.post(f"/teach/api/admin/cases/{number}/export-lesson", headers=AUTH, json=values)
    assert lesson_zip.status_code == 200 and lesson_zip.content_type == "application/zip"
    with zipfile.ZipFile(io.BytesIO(lesson_zip.data)) as zipped:
        assert len(zipped.namelist()) == 3
    meta = teach.svc.load_meta(teach.svc.id_for_number(number))
    assert meta["lesson"]["lesson_id"] == values["lesson_id"]
    assert meta["lesson"]["exported_at"]
    bundle = teach.client.get(f"/teach/api/admin/cases/{number}/ai-bundle.zip", headers=AUTH)
    assert bundle.status_code == 200 and bundle.content_type == "application/zip"
    assert bundle.headers["Content-Disposition"].startswith("attachment;")
    response = teach.client.patch(f"/teach/api/admin/cases/{number}", headers=AUTH, json={"merged": True})
    assert response.status_code == 200
    assert teach.svc.load_meta(teach.svc.id_for_number(number))["merged"] is True
