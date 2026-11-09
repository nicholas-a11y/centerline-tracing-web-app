"""Storage backends (design 7): same key layout for LocalStorage and R2Storage. R2 is tested
against a fake S3 client; the live smoke test runs only when R2_* variables are set."""

from __future__ import annotations

import os

import pytest

from teach.storage import LocalStorage, R2Storage, StorageError, submission_key

R2_VARS = ("R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")


class FakeS3:
    class NotFound(Exception):
        response = {"Error": {"Code": "NoSuchKey"}}

    def __init__(self):
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.calls: list[str] = []

    def put_object(self, Bucket, Key, Body, ContentType):
        self.calls.append(f"put {Bucket}/{Key}")
        self.objects[Key] = (bytes(Body), ContentType)

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self.NotFound()
        import io

        return {"Body": io.BytesIO(self.objects[Key][0])}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + 2]  # tiny pages to exercise pagination
        more = start + 2 < len(keys)
        out = {"Contents": [{"Key": k} for k in page], "IsTruncated": more}
        if more:
            out["NextContinuationToken"] = str(start + 2)
        return out

    def head_bucket(self, Bucket):
        return {}


def _exercise(store):
    key = submission_key("t_abcdefghijkl", "meta.json")
    assert key == "teach/submissions/t_abcdefghijkl/meta.json"
    assert store.get(key) is None and store.get_json(key, {"d": 1}) == {"d": 1}
    store.put_json(key, {"a": 1})
    store.put(submission_key("t_abcdefghijkl", "image.png"), b"png", "image/png")
    store.put(submission_key("t_abcdefghijkl", "variants/v1/image.png"), b"v", "image/png")
    store.put("teach/index/public.json", b"{}")
    assert store.get_json(key) == {"a": 1}
    assert store.list("teach/submissions/t_abcdefghijkl/") == [
        "teach/submissions/t_abcdefghijkl/image.png", "teach/submissions/t_abcdefghijkl/meta.json",
        "teach/submissions/t_abcdefghijkl/variants/v1/image.png"]
    removed = store.delete_prefix("teach/submissions/t_abcdefghijkl/", keep=(key,))
    assert removed == 2 and store.list("teach/submissions/") == [key]
    store.delete(key)
    assert store.get(key) is None
    for bad in ("other/x", "teach/../x", "/teach/x", "teach/a b", "teach\\x", "teach/x/../../y"):
        with pytest.raises(StorageError):
            store.put(bad, b"x")


def test_local_storage(tmp_path):
    store = LocalStorage(tmp_path)
    _exercise(store)
    assert not [p for p in tmp_path.rglob("*") if p.name.endswith(".tmp")]


def test_r2_storage_with_fake_client():
    fake = FakeS3()
    store = R2Storage(fake, "bucket")
    _exercise(store)
    assert "put bucket/teach/submissions/t_abcdefghijkl/image.png" in fake.calls
    assert fake.objects["teach/index/public.json"][1] == "application/octet-stream"


def test_r2_other_errors_propagate():
    class Denied(Exception):
        response = {"Error": {"Code": "AccessDenied"}}

    class Client(FakeS3):
        def get_object(self, Bucket, Key):
            raise Denied()

    with pytest.raises(Denied):
        R2Storage(Client(), "b").get("teach/x")


def test_boto3_imported_lazily():
    import subprocess
    import sys

    code = ("import sys, teach, teach.storage; from teach.storage import make_storage, LocalStorage; "
            "print('boto3' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "False", out.stderr


@pytest.mark.skipif(not all(os.environ.get(v) for v in R2_VARS), reason="live R2 smoke test needs R2_* variables")
def test_live_r2_smoke():
    import secrets

    store = R2Storage.from_config({v: os.environ[v] for v in R2_VARS})
    store.check()
    key = f"teach/smoke/{secrets.token_hex(8)}.txt"
    store.put(key, b"ok", "text/plain")
    try:
        assert store.get(key) == b"ok" and key in store.list("teach/smoke/")
    finally:
        store.delete(key)
    assert store.get(key) is None
