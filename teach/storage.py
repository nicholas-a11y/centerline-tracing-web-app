"""Teach storage: a local folder or Cloudflare R2, with the same key layout (design 7.1).

Keys always start with ``teach/``. Objects are never public; every public byte is
served by Flask after a visibility check.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

PREFIX = "teach/"
_KEY = re.compile(r"^teach/[A-Za-z0-9._\-/]+$")
_NOT_FOUND_CODES = {"NoSuchKey", "404", "NotFound"}


class StorageError(Exception):
    pass


def submission_key(sub_id: str, name: str) -> str:
    return f"{PREFIX}submissions/{sub_id}/{name}"


def check_key(key: str) -> str:
    if not _KEY.match(key) or any(part in ("", ".", "..") for part in key.split("/")):
        raise StorageError("invalid storage key")
    return key


class BaseStorage:
    kind = "base"

    def get(self, key: str) -> bytes | None:
        raise NotImplementedError

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def list(self, prefix: str) -> list[str]:
        raise NotImplementedError

    def get_json(self, key: str, default=None):
        raw = self.get(key)
        if raw is None:
            return default
        return json.loads(raw.decode("utf-8"))

    def put_json(self, key: str, obj) -> None:
        self.put(key, (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode("utf-8"), "application/json")

    def delete_prefix(self, prefix: str, keep: tuple[str, ...] = ()) -> int:
        n = 0
        for key in self.list(prefix):
            if key not in keep:
                self.delete(key)
                n += 1
        return n


class LocalStorage(BaseStorage):
    """Folder-backed storage for local development (TEACH_LOCAL_DIR)."""

    kind = "local"

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def _path(self, key: str) -> Path:
        check_key(key)
        path = (self.root / key).resolve()
        if self.root not in path.parents:
            raise StorageError("invalid storage key")
        return path

    def get(self, key: str) -> bytes | None:
        path = self._path(key)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def delete(self, key: str) -> None:
        try:
            self._path(key).unlink()
        except FileNotFoundError:
            pass

    def list(self, prefix: str) -> list[str]:
        if not prefix.startswith(PREFIX) or ".." in prefix.split("/"):
            raise StorageError("invalid storage prefix")
        base = self.root / prefix
        start = base if base.is_dir() else base.parent
        if not start.exists():
            return []
        out = []
        for p in start.rglob("*"):
            if p.is_file() and not p.name.startswith(".tmp-"):
                key = p.relative_to(self.root).as_posix()
                if key.startswith(prefix):
                    out.append(key)
        return sorted(out)


class R2Storage(BaseStorage):
    """Cloudflare R2 through the S3 API. ``client`` is a boto3 S3 client (or a test fake)."""

    kind = "r2"

    def __init__(self, client, bucket: str):
        self.client = client
        self.bucket = bucket

    @classmethod
    def from_config(cls, r2: dict) -> "R2Storage":
        import boto3  # lazy: only needed when R2 is configured
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=r2["R2_ENDPOINT"],
            aws_access_key_id=r2["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=r2["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
            config=Config(signature_version="s3v4", connect_timeout=5, read_timeout=15,
                          retries={"max_attempts": 2}),
        )
        return cls(client, r2["R2_BUCKET"])

    def check(self) -> None:
        self.client.head_bucket(Bucket=self.bucket)

    @staticmethod
    def _is_not_found(exc: Exception) -> bool:
        code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
        return code in _NOT_FOUND_CODES

    def get(self, key: str) -> bytes | None:
        check_key(key)
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if self._is_not_found(exc):
                return None
            raise
        return obj["Body"].read()

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        check_key(key)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def delete(self, key: str) -> None:
        check_key(key)
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def list(self, prefix: str) -> list[str]:
        out, token = [], None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            page = self.client.list_objects_v2(**kwargs)
            out.extend(item["Key"] for item in page.get("Contents", []))
            if not page.get("IsTruncated"):
                return sorted(out)
            token = page.get("NextContinuationToken")


def make_storage(cfg) -> tuple[BaseStorage | None, str | None]:
    """Build the configured storage. R2 is checked with one head_bucket call."""
    if cfg.storage_mode == "r2":
        try:
            storage = R2Storage.from_config(cfg.r2)
            storage.check()
        except ImportError:
            return None, "boto3 is not installed (needed for R2)"
        except Exception as exc:
            return None, f"R2 bucket check failed ({type(exc).__name__})"
        return storage, None
    try:
        cfg.local_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return None, f"TEACH_LOCAL_DIR is not writable ({type(exc).__name__})"
    return LocalStorage(cfg.local_dir), None
