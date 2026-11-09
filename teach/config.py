"""Teach configuration, read only from environment variables (design section 4).

Fail closed: any problem returns a reason string and Teach registers nothing. Reasons
name variables but never include their values or lengths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

MIN_ADMIN_TOKEN_LENGTH = 32
R2_VARS = ("R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
MAX_PROXY_HOPS = 5


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true")


def is_enabled(env: Mapping[str, str]) -> bool:
    return _truthy(env.get("TEACH_ENABLED"))


def is_railway(env: Mapping[str, str]) -> bool:
    # Same rule as centerline_web_app.IS_RAILWAY_DEPLOYMENT, without importing the app.
    return any(key.startswith("RAILWAY_") for key in env)


@dataclass(frozen=True)
class TeachConfig:
    storage_mode: str                      # "local" or "r2"
    local_dir: Path
    trusted_proxy_hops: int
    public_base_url: str | None
    llms_txt: bool
    railway: bool
    admin_token: str = field(repr=False)
    r2: dict = field(default_factory=dict, repr=False)
    alert_url: str | None = field(default=None, repr=False)

    @property
    def local_mode(self) -> bool:
        return self.storage_mode == "local"

    @property
    def alerts_configured(self) -> bool:
        return bool(self.alert_url)


def default_local_dir() -> Path:
    return Path.home() / ".centerline-teach"


def load_config(env: Mapping[str, str] | None = None) -> tuple[TeachConfig | None, str | None]:
    """Return (config, None) or (None, reason). Only call when TEACH_ENABLED is true."""
    env = os.environ if env is None else env

    token = env.get("TEACH_ADMIN_TOKEN")
    if token is None or not token.strip():
        return None, "TEACH_ENABLED is set but TEACH_ADMIN_TOKEN is missing"
    if len(token) < MIN_ADMIN_TOKEN_LENGTH:
        return None, f"TEACH_ADMIN_TOKEN is shorter than {MIN_ADMIN_TOKEN_LENGTH} characters"

    railway = is_railway(env)
    present = [name for name in R2_VARS if (env.get(name) or "").strip()]
    if present and len(present) != len(R2_VARS):
        missing = ", ".join(name for name in R2_VARS if name not in present)
        return None, f"R2 settings are incomplete (missing: {missing})"
    if present:
        storage_mode = "r2"
        r2 = {name: env[name].strip() for name in R2_VARS}
        if not r2["R2_ENDPOINT"].startswith("https://"):
            return None, "R2_ENDPOINT must start with https://"
    else:
        if railway:
            return None, "running on Railway without R2 settings (local files would be lost on deploy)"
        storage_mode = "local"
        r2 = {}

    hops_raw = (env.get("TEACH_TRUSTED_PROXY_HOPS") or "").strip()
    if hops_raw:
        if not hops_raw.isdigit() or int(hops_raw) > MAX_PROXY_HOPS:
            return None, f"TEACH_TRUSTED_PROXY_HOPS must be an integer from 0 to {MAX_PROXY_HOPS}"
        hops = int(hops_raw)
    else:
        hops = 1 if railway else 0

    base = (env.get("TEACH_PUBLIC_BASE_URL") or "").strip().rstrip("/") or None
    if base and not base.startswith(("https://", "http://")):
        return None, "TEACH_PUBLIC_BASE_URL must start with https:// or http://"

    alert = (env.get("TEACH_ALERT_WEBHOOK_URL") or "").strip() or None
    if alert and not alert.startswith(("https://", "http://")):
        return None, "TEACH_ALERT_WEBHOOK_URL must start with https:// or http://"

    local_raw = (env.get("TEACH_LOCAL_DIR") or "").strip()
    local_dir = Path(local_raw).expanduser() if local_raw else default_local_dir()

    return TeachConfig(
        storage_mode=storage_mode,
        local_dir=local_dir,
        trusted_proxy_hops=hops,
        public_base_url=base,
        llms_txt=_truthy(env.get("TEACH_LLMS_TXT")),
        railway=railway,
        admin_token=token,
        r2=r2,
        alert_url=alert,
    ), None
