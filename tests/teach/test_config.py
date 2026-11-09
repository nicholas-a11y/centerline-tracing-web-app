"""Fail-closed configuration (design 4): every problem disables all of Teach with one log line."""

from __future__ import annotations

import logging

import pytest
from flask import Flask

from teach import register_teach
from teach.config import load_config

from .conftest import TOKEN, base_env

R2_FULL = {"R2_ENDPOINT": "https://acct.r2.cloudflarestorage.com", "R2_BUCKET": "b",
           "R2_ACCESS_KEY_ID": "AKIDEXAMPLE", "R2_SECRET_ACCESS_KEY": "SECRETEXAMPLE"}


def _register(env, **overrides):
    app = Flask("cfg_test")
    ok = register_teach(app, env=env, **overrides)
    return app, ok


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "yes", "on"])
def test_flag_off_registers_nothing(tmp_path, caplog, value):
    env = base_env(tmp_path)
    if value is None:
        env.pop("TEACH_ENABLED")
    else:
        env["TEACH_ENABLED"] = value
    with caplog.at_level(logging.INFO, logger="teach"):
        app, ok = _register(env)
    assert ok is False
    assert not [r for r in app.url_map.iter_rules() if r.rule.startswith("/teach")]
    assert app.test_client().get("/teach").status_code == 404
    assert "teach" not in app.blueprints
    assert not caplog.records  # flag off is silent


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " True "])
def test_flag_on_values(tmp_path, value):
    _app, ok = _register(base_env(tmp_path, TEACH_ENABLED=value))
    assert ok is True


FAILURES = {
    "token_missing": ({"TEACH_ADMIN_TOKEN": None}, "TEACH_ADMIN_TOKEN is missing"),
    "token_blank": ({"TEACH_ADMIN_TOKEN": "   "}, "TEACH_ADMIN_TOKEN is missing"),
    "token_short": ({"TEACH_ADMIN_TOKEN": "s" * 31}, "shorter than 32 characters"),
    "r2_partial": ({"R2_ENDPOINT": R2_FULL["R2_ENDPOINT"], "R2_BUCKET": "b"},
                   "R2 settings are incomplete (missing: R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY)"),
    "r2_http": (dict(R2_FULL, R2_ENDPOINT="http://insecure.example"), "R2_ENDPOINT must start with https://"),
    "railway_without_r2": ({"RAILWAY_ENVIRONMENT": "production"}, "running on Railway without R2 settings"),
    "hops_bad": ({"TEACH_TRUSTED_PROXY_HOPS": "x"}, "TEACH_TRUSTED_PROXY_HOPS must be an integer"),
    "hops_high": ({"TEACH_TRUSTED_PROXY_HOPS": "6"}, "TEACH_TRUSTED_PROXY_HOPS must be an integer"),
    "base_url": ({"TEACH_PUBLIC_BASE_URL": "ftp://x"}, "TEACH_PUBLIC_BASE_URL must start with"),
    "alert_url": ({"TEACH_ALERT_WEBHOOK_URL": "javascript:alert(1)"}, "TEACH_ALERT_WEBHOOK_URL must start with"),
}


@pytest.mark.parametrize("name", sorted(FAILURES))
def test_misconfiguration_disables_everything(tmp_path, caplog, name):
    changes, reason = FAILURES[name]
    env = base_env(tmp_path)
    for k, v in changes.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    cfg, why = load_config(env)
    assert cfg is None and reason in why
    with caplog.at_level(logging.INFO, logger="teach"):
        app, ok = _register(env)
    assert ok is False
    assert app.test_client().get("/teach").status_code == 404
    assert not [r for r in app.url_map.iter_rules() if "teach" in r.rule or "llms" in r.rule]
    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) == 1 and lines[0].startswith("Teach disabled: ") and reason in lines[0]
    secrets_in_env = [v for k, v in env.items() if k in ("TEACH_ADMIN_TOKEN", "R2_SECRET_ACCESS_KEY",
                                                         "R2_ACCESS_KEY_ID", "TEACH_ALERT_WEBHOOK_URL") and v.strip()]
    for value in secrets_in_env:
        assert value not in lines[0]
    if name == "token_short":
        assert "31" not in lines[0]  # never the length


def test_valid_local_config(tmp_path):
    cfg, why = load_config(base_env(tmp_path))
    assert why is None and cfg.local_mode and cfg.trusted_proxy_hops == 0 and not cfg.llms_txt
    assert TOKEN not in repr(cfg)


def test_default_local_dir_is_outside_repo(tmp_path):
    env = base_env(tmp_path)
    env.pop("TEACH_LOCAL_DIR")
    cfg, _ = load_config(env)
    from .conftest import REPO

    assert REPO not in cfg.local_dir.resolve().parents and cfg.local_dir.name == ".centerline-teach"


def test_railway_with_r2(tmp_path):
    cfg, why = load_config(base_env(tmp_path, RAILWAY_ENVIRONMENT="production", **R2_FULL))
    assert why is None and cfg.storage_mode == "r2" and cfg.trusted_proxy_hops == 1
    assert "SECRETEXAMPLE" not in repr(cfg)


def test_r2_bucket_check_failure_disables(tmp_path, caplog, monkeypatch):
    from teach import storage

    class Broken:
        def head_bucket(self, **kw):
            raise RuntimeError("no network")

    monkeypatch.setattr(storage.R2Storage, "from_config", classmethod(lambda cls, r2: cls(Broken(), "b")))
    with caplog.at_level(logging.INFO, logger="teach"):
        app, ok = _register(base_env(tmp_path, **R2_FULL))
    assert ok is False
    assert caplog.records[-1].getMessage() == "Teach disabled: R2 bucket check failed (RuntimeError)"
    assert app.test_client().get("/teach").status_code == 404


def test_unexpected_error_never_raises(tmp_path, caplog, monkeypatch):
    import teach.blueprint

    def boom(*a, **k):
        raise RuntimeError("secret detail")

    monkeypatch.setattr(teach.blueprint, "create_blueprint", boom)
    with caplog.at_level(logging.INFO, logger="teach"):
        app, ok = _register(base_env(tmp_path))
    assert ok is False
    msg = caplog.records[-1].getMessage()
    assert msg == "Teach disabled: internal error during setup (RuntimeError)" and "secret detail" not in msg
    assert app.test_client().get("/teach").status_code == 404


def test_llms_txt_only_when_enabled(make_teach):
    assert make_teach().client.get("/llms.txt").status_code == 404
    t = make_teach(env_extra={"TEACH_LLMS_TXT": "1"})
    r = t.client.get("/llms.txt")
    assert r.status_code == 200 and "/teach/spec" in r.get_data(as_text=True)


def test_local_mode_banner(teach):
    assert "LOCAL MODE" in teach.client.get("/teach").get_data(as_text=True)
