"""Admin authentication and environment diagnostics use isolated local/fake storage."""

import json

import pytest
from flask import Flask

from teach import register_teach
from teach.admin import AdminAuth
from teach.blueprint import CSP

from .conftest import TOKEN, base_env, post

AUTH = {"Authorization": "Bearer " + TOKEN}


@pytest.mark.parametrize("token", [None, "", "short", "x" * 31])
def test_admin_fail_closed(tmp_path, token):
    app = Flask("admin_off")
    assert not register_teach(app, env=base_env(tmp_path, TEACH_ADMIN_TOKEN=token))
    for path in ("/teach", "/teach/admin", "/teach/api/admin/environment"):
        assert app.test_client().get(path).status_code == 404


def test_admin_requires_bearer_and_headers(teach, caplog):
    assert teach.client.get("/teach/admin").status_code == 200
    r = teach.client.get("/teach/api/admin/environment")
    assert r.status_code == 401
    r = teach.client.get("/teach/api/admin/environment", headers=AUTH)
    assert r.status_code == 200
    assert r.headers["Cache-Control"] == "no-store"
    assert r.headers["Content-Security-Policy"] == CSP
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["Referrer-Policy"] == "no-referrer"
    assert TOKEN not in r.get_data(as_text=True) + caplog.text
    html = teach.client.get("/teach/admin").get_data(as_text=True)
    assert TOKEN not in html and "admin-token" in html
    assert teach.client.get("/teach/api/admin/environment?token=" + TOKEN).status_code == 401


def test_lockout_five_tries_and_expiry(teach):
    now = [0.0]
    auth = AdminAuth(TOKEN, teach.svc.quota, clock=lambda: now[0])
    ip = "203.0.113.44"
    for _ in range(4):
        assert auth.check(ip, "Bearer wrong")[0] == 401
    assert auth.check(ip, "Bearer wrong") == (429, 900)
    assert auth.check(ip, "Bearer " + TOKEN)[0] == 429
    assert auth.check("203.0.113.45", "Bearer " + TOKEN)[0] == 200
    assert ip not in json.dumps(list(auth.failures) + list(auth.locked))
    now[0] = 901.0
    assert auth.check(ip, "Bearer " + TOKEN)[0] == 200


def test_unicode_token_constant_time(make_teach):
    t = make_teach({"TEACH_ADMIN_TOKEN": "\u00e9" * 40})
    assert t.client.get("/teach/api/admin/environment", headers={"Authorization": "Bearer " + "\u00e9" * 40}).status_code == 200


def test_environment_no_ips_or_secrets_and_storage(teach):
    post(teach.client, "/teach/api/validate")
    r = teach.client.get("/teach/api/admin/environment", headers=AUTH, environ_base={"REMOTE_ADDR": "127.0.0.1"})
    data = r.get_json()
    assert data["client_ip"] == "127.0.*.*"
    assert data["admin_token_length"] == 40
    assert data["quotas"]["daily"]["validations"] == 1
    assert "127.0.0.1" not in r.get_data(as_text=True)
    assert teach.client.post("/teach/api/admin/storage-check", headers=AUTH).get_json()["ok"]
    assert teach.storage.list("teach/diagnostics/") == []
    assert teach.client.post("/teach/api/admin/test-alert", headers=AUTH).status_code == 409


def test_admin_https_only_except_loopback(teach):
    r = teach.client.get("/teach/admin", base_url="http://example.com", environ_base={"REMOTE_ADDR": "203.0.113.4"})
    assert r.status_code == 403
    r = teach.client.get("/teach/api/admin/environment", base_url="https://example.com", headers=AUTH)
    assert r.status_code == 200
