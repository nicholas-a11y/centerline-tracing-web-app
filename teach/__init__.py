"""Teach: public submission of hard tracing cases (docs/teach/design.md).

Off unless TEACH_ENABLED is 1/true. The app calls ``register_teach(app)`` once; with
the flag off it returns at once and registers nothing. This module imports nothing
heavy; numpy, Pillow and boto3 are imported lazily by the parts that need them.
"""

from __future__ import annotations

import logging
import os
from typing import Mapping

log = logging.getLogger("teach")


def _disabled(reason: str) -> bool:
    log.warning("Teach disabled: %s", reason)
    return False


def register_teach(app, env: Mapping[str, str] | None = None, **overrides) -> bool:
    """Register the Teach blueprint on ``app`` if enabled and correctly configured.

    Never raises into app startup. Returns True if Teach was registered. ``overrides``
    are for tests only (for example ``storage=``, ``scorer=``, ``alert_sender=``).
    """
    env = os.environ if env is None else env
    try:
        from .config import is_enabled, load_config

        if not is_enabled(env):
            return False
        cfg, reason = load_config(env)
        if cfg is None:
            return _disabled(reason)

        storage = overrides.pop("storage", None)
        if storage is None:
            from .storage import make_storage

            storage, reason = make_storage(cfg)
            if storage is None:
                return _disabled(reason)

        from .blueprint import create_blueprint, llms_txt_view

        bp = create_blueprint(cfg, storage, environment={"RAILWAY_GIT_COMMIT_SHA": env.get("RAILWAY_GIT_COMMIT_SHA")}, **overrides)
        app.register_blueprint(bp)
        from .crop import install_log_redaction

        install_log_redaction()
        if cfg.llms_txt:
            app.add_url_rule("/llms.txt", endpoint="teach_llms_txt", view_func=llms_txt_view)
        log.info("Teach enabled (%s storage)", cfg.storage_mode)
        return True
    except Exception as exc:  # never take the main app down
        return _disabled(f"internal error during setup ({type(exc).__name__})")
