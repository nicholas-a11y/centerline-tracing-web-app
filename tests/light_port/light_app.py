"""Simulate the Stage 7 import switch in-process, without editing shipped code.

``centerline_web_app`` and ``centerline_core`` bind five names from the old
``centerline_engine`` at import time. :func:`light_engine_imports` temporarily
rebinds those module attributes to the ``centerline_engine_light`` objects of the
same name, so ported tests exercise the light engine end to end.
"""

from __future__ import annotations

from contextlib import contextmanager

import centerline_core
import centerline_engine_light
import centerline_web_app

OLD_ENGINE_BINDINGS = {
    centerline_web_app: (
        "CircleEvaluationSystem",
        "merge_nearby_paths",
        "optimize_path_with_circles",
        "optimize_path_with_custom_params",
        "remove_overlapping_paths",
    ),
    centerline_core: ("merge_nearby_paths",),
}


@contextmanager
def light_engine_imports():
    saved = []
    try:
        for module, names in OLD_ENGINE_BINDINGS.items():
            for name in names:
                saved.append((module, name, getattr(module, name)))
                setattr(module, name, getattr(centerline_engine_light, name))
        yield
    finally:
        for module, name, value in reversed(saved):
            setattr(module, name, value)


def bound_engine_modules() -> dict[str, str]:
    """Return ``"module.name" -> defining module`` for the switched names (for assertions/reports)."""
    out = {}
    for module, names in OLD_ENGINE_BINDINGS.items():
        for name in names:
            out[f"{module.__name__}.{name}"] = getattr(module, name).__module__
    return out
