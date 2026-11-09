"""Pytest hooks for the lesson suite: list unexpected passes (baseline lessons that now pass)."""

from __future__ import annotations


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    xpassed = terminalreporter.stats.get("xpassed", [])
    lesson_xpasses = [r.nodeid for r in xpassed if "tests/pipeline/test_lessons.py" in r.nodeid.replace("\\", "/")]
    if lesson_xpasses:
        terminalreporter.write_sep("=", "UNEXPECTED PASSES in baseline lessons (consider promoting to active)")
        for nodeid in lesson_xpasses:
            terminalreporter.write_line(nodeid)
