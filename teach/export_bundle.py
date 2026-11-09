"""Download an AI review bundle for any Teach case."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from .exporting import ExportError, build_ai_bundle, service_from_environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, help="Teach case number")
    parser.add_argument("--dest", type=Path, required=True, help="folder for the bundle zip")
    args = parser.parse_args(argv)
    try:
        service = service_from_environment()
        archive = build_ai_bundle(service, args.number)
        args.dest.mkdir(parents=True, exist_ok=True)
        destination = args.dest / f"teach-{args.number}-ai-bundle.zip"
        fd, temp = tempfile.mkstemp(prefix=".teach-bundle-", suffix=".zip", dir=args.dest)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(archive)
            os.replace(temp, destination)
        except BaseException:
            try:
                os.unlink(temp)
            except OSError:
                pass
            raise
    except (ExportError, ValueError, OSError) as exc:
        print(f"AI bundle export failed: {exc}", file=sys.stderr)
        return 1
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
