"""Export an admin-reviewed Teach case into the existing lesson folder format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .exporting import ExportError, export_defaults, service_from_environment, write_lesson_archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, help="Teach case number")
    parser.add_argument("--dest", type=Path, default=Path("tests/lessons"),
                        help="lesson directory to receive the new lesson")
    parser.add_argument("--lesson-id")
    parser.add_argument("--description")
    parser.add_argument("--expected-behaviour")
    parser.add_argument("--role", choices=("known_problem", "must_pass"))
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--region", help="JSON object {x0,y0,x1,y1}; null/omitted uses the case region")
    args = parser.parse_args(argv)
    try:
        service = service_from_environment()
        case_id = service.id_for_number(args.number)
        meta = service.load_meta(case_id) if case_id else None
        if not meta or meta["status"] == "deleted":
            raise ExportError("Case does not exist.")
        values = export_defaults(meta)
        for arg_name, field_name in (("lesson_id", "lesson_id"), ("description", "description"),
                                     ("expected_behaviour", "expected_behaviour"), ("role", "role"),
                                     ("tolerance", "tolerance_px")):
            value = getattr(args, arg_name)
            if value is not None:
                values[field_name] = value
        if args.region is not None:
            values["region"] = json.loads(args.region)
        folder = write_lesson_archive(service, args.number, values, args.dest, lessons_dir=args.dest)
    except (ExportError, ValueError, OSError) as exc:
        print(f"Lesson export failed: {exc}", file=sys.stderr)
        return 1
    print(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
