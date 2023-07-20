#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (
        (
            """
try:
    sketchy_function()
except ValueError:
    raise RuntimeError()
            """,
            """
try:
    sketchy_function()
except ValueError as error:
    raise RuntimeError() from error
            """,
        ),
        (
            """
try:
    sketchy_function()
except ValueError as foo:
    raise RuntimeError() from foo
            """,
            """
try:
    sketchy_function()
except ValueError as foo:
    raise RuntimeError() from foo
            """,
        ),
        (
            """
try:
    sketchy_function()
except ValueError:
    pass
            """,
            """
try:
    sketchy_function()
except ValueError:
    pass
            """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.fix_raise_missing_from(source)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
