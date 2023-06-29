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
if x == 100:
    y = False
else:
    y = True
            """,
            """
y = not x == 100
            """,
        ),
        (
            """
if x == 100:
    k = False
else:
    k = True
            """,
            """
k = not x == 100
            """,
        ),
        (
            """
if x == 100:
    k = False
else:
    k = 100
            """,
            """
if x == 100:
    k = False
else:
    k = 100
            """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.fix_if_assign(source)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
