#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parent))
import testing_infra


def main() -> int:
    test_cases = (
        (
            """
def f(x) -> int:
    if x > 10:
        x += 1
        x *= 12
        print(x > 30)
        return 100 - sum(x, 2, 3)
    else:
        return 13
            """,
            """
def f(x) -> int:
    if x <= 10:
        return 13
    else:
        x += 1
        x *= 12
        print(x > 30)
        return 100 - sum(x, 2, 3)
            """,
        ),
        (
            """
def f(x):
    if x > 10:
        pass
    else:
        print(2)
            """,
            """
def f(x):
    if x <= 10:
        print(2)
            """,
        ),
    )

    for content, expected_abstraction in test_cases:
        processed_content = fixes.swap_if_else(content)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
