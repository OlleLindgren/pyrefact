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
def f(x) -> int:
    if x > 10:
        x += 1
        x *= 12
        print(x > 30)
        y = 100 - sum(x, 2, 3)
    else:
        y = 13
    return y
            """,
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
        ),
        (
            """
def f(x) -> int:
    if x > 10:
        x += 1
        x *= 12
        print(x > 30)
        y = 100 - sum(x, 2, 3)
    else:
        y = 13
    print(3)
    return y
            """,
            """
def f(x) -> int:
    if x > 10:
        x += 1
        x *= 12
        print(x > 30)
        y = 100 - sum(x, 2, 3)
    else:
        y = 13
    print(3)
    return y
            """,
        ),
    )

    for content, expected_abstraction in test_cases:
        processed_content = fixes.early_return(content)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
