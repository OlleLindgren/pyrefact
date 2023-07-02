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
def f(x: int) -> int:
    if x == 100:
        return False
    return True
            """,
            """
def f(x: int) -> int:
    return not x == 100
            """,
        ),
        (
            """
def f(x: int) -> int:
    if 2 ** x >= -1 and y ** 3 == 100 and not foo(False):
        return False

    return True
            """,
            """
def f(x: int) -> int:
    return not (2 ** x >= -1 and y ** 3 == 100 and (not foo(False)))
            """,
        ),
        (
            """
def f(x: int) -> int:
    foo()
    if x == 100:
        return True
    return False
            """,
            """
def f(x: int) -> int:
    foo()
    return x == 100
            """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.fix_if_return(source)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
