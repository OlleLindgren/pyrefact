#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (
        (  # Explicit only
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
        (  # Implicit only
            """
def f(x) -> int:
    if x > 10:
        x += 1
        x *= 12
        print(x > 30)
        return 100 - sum(x, 2, 3)

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
        (  # No body
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
        (  # Combined
            """
def f(x) -> int:
    if x > 10:
        if x < 100:
            return 4
        elif x >= 12:
            return 2
        return 99
    else:
        return 14
            """,
            """
def f(x) -> int:
    if x <= 10:
        return 14
    else:
        if x < 100:
            return 4
        elif x >= 12:
            return 2
        return 99
            """,
        ),
        (  # Non-blocking body -> swap not equivalent
            """
if X % 5 == 0:
    if X % 61 == 0:
        if X % (X - 4) == 0:
            return 61
return 12
            """,
            """
if X % 5 == 0:
    if X % 61 == 0:
        if X % (X - 4) == 0:
            return 61
return 12
            """,
        ),
        (  # There's a pattern going on here, it shouldn't be disturbed.
            """
def foo(x):
    if isinstance(x, str):
        y = foo(x)
        w = faa(x)
        return y, w
    if isinstance(x, int):
        y = faf(x)
        w = wow(x)
        return y, w
    if isinstance(x, float):
        y = wowow(x)
        w = papaf(x)
        return y, w
    return 1, 2
            """,
            """
def foo(x):
    if isinstance(x, str):
        y = foo(x)
        w = faa(x)
        return y, w
    if isinstance(x, int):
        y = faf(x)
        w = wow(x)
        return y, w
    if isinstance(x, float):
        y = wowow(x)
        w = papaf(x)
        return y, w
    return 1, 2
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
