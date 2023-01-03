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
def f(a, b, c):
    return 1, 2, 3
def g(a, b, c):
    return 1, 2, 3
y = f(1, 2, 3)
h = g(1, 2, 3)
            """,
            """
def f(a, b, c):
    return 1, 2, 3
y = f(1, 2, 3)
h = f(1, 2, 3)
            """,
        ),
        (
            """
def f(a, b, c):
    w = a ** (b - c)
    return 1 + w // 2
def g(c, b, k):
    w = c ** (b - k)
    return 1 + w // 2
y = f(1, 2, 3)
h = g(1, 2, 3)
            """,
            """
def f(a, b, c):
    w = a ** (b - c)
    return 1 + w // 2
y = f(1, 2, 3)
h = f(1, 2, 3)
            """,
        ),
        (
            """
def f(a, b, c):
    w = a ** (b - c)
    return 1 + w // 2
def g(c, b, k):
    w = c ** (b - k)
    return 1 - w // 2
y = f(1, 2, 3)
h = g(1, 2, 3)
            """,
            """
def f(a, b, c):
    w = a ** (b - c)
    return 1 + w // 2
def g(c, b, k):
    w = c ** (b - k)
    return 1 - w // 2
y = f(1, 2, 3)
h = g(1, 2, 3)
            """,
        ),
        (
            """
def f(a, b, c):
    w = a ** (b - c)
    return 1 + w // 2
def g(a, b, c):
    w = a ** (c - b)
    return 1 + w // 2
y = f(1, 2, 3)
h = g(1, 2, 3)
            """,
            """
def f(a, b, c):
    w = a ** (b - c)
    return 1 + w // 2
def g(a, b, c):
    w = a ** (c - b)
    return 1 + w // 2
y = f(1, 2, 3)
h = g(1, 2, 3)
            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = fixes.remove_duplicate_functions(content, set())

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
