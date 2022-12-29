#!/usr/bin/env python3


import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (  # the original assignment becomes dead code and is removed by later steps
        (
            """
z = {a for a in range(10)}
x = sum(z)
            """,
            """
z = {a for a in range(10)}
x = sum({a for a in range(10)})
            """,
        ),
        (
            """
w = [a ** 2 for a in range(10)]
y = sum(w)
            """,
            """
w = [a ** 2 for a in range(10)]
y = sum([a ** 2 for a in range(10)])
            """,
        ),
        (
            """
k = True
w = [a ** 2 for a in range(11) if k]
k = False
y = sum(w)
            """,
            """
k = True
w = [a ** 2 for a in range(11) if k]
k = False
y = sum(w)
            """,
        ),
        (
            """
for i in range(10):
    w = [a ** 2 for a in range(10)]
    y = sum(w)
            """,
            """
for i in range(10):
    w = [a ** 2 for a in range(10)]
    y = sum([a ** 2 for a in range(10)])
            """,
        ),
        (
            """
w = []
for i in range(10):
    y = sum(w)
    w = [a ** 2 for a in range(i)]
            """,
            """
w = []
for i in range(10):
    y = sum(w)
    w = [a ** 2 for a in range(i)]
            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = fixes.inline_math_comprehensions(content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
