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
x = []
for i in range(10):
    x.append(i)
            """,
            """
x = [i for i in range(10)]
            """,
        ),
        (
            """
x = set()
for i in range(10):
    x.add(i)
            """,
            """
x = {i for i in range(10)}
            """,
        ),
        (
            """
x = set()
for i in range(10):
    x.add(1 if i > 5 else 100 - i)
            """,
            """
x = {1 if i > 5 else 100 - i for i in range(10)}
            """,
        ),
        (
            """
x = []
for i in range(10):
    x.append(i ** 2)
            """,
            """
x = [i ** 2 for i in range(10)]
            """,
        ),
        (
            """
x = []
for i in range(10):
    x.append(i ** 2)
    x.append(i + 2)
            """,
            """
x = []
for i in range(10):
    x.append(i ** 2)
    x.append(i + 2)
            """,
        ),
        (
            """
x = []
for i in range(10):
    if i > 3:
        x.append(i ** 2)
            """,
            """
x = [i ** 2 for i in range(10) if i > 3]
            """,
        ),
        (
            """
x = set()
for i in range(100):
    if i > 3:
        if i % 8 == 0:
            if i ** 2 % 5 == 7:
                x.add(i ** 2)
            """,
            """
x = {i ** 2 for i in range(100) if i > 3 and i % 8 == 0 and (i ** 2 % 5 == 7)}
            """,
        ),
        (
            """
x = set()
for i in range(100):
    if i > 3:
        if i % 8 == 0:
            if i ** 2 % 5 == 7:
                x.add(i ** 2)
            else:
                x.add(3)
            """,
            """
x = set()
for i in range(100):
    if i > 3:
        if i % 8 == 0:
            if i ** 2 % 5 == 7:
                x.add(i ** 2)
            else:
                x.add(3)
            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = fixes.replace_for_loops_with_comprehensions(content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
