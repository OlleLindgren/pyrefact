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
        (
            """
values = []
for i in range(10):
    for j in range(11):
        values.append(i * j)
            """,
            """
values = [i * j for i in range(10) for j in range(11)]
            """,
        ),
        (
            """
values = []
for i in range(10):
    for j in range(11):
        if i % j:
            values.append(i * j)
            """,
            """
values = [i * j for i in range(10) for j in range(11) if i % j]
            """,
        ),
        (
            """
values = []
for i in range(10):
    if i % 3:
        if not i % 5:
            for j in range(11):
                if i % 7 and i % 9:
                    for k in range(5555):
                        values.append(i * j + k)
            """,
            """
values = [i * j + k for i in range(10) if i % 3 and (not i % 5) for j in range(11) if i % 7 and i % 9 for k in range(5555)]
            """,
        ),
        (
            """
x = 0
for i in range(10):
    x += 1
            """,
            """
x = sum((1 for i in range(10)))
            """,
        ),
        (
            """
x = []
for i in range(10):
    x += 1
            """,
            """
x = [1 for i in range(10)]
            """,
        ),
        (
            """
x = 777
for i in range(10):
    x += 1
            """,
            """
x = 777 + sum((1 for i in range(10)))
            """,
        ),
        (
            """
x = -777
for i in range(10):
    x -= 1
            """,
            """
x = -777 - sum((1 for i in range(10)))
            """,
        ),
        (
            """
x = [1, 2, 3]
for i in range(10):
    x += 1
            """,
            """
x = [1, 2, 3] + [1 for i in range(10)]
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
