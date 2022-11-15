#!/usr/bin/env python3

import itertools
import re
import sys

from pyrefact import fixes


from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import testing_infra


def main() -> int:
    test_cases = (
        (
            """
def f(x: int) -> int:
    if x == 2:
        return 10
    else:
        return 11 - x
            """,
            """
def f(x: int) -> int:
    if x == 2:
        return 10

    return 11 - x
            """,
        ),
        (
            """
def f(x: int) -> int:
    if x == 2:
        return 10
    elif x == 12:
        return x**x - 3
    else:
        return 11 - x
            """,
            """
def f(x: int) -> int:
    if x == 2:
        return 10
    if x == 12:
        return x**x - 3

    return 11 - x
            """,
        ),
        (
            """
def f(x: int) -> int:
    if x < 0:
        if x > -100:
            return 10
        else:
            return 101
    elif x >= 12:
        if x ** 2 >= 99:
            return x**x - 3
        elif x ** 3 >= 99:
            return x**2
        else:
            return 0
    else:
        return 11 - x
            """,
            """
def f(x: int) -> int:
    if x < 0:
        if x > -100:
            return 10

        return 101
    if x >= 12:
        if x ** 2 >= 99:
            return x**x - 3
        if x ** 3 >= 99:
            return x**2

        return 0

    return 11 - x
            """,
        ),
        (
            """
for i in range(10):
    if i == 3:
        continue
    else:
        print(2)
            """,
            """
for i in range(10):
    if i == 3:
        continue

    print(2)
            """,
        ),
        (
            """
for i in range(10):
    if i == 3:
        while True:
            print(1)
            time.sleep(3)
    else:
        print(2)
            """,
            """
for i in range(10):
    if i == 3:
        while True:
            print(1)
            time.sleep(3)

    print(2)
            """,
        ),
        (
            """
for i in range(10):
    if i == 3:
        while True:
            print(1)
            time.sleep(3)
            break
    else:
        print(2)
            """,
            """
for i in range(10):
    if i == 3:
        while True:
            print(1)
            time.sleep(3)
            break
    else:
        print(2)
            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = fixes.remove_redundant_else(content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
