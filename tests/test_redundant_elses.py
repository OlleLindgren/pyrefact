#!/usr/bin/env python3

import itertools
import re
import sys

from pyrefact import fixes


def _remove_multi_whitespace(content: str) -> str:
    return re.sub("\n{2,}", "\n", f"\n{content}\n").strip()


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

        processed_content = _remove_multi_whitespace(processed_content)
        expected_abstraction = _remove_multi_whitespace(expected_abstraction)

        if processed_content != expected_abstraction:
            for i, (expected, got) in enumerate(
                itertools.zip_longest(
                    expected_abstraction.splitlines(), processed_content.splitlines()
                )
            ):
                if expected != got:
                    print(f"Line {i+1}, expected/got:\n{expected}\n{got}")
                    return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
