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
x == None or k != None
            """,
            """
x is None or k is not None
            """,
        ),
        (
            """
if a == False:
    print(1)
            """,
            """
if a is False:
    print(1)
            """,
        ),
        (
            """
print(q == True)
print(k != True)
            """,
            """
print(q is True)
print(k is not True)
            """,
        ),
        (
            """
print(q == True is x)
print(k != True != q != None is not False)
            """,
            """
print(q is True is x)
print(k is not True != q is not None is not False)
            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = fixes.singleton_eq_comparison(content)

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
