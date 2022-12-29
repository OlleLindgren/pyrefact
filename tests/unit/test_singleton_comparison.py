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
x == None and k != None
            """,
            """
x is None and k is not None
            """,
        ),
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

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
