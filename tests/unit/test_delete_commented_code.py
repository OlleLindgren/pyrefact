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
def f() -> int:
    # return 100 ** 100
    return 2

# def q() -> int:
#     if x > 1:
#         return 3
#     else:
#

#         return -3

y = 10

# def h() -> int:
#     if x > 1:
#         # return 3
#         return 99
#     else:
#         return -3

            """,
            """
def f() -> int:
    return 2

y = 10
            """,
        ),
        (
            """
def f() -> int:
    # Normal comment
    return 2
# Normal comment
def h() -> int:
    return 2  # Normal comment
# lambda: 3
            """,
            """
def f() -> int:
    # Normal comment
    return 2
# Normal comment
def h() -> int:
    return 2  # Normal comment
            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = fixes.delete_commented_code(content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
