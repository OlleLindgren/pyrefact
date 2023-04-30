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
x and False and y
            """,
            """
False
            """,
        ),
        (
            """
(x or y) and (x or y)
            """,
            """
x or y
            """,
        ),
        (
            """
x and not x
            """,
            """
False
            """,
        ),
        (
            """
x and y and f(x(3)) and not f(x(3))
            """,
            """
False
            """,
        ),
        (
            """
x or not x
            """,
            """
True
            """,
        ),
        (
            """
x or x or x or x or x
            """,
            """
x
            """,
        ),
        (
            """
x > 2 or x > 3
            """,
            """
x > 2
            """,
        ),
        (
            """
x > 2 and x > 3
            """,
            """
x > 3
            """,
        ),
        (
            """
x > 0 or x > 1 or x > 2 or x > 3
            """,
            """
x > 0
            """,
        ),
        (
            """
x > 0 and x > 1 and x > 2 and x > 3
            """,
            """
x > 3
            """,
        ),
        (
            """
x == 1 and x >= 3
            """,
            """
False
            """,
        ),
        (
            """
x == 1 or x >= 3
            """,
            """
x == 1 or x >= 3
            """,
        ),
        (
            """
x == 8 or x >= 3
            """,
            """
x >= 3
            """,
        ),
        (
            """
(x <= 5) or (x >= 3)
            """,
            """
True
            """,
        ),
        (
            """
(x > 7) and (x < 3)
            """,
            """
False
            """,
        ),
        (
            """
(x == 2) or (x != 2)
            """,
            """
True
            """,
        ),
        (
            """
(x > 5) and (x >= 3)
            """,
            """
x > 5
            """,
        ),
        (
            """
(x > 5) or (x >= 3)
            """,
            """
x >= 3
            """,
        ),
        (
            """
(x < 5) or (x <= 3)
            """,
            """
x < 5
            """,
        ),
        (
            """
(x < 5) and (x <= 3)
            """,
            """
x <= 3
            """,
        ),
        (
            """
(x < 4) and (x > 1) or (x > 7)
            """,
            """
(x < 4) and (x > 1) or (x > 7)
            """,
        ),
        (
            """
x > 4 and 5 >= x
x == 6 or x != 8
7 < x or x <= 9
3 >= x and x < 5
x != 3 and 2 == x
x <= 4 or 7 > x
8 >= x or x != 6
x < 1 and 3 > x
5 == x and x != 7
x >= 2 and x < 8
9 > x or x <= 10
x == 11 and 12 != x
4 >= x or x > 6
x != 5 or 2 == x
3 < x and x <= 8
x >= 9 and 10 > x
11 == x or x != 13
x <= 7 and 8 < x
x < 12 or x >= 14
15 > x and x != 16
x == 17 and 18 <= x
19 != x or x < 20
x >= 21 or x > 22
x < 23 and 24 >= x
            """,
            """
x > 4 and 5 >= x
x != 8
True
3 >= x
2 == x
7 > x
True
x < 1
5 == x
x >= 2 and x < 8
x <= 10
x == 11
4 >= x or x > 6
x != 5
3 < x and x <= 8
x >= 9 and 10 > x
x != 13
False
x < 12 or x >= 14
15 > x
False
True
x >= 21
x < 23
            """,
        ),
        (
            """
1 < 1 + 2 + 3
1 < -1
            """,
            """
True
False
            """,
        ),
        (
            """
x or ""
            """,
            """
x or ""
            """,
        ),
        (
            """
y or ()
            """,
            """
y or ()
            """,
        ),
        (
            """
y > 1 or (y < 0 or y > 1)
            """,
            """
y < 0 or y > 1
            """,
        ),
    )

    for source, expected_abstraction in test_cases:

        processed_content = fixes.simplify_boolean_expressions(source)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
