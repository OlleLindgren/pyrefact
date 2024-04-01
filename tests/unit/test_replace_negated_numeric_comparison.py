#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
x > y
        """,
        """
x > y
        """,
        ),
        (
            """
not x > y
        """,
        """
not x > y
        """,
        ),
        (
            """
x > 3
        """,
        """
x > 3
        """,
        ),
        (
            """
not x > 3
        """,
        """
x <= 3
        """,
        ),
        (
            """
not 3 > 3.3
        """,
        """
3 <= 3.3
        """,
        ),
        (
            """
not 500 == h
        """,
        """
500 != h
        """,
        ),
        (
            """
not a == b
not a != b
not a < b
not a <= b
not a > b
not a >= b
not a is b
not a is not b
not a in b
not a not in b
        """,
        """
a != b
a == b
not a < b
not a <= b
not a > b
not a >= b
a is not b
a is b
a not in b
a in b
        """,
        ),
        (
            """
not a == 44.1
not a != 44.1
not a < 44.1
not a <= 44.1
not a > 44.1
not a >= 44.1
not a is 44.1
not a is not 44.1
        """,
        """
a != 44.1
a == 44.1
a >= 44.1
a > 44.1
a <= 44.1
a < 44.1
a is not 44.1
a is 44.1
        """,
        ),
        (
            """
not a == -999
not a != -999
not a < -999
not a <= -999
not a > -999
not a >= -999
not a is -999
not a is not -999
        """,
        """
a != -999
a == -999
a >= -999
a > -999
a <= -999
a < -999
a is not -999
a is -999
        """,
        ),
        (
            """
not y.xa() == (hqx - 999)
not y.xa() != (hqx - 999)
not y.xa() < (hqx - 999)
not y.xa() <= (hqx - 999)
not y.xa() > (hqx - 999)
not y.xa() >= (hqx - 999)
not y.xa() is (hqx - 999)
not y.xa() is not (hqx - 999)
        """,
        """
y.xa() != hqx - 999
y.xa() == hqx - 999
y.xa() >= hqx - 999
y.xa() > hqx - 999
y.xa() <= hqx - 999
y.xa() < hqx - 999
y.xa() is not hqx - 999
y.xa() is hqx - 999
        """,
        ),
        (
            """
not (hqx - 999) == y.xa()
not (hqx - 999) != y.xa()
not (hqx - 999) < y.xa()
not (hqx - 999) <= y.xa()
not (hqx - 999) > y.xa()
not (hqx - 999) >= y.xa()
not (hqx - 999) is y.xa()
not (hqx - 999) is not y.xa()
        """,
        """
hqx - 999 != y.xa()
hqx - 999 == y.xa()
hqx - 999 >= y.xa()
hqx - 999 > y.xa()
hqx - 999 <= y.xa()
hqx - 999 < y.xa()
hqx - 999 is not y.xa()
hqx - 999 is y.xa()
        """,
        ),
        (
            """
not (0 + k) == y.xa()
not (0 + k) != y.xa()
not (0 + k) < y.xa()
not (0 + k) <= y.xa()
not (0 + k) > y.xa()
not (0 + k) >= y.xa()
not (0 + k) is y.xa()
not (0 + k) is not y.xa()
        """,
        """
0 + k != y.xa()
0 + k == y.xa()
0 + k >= y.xa()
0 + k > y.xa()
0 + k <= y.xa()
0 + k < y.xa()
0 + k is not y.xa()
0 + k is y.xa()
        """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = fixes.replace_negated_numeric_comparison(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
