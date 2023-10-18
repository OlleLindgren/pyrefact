#!/usr/bin/env python3


import sys
from pathlib import Path

from pyrefact import symbolic_math

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
(x for x in range(10) if x > 4)
    """,
        """
(x for x in range(5, 10) if True)
    """,
        ),
        (
            """
{x for x in range(10) if x > 4}
    """,
        """
{x for x in range(5, 10) if True}
    """,
        ),
        (
            """
[x for x in range(10) if x > 4]
    """,
        """
[x for x in range(5, 10) if True]
    """,
        ),
        (
            """
[x for x in range(10) if x >= 4]
    """,
        """
[x for x in range(4, 10) if True]
    """,
        ),
        (
            """
[x for x in range(10) if x < 4]
    """,
        """
[x for x in range(4) if True]
    """,
        ),
        (
            """
[x for x in range(10) if x <= 4]
    """,
        """
[x for x in range(5) if True]
    """,
        ),
        (
            """
[x for x in range(10) if x < 4 and x > 1]
    """,
        """
[x for x in range(2, 4) if True and True]
    """,
        ),
        (
            """
[x for x in range(10) if x < 4 and x > 1 and x == 88]
    """,
        """
[x for x in ()]
    """,
        ),
        (
            """
[x for x in range(-1, 89) if foo() and bar and x == 88]
    """,
        """
[x for x in range(88, 89) if foo() and bar and True]
    """,
        ),
        (
            """
[x for x in range(-1, 89) if foo() if bar if x == 88]
    """,
        """
[x for x in range(88, 89) if foo() if bar if True]
    """,
        ),
        (
            """
[x for x in range(-1, 89) if foo() and bar and x == 89]
    """,
        """
[x for x in ()]
    """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = symbolic_math.simplify_constrained_range(source)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
