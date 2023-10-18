#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
x = foo(a, b, *(c, d), e, *{f}, *[k, v, h])
    """,
        """
x = foo(a, b, c, d, e, f, k, v, h)
    """,
        ),
        (
            """
x = foo(*(1, 2))
    """,
        """
x = foo(1, 2)
    """,
        ),
        (
            """
x = foo(*[1, 2])
    """,
        """
x = foo(1, 2)
    """,
        ),
        (  # Set of > 1 length should not be unpacked
            """
x = foo(*{1, 2})
    """,
        """
x = foo(*{1, 2})
    """,
        ),
        (
            """
x = foo(*(1,))
    """,
        """
x = foo(1)
    """,
        ),
        (
            """
x = foo(*[1])
    """,
        """
x = foo(1)
    """,
        ),
        (
            """
x = foo(*{1})
    """,
        """
x = foo(1)
    """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = fixes.breakout_starred_args(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
