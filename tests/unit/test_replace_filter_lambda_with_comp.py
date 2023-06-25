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
x = filter(lambda y: y > 0, (1, 2, 3))
            """,
            """
x = (y for y in (1, 2, 3) if y > 0)
            """,
        ),
        (  # Invalid syntax
            """
x = filter(lambda y, z: y > z, zip((1, 2, 3), [3, 2, 1]))
            """,
            """
x = filter(lambda y, z: y > z, zip((1, 2, 3), [3, 2, 1]))
            """,
        ),
        (
            """
x = itertools.filterfalse(lambda y: y > 0, (1, 2, 3))
            """,
            """
x = (y for y in (1, 2, 3) if not y > 0)
            """,
        ),
        (
            """
for x in filter(lambda y: y > 0, (1, 2, 3)):
    print(x)
            """,
            """
for x in filter(lambda y: y > 0, (1, 2, 3)):
    print(x)
            """,
        ),
        (
            """
r = filter(lambda: True, (1, 2, 3))  # syntax error?
            """,
            """
r = filter(lambda: True, (1, 2, 3))  # syntax error?
            """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.replace_filter_lambda_with_comp(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
