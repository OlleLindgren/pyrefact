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
x = map(lambda y: y > 0, (1, 2, 3))
            """,
            """
x = (y > 0 for y in (1, 2, 3))
            """,
        ),
        (  # Invalid syntax
            """
x = map(lambda y, z: y > z, zip((1, 2, 3), [3, 2, 1]))
            """,
            """
x = map(lambda y, z: y > z, zip((1, 2, 3), [3, 2, 1]))
            """,
        ),
        (
            """
for x in map(lambda y: y > 0, (1, 2, 3)):
    print(x)
            """,
            """
for x in map(lambda y: y > 0, (1, 2, 3)):
    print(x)
            """,
        ),
        (
            """
r = map(lambda: True, (1, 2, 3))  # syntax error?
            """,
            """
r = map(lambda: True, (1, 2, 3))  # syntax error?
            """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.replace_map_lambda_with_comp(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
