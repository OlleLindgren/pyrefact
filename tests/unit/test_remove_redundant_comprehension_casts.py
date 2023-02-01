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
set({x for x in range(10)})
            """,
            """
{x for x in range(10)}
            """,
        ),
        (
            """
iter(x for x in range(10))
            """,
            """
(x for x in range(10))
            """,
        ),
        (
            """
list((x for x in range(10)))
            """,
            """
[x for x in range(10)]
            """,
        ),
        (
            """
list([x for y in range(10) for x in range(12 + y)])
            """,
            """
[x for y in range(10) for x in range(12 + y)]
            """,
        ),
        (
            """
list({x: 100 for x in range(10)})
            """,
            """
list({x for x in range(10)})
            """,
        ),
        (
            """
set({x: 100 for x in range(10)})
            """,
            """
{x for x in range(10)}
            """,
        ),
        (
            """
dict({x: 100 for x in range(10)})
            """,
            """
{x: 100 for x in range(10)}
            """,
        ),
        (
            """
iter({x: 100 for x in range(10)})
            """,
            """
iter({x for x in range(10)})
            """,
        ),
        (
            """
list({x for x in range(10)})
            """,
            """
list({x for x in range(10)})
            """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.remove_redundant_comprehension_casts(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
