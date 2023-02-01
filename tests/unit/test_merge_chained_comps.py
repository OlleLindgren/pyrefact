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
x = (y for y in (y for y in (3, 4, 5)))
            """,
            """
x = (y for y in (3, 4, 5))
            """,
        ),
        (
            """
x = {y for y in {y for y in (3, 4, 5)}}
            """,
            """
x = {y for y in (3, 4, 5)}
            """,
        ),
        (
            """
x = [y for y in [y for y in (3, 4, 5)]]
            """,
            """
x = [y for y in (3, 4, 5)]
            """,
        ),
        (  # Not same comprehension type
            """
x = {y for y in (y for y in (3, 4, 5))}
            """,
            """
x = {y for y in (y for y in (3, 4, 5))}
            """,
        ),
        (
            """
x = (y for y in (y for y in (3, 4, 5) if y > 3))
            """,
            """
x = (y for y in (3, 4, 5) if y > 3)
            """,
        ),
        (
            """
x = (y for y in (y for y in (3, 4, 5)) if y > 3)
            """,
            """
x = (y for y in (3, 4, 5) if y > 3)
            """,
        ),
        (
            """
x = (y for y in (y for y in (3, 4, 5) if y > 3) if y < 5)
            """,
            """
x = (y for y in (3, 4, 5) if y > 3 if y < 5)
            """,
        ),
        (
            """
x = (y ** 2 for y in (y for y in (3, 4, 5)))
            """,
            """
x = (y ** 2 for y in (3, 4, 5))
            """,
        ),
        (  # Inner elt is not same as inner target
            """
x = (y ** 2 for y in (y ** 2 for y in (3, 4, 5)))
            """,
            """
x = (y ** 2 for y in (y ** 2 for y in (3, 4, 5)))
            """,
        ),
        (
            """
x = (y ** z for y, z in ((y, z) for y, z in zip((3, 4, 5), [3, 4, 5])))
            """,
            """
x = (y ** z for y, z in zip((3, 4, 5), [3, 4, 5]))
            """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.merge_chained_comps(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
