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
{1, 99, "s", 1, sum(range(11)), sum(range(11))}
            """,
            """
{1, 99, "s", sum(range(11)), sum(range(11))}
            """,
        ),
    )

    for source, expected_abstraction in test_cases:

        processed_content = fixes.remove_duplicate_set_elts(source)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
