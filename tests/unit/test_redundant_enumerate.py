#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
(x for _, x in enumerate(y))
        """,
        """
(x for x in y)
        """,
        ),
        (
            """
(x for i, x in enumerate(y))
        """,
        """
(x for i, x in enumerate(y))
        """,
        ),
        (
            """
for _, x in enumerate(y):
    print(100 * x)
        """,
            """
for x in y:
    print(100 * x)
        """,
        ),
        (
            """
for i, x in enumerate(y):
    print(100 * x)
        """,
            """
for i, x in enumerate(y):
    print(100 * x)
        """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = fixes.redundant_enumerate(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
