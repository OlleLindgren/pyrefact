#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
x = {**{}}
        """,
        """
x = {}
        """,
        ),
        (
            """
x = {**{}, 13: 14}
        """,
        """
x = {13: 14}
        """,
        ),
        (
            """
x = {3: {}, 13: 14}
        """,
        """
x = {3: {}, 13: 14}
        """,
        ),
        (
            """
x = {1: 2, 3: 4, **{99: 109, None: None}, 4: 5, **{"asdf": 12 - 13}}
        """,
        """
x = {1: 2, 3: 4, 99: 109, None: None, 4: 5, "asdf": 12 - 13}
        """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = fixes.simplify_dict_unpacks(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
