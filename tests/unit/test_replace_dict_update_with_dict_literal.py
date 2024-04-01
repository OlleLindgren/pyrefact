#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
x = {}
x.update({10: 100})
        """,
        """
x = {**{10: 100}}
        """,
        ),
        (
            """
x = {}
x.update({10: 100})
x.update({101: 220})
x.update({103: 223})
        """,
        """
x = {**{10: 100}, **{101: 220}, **{103: 223}}
        """,
        ),
        (
            """
x = {5: 13, **{102: 101, 103: 909}, 19: 14}
x.update({10: 100, 101: 220, 103: 223})
        """,
        """
x = {5: 13, **{102: 101, 103: 909}, 19: 14, **{10: 100, 101: 220, 103: 223}}
        """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = fixes.replace_dict_update_with_dict_literal(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
