#!/usr/bin/env python3


import sys
from pathlib import Path

from pyrefact import symbolic_math

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
x and not x
        """,
        """
False
        """,
        ),
        (
            """
x or not x
        """,
        """
True
        """,
        ),
        (
            """
(A and B) and (not A and not B)
(A and B) and (A or B)
a and b and not (not c or not d)
        """,
        """
False
A and B
a and b and c and d
        """,
        ),
        (
            """
(
    testing_infra.check_fixes_equal(processed_content, expected_abstraction)
    and True and not
    testing_infra.check_fixes_equal(processed_content, expected_abstraction)
)
        """,
        """
(
    False
)
        """,
        ),
        (
            """
x = [a for a in range(10) if a % 2 == 0 and a > 5 and a % 2 == 0]
        """,
        """
x = [a for a in range(10) if a % 2 == 0 and a > 5]
        """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = symbolic_math.simplify_boolean_expressions_symmath(source)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
