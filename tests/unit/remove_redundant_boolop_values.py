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
1 or 2 or 3
0 or 4 or 5
    """,
        """
1
4
    """,
        ),
        (
        """
1 and 2 and 3
0 and 4 and 5
    """,
        """
3
0
    """,
        ),
        (
        """
print(None or os.getcwd() or False)
    """,
        """
print(os.getcwd() or False)
    """,
        ),
        (
        """
print(None or os.getcwd() or False or sys.path)
    """,
        """
print(os.getcwd() or sys.path)
    """,
        ),
        (
        """
print(None and os.getcwd() and False)
    """,
        """
print(None)
    """,
        ),
        (
        """
print(None and os.getcwd() and False and sys.path)
    """,
        """
print(None)
    """,
        ),
        (
        """
print(os.getcwd() and False)
    """,
        """
print(os.getcwd() and False)
    """,
        ),
        (
        """
print(os.getcwd() and sys.path)
    """,
        """
print(os.getcwd() and sys.path)
    """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.remove_redundant_boolop_values(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
