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
def foo():
    x = 100
    return x
            """,
            """
def foo():
    return 100
            """,
        ),
        (
            """
def foo():
    x = sorted(set(range(1000)))
    return x
            """,
            """
def foo():
    return sorted(set(range(1000)))
            """,
        ),
        (  # Variable names aren't the same
            """
y = 3
def foo():
    x = sorted(set(range(1000)))
    return y
            """,
            """
y = 3
def foo():
    x = sorted(set(range(1000)))
    return y
            """,
        ),
        (  # This pattern is ok, I don't want it removed.
            """
def foo():
    x = 100
    x = 3 - x
    return x
            """,
            """
def foo():
    x = 100
    x = 3 - x
    return x
            """,
        ),
        (  # Same variable in different places, both should be removed
            """
def foo():
    x = 100
    return x

def bar():
    x = 301 - foo()
    return x
            """,
            """
def foo():
    return 100

def bar():
    return 301 - foo()
            """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.simplify_assign_immediate_return(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
