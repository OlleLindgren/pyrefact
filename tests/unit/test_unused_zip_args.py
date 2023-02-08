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
x = (a for _, a in zip(range(3), range(1, 3)))
            """,
            """
x = (a for a in range(1, 3))
            """,
        ),
        (
            """
x = {a for _, a in zip(range(3), range(1, 3))}
            """,
            """
x = {a for a in range(1, 3)}
            """,
        ),
        (
            """
x = [a for _, a in zip(range(3), range(1, 3))]
            """,
            """
x = [a for a in range(1, 3)]
            """,
        ),
        (
            """
x = (1 for _, _ in zip(range(3), range(1, 3)))
            """,
            """
x = (1 for _ in range(3))
            """,
        ),
        (
            """
x = (1 for a, q, _, _ in zip(range(3), range(1, 3), range(3, 5), (1, 2, 3)))
            """,
            """
x = (1 for a, q in zip(range(3), range(1, 3)))
            """,
        ),
        (
            """
for _, a in zip(range(3), range(1, 3)):
    print(a - 1)
            """,
            """
for a in range(1, 3):
    print(a - 1)
            """,
        ),
        (
            """
for a, _ in zip(range(3), range(1, 3)):
    print(a - 1)
            """,
            """
for a in range(3):
    print(a - 1)
            """,
        ),
        (
            """
for _, _ in zip(range(3), range(1, 3)):
    print(10)
            """,
            """
for _ in range(3):
    print(10)
            """,
        ),
        (
            """
for a, _, e, _ in zip(range(3), range(1, 3), range(3, 5), (1, 2, 3)):
    print(a != e != e is e)
            """,
            """
for a, e in zip(range(3), range(3, 5)):
    print(a != e != e is e)
            """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.unused_zip_args(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
