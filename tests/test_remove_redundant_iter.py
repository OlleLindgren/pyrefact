#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import performance

sys.path.append(str(Path(__file__).parent))
import testing_infra


def main() -> int:
    test_cases = (
        (
            """
for key in (1, 2, 3):
    print(key)
            """,
            """
for key in (1, 2, 3):
    print(key)
            """,
        ),
        (
            """
for key in list((1, 2, 3)):
    print(key)
            """,
            """
for key in (1, 2, 3):
    print(key)
            """,
        ),
        (
            """
for q in tuple((1, 2, 3)):
    print(q)
            """,
            """
for q in (1, 2, 3):
    print(q)
            """,
        ),
        (
            """
values = (1, 2, 3)
for q in list(values):
    print(q)
            """,
            """
values = (1, 2, 3)
for q in values:
    print(q)
            """,
        ),
        (
            """
values = (1, 2, 3)
for q in sorted(values):
    print(q)
            """,
            """
values = (1, 2, 3)
for q in sorted(values):
    print(q)
            """,
        ),
        (
            """
values = range(50)
w = [x for x in list(values)]
print(w)
            """,
            """
values = range(50)
w = [x for x in values]
print(w)
            """,
        ),
        (
            """
values = range(50)
w = [x for x in iter(values)]
print(w)
            """,
            """
values = range(50)
w = [x for x in values]
print(w)
            """,
        ),
        (
            """
values = range(50)
w = {x for x in list(values)}
print(w)
            """,
            """
values = range(50)
w = {x for x in values}
print(w)
            """,
        ),
        (
            """
values = range(50)
w = (x for x in list(values))
print(w)
            """,
            """
values = range(50)
w = (x for x in values)
print(w)
            """,
        ),
    )

    for content, expected_abstraction in test_cases:
        processed_content = performance.remove_redundant_iter(content)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
