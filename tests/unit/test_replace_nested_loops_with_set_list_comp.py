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
l = []
for a in range(A):
    m = list(range(B))
    l.extend(m)
        """,
        """
l = []
l.extend(
    b
    for a in range(A)
    for b in list(range(B))
)
        """
        ),
        (
        """
q = set()
for a in [x, y, 1, 2, 3]:
    m = frozenset(range(2, a))
    q.update(m)
        """,
        """
q = set()
q.update(
    b
    for a in [x, y, 1, 2, 3]
    for b in frozenset(range(2, a))
)
        """
        ),
        (
        """
q = set()
for a in [x, y, 1, 2, 3]:
    for x in (1, 2, 3):
        if x * a > 3:
            for _ in range(11):
                m = frozenset(range(2, a, x))
                q.update(m)
        """,
        """
q = set()
for a in [x, y, 1, 2, 3]:
    for x in (1, 2, 3):
        if x * a > 3:
            q.update(
                b
                for _ in range(11)
                for b in frozenset(range(2, a, x))
            )
        """
        ),
        (  # If there isn't a loop around it, it should not be replaced.
        """
q = set()
if x * a > 3:
    q.update(
        b
        for _ in range(11)
        for b in frozenset(range(2, a, x))
    )
        """,
        """
q = set()
if x * a > 3:
    q.update(
        b
        for _ in range(11)
        for b in frozenset(range(2, a, x))
    )
        """
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.replace_nested_loops_with_set_list_comp(source)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
