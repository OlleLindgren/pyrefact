#!/usr/bin/env python3


import sys
from pathlib import Path

from pyrefact import fixes, formatting

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
l = []
for a in range(A):
    m = list(range(B))
    l.extend(m)
        """,
            """
l = []
l.extend((b for a in range(A) for b in list(range(B))))
        """,
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
q.update((
    b
    for a in [x, y, 1, 2, 3]
    for b in frozenset(range(2, a))
))
        """,
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
q.update((
    b
    for a in [x, y, 1, 2, 3]
    for x in (1, 2, 3)
    if x * a > 3
    for _ in range(11)
    for b in frozenset(range(2, a, x))
))
        """,
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
        """,
        ),
        (
            """
l = []
for a in range(A):
    if a % 2 == 0:
        m = list(range(B))
        l.extend(m)
        """,
            """
l = []
l.extend((
    b
    for a in range(A)
    if a % 2 == 0
    for b in list(range(B))
))
        """,
        ),
        (
            """
q = set()
for a in [x, y, 1, 2, 3]:
    if random() > 0.5:
        m = frozenset(range(2, a))
        q.update(m)
        """,
            """
q = set()
q.update((
    b
    for a in [x, y, 1, 2, 3]
    if random() > 0.5
    for b in frozenset(range(2, a))
))
        """,
        ),
        (
            """
q = set()
if complicated_condition(q, 1, 2):
    for a in [x, y, 1, 2, 3]:
        if random() > 0.5:
            m = frozenset(range(2, a))
            q.update(m)
        """,
            """
q = set()
if complicated_condition(q, 1, 2):
    q.update((
        b
        for a in [x, y, 1, 2, 3]
        if random() > 0.5
        for b in frozenset(range(2, a))
    ))
        """,
        ),
        (
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
        """,
            """
q = set()
q.update((
    c
    for a in [x, y, 1, 2, 3]
    for x in (1, 2, 3)
    if x * a > 3
    for c in (
        b
        for _ in range(11)
        for b in frozenset(range(2, a, x))
)))
        """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = fixes.replace_nested_loops_with_set_list_comp(source)

        # Make the formatting a bit more readable. The nested loops are already replaced,
        # but hard to read when all of it is on one line.
        processed_content = formatting.format_with_black(processed_content, line_length=60)
        processed_content = formatting.collapse_trailing_parentheses(processed_content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
