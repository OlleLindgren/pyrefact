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
x = {}
for i in range(10):
    x[i] = 10
            """,
            """
x = {i: 10 for i in range(10)}
            """,
        ),
        (
            """
x = {}
for i in range(10):
    if i % 3 == 0:
        if i % 2 == 0:
            x[i] = 10
            """,
            """
x = {i: 10 for i in range(10) if i % 3 == 0 and i % 2 == 0}
            """,
        ),
        (
            """
x = {}
for i in range(10):
    if i % 3 == 0:
        if i % 2 == 0:
            x[i] = 10
        else:
            x[i] = 2
            """,
            """
x = {}
for i in range(10):
    if i % 3 == 0:
        if i % 2 == 0:
            x[i] = 10
        else:
            x[i] = 2
            """,
        ),
        (
            """
x = {}
for i in range(10):
    if i % 3 == 0:
        if i % 2 == 0:
            x[i] = 10
    else:
        x[i] = 2
            """,
            """
x = {}
for i in range(10):
    if i % 3 == 0:
        if i % 2 == 0:
            x[i] = 10
    else:
        x[i] = 2
            """,
        ),
        (
            """
x = {}
for i in range(10):
    x[i] = 10 ** i - 1
            """,
            """
x = {i: 10 ** i - 1 for i in range(10)}
            """,
        ),
        (
            """
x = {1: 2}
for i in range(10):
    x[i] = 10 ** i - 1
            """,
            """
x = {**{1: 2}, **{i: 10 ** i - 1 for i in range(10)}}
            """,
        ),
        (
            """
x = {i: 10 - 1 for i in range(33)}
for i in range(77, 22):
    x[i] = 10 ** i - 1
            """,
            """
x = {**{i: 10 - 1 for i in range(33)}, **{i: 10 ** i - 1 for i in range(77, 22)}}
            """,
        ),
        (
            """
u = {i: 10 - 1 for i in range(33)}
v = {i: 10 ** i - 1 for i in range(77, 22)}
w = {11: 342, 'key': "value"}
x = {**u, **v, **w}
for i in range(2, 4):
    x[i] = 10 ** i - 1
            """,
            """
u = {i: 10 - 1 for i in range(33)}
v = {i: 10 ** i - 1 for i in range(77, 22)}
w = {11: 342, 'key': "value"}
x = {**u, **v, **w, **{i: 10 ** i - 1 for i in range(2, 4)}}
            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = fixes.replace_for_loops_with_dict_comp(content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
