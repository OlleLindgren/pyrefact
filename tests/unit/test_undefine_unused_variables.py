#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
def some_python_code() -> None:
    x = 3
    a, asdf, *args, _, _, q = source_of_stuff()
    do_stuff_with(args)
    a, asdf, *args2, _, _, q = source_of_stuff()
    t, e, k, u = source_of_stuff()
    print(t + k)
    return 0

STATIC_STUFF = 420
_STATIC_PRIVATE_STUFF = 69
    """,
        """
def some_python_code() -> None:
    3
    _, _, *args, _, _, _ = source_of_stuff()
    do_stuff_with(args)
    source_of_stuff()
    t, _, k, _ = source_of_stuff()
    print(t + k)
    return 0

420
69
    """,
        ),
        (  # x = 202 is not redundant, since we may not enter the loop and redefine it
            """
x = 101
x = 202
for i in range(10):
    x = i
    print(x)
print(x)
    """,
        """
101
x = 202
for i in range(10):
    x = i
    print(x)
print(x)
    """,
        ),
        (  # x = 202 is redundant, since it is only used in the loop where it is redefined
            """
x = 101
x = 202
for i in range(10):
    x = i
    print(x)
z = 200
z = 200
z = 200
    """,
        """
101
202
for i in range(10):
    x = i
    print(x)
200
200
200
    """,
        ),
        (
            """
for i in range(10):
    x = i
    x = i - 1
    x = i
    print(x)
        """,
            """
for i in range(10):
    i
    i - 1
    x = i
    print(x)
        """,
        ),
        (
            """
x = 2
print(x)
for i in range(10):
    x = i
    x = i - 1
    x = i
        """,
            """
x = 2
print(x)
for i in range(10):
    i
    i - 1
    i
        """,
        ),
        (
            """
for i in range(10):
    x = i
    print(x)
    x = i - 1
    x = i
        """,
            """
for i in range(10):
    x = i
    print(x)
    i - 1
    i
        """,
        ),
        (
            """
for i in range(10):
    x = i
    x = i - 1
    x = i
print(x)
    """,
        """
for i in range(10):
    i
    i - 1
    x = i
print(x)
    """,
        ),
        (
            """
for i in range(10):
    x = i
    x = i - 1
    x = i
    if i >= 3:
        x += 1
    elif i < 9:
        x = 4
    if i % 2:
        x = 13
print(x)
    """,
        """
for i in range(10):
    i
    i - 1
    x = i
    if i >= 3:
        x += 1
    elif i < 9:
        x = 4
    if i % 2:
        x = 13
print(x)
    """,
        ),
        (  # x is referenced at the start of the loop, last set cannot be touched
            """
x = 2
for i in range(10):
    print(x)
    x = i
    x = i - 1
    x = i
    if i % 2:
        x = 13
            """,
            """
x = 2
for i in range(10):
    print(x)
    i
    i - 1
    x = i
    if i % 2:
        x = 13
            """,
        ),
        (
            """
x = 2
for i in range(10):
    print(x)
    x = i
    if i % 2:
        x = 22
        x = 13
            """,
            """
x = 2
for i in range(10):
    print(x)
    x = i
    if i % 2:
        22
        x = 13
            """,
        ),
        (
            """
x = 2
for i in range(10):
    print(x)
    x = i
    x = i - 1
    x = i
    if i % 2:
        x = 22
        x = 13
            """,
            """
x = 2
for i in range(10):
    print(x)
    i
    i - 1
    x = i
    if i % 2:
        22
        x = 13
            """,
        ),
        (
            """
x = 2
while x < 10:
    print(x)
    x += 1
        """,
            """
x = 2
while x < 10:
    print(x)
    x += 1
        """,
        ),
        (
            """
x = 2
y = 0
while x < 10:
    x = x + 1
    y += 1
print(y)
    """,
        """
x = 2
y = 0
while x < 10:
    x = x + 1
    y += 1
print(y)
    """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = fixes.undefine_unused_variables(source)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
