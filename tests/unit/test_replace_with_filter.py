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
for x in range(10):
    if not f(x):
        continue
    print(x)
            """,
            """
for x in filter(f, range(10)):
    print(x)
            """,
        ),
        (
            """
for x in range(10):
    if f(x):
        print(3)
    else:
        continue
            """,
            """
for x in filter(f, range(10)):
    print(3)
            """,
        ),
        (
            """
for x in range(10):
    if f(x):
        print(3)
    else:
        print(76)
            """,
            """
for x in range(10):
    if f(x):
        print(3)
    else:
        print(76)
            """,
        ),
        (
            """
for x in range(10):
    if x:
        print(3)
            """,
            """
for x in filter(None, range(10)):
    print(3)
            """,
        ),
        (
            """
for x in range(10):
    if not x:
        continue
    else:
        print(3)
            """,
            """
for x in filter(None, range(10)):
    print(3)
            """,
        ),
        (  # I find itertools.filterfalse much less readable
            """
for x in range(10):
    if f(x):
        continue
    print(x)
            """,
            """
for x in range(10):
    if f(x):
        continue
    print(x)
            """,
        ),
        (
            """
for x in range(10):
    if f(x):
        print(x)
            """,
            """
for x in filter(f, range(10)):
    print(x)
            """,
        ),
        (  # Another filterfalse opportunity that I will not implement
            """
for x in range(10):
    if not f(x):
        print(x)
            """,
            """
for x in range(10):
    if not f(x):
        print(x)
            """,
        ),
        (  # Do not chain filter with filter
            """
for x in filter(bool, range(10)):
    if not f(x):
        continue
    print(x)
            """,
            """
for x in filter(bool, range(10)):
    if not f(x):
        continue
    print(x)
            """,
        ),
        (
            """
for x in filter(int, range(10)):
    if f(x):
        print(x)
            """,
            """
for x in filter(int, range(10)):
    if f(x):
        print(x)
            """,
        ),
        (  # Do not chain filter with filterfalse
            """
from itertools import filterfalse
for x in filterfalse(bool, range(10)):
    if not f(x):
        continue
    print(x)
            """,
            """
from itertools import filterfalse
for x in filterfalse(bool, range(10)):
    if not f(x):
        continue
    print(x)
            """,
        ),
        (
            """
import itertools
for x in itertools.filterfalse(int, range(10)):
    if f(x):
        print(x)
            """,
            """
import itertools
for x in itertools.filterfalse(int, range(10)):
    if f(x):
        print(x)
            """,
        ),
    )

    for content, expected_abstraction in test_cases:
        processed_content = fixes.replace_with_filter(content)
        processed_content = fixes.remove_dead_ifs(processed_content)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
