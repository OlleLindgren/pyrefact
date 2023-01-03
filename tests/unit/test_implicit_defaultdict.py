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
def f(x: int) -> int:
    return x+1
d = {}
for x in range(10):
    y = f(x)
    if x in d:
        d[x].append(y)
    else:
        d[x] = [y]
            """,
            """
def f(x: int) -> int:
    return x+1
d = collections.defaultdict(list)
for x in range(10):
    y = f(x)
    d[x].append(y)
            """,
        ),
        (
            """
def f(x: int) -> int:
    return x+1
d = {}
for x in range(10):
    y = f(x)
    if x in d:
        d[x].add(y)
    else:
        d[x] = {y}
            """,
            """
def f(x: int) -> int:
    return x+1
d = collections.defaultdict(set)
for x in range(10):
    y = f(x)
    d[x].add(y)
            """,
        ),
        (
            """
def f(x: int) -> int:
    return x+1
d = {}
for x in range(10):
    if x in d:
        d[x].append(f(x))
    else:
        d[x] = [f(x)]
            """,
            """
def f(x: int) -> int:
    return x+1
d = collections.defaultdict(list)
for x in range(10):
    d[x].append(f(x))
            """,
        ),
        (
            """
def f(x: int) -> int:
    return x+1
def h(x: int) -> int:
    return x**2
d = {}
for x in range(10):
    y = f(x)
    if x in d:
        d[x].extend([y, 2, 4])
    else:
        d[x] = [y, 2, 4]
    z = h(x)
    w = x+19
    if w in d:
        d[w].extend([z, 9, 12])
    else:
        d[w] = [z, 9, 12]
            """,
            """
def f(x: int) -> int:
    return x+1
def h(x: int) -> int:
    return x**2
d = collections.defaultdict(list)
for x in range(10):
    y = f(x)
    d[x].extend([y, 2, 4])
    z = h(x)
    w = x+19
    d[w].extend([z, 9, 12])
            """,
        ),
        (
            """
def f(x: int) -> int:
    return x+1
d = {}
for x in range(10):
    if x in d:
        d[x].append(f(x))
    else:
        d[x] = [f(x)]
    w = x * 19
    if w in d:
        d[w].add(f(x))
    else:
        d[w] = {f(x)}
            """,
            """
def f(x: int) -> int:
    return x+1
d = {}
for x in range(10):
    if x in d:
        d[x].append(f(x))
    else:
        d[x] = [f(x)]
    w = x * 19
    if w in d:
        d[w].add(f(x))
    else:
        d[w] = {f(x)}
            """,
        ),
        (
            """
def f(x: int) -> int:
    return x+1
d = {}
for x in range(10):
    if x in d:
        d[x].add(f(x))
    else:
        d[x] = [f(x)]
            """,
            """
def f(x: int) -> int:
    return x+1
d = {}
for x in range(10):
    if x in d:
        d[x].add(f(x))
    else:
        d[x] = [f(x)]
            """,
        ),
        (
            """
def f(x: int) -> int:
    return x+1
d = {}
for x in range(10):
    if x not in d:
        d[x] = []
    d[x].append(f(x))
            """,
            """
def f(x: int) -> int:
    return x+1
d = collections.defaultdict(list)
for x in range(10):
    d[x].append(f(x))
            """,
        ),
        (
            """
def f(x: int) -> int:
    return x+1
d = {}
for x in range(10):
    if x not in d:
        d[x] = set()
    d[x].add(f(x))
            """,
            """
def f(x: int) -> int:
    return x+1
d = collections.defaultdict(set)
for x in range(10):
    d[x].add(f(x))
            """,
        ),
        (
            """
d = {}
for x in range(10):
    if x in d:
        d[x].extend(i for i in range(100))
    else:
        d[x] = [i for i in range(100)]
            """,
            """
d = collections.defaultdict(list)
for x in range(10):
    d[x].extend((i for i in range(100)))
            """,
        ),
    )

    for content, expected_abstraction in test_cases:
        processed_content = fixes.implicit_defaultdict(content)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
