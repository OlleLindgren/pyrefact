#!/usr/bin/env python3


import sys
from pathlib import Path

from pyrefact import symbolic_math

sys.path.append(str(Path(__file__).parent))
import testing_infra


def main() -> int:
    test_cases = (
        (
            """
x = sum(range(10))
y = sum(range(3, 17))
h = sum((1, 2, 3))
q = sum([1, 2, 3, 99, 99, -8])
r = sum({1, 2, 3, 99, 99, -8})
            """,
            """
x = 45
y = 133
h = 6
q = 196
r = sum({1, 2, 3, 99, 99, -8})
            """,
        ),
        (
            """
y = sum([a ** 2 for a in range(10)])
            """,
            """
y = 285
            """,
        ),
        (
            """
x = 111
z = 44
y = sum([x * a ** 3 - a * z ** 2 for a in range(10)])
            """,
            """
x = 111
z = 44
y = 2025 * x - 45 * z ** 2
            """,
        ),
        (
            """
y = sum([x * a ** 3 - a * z ** 2 for a in range(10, 19) for z in range(3, 7) for x in range(1, 3)])
            """,
            """
y = 304920
            """,
        ),
        (
            """
y = sum([
    x * a ** 3 - a * z ** 2
    for a in range(10, 19, 2)
    for z in range(3, 7)
    for x in range(1, 9)
])
            """,
            """
y = 2169440
            """,
        ),
        (
            """
y = sum([x * a ** 3 - a * z ** 2 for a in range(10, 19, 2) for z in range(3, 7) for x in range(1, 9, 5)])
y = sum([x * (a * 2) ** 3 - (a * 2) * z ** 2 for a in range(5, 10) for z in range(3, 7) for x in range(1, 9, 5)])
            """,
            """
y = 419160
y = 419160
            """,
        ),
        (
            """
y = sum([x * a ** 3 - a * z ** 2 for a in range(10, 19, 2) for z in (3, 4, 5, 6) for x in range(1, 9, 5)])
h = sum(x * a ** 3 - a * z ** 2 for a in range(10, 19, 2) for z in [3, 4, 5, 6] for x in range(1, 9, 5))
w = sum([x * a ** 3 - a * z ** 2 for a in range(10, 19, 2) for z in {3, 4, 5, 6, 6, 5, 4} for x in range(1, 9, 5)])
            """,
            """
y = 419160
h = 419160
w = 419160
            """,
        ),
        (
            """
import math
x = sum([math.sqrt(x) for x in range(11)])
            """,
            """
import math
x = sum([math.sqrt(x) for x in range(11)])
            """,
        ),
        (
            """
import random
def foo():
    return random.random()
x = sum([foo() for x in range(11)])
            """,
            """
import random
def foo():
    return random.random()
x = sum([foo() for x in range(11)])
            """,
        ),
        (
            """
from math import exp
x = sum([exp(x) for x in range(11)])
            """,
            """
from math import exp
x = sum([exp(x) for x in range(11)])
            """,
        ),
        (
            """
x = sum([a + b + c + d for _ in range(k, w)])
            """,
            """
x = -(k - w) * (a + b + c + d)
            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = symbolic_math.simplify_math_iterators(content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
