#!/usr/bin/env python3


import sys
from pathlib import Path

from pyrefact import fixes, performance_numpy

sys.path.append(str(Path(__file__).parent))
import testing_infra


def main() -> int:
    test_cases = (
        (
            """
import numpy as np

a = np.random.random(n)
b = np.random.random(n)

c = sum([a_ *  b_ for a_, b_ in zip(a, b)])
d = np.sum(a_ *  b_ for a_, b_ in zip(a, b))
print(c, d)
            """,
            """
import numpy as np

a = np.random.random(n)
b = np.random.random(n)

c = np.dot(a, b)
d = np.dot(a, b)
print(c, d)
            """,
        ),
        (
            """
n = 10
def _mysterious_function(a: np.array, b: np.array):
    return sum([a_ *  b_ for a_, b_ in zip(a, b)])

a = np.random.random(n)
b = np.random.random(n)

c = _mysterious_function(a, b)
print(c, np.dot(a, b))
            """,
            """
n = 10
def _mysterious_function(a: np.array, b: np.array):
    return np.dot(a, b)

a = np.random.random(n)
b = np.random.random(n)

c = _mysterious_function(a, b)
print(c, np.dot(a, b))
            """,
        ),
        (
            """
import numpy as np

i, j, k = 10, 11, 12

a = np.random.random((i, j))
b = np.random.random((j, k))

u = np.array(
    [
        [
            np.sum(
                a__ * b__
                for a__, b__ in zip(a_, b_)
            )
            for a_ in a
        ]
        for b_ in b.T
    ]
).T

print(u)
            """,
            """
import numpy as np

i, j, k = 10, 11, 12

a = np.random.random((i, j))
b = np.random.random((j, k))

u = np.array(
    [
        [
            np.dot(a_, b_)
            for a_ in a
        ]
        for b_ in b.T
    ]
).T

print(u)
        """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = performance_numpy.replace_implicit_dot(content)
        processed_content = fixes.simplify_transposes(processed_content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
