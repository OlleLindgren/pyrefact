#!/usr/bin/env python3


import sys
from pathlib import Path

from pyrefact import performance, performance_numpy

sys.path.append(str(Path(__file__).parent))
import testing_infra


def main() -> int:
    test_cases = (
        (
            """
import numpy as np

i, j, k = 10, 11, 12

a = np.random.random((i, j))
b = np.random.random((j, k))

u = np.array([[np.dot(a_, b_) for a_ in a] for b_ in b.T]).T
v = np.array([[np.dot(b_, a_) for b_ in b.T] for a_ in a])

print(np.sum((u - np.matmul(a, b)).ravel()))
print(np.sum((v - np.matmul(a, b)).ravel()))
            """,
            """
import numpy as np

i, j, k = 10, 11, 12

a = np.random.random((i, j))
b = np.random.random((j, k))

u = np.matmul(a, b)
v = np.matmul(b.T, a.T).T

print(np.sum((u - np.matmul(a, b)).ravel()))
print(np.sum((v - np.matmul(a, b)).ravel()))
            """,
        ),
        (
            """
import numpy as np

i, j, k = 10, 11, 12

a = np.random.random((i, j))
b = np.random.random((j, k))

w = np.array([[np.dot(b[:, i], a[j, :]) for i in range(b.shape[1])] for j in range(a.shape[0])])

print(np.sum((w - np.matmul(a, b)).ravel()))
            """,
            """
import numpy as np

i, j, k = 10, 11, 12

a = np.random.random((i, j))
b = np.random.random((j, k))

w = np.matmul(b.T, a.T).T

print(np.sum((w - np.matmul(a, b)).ravel()))
            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = performance.replace_subscript_looping(content)
        processed_content = performance_numpy.replace_implicit_matmul(processed_content)
        processed_content = performance_numpy.simplify_transposes(processed_content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
