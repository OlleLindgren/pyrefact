#!/usr/bin/env python3


import sys
from pathlib import Path

from pyrefact import fixes, performance_numpy

sys.path.append(str(Path(__file__).parents[1]))
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
v = np.matmul(a, b)

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
c = np.random.random((k, j))
d = np.random.random((j, i))

u = np.array([[np.dot(b[:, i], a[j, :]) for i in range(b.shape[1])] for j in range(a.shape[0])])
v = np.array([[np.dot(c[i, :], a[j, :]) for i in range(c.shape[0])] for j in range(a.shape[0])])
w = np.array([[np.dot(b[:, i], d[:, j]) for i in range(b.shape[1])] for j in range(d.shape[1])])
z = np.array([[np.dot(a[i, :], b[:, j]) for i in range(a.shape[0])] for j in range(b.shape[1])])

print(np.sum((u - np.matmul(a, b)).ravel()))
print(np.sum((v - np.matmul(a, c.T)).ravel()))
print(np.sum((w - np.matmul(b.T, d).T).ravel()))
print(np.sum((z - np.matmul(b.T, a.T)).ravel()))
        """,
            """
import numpy as np

i, j, k = 10, 11, 12

a = np.random.random((i, j))
b = np.random.random((j, k))
c = np.random.random((k, j))
d = np.random.random((j, i))

u = np.matmul(a, b)
v = np.matmul(c, a.T).T
w = np.matmul(b.T, d).T
z = np.matmul(a, b).T

print(np.sum((u - np.matmul(a, b)).ravel()))
print(np.sum((v - np.matmul(a, c.T)).ravel()))
print(np.sum((w - np.matmul(b.T, d).T).ravel()))
print(np.sum((z - np.matmul(b.T, a.T)).ravel()))
        """,
        ),
        (
            """
for i in range(len(left)):
    for j in range(len(right[0])):
        result[i][j] = np.dot(left[i] * right.T[j])
        """,
            """
result = np.matmul(left, right)
        """,
        ),
        (
            """
for i in range(len(left)):
    for j in range(len(right[0])):
        for k in range(len(right)):
            result[i][j] += left[i][k] * right[k][j]
        """,
            """
result = np.matmul(left, right)
        """,
        ),
        (
            """
result = [
    [
        np.dot(left[i] * right.T[j])
        for j in range(len(right[0]))
    ]
    for i in range(len(left))
]
        """,
            """
result = np.matmul(left, right)
        """,
        ),
        (
            """
result = [
    [
        sum(
            left[i][k] * right[k][j]
            for k in range(len(right))
        )
        for j in range(len(right[0]))
    ]
    for i in range(len(left))
]
        """,
            """
result = np.matmul(left, right)
        """,
        ),
    )

    for source, expected_abstraction in test_cases:

        processed_content = performance_numpy.replace_implicit_matmul(source)
        processed_content = fixes.simplify_transposes(processed_content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
