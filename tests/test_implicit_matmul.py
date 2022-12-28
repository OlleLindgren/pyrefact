#!/usr/bin/env python3


import sys
from pathlib import Path

from pyrefact import performance_numpy

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

print(np.sum((u - np.matmul(a, b)).ravel()))
            """,
            """
import numpy as np

i, j, k = 10, 11, 12

a = np.random.random((i, j))
b = np.random.random((j, k))

u = np.matmul(a, b)

print(np.sum((u - np.matmul(a, b)).ravel()))
            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = performance_numpy.replace_implicit_matmul(content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
