#!/usr/bin/env python3


import sys
from pathlib import Path

from pyrefact import performance

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (
        (
            """
import numpy
[a[i] for i in range(len(a))]
[a[i, :] for i in range(len(a))]
[a[i, :] for i in range(a.shape[0])]
[a[:, i] for i in range(a.shape[1])]
            """,
            """
import numpy
list(a)
list(a)
list(a)
list(a.T)
        """,
        ),
        (
            """
import numpy
(a[i] for i in range(len(a)))
(a[i, :] for i in range(len(a)))
(a[i, :] for i in range(a.shape[0]))
(a[:, i] for i in range(a.shape[1]))
            """,
            """
import numpy
iter(a)
iter(a)
iter(a)
iter(a.T)
        """,
        ),
        (
            """
import numpy as np
[
    [
        np.dot(b[:, i], a[j, :])
        for i in range(b.shape[1])
    ]
    for j in range(a.shape[0])
]
        """,
            """
import numpy as np
[
    [
        np.dot(b_i, a_j)
        for b_i in zip(*b)
    ]
    for a_j in a
]
        """,
        ),
    )

    for source, expected_abstraction in test_cases:

        processed_content = performance.replace_subscript_looping(source)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
