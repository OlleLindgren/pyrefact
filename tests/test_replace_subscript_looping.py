#!/usr/bin/env python3


import sys
from pathlib import Path

from pyrefact import performance

sys.path.append(str(Path(__file__).parent))
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
[a_ for a_ in a]
[a_ for a_ in a.T]
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
(a_ for a_ in a)
(a_ for a_ in a.T)
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
        np.dot(b_, a_)
        for b_ in b.T
    ]
    for a_ in a
]
        """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = performance.replace_subscript_looping(content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
