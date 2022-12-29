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
np.matmul(a.T, b.T).T
np.matmul(a, b.T).T
np.matmul(a.T, b).T
np.matmul(a.T, b.T)
            """,
            """
np.matmul(b, a)
np.matmul(a, b.T).T
np.matmul(a.T, b).T
np.matmul(a.T, b.T)
            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = fixes.simplify_transposes(content)
        processed_content = performance_numpy.simplify_matmul_transposes(processed_content)
        processed_content = fixes.simplify_transposes(processed_content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
