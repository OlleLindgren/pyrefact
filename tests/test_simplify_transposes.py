#!/usr/bin/env python3


import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parent))
import testing_infra


def main() -> int:
    test_cases = (
        (
            """
arr = [[1, 2, 3], [4, 5, 6]]
assert list(zip(*arr)) == [[1, 4], [2, 5], [3, 6]]
assert list(zip(*zip(*arr))) == [[1, 4], [2, 5], [3, 6]]
            """,
            """
arr = [[1, 2, 3], [4, 5, 6]]
assert list(zip(*arr)) == [[1, 4], [2, 5], [3, 6]]
assert list(arr) == [[1, 4], [2, 5], [3, 6]]
            """,
        ),
        (
            """
arr = np.array([[1, 2, 3], [4, 5, 6]])
assert list(arr.T) == [[1, 4], [2, 5], [3, 6]]
assert list(arr.T.T) == [[1, 2, 3], [4, 5, 6]]
            """,
            """
arr = np.array([[1, 2, 3], [4, 5, 6]])
assert list(arr.T) == [[1, 4], [2, 5], [3, 6]]
assert list(arr) == [[1, 2, 3], [4, 5, 6]]
            """,
        ),
        (
            """
arr = np.array([[1, 2, 3], [4, 5, 6]])
assert list(zip(*arr.T)) == [[1, 2, 3], [4, 5, 6]]
assert list(zip(*arr.T.T)) == [[1, 4], [2, 5], [3, 6]]
assert list(zip(*zip(*arr))) == [[1, 2, 3], [4, 5, 6]]
assert list(zip(*zip(*arr.T))) == [[1, 4], [2, 5], [3, 6]]
assert list(zip(*zip(*arr.T.T))) == [[1, 2, 3], [4, 5, 6]]
            """,
            """
arr = np.array([[1, 2, 3], [4, 5, 6]])
assert list(arr) == [[1, 2, 3], [4, 5, 6]]
assert list(arr.T) == [[1, 4], [2, 5], [3, 6]]
assert list(arr) == [[1, 2, 3], [4, 5, 6]]
assert list(arr.T) == [[1, 4], [2, 5], [3, 6]]
assert list(arr) == [[1, 2, 3], [4, 5, 6]]
            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = fixes.simplify_transposes(content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
