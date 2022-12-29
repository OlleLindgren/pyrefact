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
u = list()
v = tuple()
w = dict()
a = dict(zip(range(4), range(1, 5)))
b = list((1, 2, 3, 99))
c = set([1, 2, 3])
d = iter((1, 2, 3, 5))
aa = (1 for u in (1, 2, 3, 5))
            """,
            """
u = []
v = ()
w = {}
a = dict(zip(range(4), range(1, 5)))
b = [1, 2, 3, 99]
c = {1, 2, 3}
d = iter((1, 2, 3, 5))
aa = (1 for u in (1, 2, 3, 5))
            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = fixes.replace_functions_with_literals(content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
