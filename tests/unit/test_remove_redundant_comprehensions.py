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
a = {x: y for x, y in zip(range(4), range(1, 5))}
b = [w for w in (1, 2, 3, 99)]
c = {v for v in [1, 2, 3]}
d = (u for u in (1, 2, 3, 5))
aa = (1 for u in (1, 2, 3, 5))
ww = {x: y for x, y in zip((1, 2, 3), range(3)) if x > y > 1}
ww = {x: y for y, x in zip((1, 2, 3), range(3))}
            """,
            """
a = dict(zip(range(4), range(1, 5)))
b = list((1, 2, 3, 99))
c = set([1, 2, 3])
d = iter((1, 2, 3, 5))
aa = (1 for u in (1, 2, 3, 5))
ww = {x: y for x, y in zip((1, 2, 3), range(3)) if x > y > 1}
ww = {x: y for y, x in zip((1, 2, 3), range(3))}
            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = fixes.remove_redundant_comprehensions(content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
