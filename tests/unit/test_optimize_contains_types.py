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
1 in (1, 2, 3)
x in {1, 2, ()}
x in [1, 2, []]
w in [1, 2, {}]
w in {foo, bar, "asdf", coo}
w in (foo, bar, "asdf", coo)
w in {x for x in range(10)}
w in [x for x in range(10)]
w in (x for x in range(10))
w in {x for x in [1, 3, "", 909, ()]}
w in [x for x in [1, 3, "", 909, ()]]
w in (x for x in [1, 3, "", 909, ()])
x in sorted([1, 2, 3])
            """,
            """
1 in {1, 2, 3}
x in {1, 2, ()}
x in (1, 2, [])
w in (1, 2, {})
w in {foo, bar, "asdf", coo}
w in (foo, bar, "asdf", coo)
w in {x for x in range(10)}
w in (x for x in range(10))
w in (x for x in range(10))
w in {x for x in [1, 3, "", 909, ()]}
w in (x for x in [1, 3, '', 909, ()])
w in (x for x in [1, 3, "", 909, ()])
x in [1, 2, 3]
            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = performance.optimize_contains_types(content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
