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
sorted(reversed(v))
sorted(sorted(v))
sorted(iter(v))
sorted(tuple(v))
sorted(list(v))
list(iter(v))
list(tuple(v))
list(list(v))
set(set(v))
set(reversed(v))
set(sorted(v))
set(iter(v))
set(tuple(v))
set(list(v))
iter(iter(v))
iter(tuple(v))
iter(list(v))
reversed(tuple(v))
reversed(list(v))
tuple(iter(v))
tuple(tuple(v))
tuple(list(v))
sum(reversed(v))
sum(sorted(v))
sum(iter(v))
sum(tuple(v))
sum(list(v))
sorted(foo(list(foo(iter((foo(v)))))))
            """,
            """
sorted(v)
sorted(v)
sorted(v)
sorted(v)
sorted(v)
list(v)
list(v)
list(v)
set(v)
set(v)
set(v)
set(v)
set(v)
set(v)
iter(v)
iter(v)
iter(v)
reversed(v)
reversed(v)
tuple(v)
tuple(v)
tuple(v)
sum(v)
sum(v)
sum(v)
sum(v)
sum(v)
sorted(foo(list(foo(iter((foo(v)))))))
            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = performance.remove_redundant_chained_calls(content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
