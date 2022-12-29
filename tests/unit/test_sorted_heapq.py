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
x = sorted(y)
z = sorted(p)[0]
k = sorted(p)[-1]
r = sorted([q, x], key=foo)[-1]
w = sorted(p)[:5]
w = sorted(p)[:k]
w = sorted(p)[:-5]
f = sorted(p)[-8:]
f = sorted(p)[-q():]
f = sorted(p)[13:]
sorted(x)[3:8]
print(sorted(z, key=lambda x: -x)[:94])
print(sorted(z, key=lambda x: -x)[-4:])
            """,
            """
x = sorted(y)
z = min(p)
k = max(p)
r = max([q, x], key=foo)
w = heapq.nsmallest(5, p)
w = heapq.nsmallest(k, p)
w = sorted(p)[:-5]
f = list(reversed(heapq.nlargest(8, p)))
f = list(reversed(heapq.nlargest(q(), p)))
f = sorted(p)[13:]
sorted(x)[3:8]
print(heapq.nsmallest(94, z, key=lambda x: -x))
print(list(reversed(heapq.nlargest(4, z, key=lambda x: -x))))
            """,
        ),
    )

    for content, expected_abstraction in test_cases:
        processed_content = performance.replace_sorted_heapq(content)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
