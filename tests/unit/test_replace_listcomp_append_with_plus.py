#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
x = [foo(z) ** 2 for z in range(3)]
for zua in range(3):
    x.append(zua - 1)
        """,
            """
x = [foo(z) ** 2 for z in range(3)] + [zua - 1 for zua in range(3)]
        """,
        ),
        (
            """
x = [foo(z) ** 2 for z in range(3)]
for zua in range(3):
    x.append(zua - 1)
x.extend([1, 3, 2])
x.extend({"sadf", 312})
        """,
        """
x = [foo(z) ** 2 for z in range(3)] + [zua - 1 for zua in range(3)] + list([1, 3, 2]) + list({"sadf", 312})
        """,
        ),
        (
            """
x = {foo(z) ** 2 for z in range(3)}
for zua in range(3):
    x.add(zua - 1)
        """,
            """
x = {foo(z) ** 2 for z in range(3)}
for zua in range(3):
    x.add(zua - 1)
        """,
        ),
        (
            """
x = [1, 2, 3]
for zua in range(3):
    x.append(zua - 1)
for zua in range(9):
    x.append(zua ** 3 - 1)
for fua in range(9):
    x.append(fua ** 2 - 1)
        """,
            """
x = [1, 2, 3] + [zua - 1 for zua in range(3)] + [zua ** 3 - 1 for zua in range(9)] + [fua ** 2 - 1 for fua in range(9)]
        """,
        )
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.replace_listcomp_append_with_plus(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
