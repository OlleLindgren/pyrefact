#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import constants, fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
x = {foo(z) ** 2 for z in range(3)}
for zua in range(3):
    x.add(zua - 1)
        """,
            """
x = {foo(z) ** 2 for z in range(3)} | {zua - 1 for zua in range(3)}
    """,
        ),
        (
            """
x = {foo(z) ** 2 for z in range(3)}
for zua in range(3):
    x.add(zua - 1)
x.update((1, 3, 2))
x.update({"sadf", 312})
    """,
        """
x = {foo(z) ** 2 for z in range(3)} | {zua - 1 for zua in range(3)} | set((1, 3, 2)) | set({"sadf", 312})
    """,
        )
        if constants.PYTHON_VERSION > (3, 9)
        else ("", ""),
        (
            """
x = [foo(z) ** 2 for z in range(3)]
for zua in range(3):
    x.append(zua - 1)
        """,
            """
x = [foo(z) ** 2 for z in range(3)]
for zua in range(3):
    x.append(zua - 1)
        """,
        ),
        (
            """
x = {1, 2, 3}
for zua in range(3):
    x.add(zua - 1)
for zua in range(9):
    x.add(zua ** 3 - 1)
for fua in range(9):
    x.add(fua ** 2 - 1)
        """,
            """
x = {1, 2, 3} | {zua - 1 for zua in range(3)} | {zua ** 3 - 1 for zua in range(9)} | {fua ** 2 - 1 for fua in range(9)}
    """,
        )
        if constants.PYTHON_VERSION > (3, 9)
        else ("", ""),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.replace_setcomp_add_with_union(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
