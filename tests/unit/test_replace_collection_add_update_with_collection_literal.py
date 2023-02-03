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
x = {1, 2, 3}
x.add(7)
x.add(191)
            """,
            """
x = {1, 2, 3, 7, 191}
            """,
        ),
        (
            """
x = [1, 2, 3]
x.append(7)
x.append(191)
            """,
            """
x = [1, 2, 3, 7, 191]
            """,
        ),
        (  # simplify_collection_unpacks is assumed to run after  this
            """
x = {1, 2, 3}
x.update((7, 22))
x.update((191, 191))
            """,
            """
x = {1, 2, 3, *(7, 22), *(191, 191)}
            """,
        ),
        (
            """
x = [1, 2, 3]
x.extend((7, 22))
x.extend(foo)
            """,
            """
x = [1, 2, 3, *(7, 22), *foo]
            """,
        ),
        (
            """
f = [1, 2, 3]
x.extend((7, 22))
x.extend(foo)
            """,
            """
f = [1, 2, 3]
x.extend((7, 22))
x.extend(foo)
            """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.replace_collection_add_update_with_collection_literal(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
