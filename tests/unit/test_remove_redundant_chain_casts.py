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
set(itertools.chain(range(10), range(11)))
            """,
            """
{*range(10), *range(11)}
            """,
        ),
        (
            """
set(itertools.chain())
            """,
            """
set()
            """,
        ),
        (
            """
tuple(itertools.chain(range(10), range(11)))
            """,
            """
(*range(10), *range(11))
            """,
        ),
        (
            """
tuple(itertools.chain())
            """,
            """
()
            """,
        ),
        (
            """
list(itertools.chain())
            """,
            """
[]
            """,
        ),
        (
            """
iter(itertools.chain())
            """,
            """
iter(())
            """,
        ),
        (
            """
list(itertools.chain(range(10), range(11)))
            """,
            """
[*range(10), *range(11)]
            """,
        ),
        (
            """
iter(itertools.chain(range(10), range(11)))
            """,
            """
itertools.chain(range(10), range(11))
            """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.remove_redundant_chain_casts(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
