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
[*(x for x in iterator)]
        """,
        """
list((x for x in iterator))
        """
        ),
        (
        """
(*(x for x in iterator),)
        """,
        """
tuple((x for x in iterator))
        """
        ),
        (
        """
{*[x for x in iterator]}
        """,
        """
set([x for x in iterator])
        """
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.replace_redundant_starred(source)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
