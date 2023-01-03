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
lambda: complicated_function()
lambda: pd.DataFrame()
lambda: []
lambda: {}
lambda: set()
lambda: ()
lambda: complicated_function(some_argument)
lambda: complicated_function(some_argument=2)
lambda x: []
lambda x: list()
            """,
            """
complicated_function
pd.DataFrame
list
dict
set
tuple
lambda: complicated_function(some_argument)
lambda: complicated_function(some_argument=2)
lambda x: []
lambda x: list()
            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = fixes.simplify_redundant_lambda(content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
