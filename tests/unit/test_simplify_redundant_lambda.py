#!/usr/bin/env python3


import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (
        #         (
        #             """
        # lambda: complicated_function()
        # lambda: pd.DataFrame()
        # lambda: []
        # lambda: {}
        # lambda: set()
        # lambda: ()
        #             """,
        #             """
        # complicated_function
        # pd.DataFrame
        # list
        # dict
        # set
        # tuple
        #             """,
        #         ),
        #         (
        #             """
        # lambda value: [*value]
        # lambda value: {*value,}
        # lambda value: (*value,)
        # lambda value, /: [*value]
        #             """,
        #             """
        # list
        # set
        # tuple
        # list
        #             """,
        #         ),
        #         (
        #             """
        # lambda value, /, value2: (*value, *value2)
        # lambda value, /, value2: (*value,)
        # lambda: complicated_function(some_argument)
        # lambda: complicated_function(some_argument=2)
        #             """,
        #             """
        # lambda value, /, value2: (*value, *value2)
        # lambda value, /, value2: (*value,)
        # lambda: complicated_function(some_argument)
        # lambda: complicated_function(some_argument=2)
        #             """,
        #         ),
        #         (
        #             """
        # lambda x: []
        # lambda x: list()
        #             """,
        #             """
        # lambda x: []
        # lambda x: list()
        #             """,
        #         ),
        (
            """
lambda *args: w(*args)
lambda **kwargs: r(**kwargs)
    """,
        """
w
r
    """,
        ),
        (
            """
lambda q: h(q)
lambda z, w: f(z, w)
lambda *args, **kwargs: hh(*args, **kwargs)
lambda z, k, /, w, h, *args: rrr(z, k, w, h, *args)
lambda z, k, /, w, h, *args, **kwargs: rfr(z, k, w, h, *args, **kwargs)
lambda z, k, /, w, h, *args: rrr(z, k, w, w, *args)
lambda z, k, /, w, h: rrr(z, k, w, w, *args)
    """,
        """
h
f
hh
rrr
rfr
lambda z, k, /, w, h, *args: rrr(z, k, w, w, *args)
lambda z, k, /, w, h: rrr(z, k, w, w, *args)
    """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = fixes.simplify_redundant_lambda(source)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
