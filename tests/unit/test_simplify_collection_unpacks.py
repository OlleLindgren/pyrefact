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
x = [*()]
            """,
            """
x = []
            """,
        ),
        (
            """
x = (*(),)
            """,
            """
x = ()
            """,
        ),
        (
            """
x = {*{}}
            """,
            """
x = set()
            """,
        ),
        (
            """
x = [*{}]
            """,
            """
x = []
            """,
        ),
        (
            """
x = (*{},)
            """,
            """
x = ()
            """,
        ),
        (
            """
x = {*()}
            """,
            """
x = set()
            """,
        ),
        (  # 1 element is unique, hence a dict provides no non-trivial uniqueness
            """
x = [*{1: 3}]
            """,
            """
x = [1]
            """,
        ),
        (
            """
x = (*{1: 3},)
            """,
            """
x = (1,)
            """,
        ),
        (
            """
x = {*{1: 3}}
            """,
            """
x = {1}
            """,
        ),
        (  # 2 elements need to be compared, so can only be safely moved to the set
            """
x = [*{1: 3, 2: 19}]
            """,
            """
x = [*{1: 3, 2: 19}]
            """,
        ),
        (
            """
x = (*{1: 3, 2: 19},)
            """,
            """
x = (*{1: 3, 2: 19},)
            """,
        ),
        (
            """
x = {*{1: 3, 2: 19}}
            """,
            """
x = {1, 2}
            """,
        ),
        (  # One element is unique and may be unpacked into a tuple or set
            """
x = {*(), 2, 3, *{99}, (199, 991, 2), *[], [], *tuple([1, 2, 3]), 9}
            """,
            """
x = {2, 3, 99, 199, 991, 2, [], *tuple([1, 2, 3]), 9}
            """,
        ),
        (
            """
x = (*(), 2, 3, *{99}, (199, 991, 2), *[], [], *tuple([1, 2, 3]), 9)
            """,
            """
x = (2, 3, 99, 199, 991, 2, [], *tuple([1, 2, 3]), 9)
            """,
        ),
        (
            """
x = [*(), 2, 3, *{99}, (199, 991, 2), *[], [], *tuple([1, 2, 3]), 9]
            """,
            """
x = [2, 3, 99, 199, 991, 2, [], *tuple([1, 2, 3]), 9]
            """,
        ),
        (  # Two elements must be compared/hashed to say if unique
            """
x = {*(), 2, 3, *{99, 44}, (199, 991, 2), *[], [], *tuple([1, 2, 3]), 9}
            """,
            """
x = {2, 3, 99, 44, 199, 991, 2, [], *tuple([1, 2, 3]), 9}
            """,
        ),
        (
            """
x = (*(), 2, 3, *{99, 44}, (199, 991, 2), *[], [], *tuple([1, 2, 3]), 9)
            """,
            """
x = (2, 3, *{99, 44}, 199, 991, 2, [], *tuple([1, 2, 3]), 9)
            """,
        ),
        (
            """
x = [*(), 2, 3, *{99, 44}, (199, 991, 2), *[], [], *tuple([1, 2, 3]), 9]
            """,
            """
x = [2, 3, *{99, 44}, 199, 991, 2, [], *tuple([1, 2, 3]), 9]
            """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.simplify_collection_unpacks(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
