#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import performance_pandas

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
for x, _ in df.iterrows():
    print(x)
        """,
            """
for x in df.index:
    print(x)
        """,
        ),
        (
            """
stuff = [x for x, _ in df.iterrows()]
print(stuff[-1])
        """,
        """
stuff = [x for x in df.index]
print(stuff[-1])
        """,
        ),
        (
            """
stuff = df.iterrows()
print(sum(stuff))
        """,
        """
stuff = df.iterrows()
print(sum(stuff))
        """,
        ),
        (
            """
for x, i in df.iterrows():
    print(x)
        """,
            """
for x, i in df.iterrows():
    print(x)
        """,
        ),
        (
            """
stuff = [x for x, q in df.iterrows()]
print(stuff[-1])
        """,
        """
stuff = [x for x, q in df.iterrows()]
print(stuff[-1])
        """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = performance_pandas.replace_iterrows_index(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
