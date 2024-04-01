#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (
        (  # Items to keys
            """
(x for x, _ in d.items())
        """,
        """
(x for x in d.keys())
        """,
        ),
        (
            """
{x: 100 - x for x, _ in d.items()}
        """,
        """
{x: 100 - x for x in d.keys()}
        """,
        ),
        (  # Items to values
            """
[x for _, x in d.items() if foo if bar if x if baz]
        """,
        """
[x for x in d.values() if foo if bar if x if baz]
        """,
        ),
        (
            """
{x for _, x in d.items() if foo if bar if x if baz}
        """,
        """
{x for x in d.values() if foo if bar if x if baz}
        """,
        ),
        (
            """
for x, _ in d.items():
    print(x)
        """,
            """
for x in d.keys():
    print(x)
        """,
        ),
        (
            """
for _, x in d.items():
    print(x)
        """,
            """
for x in d.values():
    print(x)
        """,
        ),
        (  # Implicit items
            """
{(x, d[x]) for x in d.keys()}
        """,
        """
{(x, d_x) for x, d_x in d.items()}
        """,
        ),
        (
            """
for x in d.keys():
    print(x)
    print(d[x])
        """,
            """
for x, d_x in d.items():
    print(x)
    print(d_x)
        """,
        ),
        (  # Implicit values
            """
[d[x] for x in d.keys()]
        """,
        """
[d_x for x, d_x in d.items()]
        """,
        ),
        (
            """
for x in d.keys():
    print(d[x])
        """,
            """
for x, d_x in d.items():
    print(d_x)
        """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = fixes.implicit_dict_keys_values_items(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
