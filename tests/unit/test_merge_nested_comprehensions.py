#!/usr/bin/env python3


import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (
        (  # GeneratorExp in SetComp
        """
{x for x in (y for y in range(10))}
        """,
        """
{x for x in range(10)}
        """,
        ),
        (  # GeneratorExp in ListComp
        """
[x for x in (y for y in range(10))]
        """,
        """
[x for x in range(10)]
        """,
        ),
        (  # GeneratorExp in DictComp
        """
{x: 1 for x in (y for y in range(10))}
        """,
        """
{x: 1 for x in range(10)}
        """,
        ),
        (  # GeneratorExp in ListComp
        """
{x for x in [y for y in range(10)]}
        """,
        """
{x for x in range(10)}
        """,
        ),
        (  # GeneratorExp in GeneratorExp
        """
(x for x in (y for y in range(10)))
        """,
        """
(x for x in range(10))
        """,
        ),
        (  # SetComp in SetComp
        """
{x for x in {y for y in range(10)}}
        """,
        """
{x for x in range(10)}
        """,
        ),
        (  # SetComp in SetComp, wrong name
        """
{x for x in {h for y in range(10)}}
        """,
        """
{x for x in {h for y in range(10)}}
        """,
        ),
        (  # SetComp in SetComp, non-trivial target
        """
{x for x in {y + 1 for y in range(10)}}
        """,
        """
{x for x in {y + 1 for y in range(10)}}
        """,
        ),
        (  # SetComp in SetComp, non-trivial iter
        """
{x for x in {y + 1 for y, z in range(10)}}
        """,
        """
{x for x in {y + 1 for y, z in range(10)}}
        """,
        ),
        (  # SetComp in ListComp
        """
[x for x in {y for y in range(10)}]
        """,
        """
[x for x in {y for y in range(10)}]
        """,
        ),
        (  # SetComp in GeneratorExp
        """
(x for x in {y for y in range(10)})
        """,
        """
(x for x in {y for y in range(10)})
        """,
        ),
        (  # SetComp in DictComp
        """
{x: 99 for x in {y for y in range(10)}}
        """,
        """
{x: 99 for x in range(10)}
        """,
        ),
        (  # DictComp in DictComp
        """
{x: 99 for x in {y: 3131 for y in range(10)}}
        """,
        """
{x: 99 for x in range(10)}
        """,
        ),
        (  # DictComp in DictComp
        """
{x: 99 for x in {3131: y for y in range(10)}}
        """,
        """
{x: 99 for x in {3131: y for y in range(10)}}
        """,
        ),
        (  # DictComp in DictComp
        """
{x: 99 for x in {z: y for y in range(10)}}
        """,
        """
{x: 99 for x in {z: y for y in range(10)}}
        """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.merge_nested_comprehensions(source)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
