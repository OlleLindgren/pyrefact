#!/usr/bin/env python3

import itertools
import re
import sys

from pyrefact import abstractions


def _remove_multi_whitespace(content: str) -> str:
    return re.sub("\n{2,}", "\n", f"\n{content}\n").strip()


def main() -> int:
    test_cases = (
        (
            """
for x in range(11):
    if x == 3:
        continue
    if x == 5:
        continue
    if x == 8:
        continue

    print(x)
    """,
            """
def _pyrefact_abstraction_1(x) -> bool:
    if x == 3:
        return False
    if x == 5:
        return False
    if x == 8:
        return False

    return True


for x in range(11):
    if not _pyrefact_abstraction_1(x):
        continue
    pass
    pass

    print(x)
    """,
        ),
        (
            """
for x in range(11):
    if x == 3:
        break
    if x == 5:
        break
    if x == 8:
        break

    print(x)
    """,
            """
def _pyrefact_abstraction_1(x) -> bool:
    if x == 3:
        return True
    if x == 5:
        return True
    if x == 8:
        return True

    return False


for x in range(11):
    if _pyrefact_abstraction_1(x):
        break
    pass
    pass

    print(x)
    """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = abstractions.create_abstractions(content)

        processed_content = _remove_multi_whitespace(processed_content)
        expected_abstraction = _remove_multi_whitespace(expected_abstraction)

        if processed_content != expected_abstraction:
            for i, (expected, got) in enumerate(
                itertools.zip_longest(
                    expected_abstraction.splitlines(), processed_content.splitlines()
                )
            ):
                if expected != got:
                    print(f"Line {i+1}, expected/got:\n{expected}\n{got}")
                    return 1


if __name__ == "__main__":
    sys.exit(main())
