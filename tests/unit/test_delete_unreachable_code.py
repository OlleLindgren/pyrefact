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
if False:
    print(2)
if True:
    print(3)
            """,
            """
if True:
    print(3)
            """,
        ),
        (
            """
if (1, 2, 3):
    print(2)
if ():
    print(3)
            """,
            """
if (1, 2, 3):
    print(2)
            """,
        ),
        (
            """
for i in range(10):
    print(2)
    continue
    import os
    print(os.getcwd())
            """,
            """
for i in range(10):
    print(2)
    continue
            """,
        ),
        (
            """
for i in range(10):
    print(2)
    if [1]:
        break
    import os
    print(os.getcwd())
            """,
            """
for i in range(10):
    print(2)
    if [1]:
        break
            """,
        ),
        (
            """
def foo():
    import random
    return random.random() > 0.5
    print(3)
for i in range(10):
    print(2)
    if foo():
        break
    else:
        continue
    import os
    print(os.getcwd())
            """,
            """
def foo():
    import random
    return random.random() > 0.5
for i in range(10):
    print(2)
    if foo():
        break
    else:
        continue
            """,
        ),
        (
            """
while 0:
    print(0)
while 3:
    print(3)
            """,
            """
while 3:
    print(3)
            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = fixes.delete_unreachable_code(content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
