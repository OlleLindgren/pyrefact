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
for x in range(100):
    if x > 10:
        y = 13
    else:
        x += 1
        x *= 12
        print(x > 30)
        y = 100 - sum(x, 2, 3)
print(x)
            """,
            """
for x in range(100):
    if x > 10:
        y = 13
        continue
    else:
        x += 1
        x *= 12
        print(x > 30)
        y = 100 - sum(x, 2, 3)
print(x)
            """,
        ),
        (
            """
for i in range(100):
    if i % 3 == 2:
        print(i ** i)
        print(i ** 3)
        import os
        import sys
        print(os.getcwd())
        print(sys is os)
            """,
            """
for i in range(100):
    if i % 3 != 2:
        continue
    else:
        print(i ** i)
        print(i ** 3)
        import os
        import sys
        print(os.getcwd())
        print(sys is os)
            """,
        ),
        (
            """
for i in range(100):
    if i % 3 == 2:
        print(i ** i)
        print(i ** 3)
        print(i ** 4)
            """,
            """
for i in range(100):
    if i % 3 == 2:
        print(i ** i)
        print(i ** 3)
        print(i ** 4)
            """,
        ),
        (
            """
for i in range(100):
    if i % 3 == 2:
        if i % 6 == 1:
            print(i ** i)
            print(i ** 3)
            print(i ** 4)
            """,
            """
for i in range(100):
    if i % 3 == 2:
        if i % 6 == 1:
            print(i ** i)
            print(i ** 3)
            print(i ** 4)
            """,
        ),
        (
            """
for i in range(100):
    if i % 3 == 2:
        print(i ** i)
        if i % 6 == 1:
            print(i ** 3)
            print(i ** 4)
            """,
            """
for i in range(100):
    if i % 3 == 2:
        print(i ** i)
        if i % 6 == 1:
            print(i ** 3)
            print(i ** 4)
            """,
        ),
        (
            """
for i in range(100):
    if i % 3 == 2:
        print(i ** i)
        print(i ** (i - 1))
        for i in range(10000):
            if i >= 1337 and i is not 99:
                print(i - i)
        if i % 6 == 1:
            print(i ** 3)
            print(i ** 4)
            """,
            """
for i in range(100):
    if i % 3 != 2:
        continue
    else:
        print(i ** i)
        print(i ** (i - 1))
        for i in range(10000):
            if i >= 1337 and i is not 99:
                print(i - i)
        if i % 6 == 1:
            print(i ** 3)
            print(i ** 4)
            """,
        ),
    )

    for content, expected_abstraction in test_cases:
        processed_content = fixes.early_continue(content)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
