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
    print(3)
if True:
    print(2)
            """,
            """
print(2)
            """,
        ),
        (
            """
if ():
    print(3)
if []:
    print(2)
            """,
            """
            """,
        ),
        (
            """
x = 0
if x == 3:
    print(x)
if []:
    print(2)
else:
    print(x + x)
if [1]:
    print(222222)
else:
    print(x ** x)
            """,
            """
x = 0
if x == 3:
    print(x)

print(x + x)
print(222222)
            """,
        ),
        (
            """
import sys
while False:
    sys.exit(0)
while sys.executable == "/usr/bin/python":
    print(7)
while True:
    sys.exit(2)
            """,
            """
import sys
while sys.executable == "/usr/bin/python":
    print(7)
while True:
    sys.exit(2)
            """,
        ),
        (
            """
x = 13
a = x if x > 3 else 0
b = x if True else 0
c = x if False else 2
d = 13 if () else {2: 3}
e = 14 if list((1, 2, 3)) else 13
print(3 if 2 > 0 else 2)
print(14 if False else 2)
            """,
            """
x = 13
a = x if x > 3 else 0
b = x
c = 2
d = {2: 3}
e = 14 if list((1, 2, 3)) else 13
print(3 if 2 > 0 else 2)
print(2)
            """,
        ),
    )

    for content, expected_abstraction in test_cases:
        processed_content = fixes.remove_dead_ifs(content)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
