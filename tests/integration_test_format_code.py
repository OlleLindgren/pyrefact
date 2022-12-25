#!/usr/bin/env python3

import sys
from pathlib import Path

import pyrefact

sys.path.append(str(Path(__file__).parent))
import testing_infra


def main() -> int:
    test_cases = (
        (
            """
            """,
            """
            """,
        ),
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
    if x <= 10:
        x += 1
        x *= 12
        print(x > 30)
print(x)
            """,
        ),
        (
            """
x = 100
z = list()
def foo() -> bool:
    if x == 1:
        return False
    elif x == 2:
        return True
    else:
        if z:
            return False
        else:
            return True
print(foo())
            """,
            """
X = 100
Z = []
def _foo() -> bool:
    if X == 1:
        return False
    if X == 2:
        return True
    if Z:
        return False

    return True
print(_foo())
            """,
        ),
        (
            """
x = sorted(list(range(100)))[::3]
z = []
for a in x:
    if a % 3 == 0 and a % 4 == 2:
        for w in x:
            if w > len(x) // 2:
                z.append(w ** 3)
if all(y in [1, 2, 5] for y in sorted(set(list(z)))):
    print(z, x)
            """,
            """
X = sorted(range(100))[::3]
Z = [w**3 for a in X if a % 3 == 0 and a % 4 == 2 for w in X if w > len(X) // 2]
if all(y in {1, 2, 5} for y in sorted(set(Z))):
    print(Z, X)
            """,
        ),
    )
    for content, expected_abstraction in test_cases:
        processed_content = pyrefact.format_code(content)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
