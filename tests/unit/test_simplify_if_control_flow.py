#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import abstractions, fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (
        (
            """
import random
x = 11
y = 12
z = random.random()
if z > 1 - z:
    do_stuff(x)
    do_stuff(y - x ** 2)
    print(doing_other_stuff(x) - do_stuff(y ** y))
else:
    do_stuff(y)
    do_stuff(x - y ** 2)
    print(doing_other_stuff(y) - do_stuff(x ** x))
            """,
            """
import random
x = 11
y = 12
z = random.random()
if z > 1 - z:
    var_1 = x
    var_2 = y
else:
    var_1 = y
    var_2 = x

do_stuff(var_1)
do_stuff(var_2 - var_1 ** 2)
print(doing_other_stuff(var_1) - do_stuff(var_2 ** var_2))
            """,
        ),
        (  # Too little code would be simplified => do not replace this
            """
import random
x = 11
y = 12
z = random.random()
if z > 1 - z:
    do_stuff(x)
else:
    do_stuff(y)
            """,
            """
import random
x = 11
y = 12
z = random.random()
if z > 1 - z:
    do_stuff(x)
else:
    do_stuff(y)
            """,
        ),
        (
            """
import random
x = 11
y = 12
z = random.random()
if z > 1 - z:
    do_stuff(x)
    if x > 1 / z:
        raise RuntimeError(f"Invalid value for {x}")

    print(random.randint(1 / z ** 2, - 1 / (z + y - x) ** 2))
else:
    do_stuff(y)
    if y > 1 / z:
        raise RuntimeError(f"Invalid value for {y}")

    print(random.randint(1 / z ** 2, - 1 / (z + x - y) ** 2))
            """,
            """
import random
x = 11
y = 12
z = random.random()
if z > 1 - z:
    var_1 = x
    var_2 = y
else:
    var_1 = y
    var_2 = x

do_stuff(var_1)
if var_1 > 1 / z:
    raise RuntimeError(f'Invalid value for {var_1}')

print(random.randint(1 / z ** 2, -1 / (z + var_2 - var_1) ** 2))
            """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = abstractions.simplify_if_control_flow(source)
        processed_content = fixes.breakout_common_code_in_ifs(processed_content)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
