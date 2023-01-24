#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (
        (  # Before
            """
import random
if random.random() < 2:
    print(100)
    print(3)
else:
    print(100)
    print(21)
            """,
            """
import random
print(100)
if random.random() < 2:
    print(3)
else:
    print(21)
            """,
        ),
        (  # After
            """
import random
if random.random() < 2:
    print(3)
    print(100)
else:
    print(21)
    print(100)
            """,
            """
import random
if random.random() < 2:
    print(3)
else:
    print(21)
print(100)
            """,
        ),
        (  # Before and after
            """
import random
if random.random() < 2:
    print(102)
    print(3)
    print(100)
else:
    print(102)
    print(21)
    print(100)
            """,
            """
import random
print(102)
if random.random() < 2:
    print(3)
else:
    print(21)
print(100)
            """,
        ),
        (  # body becomes empty
            """
import random
if random.random() < 2:
    print(100)
else:
    print(33)
    print(100)
            """,
            """
import random
if random.random() < 2:
    pass
else:
    print(33)
print(100)
            """,
        ),
        (  # orelse becomes empty
            """
import random
if random.random() < 2:
    print(100)
    print(33)
else:
    print(100)
            """,
            """
import random
print(100)
if random.random() < 2:
    print(33)
else:
    pass
            """,
        ),
        (  # nested if, match start
            """
import random
if random.random() < 2:
    print(100)
    print(33)
elif random.random() >= 2:
    print(100)
else:
    print(100)
    print(21)
            """,
            """
import random
print(100)
if random.random() < 2:
    print(33)
elif random.random() >= 2:
    pass
else:
    print(21)
            """,
        ),
        (  # nested if, match end
            """
import random
import heapq
if random.random() < 2:
    print(33)
    print(100)
elif random.random() >= 2:
    if random is heapq:
        print(100)
    else:
        print(300)
        print(100)
else:
    print(21)
    print(100)
            """,
            """
import random
import heapq
if random.random() < 2:
    print(33)
elif random.random() >= 2:
    if random is heapq:
        pass
    else:
        print(300)
else:
    print(21)
print(100)
            """,
        ),
        (  # nested if, match inner but not outer
            """
import random
import heapq
if random.random() < 2:
    print(33)
    print(100)
elif random.random() >= 2:
    if random is heapq:
        print(100)
    else:
        print(100)
        print(300)
else:
    print(21)
    print(100)
            """,
            """
import random
import heapq
if random.random() < 2:
    print(33)
    print(100)
elif random.random() >= 2:
    print(100)
    if random is heapq:
        pass
    else:
        print(300)
else:
    print(21)
    print(100)
            """,
        ),
        (  # nested if, no match since the else is missing in the inner condition
            """
import random
import heapq
if random.random() < 2:
    print(33)
    print(100)
elif random.random() >= 2:
    if random is heapq:
        print(100)
else:
    print(21)
    print(100)
            """,
            """
import random
import heapq
if random.random() < 2:
    print(33)
    print(100)
elif random.random() >= 2:
    if random is heapq:
        print(100)
else:
    print(21)
    print(100)
            """,
        ),
        (  # nested if, inner ifs are identical
            """
import random
import heapq
if random.random() < 2:
    if random is heapq:
        print(100)
else:
    if random is heapq:
        print(100)
            """,
            """
import random
import heapq
if random is heapq:
    print(100)
if random.random() < 2:
    pass
else:
    pass
            """,
        ),
        (  # TODO nested if with identical inner parts
            """
import random
import heapq
if random.random() < 2:
    if random is heapq:
        print(100)
elif random.random() >= 2:
    if random is heapq:
        print(100)
else:
    if random is heapq:
        print(100)
            """,
            """
import random
import heapq
if random.random() < 2:
    if random is heapq:
        print(100)
elif random.random() >= 2:
    if random is heapq:
        print(100)
else:
    if random is heapq:
        print(100)
            """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.breakout_common_code_in_ifs(source)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
