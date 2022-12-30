#!/usr/bin/env python3

import sys
from pathlib import Path

import pyrefact

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:

    test_cases = (
        (
            """
def q() -> None:
    print(1)
    Spam.weeee()

class Spam:
    @staticmethod
    def weeee():
        print(3)

"Very important string statement"

class Foo:
    def __init__(self):
        self.x = 2

    @staticmethod
    def some_static_function(x, y) -> int:
        return 2 + x + y

    @staticmethod
    def some_other_static_function():
        print(3)
            """,
            """
def q() -> None:
    print(1)
    Spam.weeee()

class Spam:
    @staticmethod
    def weeee():
        print(3)

class Foo:
    def __init__(self):
        self.x = 2

    @staticmethod
    def some_static_function(x, y) -> int:
        return 2 + x + y

    @staticmethod
    def some_other_static_function():
        print(3)
            """,
        ),
    )

    for content, expected_abstraction in test_cases:
        processed_content = pyrefact.format_code(content, safe=True)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
