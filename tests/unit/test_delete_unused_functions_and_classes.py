#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """

def function() -> int:
    return 0

def _private_function() -> bool:
    return 11

def _used_function() -> str:
    '''Docstring mentioning _private_function()'''
    return "this function is used"

def _user_of_used_function() -> str:
    return _used_function()

class Foo:
    def bar(self) -> bool:
        return False

    @property
    def spammy(self) -> bool:
        return True

if __name__ == "__main__":
    Foo().spammy
    _user_of_used_function()

        """,
        """
def _used_function() -> str:
    '''Docstring mentioning _private_function()'''
    return "this function is used"

def _user_of_used_function() -> str:
    return _used_function()

class Foo:
    @property
    def spammy(self) -> bool:
        return True

if __name__ == "__main__":
    Foo().spammy
    _user_of_used_function()

        """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.delete_unused_functions_and_classes(source)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
