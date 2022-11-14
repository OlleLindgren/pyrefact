#!/usr/bin/env python3

import itertools
import re
import sys

from pyrefact import object_oriented


def _remove_multi_whitespace(content: str) -> str:
    return re.sub("\n{2,}", "\n", f"\n{content}\n").strip()


def _check_fixes(processed_content: str, expected_abstraction: str) -> int:
    processed_content = _remove_multi_whitespace(processed_content)
    expected_abstraction = _remove_multi_whitespace(expected_abstraction)

    if processed_content != expected_abstraction:
        for i, (expected, got) in enumerate(
            itertools.zip_longest(expected_abstraction.splitlines(), processed_content.splitlines())
        ):
            if expected != got:
                print(f"Line {i+1}, expected/got:\n{expected}\n{got}")
                return 1

    return 0


def _test_remove_unused_self_cls() -> int:
    test_cases = (
        (
            """
class Foo:
    def __init__(self):
        self.bar = 3

    def do_stuff(self):
        print(self.bar)

    @staticmethod
    def do_stuff_static(var, arg):
        print(var + arg)

    @classmethod
    def do_stuff_classmethod(cls, arg):
        cls.do_stuff_static(1, arg)

    def do_stuff_classmethod_2(self, arg):
        self.do_stuff_static(1, arg)

    @classmethod
    def do_stuff_classmethod_unused(cls, arg):
        print(arg)

    def do_stuff_no_self(self):
        print(3)

    @classmethod
    @functools.lru_cache(maxsize=None)
    @custom_decorator
    def i_have_many_decorators(cls):
        return 10
            """,
            """
class Foo:
    def __init__(self):
        self.bar = 3

    def do_stuff(self):
        print(self.bar)

    @staticmethod
    def do_stuff_static(var, arg):
        print(var + arg)

    @classmethod
    def do_stuff_classmethod(cls, arg):
        cls.do_stuff_static(1, arg)

    @classmethod
    def do_stuff_classmethod_2(cls, arg):
        cls.do_stuff_static(1, arg)

    @staticmethod
    def do_stuff_classmethod_unused(arg):
        print(arg)

    @staticmethod
    def do_stuff_no_self():
        print(3)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    @custom_decorator
    def i_have_many_decorators():
        return 10
            """,
        ),
    )

    for content, expected_abstraction in test_cases:
        processed_content = object_oriented.remove_unused_self_cls(content)
        returncode = _check_fixes(processed_content, expected_abstraction)
        if returncode:
            return returncode

    return 0


def _test_move_staticmethod_global() -> int:
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
    _weeee()

def _weeee():
    print(3)

class Spam:
    pass

"Very important string statement"

def _some_static_function(x, y) -> int:
    return 2 + x + y

def _some_other_static_function():
    print(3)

class Foo:
    def __init__(self):
        self.x = 2
            """,
        ),
    )

    for content, expected_abstraction in test_cases:
        processed_content = object_oriented.move_staticmethod_static_scope(content, preserve=set())
        returncode = _check_fixes(processed_content, expected_abstraction)
        if returncode != 0:
            return returncode

    return 0


def main() -> int:
    returncode = 0
    returncode += _test_remove_unused_self_cls()
    returncode += _test_move_staticmethod_global()

    return returncode


if __name__ == "__main__":
    sys.exit(main())
