#!/usr/bin/env python3


import sys
from pathlib import Path

from pyrefact import object_oriented

sys.path.append(str(Path(__file__).parent))
import testing_infra


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
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

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
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


def main() -> int:
    returncode = 0
    returncode += _test_remove_unused_self_cls()
    returncode += _test_move_staticmethod_global()

    return returncode


if __name__ == "__main__":
    sys.exit(main())
