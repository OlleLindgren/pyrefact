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
        (
            """
def asdf():
    x = None
    if 2 in {1, 2, 3}:
        print(3)
class Foo:
    @staticmethod
    def asdf():
        x = None
        if 2 in {1, 2, 3}:
            y = x is not None
            z = y or not y
            print(3)
            """,
            """
def asdf():
    if 2 in {1, 2, 3}:
        print(3)
class Foo:
    @staticmethod
    def asdf():
        if 2 in {1, 2, 3}:
            print(3)
            """,
        ),
        (
            """
class TestSomeStuff(unittest.TestCase):
    def test_important_stuff(self):
        assert 1 == 3
    @classmethod
    def test_important_stuff2(cls):
        assert 1 == 3
    def test_nonsense(self):
        self.assertEqual(1, 3)
            """,
            """
import unittest
class TestSomeStuff(unittest.TestCase):
    @staticmethod
    def test_important_stuff():
        assert 1 == 3
    @staticmethod
    def test_important_stuff2():
        assert 1 == 3
    def test_nonsense(self):
        self.assertEqual(1, 3)
            """,
        ),
        (
            '''
def foo() -> int:
    """This seems useless, but pyrefact shouldn't remove it with --safe"""
    return 10
            ''',
            '''
def foo() -> int:
    """This seems useless, but pyrefact shouldn't remove it with --safe"""
    return 10
            ''',
        ),
    )

    for content, expected_abstraction in test_cases:
        processed_content = pyrefact.format_code(content, safe=True)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
