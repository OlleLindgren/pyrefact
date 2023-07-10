#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import abstractions

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (
        (  # Small things are ok
            """
x = 404
y = 404
z = 404
w = 404
            """,
            """
x = 404
y = 404
z = 404
w = 404
            """,
        ),
        (  # Bigger things are not ok
            """
x = "some/path/to/something/cool"
y = "some/path/to/something/cool"
z = "some/path/to/something/cool"
w = "some/path/to/something/cool"
h = "some/path/to/something/cool"
g = "some/path/to/something/cool"
            """,
            """
SOME_PATH_TO_SOMETHING_COOL = "some/path/to/something/cool"
x = SOME_PATH_TO_SOMETHING_COOL
y = SOME_PATH_TO_SOMETHING_COOL
z = SOME_PATH_TO_SOMETHING_COOL
w = SOME_PATH_TO_SOMETHING_COOL
h = SOME_PATH_TO_SOMETHING_COOL
g = SOME_PATH_TO_SOMETHING_COOL
            """,
        ),
        (  # Best common scope is module scope
            """
import math

def foo():
    if False:
        x = "some/path/to/something/cool"
        y = "some/path/to/something/cool" + "qwerty"
    else:
        z = "some/path/to/something/cool"
    w = "some/path/to/something/cool" is True

def bar():
    h = "some/path/to/something/cool"
    g = "some/path/to/something/cool"
    print(h, g)
            """,
            """
import math
SOME_PATH_TO_SOMETHING_COOL = "some/path/to/something/cool"

def foo():
    if False:
        x = SOME_PATH_TO_SOMETHING_COOL
        y = SOME_PATH_TO_SOMETHING_COOL + "qwerty"
    else:
        z = SOME_PATH_TO_SOMETHING_COOL
    w = SOME_PATH_TO_SOMETHING_COOL is True

def bar():
    h = SOME_PATH_TO_SOMETHING_COOL
    g = SOME_PATH_TO_SOMETHING_COOL
    print(h, g)
            """,
        ),
        (  # Best common scope is not module scope
            """
import math

def foo():
    if False:
        x = "some/path/to/something/cool"
        y = "some/path/to/something/cool" + "qwerty"
    else:
        z = "some/path/to/something/cool"
    w = "some/path/to/something/cool" is True

    def bar():
        h = "some/path/to/something/cool"
        g = "some/path/to/something/cool"
        print(h, g)

    bar()
            """,
            """
import math

def foo():
    some_path_to_something_cool = "some/path/to/something/cool"
    if False:
        x = some_path_to_something_cool
        y = some_path_to_something_cool + "qwerty"
    else:
        z = some_path_to_something_cool
    w = some_path_to_something_cool is True

    def bar():
        h = some_path_to_something_cool
        g = some_path_to_something_cool
        print(h, g)

    bar()
            """,
        ),
        (  # Best common scope is not module scope
            """
def foo():
    d = {"spam": 3, "eggs": 2, "snake": 1336}
    print(d.get("spam"))

def boo():
    r = {"spam": 3, "eggs": 2, "snake": 1336}
    print(r.get("spam"))

def moo():
    def zoo():
        s = {"spam": 3, "eggs": 2, "snake": 1336}
        print(s.get("spam"))

    def qoo():
        d = {"spam": 3, "eggs": 2, "snake": 1336}
        print(d.get("spam"))

    zoo() is zoo()

def koo():
    print({"spam": 3, "eggs": 2, "snake": 1336}.get("spam"))
            """,
            """
PYREFACT_OVERUSED_CONSTANT_0 = {"spam": 3, "eggs": 2, "snake": 1336}
def foo():
    d = PYREFACT_OVERUSED_CONSTANT_0
    print(d.get("spam"))

def boo():
    r = PYREFACT_OVERUSED_CONSTANT_0
    print(r.get("spam"))

def moo():
    def zoo():
        s = PYREFACT_OVERUSED_CONSTANT_0
        print(s.get("spam"))

    def qoo():
        d = PYREFACT_OVERUSED_CONSTANT_0
        print(d.get("spam"))

    zoo() is zoo()

def koo():
    print(PYREFACT_OVERUSED_CONSTANT_0.get("spam"))
            """,
        ),
        (  # Best common scope is not module scope
            """
PYREFACT_OVERUSED_CONSTANT_0 = "foo"
_pyrefact_overused_constant_1 = "bar"
one("asdfasdfasdfasdfasdfasdfasdf")
two("asdfasdfasdfasdfasdfasdfasdf")
three("asdfasdfasdfasdfasdfasdfasdf")
four("asdfasdfasdfasdfasdfasdfasdf")
five("asdfasdfasdfasdfasdfasdfasdf")
six("asdfasdfasdfasdfasdfasdfasdf")
one("fdsafdsafdsafdsafdsafdsafdsa")
two("fdsafdsafdsafdsafdsafdsafdsa")
three("fdsafdsafdsafdsafdsafdsafdsa")
four("fdsafdsafdsafdsafdsafdsafdsa")
five("fdsafdsafdsafdsafdsafdsafdsa")
            """,
            """
ASDFASDFASDFASDFASDFASDFASDF = "asdfasdfasdfasdfasdfasdfasdf"
FDSAFDSAFDSAFDSAFDSAFDSAFDSA = "fdsafdsafdsafdsafdsafdsafdsa"
PYREFACT_OVERUSED_CONSTANT_0 = "foo"
_pyrefact_overused_constant_1 = "bar"
one(ASDFASDFASDFASDFASDFASDFASDF)
two(ASDFASDFASDFASDFASDFASDFASDF)
three(ASDFASDFASDFASDFASDFASDFASDF)
four(ASDFASDFASDFASDFASDFASDFASDF)
five(ASDFASDFASDFASDFASDFASDFASDF)
six(ASDFASDFASDFASDFASDFASDFASDF)
one(FDSAFDSAFDSAFDSAFDSAFDSAFDSA)
two(FDSAFDSAFDSAFDSAFDSAFDSAFDSA)
three(FDSAFDSAFDSAFDSAFDSAFDSAFDSA)
four(FDSAFDSAFDSAFDSAFDSAFDSAFDSA)
five(FDSAFDSAFDSAFDSAFDSAFDSAFDSA)
            """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = abstractions.overused_constant(source, root_is_static=True)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
