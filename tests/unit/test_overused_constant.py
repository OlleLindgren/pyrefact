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
PYREFACT_OVERUSED_CONSTANT_0 = 'some/path/to/something/cool'
x = PYREFACT_OVERUSED_CONSTANT_0
y = PYREFACT_OVERUSED_CONSTANT_0
z = PYREFACT_OVERUSED_CONSTANT_0
w = PYREFACT_OVERUSED_CONSTANT_0
h = PYREFACT_OVERUSED_CONSTANT_0
g = PYREFACT_OVERUSED_CONSTANT_0
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
PYREFACT_OVERUSED_CONSTANT_0 = 'some/path/to/something/cool'

def foo():
    if False:
        x = PYREFACT_OVERUSED_CONSTANT_0
        y = PYREFACT_OVERUSED_CONSTANT_0 + "qwerty"
    else:
        z = PYREFACT_OVERUSED_CONSTANT_0
    w = PYREFACT_OVERUSED_CONSTANT_0 is True

def bar():
    h = PYREFACT_OVERUSED_CONSTANT_0
    g = PYREFACT_OVERUSED_CONSTANT_0
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
    pyrefact_overused_constant_0 = 'some/path/to/something/cool'
    if False:
        x = pyrefact_overused_constant_0
        y = pyrefact_overused_constant_0 + "qwerty"
    else:
        z = pyrefact_overused_constant_0
    w = pyrefact_overused_constant_0 is True

    def bar():
        h = pyrefact_overused_constant_0
        g = pyrefact_overused_constant_0
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
PYREFACT_OVERUSED_CONSTANT_0 = {'spam': 3, 'eggs': 2, 'snake': 1336}
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
    )

    for source, expected_abstraction in test_cases:
        processed_content = abstractions.overused_constant(source, root_is_static=True)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
