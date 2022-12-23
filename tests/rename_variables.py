#!/usr/bin/env python3
"""Test rename variables logic"""


import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parent))
import testing_infra


def _unit():
    print(f"Running tests in {__file__}...")
    assert fixes._list_words("AbCDEFghi0K123___LM4Nopq_rstUvwxYz") == [
        "Ab",
        "CDE",
        "Fghi0",
        "K123",
        "LM4",
        "Nopq",
        "rst",
        "Uvwx",
        "Yz",
    ]
    assert fixes._make_camelcase("snake_case_name") == "SnakeCaseName"
    assert fixes._make_snakecase("snake_case_name") == "snake_case_name"
    assert fixes._make_camelcase("CamelCaseName") == "CamelCaseName"
    assert fixes._make_snakecase("CamelCaseName") == "camel_case_name"
    assert (
        fixes._make_snakecase("AbCDEFghi0K123___LM4Nopq_rstUvwxYz")
        == "ab_cde_fghi0_k123_lm4_nopq_rst_uvwx_yz"
    )
    assert (
        fixes._make_camelcase("AbCDEFghi0K123___LM4Nopq_rstUvwxYz")
        == "AbCdeFghi0K123Lm4NopqRstUvwxYz"
    )
    assert fixes._make_camelcase("_____x___") == "X"
    assert fixes._make_snakecase("_____x___") == "x"

    assert fixes._rename_class("CLASS_NAME", private=False) == "ClassName"
    assert fixes._rename_class("CLASS_NAME", private=True) == "_ClassName"

    assert fixes._rename_variable("VAR_NAME", private=False, static=False) == "var_name"
    assert fixes._rename_variable("VAR_NAME", private=False, static=True) == "VAR_NAME"
    assert fixes._rename_variable("VAR_NAME", private=True, static=False) == "_var_name"
    assert fixes._rename_variable("VAR_NAME", private=True, static=True) == "_VAR_NAME"
    assert fixes._rename_variable("__var_Name__", private=False, static=False) == "var_name"
    assert fixes._rename_variable("__var_Name__", private=False, static=True) == "VAR_NAME"
    assert fixes._rename_variable("__var_Name__", private=True, static=False) == "_var_name"
    assert fixes._rename_variable("__var_Name__", private=True, static=True) == "_VAR_NAME"


def _integration() -> int:
    test_cases = (
        (
            """
some_variable = collections.namedtuple("some_variable", ["field", "foo", "bar"])
variable = TypeVar("variable")
T = Mapping[Tuple[int, int], Collection[str]]
something_else = 1


def foo() -> Tuple[some_variable, T]:
    _ax = 4
    print(_ax)
    R = 3
    print(R)
    s = 2
    print(s)
    return some_variable(1, 2, 3)

moose = namedtuple("moose", ["field", "foo", "bar"])

ax = 22
print(ax)


def main() -> None:
    bar: some_variable = foo()
    print(bar)
    return 0
        """,
            """
SomeVariable = collections.namedtuple("some_variable", ["field", "foo", "bar"])
Variable = TypeVar("variable")
T = Mapping[Tuple[int, int], Collection[str]]
SOMETHING_ELSE = 1


def _foo() -> Tuple[SomeVariable, T]:
    ax = 4
    print(ax)
    r = 3
    print(r)
    s = 2
    print(s)
    return SomeVariable(1, 2, 3)

Moose = namedtuple("moose", ["field", "foo", "bar"])

AX = 22
print(AX)


def _main() -> None:
    bar: SomeVariable = _foo()
    print(bar)
    return 0
        """,
        ),
    )
    for content, expected_abstraction in test_cases:

        processed_content = fixes.align_variable_names_with_convention(content, set())
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


def main():
    _unit()
    return _integration()


if __name__ == "__main__":
    sys.exit(main())
