"""Test rename variables logic"""
import sys

from pyrefact import fixes


def main():
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


if __name__ == "__main__":
    main()
    sys.exit(0)
