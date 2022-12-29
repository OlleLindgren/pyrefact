#!/usr/bin/env python3

import ast
import sys

from pyrefact import abstractions, constants, parsing


def _error_message(left: ast.AST, right: ast.AST, *, positive: bool) -> str:
    if positive:
        msg = "Hashes are different, but should be the same for"
    else:
        msg = "Hashes are the same, but should be different for"
    return f"""{msg}:

    {ast.dump(left, indent=2)}

    and

    {ast.dump(right, indent=2)}
    """


def main() -> None:
    preserved_names = constants.BUILTIN_FUNCTIONS
    positives = (
        ("lambda x: x", "lambda y: y"),
        ("lambda x: (x**3 - x**2) // 11", "lambda aaaa: (aaaa**3 - aaaa**2) // 11"),
        (
            """
def q(a, b, c):
    if a and b and c:
        return q(a)
    elif c and d > 0:
        return -a
    return a*b +c*a
        """,
            """
def qz(aaa, bbb, ccc):
    if aaa and bbb and ccc:
        return qz(aaa)
    elif ccc and ddd > 0:
        return -aaa
    return aaa*bbb +ccc*aaa
        """,
        ),
    )
    negatives = (("lambda x: list(x)", "lambda x: set(x)"),)

    for left_expression, right_expression in positives:
        left_node = parsing.parse(left_expression).body[0]
        right_node = parsing.parse(right_expression).body[0]
        left_hash = abstractions.hash_node(left_node, preserved_names)
        right_hash = abstractions.hash_node(right_node, preserved_names)
        assert left_hash == right_hash, _error_message(left_node, right_node, positive=True)

    for left_expression, right_expression in negatives:
        left_node = parsing.parse(left_expression).body[0]
        right_node = parsing.parse(right_expression).body[0]
        left_hash = abstractions.hash_node(left_node, preserved_names)
        right_hash = abstractions.hash_node(right_node, preserved_names)
        assert left_hash != right_hash, _error_message(left_node, right_node, positive=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
