#!/usr/bin/env python3

import ast
import sys

from pyrefact import abstractions


def main() -> None:
    for left_expression, right_expression in (
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
    ):
        left_node = ast.parse(left_expression).body[0]
        right_node = ast.parse(right_expression).body[0]
        preserved_names = set()
        left_hash = abstractions.hash_node(left_node, preserved_names)
        right_hash = abstractions.hash_node(right_node, preserved_names)
        assert (
            left_hash == right_hash
        ), f"""Hashes differ for:

{ast.dump(left_node, indent=2)}

and

{ast.dump(right_node, indent=2)}
"""


if __name__ == "__main__":
    main()
    sys.exit(0)
