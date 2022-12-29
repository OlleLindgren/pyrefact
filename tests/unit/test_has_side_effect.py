#!/usr/bin/env python3

import ast
import sys

from pyrefact import constants, parsing


def main() -> int:
    """Test parsing.has_side_effect

    Returns:
        int: 1 if the function behaves incorrectly, otherwise 0
    """
    whitelist = constants.BUILTIN_FUNCTIONS - {"print", "exit"}
    for source in (
        "{}",
        "()",
        "[]",
        "1",
        "-1",
        "2-1",
        "2.-1",
        "False",
        "pass",
        "print",
        "exit",
        "x",
        "x+1",
        "x**10",
        "x > 2",
        "x < 2 < x",
        "{1, 2, 3}",
        "[1, 2, 3]",
        "(1, 2, 3)",
        "[1]*2",
        "[] + []",
        "range(2)",
        "list(range(2, 6))",
        "[x for x in range(3)]",
        "(x for x in range(3))",
        "{x for x in range(3)}",
        "{x: x-1 for x in range(3)}",
        "{**{1: 2}}",
        "lambda: 2",
        "{1: 2}[1]",
        "{1: sum}[1]((3, 3, 3))",
        '{1: sum((2, 3, 6, 0)), "asdf": 13-12}',
        "_=2",
        "(_:=3)",
        "_+=11",
        "_: int = 'q'",
        """for _ in range(10):
    1
""",
    ):
        node = parsing.parse(source).body[0]
        if not parsing.has_side_effect(node, whitelist):
            continue

        print("Ast has side effect, but should not:")
        print(source)
        print("Ast structure:")
        print(ast.dump(node, indent=2))
        return 1

    for source in (
        "x=100",
        "requests.post(*args, **kwargs)",
        "x.y=z",
        "print()",
        "exit()",
        """def f() -> None:
    return 1
        """,
        "g=2",
        "(h:=3)",
        """for i in range(10):
    1
""",
        "mysterious_function()",
        "flat_dict[value] = something",
        "nested_dict[value][item] = something",
        "deep_nested_dict[a][b][c][d][e][f][g] = something",
    ):
        node = parsing.parse(source).body[0]
        if parsing.has_side_effect(node, whitelist):
            continue

        print("Ast has no side effect, but should:")
        print(source)
        print("Ast structure:")
        print(ast.dump(node, indent=2))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
