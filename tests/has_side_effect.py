import ast
import builtins
import sys

from pyrefact.parsing import has_side_effect


def main() -> int:
    """Test parsing.has_side_effect

    Returns:
        int: 1 if the function behaves incorrectly, otherwise 0
    """
    whitelist = set(dir(builtins)) - {"print", "exit"}
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
    ):
        node = ast.parse(source).body[0]
        if has_side_effect(node, whitelist):
            print("Statement has side effect, but should not:")
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
    ):
        node = ast.parse(source).body[0]
        if not has_side_effect(node, whitelist):
            print("Statement has no side effect, but should:")
            print(source)
            print("Ast structure:")
            print(ast.dump(node, indent=2))
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
