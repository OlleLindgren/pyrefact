import ast
import heapq
from typing import Collection, Iterable, Mapping

from . import parsing


def remove_nodes(content: str, nodes: Iterable[ast.AST], root: ast.Module) -> str:
    """Remove ast nodes from code

    Args:
        content (str): Python source code
        nodes (Iterable[ast.AST]): Nodes to delete from code
        root (ast.Module): Complete corresponding module

    Returns:
        str: Code after deleting nodes
    """
    keep_mask = [True] * len(content)
    nodes = list(nodes)
    for node in nodes:
        start, end = parsing.get_charnos(node, content)
        print(f"Removing:\n{content[start:end]}")
        keep_mask[start:end] = [False] * (end - start)

    passes = [len(content) + 1]

    for node in ast.walk(root):
        for bodytype in "body", "finalbody", "orelse":
            if body := getattr(node, bodytype, []):
                if isinstance(body, list) and all(child in nodes for child in body):
                    print(f"Found empty {bodytype}")
                    start_charno, _ = parsing.get_charnos(body[0], content)
                    passes.append(start_charno)

    heapq.heapify(passes)

    next_pass = heapq.heappop(passes)
    chars = []
    for i, char, keep in zip(range(len(content)), content, keep_mask):
        if i == next_pass:
            chars.extend("pass")
        elif next_pass < i < next_pass + 3:
            continue
        else:
            if i > next_pass:
                next_pass = heapq.heappop(passes)
            if keep:
                chars.append(char)

    return "".join(chars)


def replace_nodes(content: str, replacements: Mapping[ast.AST, ast.AST]) -> str:
    for node, replacement in sorted(
        replacements.items(), key=lambda tup: (tup[0].lineno, tup[0].end_lineno), reverse=True
    ):
        start, end = parsing.get_charnos(node, content)
        code = content[start:end]
        new_code = ast.unparse(replacement)
        indent = " " * node.col_offset
        new_code = "".join(
            f"{indent * int(i > 0)}{code}"
            for i, code in enumerate(new_code.splitlines(keepends=True))
        )
        print(f"Replacing \n{code}\nWith      \n{new_code}")
        content = content[:start] + new_code + content[end:]

    return content


def insert_nodes(content: str, additions: Collection[ast.AST]) -> str:
    """Insert ast nodes in python source code.

    Args:
        content (str): Python source code before insertions.
        additions (Collection[ast.AST]): Ast nodes to add. Linenos must be accurate.

    Returns:
        str: Code with added asts.
    """
    lines = content.splitlines(keepends=True)

    for node in sorted(additions, key=lambda n: n.lineno, reverse=True):
        addition = ast.unparse(node)
        print(f"Adding:\n{addition}")
        lines = (
            lines[: node.lineno]
            + ["\n"] * 3
            + addition.splitlines(keepends=True)
            + ["\n"] * 3
            + lines[node.lineno :]
        )

    return "".join(lines)
