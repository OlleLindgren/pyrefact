import ast
import heapq
from types import MappingProxyType
from typing import Collection, Iterable, Mapping, Optional

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
        if isinstance(node, ast.Module):
            continue
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


def replace_nodes(content: str, replacements: Mapping[ast.AST, Optional[ast.AST]]) -> str:
    for node, replacement in sorted(
        replacements.items(), key=lambda tup: (tup[0].lineno, tup[0].end_lineno), reverse=True
    ):
        start, end = parsing.get_charnos(node, content)
        code = content[start:end]
        new_code = ast.unparse(replacement) if replacement is not None else ""
        indent = " " * node.col_offset
        new_code = "".join(
            f"{indent * int(i > 0)}{code}"
            for i, code in enumerate(new_code.splitlines(keepends=True))
        )
        if new_code:
            print(f"Replacing \n{code}\nWith      \n{new_code}")
        else:
            print(f"Removing \n{code}")
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
        col_offset = getattr(node, "col_offset", 0)
        print(f"Adding:\n{addition}")
        lines = (
            lines[: node.lineno]
            + ["\n"] * 3
            + [" " * col_offset + line for line in addition.splitlines(keepends=True)]
            + ["\n"] * 3
            + lines[node.lineno :]
        )

    return "".join(lines)


def alter_code(
    content: str,
    root: ast.AST,
    *,
    additions: Collection[ast.AST] = frozenset(),
    removals: Collection[ast.AST] = frozenset(),
    replacements: Mapping[ast.AST, ast.AST] = MappingProxyType({}),
) -> str:
    """Alter python code.

    This coordinates additions, removals and replacements in a safe way.

    Args:
        content (str): Python source code
        root (ast.AST): Parsed AST tree corresponding to source code
        additions (Collection[ast.AST], optional): Nodes to add
        removals (Collection[ast.AST], optional): Nodes to remove
        replacements (Mapping[ast.AST, ast.AST], optional): Nodes to replace

    Raises:
        ValueError: _description_

    Returns:
        str: _description_
    """
    actions = []

    # Yes, this unparsing is an expensive way to sort the nodes.
    # However, this runs relatively infrequently and should not have a big
    # performance impact.
    actions.extend((x.lineno, "add", ast.unparse(x), x) for x in additions)
    actions.extend((x.lineno, "delete", ast.unparse(x), x) for x in removals)
    actions.extend((x.lineno, "replace", ast.unparse(x), {x: y}) for x, y in replacements.items())

    # a < d => deletions will go before additions if same lineno and reversed sorting.
    for _, action, _, value in sorted(actions, reverse=True):
        if action == "add":
            content = insert_nodes(content, [value])
        elif action == "delete":
            content = remove_nodes(content, [value], root)
        elif action == "replace":
            content = replace_nodes(content, value)
        else:
            raise ValueError(f"Invalid action: {action}")

    return content
