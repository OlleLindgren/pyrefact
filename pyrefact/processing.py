import ast
import heapq
import re
from types import MappingProxyType
from typing import Collection, Iterable, Mapping, Union

import black

from pyrefact import constants, parsing


def _get_indent(content: str) -> int:
    indentation_whitespace = [x.group() for x in re.finditer(r"(?<![^\n]) *(?=[^\n])", content)]
    if indentation_whitespace:
        return min(len(x) for x in indentation_whitespace)

    return 0


def _deindent_code(content: str, indent: int) -> str:
    lines = content.splitlines(keepends=True)
    return "".join(line[indent:] if line.strip() else line for line in lines)


def _indent_code(content: str, indent: int) -> str:
    lines = content.splitlines(keepends=True)
    return "".join(" " * indent + line for line in lines)


def _match_wrapping_whitespace(new: str, initial: str) -> str:
    prefix_whitespace = max(re.findall(r"\A^[\s\n]*", initial), key=len)
    suffix_whitespace = max(re.findall(r"[\s\n]*\Z$", initial), key=len)
    return prefix_whitespace + new.strip() + suffix_whitespace


def format_with_black(content: str, *, line_length: int = 100) -> str:
    """Format code with black.

    Args:
        content (str): Python source code

    Returns:
        str: Formatted source code.
    """
    indent = _get_indent(content)
    deindented_code = _deindent_code(content, indent)
    formatted_deindented_code = black.format_str(
        deindented_code, mode=black.Mode(line_length=max(60, line_length - indent))
    )
    formatted_content = _indent_code(formatted_deindented_code, indent)
    whitespace_adjusted_content = _match_wrapping_whitespace(formatted_content, content)

    return whitespace_adjusted_content


def unparse(node: ast.AST) -> str:
    if constants.PYTHON_VERSION >= (3, 9):
        return ast.unparse(node)

    import astunparse

    content = astunparse.unparse(node)

    if not isinstance(
        node,
        (
            ast.FunctionDef,
            ast.ClassDef,
            ast.AsyncFunctionDef,
            ast.Expr,
            ast.Assign,
            ast.AnnAssign,
            ast.AugAssign,
            ast.If,
            ast.IfExp,
        ),
    ):
        content = content.rstrip()

    line_length = max(60, 100 - getattr(node, "col_offset", 0))
    content = format_with_black(content, line_length=line_length)
    indent = _get_indent(content)
    content = _deindent_code(content, indent).lstrip()

    return content


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
            chars.extend("pass\n")
        elif next_pass < i < next_pass + 3:
            continue
        else:
            if i > next_pass:
                next_pass = heapq.heappop(passes)
            if keep:
                chars.append(char)

    return "".join(chars)


def replace_nodes(content: str, replacements: Mapping[ast.AST, Union[ast.AST, str]]) -> str:
    for node, replacement in sorted(
        replacements.items(), key=lambda tup: (tup[0].lineno, tup[0].end_lineno), reverse=True
    ):
        start, end = parsing.get_charnos(node, content)
        code = content[start:end]
        if isinstance(replacement, str):
            new_code = replacement
            if new_code == code:
                continue
        elif isinstance(replacement, ast.AST):
            new_code = unparse(replacement)
        else:
            raise TypeError(f"Invalid replacement type: {type(replacement)}")
        lines = new_code.splitlines(keepends=True)
        indent = " " * node.col_offset
        start_indent = " " * (len(code) - len(code.lstrip(" ")))
        new_code = "".join(
            f"{indent if i > 0 else start_indent}{code}".rstrip()
            + ("\n" if code.endswith("\n") else "")
            for i, code in enumerate(lines)
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
        addition = unparse(node)
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
    actions.extend((x.lineno, "add", unparse(x), x) for x in additions)
    actions.extend((x.lineno, "delete", unparse(x), x) for x in removals)
    actions.extend((x.lineno, "replace", unparse(x), {x: y}) for x, y in replacements.items())

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
