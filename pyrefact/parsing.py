import ast
import dataclasses
import functools
import itertools
import re
from typing import Collection, Iterable, Sequence, Tuple

# match :=, do not match  =, >=, <=, ==, !=
#  match =,  do not match :=, >=, <=, ==, !=


re.compile(r"( |\n)+")
re.compile(r"(?<![^\n]) *")


@dataclasses.dataclass()
class Statement:
    start: int
    end: int
    ast_node: ast.AST
    indent: int
    paranthesis_depth: int
    statement: str


def is_valid_python(content: str) -> bool:
    """Determine if source code is valid python.

    Args:
        content (str): Python source code

    Returns:
        bool: True if content is valid python.
    """
    try:
        ast.parse(content, "")
        return True
    except SyntaxError:
        return False


@functools.lru_cache()
def get_is_code_mask(content: str) -> Sequence[bool]:
    """Get boolean mask of whether content is code or not.

    Args:
        content (str): Python source code

    Returns:
        Sequence[bool]: True if source code, False if comment or string, for every character.
    """
    regexes = {
        '"""': '"""',
        '"': '"',
        "'''": "'''",
        "'": "'",
        "#": "#",
        "{": r"(?<![{]){(?![{])",
        "}": r"(?<![}])}(?![}])",
        "\n": "\n",
    }

    statement_breaks = set()

    for key, regex in regexes.items():
        statement_breaks.update(
            (hit.start(), hit.end(), key, content[hit.start() - 1] == "f")
            for hit in re.finditer(regex, content)
        )

    triple_overlapping_singles = set()

    for hit in statement_breaks:
        if hit[2] in ("'", '"'):
            triple_matches = [
                candidate
                for candidate in statement_breaks
                if candidate[2] == hit[2] * 3 and candidate[0] <= hit[0] <= hit[1] <= candidate[1]
            ]
            if triple_matches:
                triple_overlapping_singles.add(hit)

    for hit in triple_overlapping_singles:
        statement_breaks.discard(hit)

    string_types = {
        '"""',
        '"',
        "'''",
        "'",
    }
    inline_comment = "#"
    newline_character = "\n"
    f_escape_start = "{"
    f_escape_end = "}"

    comment_super_ranges = []

    context = []

    while statement_breaks:
        start_item = min(statement_breaks)
        statement_breaks.remove(start_item)
        start, s_end, key, is_f = start_item
        if key in string_types:
            end_key = key
        elif key == inline_comment:
            end_key = newline_character
        elif key == f_escape_start:
            end_key = None
            comment_super_ranges.append([start, None, key, False])
            continue
        elif key == f_escape_end:
            for rng in reversed(comment_super_ranges):
                if rng[2] == f_escape_start and rng[1] is None:
                    rng[1] = s_end
                    break
            else:
                raise ValueError(f"Cannot find corresponding start for {key}")
            continue
        elif key == newline_character:
            continue
        else:
            raise ValueError(f"Unknown delimiter: {key}")

        if not statement_breaks:
            break

        end_item = min(item for item in statement_breaks if item[2] == end_key)
        statement_breaks.remove(end_item)
        _, end, _, _ = end_item
        if is_f:
            context.append(key)
        else:
            statement_breaks = {item for item in statement_breaks if item[0] >= end}

        comment_super_ranges.append([start, end, key, is_f])

    mask = [True] * len(content)
    for start, end, key, is_f in sorted(comment_super_ranges):
        if key in string_types or key == inline_comment:
            value = False
        elif key == f_escape_start:
            value = True
        else:
            raise ValueError(f"Invalid range start delimiter: {key}")

        if value is False:
            if is_f:
                mask[start - 1 : end] = [value] * (end + 1 - start)
            else:
                mask[start:end] = [value] * (end - start)
        else:
            mask[start + 1 : end - 1] = [value] * (end - start - 2)

    return tuple(mask)


def _unpack_ast_target(target: ast.AST) -> Iterable[str]:
    if isinstance(target, ast.Name):
        yield target
        return
    if isinstance(target, ast.Tuple):
        for subtarget in target.elts:
            yield from _unpack_ast_target(subtarget)


def iter_assignments(ast_tree: ast.Module) -> Iterable[str]:
    """Iterate over defined variables in code

    Args:
        content (str): Python source code

    Yields:
        Tuple[Statement, str]: Statement, and lvalue assigned in statement
    """
    for node in ast_tree.body:
        if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            yield from _unpack_ast_target(node.target)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                yield from _unpack_ast_target(target)


def iter_funcdefs(ast_tree: ast.Module) -> Iterable[ast.FunctionDef]:
    """Iterate over defined variables in code

    Args:
        content (str): Python source code

    Yields:
        Tuple[Statement, str]: Statement, and lvalue assigned in statement
    """
    for node in ast_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def iter_classdefs(ast_tree: ast.Module) -> Iterable[ast.ClassDef]:
    """Iterate over defined variables in code

    Args:
        content (str): Python source code

    Yields:
        Tuple[Statement, str]: Statement, and lvalue assigned in statement
    """
    for node in ast_tree.body:
        if isinstance(node, (ast.ClassDef)):
            yield node


def has_side_effect(
    node: ast.AST,
    safe_callable_whitelist: Collection[str] = frozenset(),
) -> bool:
    """Determine if a statement has a side effect.

    Args:
        statement (Statement): Statement to check
        safe_callable_whitelist (Collection[str]): Items known to not have a side effect
        used_variables (Collection[str]): Used variables and names

    Returns:
        bool: True if it may have a side effect.

    """
    if isinstance(node, (ast.Constant, ast.Pass)):
        return False

    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        return any(has_side_effect(value, safe_callable_whitelist) for value in node.elts)

    if isinstance(node, ast.Dict):
        return any(
            has_side_effect(value, safe_callable_whitelist)
            for value in itertools.chain(node.keys, node.values)
        )

    if isinstance(node, ast.Expr):
        return has_side_effect(node.value, safe_callable_whitelist)

    if isinstance(node, ast.Expression):
        return has_side_effect(node.body, safe_callable_whitelist)

    if isinstance(node, ast.UnaryOp):
        return has_side_effect(node.operand, safe_callable_whitelist)

    if isinstance(node, ast.BinOp):
        return any(
            has_side_effect(child, safe_callable_whitelist) for child in (node.left, node.right)
        )

    if isinstance(node, ast.BoolOp):
        return any(has_side_effect(value, safe_callable_whitelist) for value in node.values)

    if isinstance(node, ast.Name):
        return isinstance(node.ctx, ast.Load)

    if isinstance(node, ast.Subscript):
        return (
            has_side_effect(node.value, safe_callable_whitelist)
            or has_side_effect(node.slice, safe_callable_whitelist)
            or not isinstance(node.ctx, ast.Load)
        )

    if isinstance(node, ast.Slice):
        return any(
            has_side_effect(child, safe_callable_whitelist) for child in (node.lower, node.upper)
        )

    if isinstance(node, (ast.DictComp)) and has_side_effect(node.value, safe_callable_whitelist):
        return True

    if isinstance(node, (ast.SetComp, ast.ListComp, ast.GeneratorExp, ast.DictComp)):
        return any(has_side_effect(item, safe_callable_whitelist) for item in node.generators)

    if isinstance(node, ast.comprehension):
        if (
            isinstance(node.target, ast.Name)
            and isinstance(node.iter, ast.Name)
            and not any(has_side_effect(value, safe_callable_whitelist) for value in node.ifs)
        ):
            return False
        return True

    if isinstance(node, ast.Call):
        return (
            not isinstance(node.func, ast.Name)
            or node.func.id not in safe_callable_whitelist
            or any(has_side_effect(item, safe_callable_whitelist) for item in node.args)
            or any(has_side_effect(item.value, safe_callable_whitelist) for item in node.keywords)
        )

    if isinstance(node, ast.Starred):
        return has_side_effect(node.value, safe_callable_whitelist)

    if isinstance(node, ast.IfExp):
        return any(
            has_side_effect(child, safe_callable_whitelist)
            for child in (node.test, node.body, node.orelse)
        )

    return True


@functools.lru_cache(maxsize=1)
def _get_line_lengths(content: str) -> Sequence[int]:
    return tuple([len(line) for line in content.splitlines(keepends=True)])


def get_charnos(node: ast.AST, content: str) -> Tuple[int, int]:
    """Get start and end character numbers in source code from ast node.

    Args:
        node (ast.AST): Node to fetch character numbers for
        content (str): Python source code

    Returns:
        Tuple[int, int]: start, end
    """
    line_lengths = _get_line_lengths(content)

    start_charno = sum(line_lengths[: node.lineno - 1]) + node.col_offset
    end_charno = sum(line_lengths[: node.end_lineno - 1]) + node.end_col_offset

    return start_charno, end_charno


def get_code(node: ast.AST, content: str) -> str:
    """Get python code from ast

    Args:
        node (ast.AST): ast to get code from
        content (str): Python source code that ast was parsed from

    Returns:
        str: Python source code

    """
    start_charno, end_charno = get_charnos(node, content)
    return content[start_charno:end_charno]
