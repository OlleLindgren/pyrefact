import ast
import dataclasses
import enum
import functools
import re
from typing import Collection, Iterable, Sequence, Tuple
from unicodedata import name

WALRUS_RE_PATTERN = (
    r"(?<![<>=!:]):=(?![=])"  # match :=, do not match  =, >=, <=, ==, !=
)
ASSIGN_RE_PATTERN = (
    r"(?<![<>=!:])=(?![=])"  #  match =,  do not match :=, >=, <=, ==, !=
)
ASSIGN_OR_WALRUS_RE_PATTERN = r"(?<![<>=!:]):?=(?![=])"
VARIABLE_RE_PATTERN = r"(?<![a-zA-Z0-9_])[a-zA-Z_]+[a-zA-Z0-9_]*"
SCOPED_VAR_RE_PATTERN = r"(?<![a-zA-Z0-9_])([a-zA-Z_]+[a-zA-Z0-9_]*\.?)+"
STATEMENT_DELIMITER_RE_PATTERN = (
    r"[\(\)\[\]\{\}\n]|(?<![a-zA-Z_])class|async def|def(?![a-zA-Z_])"
)
_WHITESPACE_RE_PATTERN = re.compile(r"( |\n)+")
_INDENT_RE_PATTERN = re.compile(r"(?<![^\n]) *")

from .constants import PYTHON_KEYWORDS


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
                if candidate[2] == hit[2] * 3
                and candidate[0] <= hit[0] <= hit[1] <= candidate[1]
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


@functools.lru_cache()
def get_paren_depths(content: str, code_mask_subset: Sequence[bool]) -> Sequence[int]:
    """Get paranthesis depths of every character in content.

    Args:
        content (str): Python source code

    Returns:
        Sequence[int]: A list of non-negative paranthesis depths, corresponding to every character.
    """
    depth = 0
    depths = []
    for is_code, character in zip(code_mask_subset, content):
        if not is_code:
            depths.append(depth)
            continue
        if character in ")]}":
            depth -= 1
        depths.append(depth)
        if character in "([{":
            depth += 1

    return depths


def get_line(content: str, charno: int) -> str:
    for hit in re.finditer(".*\n", content):
        if hit.start() <= charno < hit.end():
            return hit.group()

    return content.splitlines()[-1]


def get_indent(line: str) -> int:
    return len(next(re.finditer(r"^ *", line)).group())


def _unpack_ast_target(target: ast.AST) -> Iterable[str]:
    if isinstance(target, ast.Name):
        yield target.id
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
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
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


def iter_usages(ast_tree: ast.Module) -> Iterable[str]:
    """Iterate over all names referenced in

    Args:
        ast_tree (ast.Module): Module to search

    Yields:
        str: A varible or object that is used somewhere.
    """
    for node in ast_tree.body:
        if hasattr(node, "body"):
            yield from iter_usages(node)
        if hasattr(node, "bases"):
            for base in node.bases:
                yield base.id
        if hasattr(node, "keywords"):
            for kwd in node.keywords:
                if isinstance(kwd.value, ast.Name):
                    yield kwd.value.id
        if hasattr(node, "decorator_list"):
            for decorator in node.decorator_list:
                yield decorator.id
        if hasattr(node, "value") and isinstance(node.value, ast.Name):
            yield node.value.id


def has_side_effect(
    statement: Statement,
    safe_callable_whitelist: Collection[str] = frozenset(),
    used_variables: Collection[str] = frozenset(),
) -> bool:
    """Determine if a statement has a side effect.

    Args:
        statement (Statement): Statement to check
        safe_callable_whitelist (Collection[str]): Items known to not have a side effect
        used_variables (Collection[str]): Used variables and names

    Returns:
        bool: True if it may have a side effect.

    """
    nonempty_lines = [line for line in statement.statement.splitlines() if line.strip()]
    if not nonempty_lines:
        return False
    indent = min(get_indent(line) for line in nonempty_lines)
    deindented_code = "".join(
        line[indent:] if len(line) > indent else line
        for line in statement.statement.splitlines(keepends=True)
    )
    code_mask = get_is_code_mask(deindented_code)
    if not any(code_mask):
        return False
    try:
        ast.literal_eval(deindented_code)
    except (SyntaxError, ValueError):
        pass

    builtins = set(dir(__builtins__))

    for hit in re.finditer(VARIABLE_RE_PATTERN, deindented_code):
        start = hit.start()
        end = hit.end()
        value = hit.group()
        is_code_states = set(code_mask[start:end])
        assert (
            len(is_code_states) == 1
        ), f"Got ambiguous regex hit for VARIABLE_RE_PATTERN:\n{value}"
        is_code = is_code_states.pop()
        if not is_code:
            continue
        if value in {"raise", "assert"}:
            return True
        if value in PYTHON_KEYWORDS:
            continue
        if value in builtins:
            continue
        if value in used_variables:
            return True
        if value in safe_callable_whitelist:
            continue

    return False


def get_code(node: ast.AST, content: str) -> str:
    """Get python code from ast

    Args:
        node (ast.AST): ast to get code from
        content (str): Python source code that ast was parsed from

    Returns:
        str: Python source code

    """
    lines = content.splitlines(keepends=True)
    code_lines = lines[node.lineno - 1 : node.end_lineno]
    print(code_lines)
    if node.lineno == node.end_lineno:
        return code_lines[0][node.col_offset : node.end_col_offset]
    return "".join(
        [lines[0][node.col_offset :]] + lines[1:-1] + lines[-1][: node.end_col_offset]
    )
