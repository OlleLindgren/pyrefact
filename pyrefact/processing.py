import ast
import functools
import heapq
from types import MappingProxyType
from typing import Callable, Collection, Iterable, Mapping, NamedTuple, Union

from pyrefact import logs as logger
from pyrefact import parsing

MSG_INFO_REPLACE = """{fix_function_name:<40}: Replacing code:
{old_code}
* -> *****************
{new_code}
**********************"""
MSG_INFO_REMOVE = """{fix_function_name:<40}: Removing code:
{old_code}
**********************"""


class Range(NamedTuple):
    start: int  # Character number
    end: int  # Character number


class _Rewrite(NamedTuple):
    old: Union[ast.AST, Range]  # TODO replace with (start_char, end_char)
    new: Union[ast.AST, str]  # "" indicates a removal


def remove_nodes(source: str, nodes: Iterable[ast.AST], root: ast.Module) -> str:
    """Remove ast nodes from code

    Args:
        source (str): Python source code
        nodes (Iterable[ast.AST]): Nodes to delete from code
        root (ast.Module): Complete corresponding module

    Returns:
        str: Code after deleting nodes
    """
    keep_mask = [True] * len(source)
    nodes = list(nodes)
    for node in nodes:
        start, end = parsing.get_charnos(node, source)
        print(f"Removing:\n{source[start:end]}")
        keep_mask[start:end] = [False] * (end - start)

    passes = [len(source) + 1]

    for node in ast.walk(root):
        if isinstance(node, ast.Module):
            continue
        for bodytype in "body", "finalbody", "orelse":
            if body := getattr(node, bodytype, []):
                if isinstance(body, list) and all(child in nodes for child in body):
                    print(f"Found empty {bodytype}")
                    start_charno, _ = parsing.get_charnos(body[0], source)
                    passes.append(start_charno)

    heapq.heapify(passes)

    next_pass = heapq.heappop(passes)
    chars = []
    for i, char, keep in zip(range(len(source)), source, keep_mask):
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


def _do_rewrite(source: str, rewrite: _Rewrite, *, fix_function_name: str = "") -> str:
    old, new = rewrite
    start, end = _get_charnos(rewrite, source)
    code = source[start:end]
    if isinstance(new, str):
        new_code = new
        if new_code == code:
            return source
    elif isinstance(new, ast.AST):
        new_code = parsing.unparse(new)
    else:
        raise TypeError(f"Invalid replacement type: {type(new)}")
    lines = new_code.splitlines(keepends=True)
    indent = " " * getattr(old, "col_offset", getattr(new, "col_offset", 0))
    start_indent = " " * (len(code) - len(code.lstrip(" ")))
    new_code = "".join(
        f"{indent if i > 0 else start_indent}{code}".rstrip()
        + ("\n" if code.endswith("\n") else "")
        for i, code in enumerate(lines)
    )
    if new_code:
        logger.info(
            MSG_INFO_REPLACE, fix_function_name=fix_function_name, old_code=code, new_code=new_code
        )
    else:
        logger.info(MSG_INFO_REMOVE, fix_function_name=fix_function_name, old_code=code)

    return source[:start] + new_code + source[end:]


def replace_nodes(source: str, replacements: Mapping[ast.AST, Union[ast.AST, str]]) -> str:
    rewrites = sorted(
        replacements.items(), key=lambda tup: (tup[0].lineno, tup[0].end_lineno), reverse=True
    )
    for old, new in rewrites:
        rewrite = _Rewrite(old, new)
        source = _do_rewrite(source, rewrite)

    return source


def insert_nodes(source: str, additions: Collection[ast.AST]) -> str:
    """Insert ast nodes in python source code.

    Args:
        source (str): Python source code before insertions.
        additions (Collection[ast.AST]): Ast nodes to add. Linenos must be accurate.

    Returns:
        str: Code with added asts.
    """
    lines = source.splitlines(keepends=True)

    for node in sorted(additions, key=lambda n: n.lineno, reverse=True):
        addition = parsing.unparse(node)
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
    source: str,
    root: ast.AST,
    *,
    additions: Collection[ast.AST] = frozenset(),
    removals: Collection[ast.AST] = frozenset(),
    replacements: Mapping[ast.AST, ast.AST] = MappingProxyType({}),
) -> str:
    """Alter python code.

    This coordinates additions, removals and replacements in a safe way.

    Args:
        source (str): Python source code
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
    actions.extend((x.lineno, "add", parsing.unparse(x), x) for x in additions)
    actions.extend((x.lineno, "delete", parsing.unparse(x), x) for x in removals)
    actions.extend(
        (x.lineno, "replace", parsing.unparse(x), (x, y)) for x, y in replacements.items()
    )

    # a < d => deletions will go before additions if same lineno and reversed sorting.
    for _, action, _, value in sorted(actions, reverse=True):
        if action == "add":
            source = insert_nodes(source, [value])
        elif action == "delete":
            source = remove_nodes(source, [value], root)
        elif action == "replace":
            source = replace_nodes(source, {value[0]: value[1]})
        else:
            raise ValueError(f"Invalid action: {action}")

    return source


def _get_charnos(obj: _Rewrite, source: str):
    old, new = obj
    if isinstance(old, Range):
        return old.start, old.end

    if old is not None:
        return parsing.get_charnos(old, source)

    return parsing.get_charnos(new, source)


def fix(*maybe_func, restart_on_replace: bool = False, sort_order: bool = True) -> Callable:
    def fix_decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(source, *args, **kwargs):

            # Track rewrite history as an infinite loop guard
            history = set()
            if restart_on_replace:
                while True:
                    try:
                        old, new = next(
                            r for r in func(source, *args, **kwargs) if r not in history
                        )
                        rewrite = _Rewrite(old, new or "")
                        history.add(rewrite)
                        source = _do_rewrite(source, rewrite, fix_function_name=func.__name__)
                    except StopIteration:
                        return source

            rewrites = (_Rewrite(old, new or "") for old, new in func(source, *args, **kwargs))
            if sort_order:
                rewrites = sorted(
                    rewrites,
                    key=functools.partial(_get_charnos, source=source),
                    reverse=True,
                )

            for rewrite in rewrites:
                source = _do_rewrite(source, rewrite, fix_function_name=func.__name__)

            return source

        return wrapper

    if not maybe_func:
        return fix_decorator

    if len(maybe_func) == 1 and callable(maybe_func[0]):
        return fix_decorator(maybe_func[0])

    raise ValueError(f"Exactly 1 or 0 arguments must be given as maybe_func, not {len(maybe_func)}")
