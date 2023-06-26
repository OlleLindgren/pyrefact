from __future__ import annotations

import ast
import collections
import difflib
import functools
import heapq
import re
import textwrap
from types import MappingProxyType
from typing import Callable, Collection, Iterable, Literal, Mapping, NamedTuple, Sequence, Tuple

from pyrefact import core, formatting, logs as logger


MSG_INFO_REPLACE = """{fix_function_name:<40}: Replacing code:
{old_code}
* -> *****************
{new_code}
**********************"""
MSG_ERROR_REPLACE = """{fix_function_name:<40}: Failed to replace code:
{old_code}
* -> *****************
{new_code}
**********************"""
MSG_INFO_REMOVE = """{fix_function_name:<40}: Removing code:
{old_code}
**********************"""
MSG_ERROR_REMOVE = """{fix_function_name:<40}: Failed to remove code:
{old_code}
**********************"""


def log_replacement(old: str, new: str | None, fix_function_name: str, valid: bool) -> None:
    if new and valid:
        logger.info(MSG_INFO_REPLACE, fix_function_name=fix_function_name, old_code=old, new_code=new)
    elif new and not valid:
        logger.error(MSG_ERROR_REPLACE, fix_function_name=fix_function_name, old_code=old, new_code=new)
    elif not new and valid:
        logger.info(MSG_INFO_REMOVE, fix_function_name=fix_function_name, old_code=old)
    elif not new and not valid:
        logger.error(MSG_ERROR_REMOVE, fix_function_name=fix_function_name, old_code=old)


class _Rewrite(NamedTuple):
    old: ast.AST | core.Range  # TODO replace with (start_char, end_char)
    new: ast.AST | str  # "" indicates a removal


def _substitute_original_strings(original_source: str, new_source: str) -> str:
    """Ensure consistent string formattings in new and old source.

    The reason for this to exist is that, without it, pyrefact will change the string
    formattings of values in a very opinionated, and frankly not very nice way. For
    example, multiline f-strings would be replaced by really long regular strings with
    a bunch of newline characters in them, which looks horrible.

    Args:
        original_source (str): Original source code
        new_source (str): New source code

    Returns:
        str: new_source, but with consistent string formattings as in original_source
    """
    original_ast = core.parse(original_source)

    original_string_formattings = collections.defaultdict(set)
    for node in core.walk(original_ast, ast.Constant(value=str)):
        original_string_formattings[node.value].add(core.get_code(node, original_source))

    for value, sources in original_string_formattings.items():
        template = ast.Module(body=[ast.Expr(value=ast.Constant(value=value))])
        # Temporary set created and orignal sources object is overwritten with this, which is
        # preferable to assigning a new set. Not so elegant perhaps.
        tmp = {
            src
            for src in sources
            if core.is_valid_python(src) and core.match_template(core.parse(src), template)
        }
        sources.clear()
        sources.update(tmp)

    replacements = {}
    new_ast = core.parse(new_source)
    for node in core.walk(new_ast, ast.Constant(value=tuple(original_string_formattings))):
        # If this new string formatting doesn't exist in the orignal source, find the most
        # common orignal equivalent string formatting and use that instead.
        original_formattings = original_string_formattings[node.value]
        new_formatting = core.get_code(node, new_source)
        template = ast.Module(body=[ast.Expr(value=ast.Constant(value=node.value))])
        if (
            original_formattings
            and core.is_valid_python(new_formatting)
            and core.match_template(core.parse(new_formatting), template)
            and new_formatting not in original_formattings
        ):
            most_common_original_formatting = collections.Counter(original_formattings).most_common(
                1
            )[0][0]
            replacements[node] = most_common_original_formatting

    return replace_nodes(new_source, replacements)


def _substitute_original_fstrings(original_source: str, new_source: str) -> str:
    """Ensure consistent string formattings in new and old source.

    The reason for this to exist is that, without it, pyrefact will change the string
    formattings of values in a very opinionated, and frankly not very nice way. For
    example, multiline f-strings would be replaced by really long regular strings with
    a bunch of newline characters in them, which looks horrible.

    Args:
        original_source (str): Original source code
        new_source (str): New source code

    Returns:
        str: new_source, but with consistent string formattings as in original_source
    """
    original_ast = core.parse(original_source)
    original_string_formattings = collections.defaultdict(set)
    for node in core.walk(original_ast, ast.JoinedStr):
        code = core.get_code(node, original_source)
        unparsed_code = core.unparse(node)
        if core.is_valid_python(code):
            original_string_formattings[unparsed_code].add(code)

    replacements = {}
    new_ast = core.parse(new_source)
    for node in core.walk(new_ast, ast.JoinedStr):
        # If this new string formatting doesn't exist in the orignal source, find the most
        # common orignal equivalent string formatting and use that instead.
        unparsed_code = core.unparse(node)
        original_formattings = original_string_formattings[unparsed_code]
        new_formatting = core.get_code(node, new_source)
        if (
            original_formattings
            and core.is_valid_python(new_formatting)
            and new_formatting not in original_formattings
        ):
            most_common_original_formatting = collections.Counter(original_formattings).most_common(
                1
            )[0][0]
            replacements[node] = most_common_original_formatting

    return replace_nodes(new_source, replacements)


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
        start, end = core.get_charnos(node, source)

        # If multiple "lines" are on the same line, with a semicolon in between,
        # we also need to purge the semicolon and any whitespace before and after it
        semicolon_anti_delimiters = re.findall(r"^\s*;\s*", source[end:])
        if semicolon_anti_delimiters:
            end += len(semicolon_anti_delimiters[0])

        logger.debug("Removing:\n{old}", old=source[start:end])
        keep_mask[start:end] = [False] * (end - start)

    passes = [len(source) + 1]

    for node in core.walk(root, ast.AST):
        if isinstance(node, ast.Module):
            continue
        for bodytype in "body", "finalbody", "orelse":
            if body := getattr(node, bodytype, []):
                if (
                    isinstance(body, list)
                    and all(child in nodes for child in body)
                    and node not in nodes
                ):
                    logger.debug("Found empty {bodytype}", bodytype=bodytype)
                    start_charno, _ = core.get_charnos(body[0], source)
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


def _asts_equal(ast1: ast.AST, ast2: ast.AST):
    """Determine if two ASTs are the same."""
    return core.unparse(ast1) == core.unparse(ast2)


def _sources_equivalent(source1: str, source2: ast.AST) -> bool:
    """Determine if two source code snippets produce the same ASTs."""
    return _asts_equal(core.parse(source1), core.parse(source2))


def minimize_whitespace_line_differences(source: str, new_source: str) -> Tuple[str, str, str]:
    old_lines = source.splitlines(keepends=True)
    new_lines = new_source.splitlines(keepends=True)

    differ = difflib.Differ()
    diffs = list(differ.compare(old_lines, new_lines))

    segments = []
    # Iterate through the differences
    for identifier, _, *line in diffs:
        line = "".join(line)
        if not segments:
            segments.append((identifier, [line]))
        elif segments[-1][0] == identifier:
            segments[-1][1].append(line)
        else:
            segments.append((identifier, [line]))

    new_lines = []
    for identifier, lines in segments:
        if identifier == " ":  # Present in both
            new_lines.extend(lines)
        elif identifier == "+" and "".join(lines).strip():  # Present in new, not only whitespace
            new_lines.extend(lines)
        elif identifier == "-" and not "".join(lines).strip():  # Present in old, only whitespace
            new_lines.extend(lines)

    new_source = "".join(new_lines)

    while old_lines and new_lines and old_lines[0] == new_lines[0]:
        old_lines.pop(0)
        new_lines.pop(0)

    while old_lines and new_lines and old_lines[-1] == new_lines[-1]:
        old_lines.pop()
        new_lines.pop()

    found = "".join(old_lines)
    replaced = "".join(new_lines)

    return new_source, found, replaced


def _do_rewrite(source: str, rewrite: _Rewrite, *, fix_function_name: str = "") -> str:
    old, new = rewrite
    start, end = _get_charnos(rewrite, source)
    code = source[start:end]
    if isinstance(new, str):
        new_code = new
        if new_code == code:
            return source

    elif isinstance(new, ast.AST):
        new_code = core.unparse(new)
    else:
        raise TypeError(f"Invalid replacement type: {type(new)}")

    if isinstance(old, core.Range):
        # Prevent whitespace-only changes from being applied
        new_code_lines = [l.rstrip() for l in new_code.splitlines() if l.strip()]
        code_lines = [l.rstrip() for l in code.splitlines() if l.strip()]
        if new_code_lines == code_lines:
            return source

        candidate = source[: old.start] + new_code + source[old.end :]
        if new_code or core.is_valid_python(candidate):
            choice = candidate
        else:
            pass_candidate = source[: old.start] + "pass" + source[old.end :]
            if core.is_valid_python(pass_candidate):
                choice = pass_candidate
            else:
                choice = candidate

        new_source, old, new = minimize_whitespace_line_differences(source, choice)
        valid = core.is_valid_python(new_source)
        log_replacement(old, new, fix_function_name, valid)

        if not core.is_valid_python(new_source):
            return source

        return new_source

    lines = new_code.splitlines(keepends=True)
    indent = getattr(old, "col_offset", getattr(new, "col_offset", 0))
    indents = {**{i: indent for i in range(len(lines))}, 0: len(code) - len(code.lstrip(" "))}

    try:
        new_code_ast = core.parse(new_code)
    except SyntaxError:
        pass  # new_code is not necessarily valid python syntax in all cases
    else:
        for node in core.walk(new_code_ast, (ast.Constant(value=str), ast.JoinedStr)):
            node_code = core.get_code(node, new_code)
            if any(
                node_code.startswith(prefix) and node_code.endswith(prefix[-3:])
                for prefix in ("b'''", "r'''", "f'''", "'''", 'b"""', 'r"""', 'f"""', '"""')
            ):
                for lineno in range(node.lineno, node.end_lineno):
                    indents[lineno] = 0

    new_code = "".join(
        f"{' ' * indents[i]}{code}".rstrip() + ("\n" if code.endswith("\n") else "")
        for i, code in enumerate(lines)
    )

    candidate = source[:start] + new_code + source[end:]

    if new_code.strip() and isinstance(old, ast.GeneratorExp):
        candidate_parenthesized = source[:start] + "(" + new_code + ")" + source[end:]
        if not _sources_equivalent(candidate, candidate_parenthesized):
            candidate = candidate_parenthesized

    elif not new_code.strip() and not core.is_valid_python(candidate):
        pass_candidate = source[:start] + "pass" + source[end:]
        if core.is_valid_python(pass_candidate):
            candidate = pass_candidate

    new_source, old, new = minimize_whitespace_line_differences(source, candidate)
    valid = core.is_valid_python(new_source)
    log_replacement(old, new, fix_function_name, valid)

    if not core.is_valid_python(new_source):
        return source

    return new_source


def replace_nodes(source: str, replacements: Mapping[ast.AST, ast.AST | str]) -> str:
    rewrites = sorted(
        replacements.items(),
        key=lambda tup: (
            tup[0].lineno,
            tup[0].end_lineno,
            getattr(tup[0], "col_offset", 0),
            getattr(tup[0], "end_col_offset", 0),
        ),
        reverse=True,
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
        addition = core.unparse(node)
        col_offset = getattr(node, "col_offset", 0)
        logger.debug("Adding:\n{new}", new=addition)
        lines = (
            lines[: node.lineno]
            + ["\n"]
            + [" " * col_offset + line for line in addition.splitlines(keepends=True)]
            + ["\n"] * (not addition.endswith("\n"))
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
    priority: Sequence[Literal["additions", "removals", "replacements"]] = (),
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
    # If priority specified, prioritize some actions over others. This goes on a line number
    # level, so col_offset will be overridden by this.
    original_source = source
    priorities = {
        modification_type: priority.index(modification_type)
        if modification_type in priority
        else 3 + len(priority)
        for modification_type in ("additions", "removals", "replacements")
    }
    # Yes, this unparsing is an expensive way to sort the nodes.
    # However, this runs relatively infrequently and should not have a big
    # performance impact.
    actions = [
        *((
            x.lineno,
            -priorities["additions"],
            getattr(x, "col_offset", 0),
            "add",
            core.unparse(x),
            x,
            )
            for x in additions
        ),
        *((
            x.lineno,
            -priorities["removals"],
            getattr(x, "col_offset", 0),
            "delete",
            core.unparse(x),
            x,
            )
            for x in removals
        ),
        *((
            x.lineno,
            -priorities["replacements"],
            getattr(x, "col_offset", 0),
            "replace",
            core.unparse(x),
            (x, y),
            )
            for x, y in replacements.items()
    ),]
    # a < d => deletions will go before additions if same lineno and reversed sorting.
    for *_, action, _, value in sorted(actions, reverse=True):
        if action == "add":
            source = insert_nodes(source, [value])
        elif action == "delete":
            source = remove_nodes(source, [value], root)
        elif action == "replace":
            source = replace_nodes(source, {value[0]: value[1]})
        else:
            raise ValueError(f"Invalid action: {action}")

    source = _substitute_original_strings(original_source, source)
    source = _substitute_original_fstrings(original_source, source)

    return source


def _get_charnos(obj: _Rewrite, source: str) -> core.Range:
    old, new = obj
    if isinstance(old, core.Range):
        return core.Range(old.start, old.end)

    if old is not None:
        return core.get_charnos(old, source)

    return core.get_charnos(new, source)


def fix(*maybe_func, restart_on_replace: bool = False, sort_order: bool = True) -> Callable:
    """Convert an iterator of (before, after) asts to a function that fixes source code.

    Args:
        maybe_func (Callable): Function to fix source code. If not provided, this is a decorator.
        restart_on_replace (bool, optional): If True, restarts the rewrite loop on replacements.
            This is needed when a replacement may alter the code in a way that makes it hard for
            the rewriter to properly place the next rewrite. Defaults to False.
        sort_order (bool, optional): If True, sorts the rewrites by line number and col_offset.
    """

    def fix_decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(source, *args, **kwargs):
            # Track rewrite history as an infinite loop guard
            history = set()
            for _ in range(1000):  # Max 1000 iterations
                if restart_on_replace:
                    try:
                        old, new = next(
                            r for r in func(source, *args, **kwargs) if r not in history
                        )
                        rewrite = _Rewrite(old, new or "")
                        history.add(rewrite)
                        source = _do_rewrite(source, rewrite, fix_function_name=func.__name__)
                    except StopIteration:
                        return source

                else:
                    rewrites = (_Rewrite(old, new or "") for old, new in func(source, *args, **kwargs))
                    if sort_order:
                        rewrites = sorted(
                            ((_get_charnos(rewrite, source), rewrite) for rewrite in rewrites),
                            key=lambda tup: tup[0],
                            reverse=True,
                        )

                    rewritten_ranges = []
                    for rewrite_range, rewrite in rewrites:
                        if any(rewrite_range & other for other in rewritten_ranges):
                            logger.debug("Found rewrite that overlaps with previous rewrite. Restarting @fix loop.")
                            break

                        source = _do_rewrite(source, rewrite, fix_function_name=func.__name__)
                    else:
                        return source

            return source

        return wrapper

    if not maybe_func:
        return fix_decorator

    if len(maybe_func) == 1 and callable(maybe_func[0]):
        return fix_decorator(maybe_func[0])

    raise ValueError(f"Exactly 1 or 0 arguments must be given as maybe_func, not {len(maybe_func)}")


def find_replace(
    source: str,
    root: ast.AST = None,
    *,
    find: str | ast.AST,
    replace: str,
    expand_first: bool = None,
    expand_last: bool = None,
    yield_match: bool = False,
    **callables_or_templates,
) -> Iterable[core.Range, str]:
    """Swap a find with a replacement.

    Args:
        source (str): Python source code.
        root (ast.AST): Parsed AST to search. If not provided, source is parsed.
        find (ast.AST | str | tuple[str]): Template, tuple of templates, or compiled template to find.
        replace (ast.AST): Uncompiled template to replace found find with.
        expand_first (bool, optional): If passed, forwarded to `core.walk_wildcard`.
        expand_last (bool, optional): If passed, forwarded to `core.walk_wildcard`.
        yield_match (bool, optional): If True, yields the matches as well as the replacements.

    Yields:
        Iterable[Range, str]: Range of the replacement and the replacement itself.
    """
    callables = {}
    templates = {}
    for name, value in callables_or_templates.items():
        if callable(value) and not isinstance(value, type):
            callables[name] = value
        else:
            templates[name] = value
    if root is None:
        root = ast.parse(source)
    if isinstance(find, str):
        find = (find,)
    if isinstance(find, (tuple, set)) and isinstance(find[0], str):
        find = type(find)(
            core.compile_template(
                f,
                **{
                    **{name.strip("{}"): object for name in re.findall(r"\{\{\w+\}\}", f)},
                    **templates
                }
            )
            for f in find
        )
    if isinstance(find, tuple) and len(find) == 1:
        find = find[0]

    if isinstance(find, list):
        walk_sequence_kwargs = {"expand_first": expand_first, "expand_last": expand_last}
        walk_sequence_kwargs = {k: v for k, v in walk_sequence_kwargs.items() if v is not None}
        iterator = core.walk_sequence(root, *find, **walk_sequence_kwargs)
    else:
        iterator = [[m] for m in core.walk_wildcard(root, find)]

    for matches in iterator:
        if isinstance(find, list):
            combined_match = core.merge_matches(root, matches)
        else:
            combined_match = matches[0]

        if not combined_match:
            continue

        ranges = [core.get_charnos(m[0], source) for m in matches]
        range_start = min(r[0] for r in ranges)
        range_end = max(r[1] for r in ranges)
        replacement_range = core.Range(range_start, range_end)

        template_replacement = core.format_template(replace, combined_match, **callables)

        indentation = formatting.indentation_level(source[range_start:range_end])

        template_replacement = textwrap.dedent(template_replacement)
        template_replacement = textwrap.indent(template_replacement, " " * indentation)

        if yield_match:
            yield replacement_range, template_replacement, combined_match
        else:
            yield replacement_range, template_replacement
