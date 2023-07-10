from __future__ import annotations

import ast
import collections
import dataclasses
import functools
import itertools
import re
import textwrap
import traceback
from typing import Collection, Iterable, List, Mapping, NamedTuple, Sequence, Set, Tuple

from pyrefact import constants, formatting, logs as logger


def unparse(node: ast.AST | str) -> str:
    if isinstance(node, str):
        return node  # Hack to allow format_template to accept strings matched by wildcards.
    if constants.PYTHON_VERSION >= (3, 9):
        return ast.unparse(node)

    import astunparse

    source = astunparse.unparse(node)

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
    ),):
        source = source.rstrip()

    line_length = max(60, 100 - getattr(node, "col_offset", 0))
    try:
        ast.parse(source)
    except (SyntaxError, ValueError):
        source = formatting.format_with_black(source, line_length=line_length)
    source = formatting.collapse_trailing_parentheses(source)
    source = textwrap.dedent(source).lstrip()

    return source


@dataclasses.dataclass(eq=True, frozen=True)
class Wildcard(ast.AST):
    """A wildcard matches a specific template, and extracts its name."""

    name: str
    template: object = object
    common: bool = True

    @staticmethod
    def __iter__() -> Iterable:
        yield from ()

    def __repr__(self) -> str:
        if self.template is object and self.common is True:
            return f"Wildcard({self.name!r})"
        if self.template is object and self.common is not True:
            return f"Wildcard({self.name!r}, common={self.common!r})"
        if self.template is not object and self.common is True:
            return f"Wildcard({self.name!r}, {self.template!r})"

        return f"Wildcard({self.name!r}, {self.value!r}, {self.common!r})"


@functools.lru_cache(maxsize=10000)
def _make_match_type(fields: Tuple[str, ...]) -> type:
    return collections.namedtuple("Match", fields)


def _all_fields_consistent(
    matches: Iterable[Tuple[object]], ignore: Collection[str] = frozenset()
) -> bool:
    field_options = collections.defaultdict(set)
    for m in matches:
        for key, value in zip(getattr(m, "_fields", ()), m):
            if key != "root" and key not in ignore:
                options = field_options[key]
                options.add(unparse(value) if isinstance(value, ast.AST) else str(value))
                if len(options) > 1:
                    return False

    return True


def merge_matches(root: ast.AST, matches: Iterable[Tuple[object]]) -> Tuple[object]:
    namedtuple_matches = []
    for match in matches:
        if not match:
            return ()

        if type(match) is not tuple:
            namedtuple_matches.append(match)

    if not namedtuple_matches:
        return (root,)

    if not _all_fields_consistent(namedtuple_matches):
        return ()

    namedtuple_vars = {
        **{key: value for match in namedtuple_matches for key, value in zip(match._fields, match)},
        "root": root,
    }

    # Always store node in special "root" field. Other contents of this field are discarded.

    # Sort in alphabetical order, but always with "root" first.
    fields = tuple(sorted(namedtuple_vars.keys(), key=lambda k: (k != "root", k)))
    namedtuple_type = _make_match_type(fields)
    return namedtuple_type(*(namedtuple_vars[field] for field in fields))


def _match_tuple(node: ast.AST, template: Tuple[ast.AST, ...], ignore: Collection[str]) -> Tuple:
    for child in template:
        if match := match_template(node, child, ignore=ignore):
            return match

    return ()


def _match_set(node: ast.AST, template: Set[ast.AST], ignore: Collection[str]) -> Tuple:
    if not isinstance(node, list):
        return ()

    matches = (match_template(node_child, tuple(template), ignore=ignore) for node_child in node)
    return merge_matches(node, matches)


def _match_list(node: ast.AST, template: List[ast.AST], ignore: Collection[str]) -> Tuple:
    if not isinstance(node, list):
        return ()
    if len(node) != len(template):
        return ()

    matches = (
        match_template(child, template_child, ignore=ignore)
        for child, template_child in zip(node, template)
    )
    return merge_matches(node, matches)


def _match_wildcard(node: ast.AST, template: Wildcard, ignore: Collection[str]) -> Tuple:
    namedtuple_type = _make_match_type((template.name,))
    template_match = match_template(node, template.template, ignore=ignore)
    return namedtuple_type(template_match[0]) if len(template_match) == 1 else ()


def _match_template_vars(
    node: ast.AST,
    template: ast.AST,
    ignore: Collection[str] = frozenset(("lineno", "end_lineno", "col_offset", "end_col_offset")),
) -> Tuple:
    t_vars = vars(template)
    n_vars = vars(node)

    for k in t_vars:
        if k in ignore:
            continue
        if k not in n_vars:
            return ()

    matches = (
        match_template(n_vars[key], t_vars[key], ignore=ignore) for key in t_vars.keys() - ignore
    )
    return merge_matches(node, matches)


@functools.lru_cache(maxsize=10000)
def _issubclas_cache(obj: type, types: type | Tuple[type, ...]) -> bool:
    return issubclass(obj, types)


def _isinstance_cache(obj: object, types: type | Tuple[type, ...]) -> bool:
    return _issubclas_cache(type(obj), types)


def match_template(
    node: ast.AST,
    template: ast.AST,
    ignore: Collection[str] = frozenset(("lineno", "end_lineno", "col_offset", "end_col_offset")),
) -> Tuple:
    """Match a node against a provided ast template.

    Args:
        node (ast.AST): Node to match against template
        template (ast.AST): Template to match node with

    Returns:
        Tuple:
            If node matches, a namedtuple with node as the first element, and
            any wildcards as other fields. Otherwise, an empty tuple.

    Essentially, asts match themselves. However, there are a few special rules:
    * Types match instances of that type, and nothing else.
    * Tuples denote OR syntax, where the node must match any of the templates in the tuple.
    * Sets denote variable length lists, where all elements must match against at least one of the templates in it.
    * Wildcards have a template, and match any node that matches that template. They also extract the matched node
        into a field with the wildcard's name. If the same wildcard is used multiple times, all the matched nodes
        must unparse to the same source code.

    Examples of templates, and what they match:
    # Match the int 1
    `ast.Constant(value=1)`
    # Match the float 1.0
    `ast.Constant(value=1.0)`
    # Match any int
    `ast.Constant(value=int)`
    # Tuples denote OR syntax, match any int or float
    `ast.Constant(value=(int, float))`
    # Match the variable x
    `ast.Name(id="x")`
    # Match x(1) and y(1)
    `ast.Call(func=ast.Name(id=("x", "y")), args=[ast.Constant(value=1)])`

    """
    # A type indicates that the node should be an instance of that type,
    # and nothing else. A tuple indicates that the node should be any of
    # the types in it.
    if _isinstance_cache(template, type):
        return (node,) if _isinstance_cache(node, template) else ()

    # A tuple indicates an or condition; the node must comply with any of
    # the templates in the child.
    # They may all be types for example, which boils down to the traditional
    # isinstance logic.
    # If there are wildcards, the first match is chosen.
    if _isinstance_cache(template, tuple):
        return _match_tuple(node, template, ignore)

    # A set indicates a variable length list, where all elements must match
    # against at least one of the templates in it.
    # If there are wildcards, the first match is chosen.
    # If there are inconsistencies between different elements, the match is discarded.
    if _isinstance_cache(template, set):
        return _match_set(node, template, ignore)

    # A list indicates that the node must also be a list, and for every
    # element, it must match against the corresponding node in the template.
    # It must also be equal length.
    if _isinstance_cache(template, list):
        return _match_list(node, template, ignore)

    if template is True or template is False or template is None:
        return (node,) if node is template else ()

    if _isinstance_cache(template, Wildcard):
        return _match_wildcard(node, template, ignore)

    if _isinstance_cache(template, ast.AST):
        if isinstance(node, type(template)):
            return _match_template_vars(node, template, ignore=ignore)

        return ()

    if node == template:
        return (node,)

    return ()


@functools.lru_cache(maxsize=100)
def parse(source_code: str) -> ast.AST:
    """Parse python source code and cache

    Args:
        source_code (str): Python source code

    Returns:
        ast.AST: Parsed AST
    """
    try:
        return ast.parse(source_code)
    except SyntaxError as error:
        stack_trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        logger.error(
            "Failed to parse source with error:\n{}\n\n\nCode:\n\n{}", stack_trace, source_code
        )
        raise error


@functools.lru_cache(maxsize=100)
def _group_nodes_in_scope(scope: ast.AST) -> Mapping[ast.AST, Sequence[ast.AST]]:
    node_types = collections.defaultdict(list)
    for node in ast.walk(scope):
        node_types[type(node)].append(node)

    return node_types


def walk_wildcard(
    scope: ast.AST, node_template: ast.AST | Tuple[ast.AST, ...], ignore: Collection[str] = ()
) -> Sequence[Tuple[ast.AST, ...]]:
    """Iterate over all nodes in scope that match a particular type or template.

    The `node_template` argument supports the same syntax as in match_template().

    Args:
        scope (ast.AST): Scope to search
        node_template (ast.AST): Node type to filter on

    Returns:
        Sequence[ast.AST]: All nodes in scope of that type
    """
    types_in_scope = _group_nodes_in_scope(scope)
    if not isinstance(node_template, tuple):
        node_template = (node_template,)

    yielded_nodes = set()
    for template in node_template:
        type_matcher = template if isinstance(template, type) else type(template)
        nodes = itertools.chain.from_iterable(
            children
            for child_type, children in types_in_scope.items()
            if issubclass(child_type, type_matcher)
        )
        for node in nodes:
            if node not in yielded_nodes:
                if template_match := match_template(node, template, ignore=ignore):
                    yielded_nodes.add(node)
                    yield template_match


def walk(
    scope: ast.AST, node_template: ast.AST | Tuple[ast.AST, ...], ignore: Collection[str] = ()
) -> Sequence[ast.AST]:
    """Iterate over all nodes in scope that match a particular type or template.

    The `node_template` argument supports the same syntax as in match_template(), but
    only the matched node is returned, and not the full match with wildcards. To get
    the full match, use `walk_wildcard()` instead.

    Args:
        scope (ast.AST): Scope to search
        node_template (ast.AST): Node type to filter on

    Returns:
        Sequence[ast.AST]: All nodes in scope of that type
    """
    for node, *_ in walk_wildcard(scope, node_template, ignore=ignore):
        yield node


def _iter_wildcards(
    template: ast.AST, recursion_blacklist: Collection = None
) -> Iterable[Wildcard]:
    if recursion_blacklist is None:
        recursion_blacklist = set()
    if id(template) in recursion_blacklist:
        return
    recursion_blacklist = set.union(recursion_blacklist, {id(template)})
    if isinstance(template, Wildcard):
        yield template
        return
    if isinstance(template, type):
        return
    if isinstance(template, Iterable):
        for item in template:
            yield from _iter_wildcards(item, recursion_blacklist=recursion_blacklist)
    if isinstance(template, Mapping):
        for item in template.values():
            yield from _iter_wildcards(item, recursion_blacklist=recursion_blacklist)
    if hasattr(template, "__dict__"):
        for item in vars(template).values():
            yield from _iter_wildcards(item, recursion_blacklist=recursion_blacklist)


def walk_sequence(
    scope: ast.Module, *templates: ast.AST, expand_first: bool = False, expand_last: bool = False
) -> Iterable[Sequence[ast.AST]]:
    """Iterate over all sequences of nodes in scope that match a sequence of templates."""
    uncommon = set()
    for node in walk(
        scope, tuple({*constants.AST_TYPES_WITH_BODY, *constants.AST_TYPES_WITH_ORELSE})
    ):
        for body in [getattr(node, "body", []), getattr(node, "orelse", [])]:
            if not body:
                continue

            for nodes in zip(
                *(body[i : len(body) - len(templates) + i + 1] for i in range(len(templates)))
            ):
                all_matching = True
                matches = []
                for node, template in zip(nodes, templates):
                    if m := match_template(node, template):
                        matches.append(m)
                    else:
                        all_matching = False
                        break

                if not all_matching:
                    continue

                # This is really expensive so it's better to do it way in here even though it's
                # the same on all iterations.
                # I assume we probably don't get in here very often and that seems true by the
                # speedup this gives us.
                if not uncommon:
                    uncommon = {
                        wildcard.name
                        for template in templates
                        for wildcard in _iter_wildcards(template)
                        if not wildcard.common
                    }

                if not _all_fields_consistent(matches):
                    continue

                if expand_first:
                    pre = body[: body.index(nodes[0])]
                    for child in reversed(pre):
                        template_match = match_template(child, templates[0])
                        if not template_match or not _all_fields_consistent(
                            matches + [template_match], ignore=uncommon
                        ):
                            break

                        matches.insert(0, template_match)

                if expand_last:
                    post = body[body.index(nodes[-1]) + 1 :]
                    for child in post:
                        template_match = match_template(child, templates[-1])
                        if not template_match or not _all_fields_consistent(
                            matches + [template_match], ignore=uncommon
                        ):
                            break

                        matches.append(template_match)

                yield tuple(matches)


def filter_nodes(
    nodes: Iterable[ast.AST], node_type: ast.AST | Sequence[ast.AST]
) -> Sequence[ast.AST]:
    for node in nodes:
        if match_template(node, node_type):
            yield node


def is_valid_python(source: str) -> bool:
    """Determine if source code is valid python.

    Args:
        source (str): Python source code

    Returns:
        bool: True if source is valid python.
    """
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def has_side_effect(node: ast.AST, safe_callable_whitelist: Collection[str] = frozenset()) -> bool:
    """Determine if a statement has a side effect.

    A statement has a side effect if it can influence the outcome of a subsequent statement.
    A slight exception to this rule exists: Anything named "_" is assumed to be unused and
    meaningless. So a statement like "_ = 100" is assumed to have no side effect.

    Args:
        node (ast.AST): Node to check
        safe_callable_whitelist (Collection[str]): Items known to not have a side effect

    Returns:
        bool: True if it may have a side effect.

    """
    if isinstance(
        node, (ast.Yield, ast.YieldFrom, ast.Return, ast.Raise, ast.Continue, ast.Break, ast.Assert)
    ):
        return True

    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name != "_"

    if isinstance(node, ast.For):
        return any(
            has_side_effect(item, safe_callable_whitelist)
            for item in itertools.chain([node.target], [node.iter], node.body)
        )

    if isinstance(node, ast.Lambda):
        return has_side_effect(node.args, safe_callable_whitelist) or has_side_effect(
            node.body, safe_callable_whitelist
        )

    if isinstance(node, ast.arguments):
        return any(
            has_side_effect(item, safe_callable_whitelist)
            for item in itertools.chain(
                node.posonlyargs, node.args, node.kwonlyargs, node.kw_defaults, node.defaults
        ))

    if node is None:
        return False

    if isinstance(node, ast.Module):
        return any(has_side_effect(value, safe_callable_whitelist) for value in node.body)

    if isinstance(node, (ast.Constant, ast.Pass)):
        return False

    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return True

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

    if isinstance(node, ast.Compare):
        return any(
            has_side_effect(child, safe_callable_whitelist)
            for child in [node.left] + node.comparators
        )

    if isinstance(node, ast.BoolOp):
        return any(has_side_effect(value, safe_callable_whitelist) for value in node.values)

    if isinstance(node, ast.Name):
        return isinstance(node.ctx, ast.Store) and node.id != "_"

    if isinstance(node, ast.Attribute):
        return isinstance(node.ctx, ast.Store) or has_side_effect(node.value)

    if isinstance(node, ast.Subscript):
        return (
            has_side_effect(node.value, safe_callable_whitelist)
            or has_side_effect(node.slice, safe_callable_whitelist)
            or (
                isinstance(node.ctx, ast.Store)
                and not (isinstance(node.value, ast.Name) and node.value.id == "_")
        ))

    if isinstance(node, ast.Slice):
        return any(
            has_side_effect(child, safe_callable_whitelist) for child in (node.lower, node.upper)
        )

    if isinstance(node, (ast.DictComp)) and has_side_effect(node.value, safe_callable_whitelist):
        return True

    if isinstance(node, (ast.SetComp, ast.ListComp, ast.GeneratorExp, ast.DictComp)):
        return any(has_side_effect(item, safe_callable_whitelist) for item in node.generators)

    if isinstance(node, ast.comprehension):
        return not (
            isinstance(node.target, ast.Name)
            and (not has_side_effect(node.iter, safe_callable_whitelist))
            and (not any((has_side_effect(value, safe_callable_whitelist) for value in node.ifs)))
        )

    if isinstance(node, ast.Call):
        return (
            not all(
                child.id in safe_callable_whitelist or child.id == "_"
                for child in ast.walk(node.func)
                if isinstance(child, ast.Name)
            )
            or any(has_side_effect(item, safe_callable_whitelist) for item in node.args)
            or any(has_side_effect(item.value, safe_callable_whitelist) for item in node.keywords)
            or not all(child.attr in safe_callable_whitelist for child in walk(node, ast.Attribute))
        )

    if isinstance(node, ast.Starred):
        return has_side_effect(node.value, safe_callable_whitelist)

    if isinstance(node, ast.If):
        return any(
            has_side_effect(item, safe_callable_whitelist)
            for item in itertools.chain(node.body, [node.test], node.orelse)
        )

    if isinstance(node, ast.IfExp):
        return any(
            has_side_effect(child, safe_callable_whitelist)
            for child in (node.test, node.body, node.orelse)
        )

    if isinstance(node, ast.Subscript):
        return isinstance(node.ctx, ast.Store) or any(
            has_side_effect(child, safe_callable_whitelist) for child in (node.value, node.slice)
        )

    # NamedExpr is :=
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        if isinstance(node, ast.Assign):
            targets = node.targets
        else:
            targets = [node.target]
        return has_side_effect(node.value) or any(has_side_effect(target) for target in targets)

    if isinstance(node, ast.Index):
        return has_side_effect(node.value)

    if isinstance(node, ast.JoinedStr):
        return any(has_side_effect(child) for child in node.values)

    if isinstance(node, ast.FormattedValue):
        return any(has_side_effect(child) for child in (node.value, node.format_spec))

    return True


@functools.lru_cache(maxsize=100)
def _get_line_start_charnos(source: str) -> Sequence[int]:
    start = 0
    charnos = []
    for line in source.splitlines(keepends=True):
        charnos.append(start)
        start += len(line)
    return tuple(charnos)


class Range(NamedTuple):
    start: int  # Character number
    end: int  # Character number

    def overlaps(self, other: "Range") -> bool:
        return self.start < other.end and other.start < self.end

    # Use & operator for overlaps()
    def __and__(self, other: "Range") -> bool:
        return self.overlaps(other)


def get_charnos(node: ast.AST, source: str, keep_first_indent: bool = False) -> Range:
    """Get start and end character numbers in source code from ast node.

    Args:
        node (ast.AST): Node to fetch character numbers for
        source (str): Python source code

    Returns:
        Tuple[int, int]: start, end
    """
    line_start_charnos = _get_line_start_charnos(source)
    if match_template(node, ast.AST(decorator_list=list)) and node.decorator_list:
        start = min(node.decorator_list, key=lambda n: (n.lineno, n.col_offset))
    else:
        start = node

    start_charno = line_start_charnos[start.lineno - 1] + start.col_offset
    if getattr(node, "end_lineno", None) is None:
        return Range(start_charno, start_charno)

    end_charno = line_start_charnos[node.end_lineno - 1] + node.end_col_offset

    code = source[start_charno:end_charno]
    if code[0] == " ":
        whitespace = max(re.findall(r"\A^ *", code), key=len)
        start_charno += len(whitespace)
    if code[-1] == " ":
        whitespace = max(re.findall(r" *\Z$", code), key=len)
        end_charno -= len(whitespace)
    if source[start_charno - 1] == "@" and isinstance(
        node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        start_charno -= 1
    if keep_first_indent:
        whitespace = max(re.findall(r" *\Z$", source[:start_charno]), key=len)
        start_charno -= len(whitespace)

    if (
        constants.PYTHON_VERSION < (3, 9)
        and not is_valid_python(source[start_charno:end_charno])
        and not code.startswith(" ")
        and not match_template(node, ast.AST(body=list))
    ):
        start_charno += 2
        end_charno += 2

    return Range(start_charno, end_charno)


def get_code(node: ast.AST | Range, source: str) -> str:
    """Get python code from ast

    Args:
        node (ast.AST): ast to get code from
        source (str): Python source code that ast was parsed from

    Returns:
        str: Python source code

    """
    if isinstance(node, Range):
        return source[node.start : node.end]

    start_charno, end_charno = get_charnos(node, source)
    return source[start_charno:end_charno]


def literal_value(node: ast.AST) -> bool:
    if has_side_effect(node):
        raise ValueError("Cannot find a deterministic value for a node with a side effect")

    if match_template(
        node, ast.BinOp(op=tuple(constants.COMPARISON_OPERATORS), left=object, right=object)
    ):
        left = literal_value(node.left)
        right = literal_value(node.right)
        return constants.COMPARISON_OPERATORS[type(node.op)](left, right)

    if match_template(node, ast.Compare(left=object, ops={object}, comparators={object})):
        return all(
            constants.COMPARISON_OPERATORS[type(op)](literal_value(left), literal_value(comparator))
            for left, op, comparator in zip(
                [node.left] + node.comparators, node.ops, node.comparators
        ))

    if match_template(node, ast.UnaryOp(op=ast.Not, operand=object)):
        return not literal_value(node.operand)

    if match_template(node, ast.BoolOp(op=(ast.And, ast.Or))):
        if not node.values:
            raise ValueError("Cannot find a deterministic value for an empty BoolOp")
        if isinstance(node.op, ast.And):
            # Return the first falsy value, if any.
            # Otherwise return the last value.
            for value in node.values:
                result = literal_value(value)
                if not result:
                    return result

            return result

        if isinstance(node.op, ast.Or):
            # Return the first non-falsy value, if any.
            # Otherwise return the last value.
            for value in node.values:
                result = literal_value(value)
                if result:
                    return result

            return result

    return ast.literal_eval(node)


def _is_exception(node: ast.AST) -> bool:
    """Check if a node is an exception.

    Args:
        node (ast.AST): Node to check

    Returns:
        bool: True if it will always raise an exception
    """
    if isinstance(node, ast.Raise):
        return True

    if isinstance(node, ast.Assert):
        try:
            return not literal_value(node.test)
        except (ValueError, AttributeError):
            return False

    return False


def is_blocking(node: ast.AST, parent_type: ast.AST = None) -> bool:
    """Check if a node is impossible to get past.

    Args:
        node (ast.AST): Node to check

    Returns:
        bool: True if no code after this node can ever be executed.
    """
    if _is_exception(node):
        return True

    if parent_type is None:
        blocking_types = (ast.Return, ast.Continue, ast.Break)
    elif parent_type in (ast.For, ast.While):
        blocking_types = (ast.Return,)

    if isinstance(node, blocking_types):
        return True

    if isinstance(node, ast.If):
        try:
            branch = node.body if literal_value(node.test) else node.orelse
        except ValueError:
            branches = [node.body, node.orelse]
            return all(
                any(is_blocking(child, parent_type) for child in branch) for branch in branches
            )
        else:
            return any(is_blocking(child, parent_type) for child in branch)

    if isinstance(node, ast.While):
        try:
            test_value = literal_value(node.test)
        except ValueError:
            pass
        else:
            if not test_value:
                return False

            for child in node.body:
                if isinstance(child, ast.Break):
                    return False
                if is_blocking(child, type(node)):
                    return True

    if isinstance(node, ast.For):
        try:
            iterator = literal_value(node.iter)
        except ValueError:
            return False
        if not any(True for _ in iterator):
            return False

    if isinstance(node, (ast.For, ast.While)):
        for child in node.body:
            if is_blocking(child, type(node)):
                return True
            if is_blocking(child, parent_type):
                return False
            if isinstance(child, ast.If) and any(walk(child, (ast.Break, ast.Continue))):
                try:
                    test = literal_value(child.test)
                except ValueError:
                    return False
                if test:
                    return False

        if isinstance(node, ast.For):
            return False
        try:
            return literal_value(node.test)
        except ValueError:
            return False

    if isinstance(node, ast.With):
        return any(is_blocking(child, parent_type) for child in node.body)

    return False


class _NameWildcardTransformer(ast.NodeTransformer):
    def __init__(
        self,
        name_wildcard_mapping: Mapping[str, Wildcard],
        expand: Collection[str],
        ignore: Collection[str],
    ):
        self.name_wildcard_mapping = name_wildcard_mapping
        self.expand = expand
        self.ignore = frozenset(ignore)

    def visit(self, node):
        # This is where the recursion happens, so without this we won't be
        # visiting any nodes.
        if isinstance(node, Wildcard):
            template = self.visit(node.template)
            node = Wildcard(node.name, template, node.common)
        elif isinstance(node, tuple):
            node = tuple(self.visit(child) for child in node)
        elif isinstance(node, set):
            node = {self.visit(child) for child in node}
        elif isinstance(node, list):
            node = [self.visit(child) for child in node]
        elif isinstance(node, type):
            return node
        else:
            node = super().visit(node)

        # Delete attributes potentially present in ignore.
        # This may happen with on the fly compiled templates, where these
        # attributes are not explicitly set.
        attrs_in_ignore = set(dir(node)) & self.ignore
        if attrs_in_ignore:
            new_attrs = {
                attr: value for attr, value in vars(node).items() if attr not in self.ignore
            }
            node = type(node)(**new_attrs)

        if not self.expand:
            return node

        changes = False
        node_vars = dict(vars(node))
        for key, value in node_vars.items():
            if (
                isinstance(value, list)
                and value
                and isinstance(value[0], Wildcard)
                and value[0].name in self.expand
            ):
                assert len(value) == 1, "Cannot expand more than one wildcard"
                node_vars[key] = {value[0]}
                changes = True

        if not changes:
            return node

        return type(node)(**node_vars)

    def visit_arg(self, node):
        return self.name_wildcard_mapping.get(node.arg, node)

    def visit_Name(self, node):
        return self.name_wildcard_mapping.get(node.id, node)

    def visit_Attribute(self, node):
        new_attr = self.name_wildcard_mapping.get(node.attr, node.attr)
        new_node = ast.Attribute(value=self.visit(node.value), attr=new_attr, ctx=node.ctx)
        return ast.copy_location(new_node, node)

    def visit_alias(self, node):
        new_name = self.name_wildcard_mapping.get(node.name, node.name)
        new_asname = self.name_wildcard_mapping.get(node.asname, node.asname)

        if (new_name, new_asname) == (node.name, node.asname):
            return node

        new_node = ast.alias(name=new_name, asname=new_asname)
        return ast.copy_location(new_node, node)


@functools.lru_cache(maxsize=10_000)
def compile_template(
    source: str | Set[str] | Tuple[str, ...],
    ignore: Collection[str] = frozenset(
        ("lineno", "col_offset", "end_lineno", "end_col_offset", "ctx")
    ),
    keep_expr: bool = False,
    expand: Collection[str] = frozenset(),
    **wildcards: ast.AST,
) -> ast.AST:
    if isinstance(source, tuple):
        return tuple(compile_template(s, ignore, keep_expr, **wildcards) for s in source)
    if isinstance(source, set):
        return {compile_template(s, ignore, keep_expr, **wildcards) for s in source}
    if isinstance(expand, str):
        expand = frozenset((expand,))

    name_wildcard_mapping = {}
    wildcards = {
        **{name.strip("{}"): object for name in re.findall(r"\{\{\w+\}\}", source)},
        **wildcards,
    }
    transformer = _NameWildcardTransformer(name_wildcard_mapping, expand, ignore)
    for name, template in wildcards.items():
        wildcard_placeholder_name = f"____wildcard__{name}____"
        assert (
            wildcard_placeholder_name not in source
        ), f"Bad wildcard name: `{name}` found in source."

        source = source.replace("{{" + name + "}}", wildcard_placeholder_name)

        wildcard = Wildcard(name, transformer.visit(template))
        name_wildcard_mapping[wildcard_placeholder_name] = wildcard

    if unfilled_wildcards := re.findall(r"\{\{\w+\}\}", source):
        raise ValueError(f"Unfilled wildcards found in source: {unfilled_wildcards}")

    source = textwrap.dedent(source)
    template = ast.parse(source)

    if not template.body:
        raise ValueError("Template is empty.")

    template = transformer.visit(template)
    template = template.body

    if len(template) > 1:
        return template

    template = template[0]

    if isinstance(template, ast.Expr) and not keep_expr:
        return template.value

    return template


def format_template(source: str, template_match: NamedTuple, **callables) -> str:
    template_match_asdict = template_match._asdict() if hasattr(template_match, "_asdict") else {}
    for name, value in template_match_asdict.items():
        source = source.replace("{{" + name + "}}", unparse(value))

    # It's ok that some of the template_match isn't used, just like str.format()
    # may not use all of the arguments.

    if unfilled_wildcards := re.findall(r"\{\{\w+\}\}", source):
        raise ValueError(f"Unfilled wildcards found in source: {unfilled_wildcards}")

    for callable_slot in re.finditer(r"\{\{\w+\((\w+,?)+\)\}\}", source):
        callable_slot = callable_slot.group()
        callable_name, *arg_names = re.findall(r"\w+", callable_slot)
        callable_result = callables[callable_name](
            *[template_match_asdict[name] for name in arg_names]
        )
        source = source.replace(callable_slot, callable_result)

    return source
