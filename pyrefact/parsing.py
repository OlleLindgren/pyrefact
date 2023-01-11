import ast
import collections
import dataclasses
import functools
import itertools
import re
from typing import Collection, Iterable, Mapping, Sequence, Tuple, Union

from pyrefact import constants


@dataclasses.dataclass(eq=True, frozen=True)
class Wildcard:
    """A wildcard matches a specific template, and extracts its name."""
    name: str
    template: object


def _merge_matches(root: ast.AST, matches: Iterable[Tuple[object]]) -> Tuple[object]:
    if not all(matches):
        return ()

    namedtuple_matches = []
    tuple_matches = []
    for match in matches:
        if type(match) is tuple:
            tuple_matches.append(match)
        else:
            namedtuple_matches.append(match)
    if not namedtuple_matches:
        return (root,)
    namedtuple_vars = collections.defaultdict(list)  # May not be hashable

    for match in namedtuple_matches:
        for key, value in zip(match._fields, match):
            namedtuple_vars[key].append(value)

    # Always store node in special "root" field. Other contents of this field are discarded.
    namedtuple_vars["root"] = [root]

    # Sort in alphabetical order, but always with "root" first.
    if not all(
        len({ast.dump(value) if isinstance(value, ast.AST) else str(value) for value in values})
        == 1
        for values in namedtuple_vars.values()
    ):
        return ()

    fields = sorted(namedtuple_vars.keys(), key=lambda k: (k != "root", k))
    namedtuple_type = collections.namedtuple("Match", fields)
    return namedtuple_type(*(namedtuple_vars[field][0] for field in fields))


def match_template(node: ast.AST, template: ast.AST) -> Tuple:
    """Match a node against a provided ast template.

    Args:
        node (ast.AST): Node to match against template
        template (ast.AST): Template to match node with

    Returns:
        Tuple:
            If node matches, a namedtuple with node as the first element, and
            any wildcards as other fields. Otherwise, an empty tuple.
    """
    # A type indicates that the node should be an instance of that type,
    # and nothing else. A tuple indicates that the node should be any of
    # the types in it.

    if isinstance(template, type):
        return (node,) if isinstance(node, template) else ()

    # A tuple indicates an or condition; the node must comply with any of
    # the templates in the child.
    # They may all be types for example, which boils down to the traditional
    # isinstance logic.
    # If there are wildcards, the first match is chosen.
    if isinstance(template, tuple):
        return next(filter(None, (match_template(node, child) for child in template)), ())

    # A set indicates a variable length list, where all elements must match
    # against at least one of the templates in it.
    # If there are wildcards, the first match is chosen.
    # If there are inconsistencies between different elements, the match is discarded.
    if isinstance(template, set):
        if not isinstance(node, list):
            return ()

        matches = [match_template(node_child, tuple(template)) for node_child in node]
        return _merge_matches(node, matches)

    # A list indicates that the node must also be a list, and for every
    # element, it must match against the corresponding node in the template.
    # It must also be equal length.
    if isinstance(template, list):
        if isinstance(node, list) and len(node) == len(template):
            matches = [
                match_template(child, template_child)
                for child, template_child in zip(node, template)
            ]
            return _merge_matches(node, matches)

        return ()

    if template is True or template is False or template is None:
        return (node,) if node is template else ()

    if isinstance(template, Wildcard):
        namedtuple_type = collections.namedtuple("Match", (template.name,))
        template_match = match_template(node, template.template)
        return namedtuple_type(template_match[0]) if len(template_match) == 1 else ()

    # If the node is not an ast, we presume it is a string or something like
    # that, and just assert it should be equal.
    if not isinstance(node, ast.AST):
        return (node,) if node == template else ()

    t_vars = vars(template)
    n_vars = vars(node)

    if issubclass(type(node), type(template)) and t_vars.keys() <= n_vars.keys():
        matches = [match_template(n_vars[key], t_vars[key]) for key in t_vars.keys()]
        return _merge_matches(node, matches)

    return ()


@functools.lru_cache(maxsize=100)
def parse(source_code: str) -> ast.AST:
    """Parse python source code and cache

    Args:
        source_code (str): Python source code

    Returns:
        ast.AST: Parsed AST
    """
    return ast.parse(source_code)


@functools.lru_cache(maxsize=100)
def _group_nodes_in_scope(scope: ast.AST) -> Mapping[ast.AST, Sequence[ast.AST]]:
    node_types = collections.defaultdict(list)
    for node in ast.walk(scope):
        node_types[type(node)].append(node)

    return node_types


def walk_wildcard(
    scope: ast.AST, node_template: Union[ast.AST, Tuple[ast.AST, ...]]
) -> Sequence[Tuple[ast.AST, ...]]:
    """Get nodes in scope of a particular type. Match wildcards.

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
                if template_match := match_template(node, template):
                    yielded_nodes.add(node)
                    yield template_match


def walk(scope: ast.AST, node_template: Union[ast.AST, Tuple[ast.AST, ...]]) -> Sequence[ast.AST]:
    """Get nodes in scope of a particular type

    Args:
        scope (ast.AST): Scope to search
        node_template (ast.AST): Node type to filter on

    Returns:
        Sequence[ast.AST]: All nodes in scope of that type
    """
    for node, *_ in walk_wildcard(scope, node_template):
        yield node


def walk_sequence(
    scope: ast.Module, *templates: ast.AST, expand_first: bool = False, expand_last: bool = False
) -> Iterable[Sequence[ast.AST]]:
    for node in walk(scope, (ast.AST(body=list), ast.AST(orelse=list))):
        for body in [
            getattr(node, "body", []),
            getattr(node, "orelse", []),
        ]:
            if not body:
                continue

            for nodes in zip(
                *(body[i : len(body) - len(templates) + i + 1] for i in range(len(templates)))
            ):
                matches = [
                    match_template(node, template) for node, template in zip(nodes, templates)
                ]

                if not all(matches):
                    continue

                if expand_first:
                    pre = body[: body.index(nodes[0])]
                    for child in reversed(pre):
                        if template_match := match_template(child, templates[0]):
                            matches.insert(0, template_match)

                if expand_last:
                    post = body[body.index(nodes[-1]) + 1 :]
                    for child in post:
                        if template_match := match_template(child, templates[-1]):
                            matches.append(template_match)

                yield tuple(matches)


def filter_nodes(
    nodes: Iterable[ast.AST], node_type: Union[ast.AST, Sequence[ast.AST]]
) -> Sequence[ast.AST]:
    for node in nodes:
        if match_template(node, node_type):
            yield node


def iter_similar_nodes(
    root: ast.AST, content: str, node_type: ast.AST, count: int, length: int
) -> Collection[ast.AST]:
    for sequence in walk_sequence(root, *[node_type] * count):
        sequence = [node for node, *_ in sequence]
        for i, chars in enumerate(
            zip(*(re.sub(r"\s", "", get_code(node, content)) for node in sequence))
        ):
            if len(set(chars)) != 1:
                break
            if i - 1 >= length:
                yield sequence
                break


def is_valid_python(content: str) -> bool:
    """Determine if source code is valid python.

    Args:
        content (str): Python source code

    Returns:
        bool: True if content is valid python.
    """
    try:
        parse(content)
        return True
    except SyntaxError:
        return False


def is_private(variable: str) -> bool:
    return variable.startswith("_")


def is_magic_method(node: ast.AST) -> bool:
    """Determine if a node is a magic method function definition, like __init__ for example.

    Args:
        node (ast.AST): AST to check.

    Returns:
        bool: True if it is a magic method function definition.
    """
    return (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("__")
        and node.name.endswith("__")
    )


def _unpack_ast_target(target: ast.AST) -> Iterable[ast.Name]:
    if isinstance(target, ast.Name):
        yield target
        return
    if isinstance(target, ast.Tuple):
        for subtarget in target.elts:
            yield from _unpack_ast_target(subtarget)


def iter_assignments(ast_tree: ast.Module) -> Iterable[ast.Name]:
    """Iterate over defined variables in code

    Args:
        content (str): Python source code

    Yields:
        ast.Name: A name that is being assigned.
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
        ast.FunctionDef: A function definition node
    """
    for node in ast_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def iter_classdefs(ast_tree: ast.Module) -> Iterable[ast.ClassDef]:
    """Iterate over defined variables in code

    Args:
        ast_tree (ast.Module): Module to parse

    Yields:
        ast.ClassDef: A class definition node
    """
    for node in ast_tree.body:
        if isinstance(node, (ast.ClassDef)):
            yield node


def iter_typedefs(ast_tree: ast.Module) -> Iterable[ast.Name]:
    """Iterate ove all TypeVars and custom type annotations in code

    Args:
        ast_tree (ast.Module): Module to parse

    Yields:
        ast.Assign: An assignment of a custom type annotation or typevar
    """
    for node in filter_nodes(ast_tree.body, ast.Assign(targets=[object])):
        for child in ast.walk(node.value):
            if isinstance(child, ast.Name) and (
                child.id in constants.ASSUMED_SOURCES["typing"] or "namedtuple" in child.id
            ):
                yield node
                break
            if match_template(
                child, ast.Attribute(value=ast.Name(id=("collections", "typing")))
            ) and ("namedtuple" in child.attr or child.attr in constants.ASSUMED_SOURCES["typing"]):
                yield node
                break


def slice_of(node: ast.Subscript) -> ast.AST:
    node_slice = node.slice
    if constants.PYTHON_VERSION < (3, 9):
        if isinstance(node_slice, ast.Index):
            return node_slice.value
        if isinstance(node_slice, ast.ExtSlice):
            return ast.Tuple(elts=node_slice.dims)

    return node_slice


def has_side_effect(
    node: ast.AST,
    safe_callable_whitelist: Collection[str] = frozenset(),
) -> bool:
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
        node,
        (
            ast.Yield,
            ast.YieldFrom,
            ast.Return,
            ast.Raise,
            ast.Continue,
            ast.Break,
            ast.Assert,
        ),
    ):
        return True

    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name != "_"

    if isinstance(node, ast.For):
        return any(
            has_side_effect(item, safe_callable_whitelist)
            for item in itertools.chain(
                [node.target],
                [node.iter],
                node.body,
            )
        )

    if isinstance(node, ast.Lambda):
        return has_side_effect(node.args, safe_callable_whitelist) or has_side_effect(
            node.body, safe_callable_whitelist
        )

    if isinstance(node, ast.arguments):
        return any(
            has_side_effect(item, safe_callable_whitelist)
            for item in itertools.chain(
                node.posonlyargs,
                node.args,
                node.kwonlyargs,
                node.kw_defaults,
                node.defaults,
            )
        )

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

    if isinstance(node, ast.Subscript):
        return (
            has_side_effect(node.value, safe_callable_whitelist)
            or has_side_effect(node.slice, safe_callable_whitelist)
            or (
                isinstance(node.ctx, ast.Store)
                and not (isinstance(node.value, ast.Name) and node.value.id == "_")
            )
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
            and not has_side_effect(node.iter, safe_callable_whitelist)
            and not any(has_side_effect(value, safe_callable_whitelist) for value in node.ifs)
        ):
            return False
        return True

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

    return True


@functools.lru_cache(maxsize=1)
def _get_line_start_charnos(content: str) -> Sequence[int]:
    start = 0
    charnos = []
    for line in content.splitlines(keepends=True):
        charnos.append(start)
        start += len(line)
    return charnos


def get_charnos(node: ast.AST, content: str, keep_first_indent: bool=False) -> Tuple[int, int]:
    """Get start and end character numbers in source code from ast node.

    Args:
        node (ast.AST): Node to fetch character numbers for
        content (str): Python source code

    Returns:
        Tuple[int, int]: start, end
    """
    line_start_charnos = _get_line_start_charnos(content)
    if match_template(node, ast.AST(decorator_list=list)) and node.decorator_list:
        start = min(node.decorator_list, key=lambda n: (n.lineno, n.col_offset))
    else:
        start = node

    start_charno = line_start_charnos[start.lineno - 1] + start.col_offset
    end_charno = line_start_charnos[node.end_lineno - 1] + node.end_col_offset

    code = content[start_charno:end_charno]
    if code[0] == " ":
        whitespace = max(re.findall(r"\A^ *", code), key=len)
        start_charno += len(whitespace)
    if code[-1] == " ":
        whitespace = max(re.findall(r" *\Z$", code), key=len)
        end_charno -= len(whitespace)
    if content[start_charno - 1] == "@" and isinstance(
        node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        start_charno -= 1
    if keep_first_indent:
        whitespace = max(re.findall(r" *\Z$", content[:start_charno]), key=len)
        start_charno -= len(whitespace)

    if (
        constants.PYTHON_VERSION < (3, 9)
        and not is_valid_python(content[start_charno:end_charno])
        and not isinstance(
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
        )
    ):
        start_charno += 2
        end_charno += 2

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


def literal_value(node: ast.AST) -> bool:
    if has_side_effect(node):
        raise ValueError("Cannot find a deterministic value for a node with a side effect")

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


def iter_bodies_recursive(
    ast_root: ast.Module,
) -> Iterable[Union[ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef]]:
    try:
        left = list(ast_root.body)
    except AttributeError:
        return
    while left:
        for node in left.copy():
            left.remove(node)
            if match_template(node, ast.AST(body=list, orelse=list)):
                left.extend(node.body)
                left.extend(node.orelse)
                yield node
            elif match_template(node, ast.AST(body=list)):
                left.extend(node.body)
                yield node


def _get_imports(ast_tree: ast.Module) -> Iterable[Union[ast.Import, ast.ImportFrom]]:
    """Iterate over all import nodes in ast tree. __future__ imports are skipped.

    Args:
        ast_tree (ast.Module): Ast tree to search for imports

    Yields:
        str: An import node
    """
    for node in walk(ast_tree, ast.Import):
        yield node
    for node in walk(ast_tree, ast.ImportFrom):
        if node.module != "__future__":
            yield node


def get_imported_names(ast_tree: ast.Module) -> Collection[str]:
    """Get all names that are imported in module.

    Args:
        ast_tree (ast.Module): Module to search

    Returns:
        Collection[str]: All imported names.
    """
    imports = {
        alias.name if alias.asname is None else alias.asname
        for node in _get_imports(ast_tree)
        for alias in node.names
    }

    return imports


def safe_callable_names(root: ast.Module) -> Collection[str]:
    """Compute what functions can safely be called without having a side effect.

    This is also to compute the inverse, i.e. what function calls may be removed
    without breaking something.

    Args:
        root (ast.Module): Module to find function definitions in

    Returns:
        Collection[str]: Names of all functions that have no side effect when called.
    """
    defined_names = {node.id for node in walk(root, ast.Name(ctx=ast.Store))}
    function_defs = list(walk(root, (ast.FunctionDef, ast.AsyncFunctionDef)))
    safe_callables = set(constants.BUILTIN_FUNCTIONS) - {"print", "exit"}
    safe_callable_nodes = set()
    changes = True
    while changes:
        changes = False
        for node in function_defs:
            if node.name in defined_names:
                continue
            nonreturn_children = []
            for child in node.body:
                if is_blocking(child):
                    break

                nonreturn_children.append(child)
            return_children = [child.value for child in walk(node, ast.Return)]

            if not any(
                has_side_effect(child, safe_callables)
                for child in itertools.chain(nonreturn_children, return_children)
            ):
                safe_callable_nodes.add(node)
                safe_callables.add(node.name)
                changes = True

        function_defs = [node for node in function_defs if node.name not in safe_callables]

    for node in walk(root, ast.ClassDef):
        constructors = {
            child
            for child in node.body
            if match_template(child, ast.FunctionDef(name=("__init__", "__post_init__", "__new__")))
        }
        if not constructors - safe_callable_nodes:
            safe_callables.add(node.name)

    return safe_callables


def module_dependencies(root: ast.Module) -> Iterable[str]:
    """Iterate over all packages that a module depends on."""
    for node in walk(root, ast.Import):
        for alias in node.names:
            yield alias.name

    for node in walk(root, ast.ImportFrom):
        yield node.module


def get_comp_wrapper_func_equivalent(node: ast.AST) -> str:
    if isinstance(node, ast.DictComp):
        return "dict"
    if isinstance(node, ast.ListComp):
        return "list"
    if isinstance(node, ast.SetComp):
        return "set"
    if isinstance(node, ast.GeneratorExp):
        return "iter"

    raise ValueError(f"Unexpected type of node: {type(node)}")


def is_transpose_operation(node: ast.AST) -> bool:
    numpy_transpose_template = ast.Attribute(value=object, attr="T")
    zip_transpose_template = ast.Call(func=ast.Name(id="zip"), args=[ast.Starred])

    return match_template(node, (numpy_transpose_template, zip_transpose_template))


def transpose_target(node: ast.AST) -> ast.AST:
    if isinstance(node, ast.Attribute):
        return node.value

    if isinstance(node, ast.Call) and len(node.args) > 0:
        return node.args[0].value

    raise ValueError(f"Node {node} is not a transpose operation.")


def is_call(node: ast.AST, qualified_name: Union[str, Collection[str]]) -> bool:
    if not isinstance(node, ast.Call):
        return False

    if isinstance(qualified_name, str):
        qualified_name = (qualified_name,)

    func = node.func

    if isinstance(func, ast.Name):
        return func.id in qualified_name

    if match_template(func, ast.Attribute(value=ast.Name)):
        return f"{func.value.id}.{func.attr}" in qualified_name

    return False
