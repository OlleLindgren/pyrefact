from __future__ import annotations

import ast
import copy
import itertools
import re
from typing import Collection, Iterable

from pyrefact import constants, core


def iter_similar_nodes(
    root: ast.AST, source: str, node_type: ast.AST, count: int, length: int
) -> Collection[ast.AST]:
    for sequence in core.walk_sequence(root, *[node_type] * count):
        sequence = [node for node, *_ in sequence]
        for i, chars in enumerate(
            zip(*(re.sub(r"\s", "", core.get_code(node, source)) for node in sequence))
        ):
            if len(set(chars)) != 1:
                break
            if i - 1 >= length:
                yield sequence
                break


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


def iter_assignments(root: ast.Module) -> Iterable[ast.Name]:
    """Iterate over defined variables in code

    Args:
        source (str): Python source code

    Yields:
        ast.Name: A name that is being assigned.
    """
    for node in root.body:
        if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            yield from _unpack_ast_target(node.target)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                yield from _unpack_ast_target(target)


def iter_funcdefs(root: ast.Module) -> Iterable[ast.FunctionDef]:
    """Iterate over defined variables in code

    Args:
        root (ast.Module): Module to parse

    Yields:
        ast.FunctionDef: A function definition node
    """
    for node in root.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def iter_classdefs(root: ast.Module) -> Iterable[ast.ClassDef]:
    """Iterate over defined variables in code

    Args:
        root (ast.Module): Module to parse

    Yields:
        ast.ClassDef: A class definition node
    """
    for node in root.body:
        if isinstance(node, (ast.ClassDef)):
            yield node


def iter_typedefs(root: ast.Module) -> Iterable[ast.Name]:
    """Iterate over all TypeVars and custom type annotations in code

    Args:
        root (ast.Module): Module to parse

    Yields:
        ast.Assign: An assignment of a custom type annotation or typevar
    """
    for node in core.filter_nodes(root.body, ast.Assign(targets=[object])):
        for child in ast.walk(node.value):
            if isinstance(child, ast.Name) and (
                child.id in constants.ASSUMED_SOURCES["typing"] or "namedtuple" in child.id
            ):
                yield node
                break
            if core.match_template(
                child, ast.Attribute(value=ast.Name(id=("collections", "typing")))
            ) and ("namedtuple" in child.attr or child.attr in constants.ASSUMED_SOURCES["typing"]):
                yield node
                break


def iter_bodies_recursive(
    root: ast.Module,
) -> Iterable[ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef]:
    try:
        left = list(root.body)
    except AttributeError:
        return
    while left:
        for node in left.copy():
            left.remove(node)
            if core.match_template(node, ast.AST(body=list, orelse=list)):
                left.extend(node.body)
                left.extend(node.orelse)
                yield node
            elif core.match_template(node, ast.AST(body=list)):
                left.extend(node.body)
                yield node


def safe_callable_names(root: ast.Module) -> Collection[str]:
    """Compute what functions can safely be called without having a side effect.

    This is also to compute the inverse, i.e. what function calls may be removed
    without breaking something.

    Args:
        root (ast.Module): Module to find function definitions in

    Returns:
        Collection[str]: Names of all functions that have no side effect when called.
    """
    defined_names = {node.id for node in core.walk(root, ast.Name(ctx=ast.Store))}
    function_defs = list(core.walk(root, (ast.FunctionDef, ast.AsyncFunctionDef)))
    safe_callables = set(constants.SAFE_CALLABLES)
    safe_callable_nodes = set()
    changes = True
    while changes:
        changes = False
        for node in function_defs:
            if node.name in defined_names:
                continue
            nonreturn_children = []
            for child in node.body:
                if core.is_blocking(child):
                    break

                nonreturn_children.append(child)
            return_children = [child.value for child in core.walk(node, ast.Return)]

            if not any(
                core.has_side_effect(child, safe_callables)
                for child in itertools.chain(nonreturn_children, return_children)
            ):
                safe_callable_nodes.add(node)
                safe_callables.add(node.name)
                changes = True

        function_defs = [node for node in function_defs if node.name not in safe_callables]

    for node in core.walk(root, ast.ClassDef):
        constructors = {
            child
            for child in node.body
            if core.match_template(
                child, ast.FunctionDef(name=("__init__", "__post_init__", "__new__"))
        )}
        if not constructors - safe_callable_nodes:
            safe_callables.add(node.name)

    return safe_callables


def module_dependencies(root: ast.Module) -> Iterable[str]:
    """Iterate over all packages that a module depends on."""
    for node in core.walk(root, ast.Import):
        for alias in node.names:
            yield alias.name

    for node in core.walk(root, ast.ImportFrom):
        yield node.module


def is_transpose_operation(node: ast.AST) -> bool:
    numpy_transpose_template = ast.Attribute(value=object, attr="T")
    zip_transpose_template = ast.Call(func=ast.Name(id="zip"), args=[ast.Starred])

    return core.match_template(node, (numpy_transpose_template, zip_transpose_template))


def transpose_target(node: ast.AST) -> ast.AST:
    if isinstance(node, ast.Attribute):
        return node.value

    if isinstance(node, ast.Call) and len(node.args) > 0:
        return node.args[0].value

    raise ValueError(f"Node {node} is not a transpose operation.")


def is_call(node: ast.AST, qualified_name: str | Collection[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False

    if isinstance(qualified_name, str):
        qualified_name = (qualified_name,)

    func = node.func

    if isinstance(func, ast.Name):
        return func.id in qualified_name

    if core.match_template(func, ast.Attribute(value=ast.Name)):
        return f"{func.value.id}.{func.attr}" in qualified_name

    return False


def assignment_targets(
    node: ast.Assign | ast.AnnAssign | ast.AugAssign | ast.For,
) -> Collection[ast.Name]:
    targets = set()
    if isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For)):
        return set(core.walk(node.target, ast.Name(ctx=ast.Store)))
    if isinstance(node, ast.Assign):
        for target in node.targets:
            targets.update(core.walk(target, ast.Name(ctx=ast.Store)))
        return targets
    raise TypeError(f"Expected Assignment type, got {type(node)}")


def with_added_indent(node: ast.AST, indent: int):
    clone = copy.deepcopy(node)
    for child in ast.walk(clone):
        if isinstance(child, ast.AST) and hasattr(child, "col_offset"):
            child.col_offset += indent

    return clone
