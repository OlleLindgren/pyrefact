from __future__ import annotations

import ast
import collections
import functools
import importlib
from pathlib import Path
from typing import Collection, Iterable, Sequence, Tuple, NamedTuple

from pyrefact import constants, parsing, processing


def _get_imports(ast_tree: ast.Module) -> Iterable[ast.Import | ast.ImportFrom]:
    """Iterate over all import nodes in ast tree. __future__ imports are skipped.

    Args:
        ast_tree (ast.Module): Ast tree to search for imports

    Yields:
        str: An import node
    """
    for node in parsing.walk(ast_tree, ast.Import):
        yield node
    for node in parsing.walk(ast_tree, ast.ImportFrom):
        if node.module != "__future__":
            yield node


def get_imported_names(ast_tree: ast.Module) -> Collection[str]:
    """Get all names that are imported in module.

    Args:
        ast_tree (ast.Module): Module to search

    Returns:
        Collection[str]: All imported names.
    """

    return {
        alias.name if alias.asname is None else alias.asname
        for node in _get_imports(ast_tree)
        for alias in node.names
    }


def code_dependencies_outputs(
    code: Sequence[ast.AST],
) -> Tuple[Collection[str], Collection[str], Collection[str]]:
    """Get required and created names in code.

    Args:
        code (Sequence[ast.AST]): Nodes to find required and created names by

    Raises:
        ValueError: If any node is a try, class or function def.

    Returns:
        Tuple[Collection[str], Collection[str]]: created_names, maybe_created_names, required_names
    """
    required_names = set()
    created_names = set()
    created_names_original = created_names
    maybe_created_names = set()

    for node in code:
        temp_children = []
        children = []
        if isinstance(node, (ast.While, ast.For, ast.If)):
            temp_children = (
                [node.test] if isinstance(node, (ast.If, ast.While)) else [node.target, node.iter]
            )
            children = [node.body, node.orelse]
            if any(parsing.is_blocking(child) for child in ast.walk(node)):
                created_names = maybe_created_names

        elif isinstance(node, ast.With):
            temp_children = tuple(node.items)
            children = [node.body]

        elif isinstance(node, (ast.Try, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            required_names.update(name.id for name in parsing.walk(node, ast.Name))
            required_names.update(
                func.name
                for func in parsing.walk(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ))
            if isinstance(node, ast.Try):
                maybe_created_names.update(
                    name.id for name in parsing.walk(node, ast.Name(ctx=ast.Store))
                )
            continue

        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            created_names.update(
                alias.name if alias.asname is None else alias.asname for alias in node.names
            )
            continue

        else:
            node_created = set()
            node_needed = set()
            generator_internal_names = set()
            for child in parsing.walk(
                node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
            ):
                comp_created = set()
                for comp in child.generators:
                    comp_created.update(parsing.walk(comp.target, ast.Name(ctx=ast.Store)))
                for grandchild in ast.walk(child):
                    if isinstance(grandchild, ast.Name) and grandchild.id in comp_created:
                        generator_internal_names.add(grandchild)

            if isinstance(node, ast.AugAssign):
                node_needed.update(n.id for n in parsing.assignment_targets(node))

            for child in parsing.walk(node, ast.Attribute(ctx=ast.Load)):
                for n in parsing.walk(child, ast.Name):
                    if n not in generator_internal_names:
                        node_needed.add(n.id)

            for child in parsing.walk(node, ast.Name):
                if child.id not in node_needed and child not in generator_internal_names:
                    if isinstance(child.ctx, ast.Load):
                        node_needed.add(child.id)
                    elif isinstance(child.ctx, ast.Store):
                        node_created.add(child.id)
                    else:
                        # Del
                        node_created.discard(child.id)
                        created_names.discard(child.id)

                elif isinstance(child.ctx, ast.Store):
                    maybe_created_names.add(child.id)

            node_needed -= created_names
            created_names.update(node_created)
            maybe_created_names.update(created_names)
            required_names.update(node_needed)

            continue

        temp_created, temp_maybe_created, temp_needed = code_dependencies_outputs(temp_children)
        maybe_created_names.update(temp_maybe_created)
        created = []
        needed = []

        for nodes in children:
            c_created, c_maybe_created, c_needed = code_dependencies_outputs(nodes)
            created.append(c_created)
            maybe_created_names.update(c_maybe_created)
            needed.append(c_needed - temp_created)

        node_created = set.intersection(*created) if created else set()
        node_needed = set.union(*needed) if needed else set()
        node_needed -= created_names
        node_needed -= temp_created
        node_needed |= temp_needed
        created_names.update(node_created)
        required_names.update(node_needed)

    return created_names_original, maybe_created_names, required_names


class TraceResult(NamedTuple):
    """Result of tracing."""

    source: str
    ast: ast.AST
    lineno: int


@functools.lru_cache(maxsize=100_000)
def trace_origin(name: str, source: str, working_directory: Path = None) -> Tuple[int, str]:
    if working_directory is None:
        working_directory = Path.cwd()
    else:
        working_directory = Path(working_directory)

    root = parsing.parse(source)
    nodes = set(parsing.walk(root, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign, ast.NamedExpr)))

    all_template = ast.Assign(targets=[ast.Name(id="all")], value=(ast.List, ast.Tuple))

    for node in sorted(nodes, key=lambda n: (n.lineno, n.col_offset), reverse=True):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.asname == name:
                    return TraceResult(parsing.get_code(node, source), node.lineno, node)
                if alias.asname is None and alias.name == name:
                    return TraceResult(parsing.get_code(node, source), node.lineno, node)
                if alias.name == "*":
                    if node.module in constants.PYTHON_311_STDLIB and name in dir(__import__(node.module)):
                        return TraceResult(parsing.get_code(node, source), node.lineno, node)

                    if module_spec := importlib.util.find_spec(node.module):
                        if module_spec.origin in {"frozen", "built-in"}:
                            if name in dir(__import__(node.module)):
                                return TraceResult(parsing.get_code(node, source), node.lineno, node)

                            continue

                        if module_spec.origin is None:
                            continue

                        with Path(module_spec.origin).open("r", encoding="utf-8") as stream:
                            module_source = stream.read()

                        module_root = parsing.parse(module_source)
                        # if __all__ exists, name is importable from there iff it is in __all__
                        if all_nodes := tuple(parsing.filter_nodes(module_root.body, all_template)):
                            for all_node in all_nodes:
                                if name in all_node.elts:
                                    return TraceResult(parsing.get_code(node, source), node.lineno, node)

                        elif trace_origin(name, module_source, working_directory):
                            return TraceResult(parsing.get_code(node, source), node.lineno, node)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return TraceResult(parsing.get_code(node, source), node.lineno, node)

        if isinstance(node, (ast.Assign, ast.AnnAssign)) and any(target.id == name for target in parsing.assignment_targets(node)):
            return TraceResult(parsing.get_code(node, source), node.lineno, node)

        if isinstance(node, ast.NamedExpr) and parsing.match_template(node.target, ast.Name(id=name)):
            return TraceResult(parsing.get_code(node, source), node.lineno, node)

    return None


def get_undefined_variables(source: str) -> Collection[str]:
    root = parsing.parse(source)
    imported_names = get_imported_names(root)
    defined_names = set()
    referenced_names = set()
    for node in parsing.walk(root, ast.Name):
        if isinstance(node.ctx, ast.Load):
            referenced_names.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            defined_names.add(node.id)
    for node in parsing.walk(root, ast.arg):
        defined_names.add(node.arg)

    return (
        referenced_names
        - defined_names
        - imported_names
        - {name.split(".")[0] for name in imported_names}
        - constants.BUILTIN_FUNCTIONS
    )


@processing.fix
def fix_starred_imports(source: str) -> str:
    """Replace starred imports with normal `from x import y, z` style imports."""

    # This is needlessly complicated because the cache has a way of getting invalidated
    # in the middle of it. I don't know why but it shows up on python3.12 on the main
    # test. If a fix is found (a fix should be found), this could be simplified to better
    # use set/dict keys() - keys() operations etc.

    root = parsing.parse(source)

    template = ast.ImportFrom(names=[ast.alias(name="*")])

    starred_import_name_mapping = collections.defaultdict(set)

    template = tuple(parsing.filter_nodes(root.body, template))

    if not template:
        return source

    undefined_names = get_undefined_variables(source)
    for name in undefined_names:
        if trace_result := trace_origin(name, source):
            node_source, lineno, node = trace_result

            if parsing.match_template(node, template):
                starred_import_name_mapping[node].add(name)

    for node, names in starred_import_name_mapping.items():
        if names:
            yield node, ast.ImportFrom(
                module=node.module,
                names=[ast.alias(name=name, asname=None) for name in sorted(names)],
                level=0,
            )

    # Remove remaining starred imports
    for node in parsing.filter_nodes(root.body, template):
        if not parsing.match_template(node, tuple(starred_import_name_mapping)):
            yield node, None
