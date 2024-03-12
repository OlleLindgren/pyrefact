from __future__ import annotations

import ast
import collections
import functools
import importlib
import sys
import threading
from pathlib import Path
from typing import Iterable, NamedTuple, Set, Sequence, Tuple

from pyrefact import constants, core, parsing, processing


def _get_imports(ast_tree: ast.Module) -> Iterable[ast.Import | ast.ImportFrom]:
    """Iterate over all import nodes in ast tree. __future__ imports are skipped.

    Args:
        ast_tree (ast.Module): Ast tree to search for imports

    Yields:
        str: An import node
    """
    for node in core.walk(ast_tree, ast.Import):
        yield node
    for node in core.walk(ast_tree, ast.ImportFrom):
        if node.module != "__future__":
            yield node


def get_imported_names(ast_tree: ast.Module) -> Set[str]:
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
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Get required and created names in code.

    Args:
        code (Sequence[ast.AST]): Nodes to find required and created names by

    Raises:
        ValueError: If any node is a try, class or function def.

    Returns:
        Tuple[Collection[str], Collection[str]]: created_names, maybe_created_names, required_names
    """
    required_names: Set[str] = set()
    created_names: Set[str] = set()
    created_names_original = created_names
    maybe_created_names: Set[str] = set()

    for node in code:
        temp_children = []
        children = []
        if isinstance(node, (ast.While, ast.For, ast.If)):
            temp_children = (
                [node.test] if isinstance(node, (ast.If, ast.While)) else [node.target, node.iter]
            )
            children = [node.body, node.orelse]
            if any(core.is_blocking(child) for child in ast.walk(node)):
                created_names = maybe_created_names

        elif isinstance(node, ast.With):
            temp_children = tuple(node.items)
            children = [node.body]

        elif isinstance(node, (ast.Try, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            required_names.update(name.id for name in core.walk(node, ast.Name))
            required_names.update(
                func.name
                for func in core.walk(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            )
            if isinstance(node, ast.Try):
                maybe_created_names.update(
                    name.id for name in core.walk(node, ast.Name(ctx=ast.Store))
                )
            continue

        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            created_names.update(
                alias.name if alias.asname is None else alias.asname for alias in node.names
            )
            continue

        else:
            node_created: Set[str] = set()
            node_needed: Set[str] = set()
            generator_internal_names = set()
            for child in core.walk(
                node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
            ):
                comp_created: Set[str] = set()
                for comp in child.generators:
                    for name in core.walk(comp.target, ast.Name(ctx=ast.Store)):
                        comp_created.add(name.id)
                for grandchild in ast.walk(child):
                    if isinstance(grandchild, ast.Name) and grandchild.id in comp_created:
                        generator_internal_names.add(grandchild)

            if isinstance(node, ast.AugAssign):
                node_needed.update(n.id for n in parsing.assignment_targets(node))

            for child in core.walk(node, ast.Attribute(ctx=ast.Load)):
                for n in core.walk(child, ast.Name):
                    if n not in generator_internal_names:
                        node_needed.add(n.id)

            for child in core.walk(node, ast.Name):
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


class _TraceResult(NamedTuple):
    """Result of tracing."""

    source: str
    lineno: int
    ast: ast.AST


# To prevent concurrent modification of sys.path
SYS_PATH_LOCK = threading.Lock()


def _trace_module_source_file(module: str) -> str | None:
    with SYS_PATH_LOCK:
        try:
            sys.path.append(str(Path.cwd()))

            try:
                module_spec = importlib.util.find_spec(module)
            except ImportError:
                return None

            if module_spec is None:
                return None

            return module_spec.origin

        finally:
            sys.path.pop()


@functools.lru_cache(maxsize=100_000)
def trace_origin(name: str, source: str, *, __all__: bool = False) -> _TraceResult | None:
    """Trace the origin of a name in python source code.

    Args:
        name (str): Name to trace
        source (str): Source code to trace name in
        __all__ (bool, optional): If True, and __all__ is defined in source,
            use __all__ as a filter for importable names. Defaults to False.

    Returns:
        (source, ast, lineno) of the origin of name in source.
    """
    root = core.parse(source)
    nodes = set(
        core.walk(
            root,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Assign,
                ast.AnnAssign,
                ast.NamedExpr,
    ),))

    # Try to figure out what the __all__ variable in source may contain at runtime.
    # The point of this is that, if __all__ is defined, only the names in __all__ are
    # importable from outside the module.
    # Without this, we could for example think that `os` was accessible in `pathlib`,
    # and end up putting `from pathlib import os` in generated code.
    if __all__:
        all_template = ast.Assign(
            targets=[ast.Name(id="__all__")], value=ast.List(elts={ast.Constant(value=str)})
        )
        all_extend_template = ast.Call(
            func=ast.Attribute(value=ast.Name(id="__all__"), attr="extend"),
            args=[(
                ast.Tuple(elts={ast.Constant(value=str)}),
                ast.List(elts={ast.Constant(value=str)}),
        )],)
        all_append_template = ast.Call(
            func=ast.Attribute(value=ast.Name(id="__all__"), attr="append"), args=[str]
        )
        all_filter: Set[str] = set()
        all_nodes = tuple(core.filter_nodes(root.body, all_template))

        if all_nodes:
            for node in all_nodes:
                all_filter.update(constant.value for constant in node.value.elts)

            for node in core.walk(root, all_extend_template):
                all_filter.update(constant.value for constant in node.args[0].elts)

            for node in core.walk(root, all_append_template):
                all_filter.add(node.args[0])

            if name not in all_filter:
                return None

    for node in sorted(nodes, key=lambda n: (n.lineno, n.col_offset), reverse=True):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.asname == name:
                    return _TraceResult(core.get_code(node, source), node.lineno, node)
                if alias.asname is None and alias.name == name:
                    return _TraceResult(core.get_code(node, source), node.lineno, node)

                if alias.name != "*":
                    continue

                if node.module in constants.PYTHON_311_STDLIB:
                    # Logic copied from _get_exports_list() in os.py from python3.12.0b2
                    module = __import__(node.module)
                    exports = getattr(
                        module, "__all__", [x for x in dir(module) if not x.startswith("_")]
                    )
                    if name in exports:
                        return _TraceResult(core.get_code(node, source), node.lineno, node)

                if node.module is None:
                    continue

                origin = _trace_module_source_file(node.module)

                # This is likely the best way to truly check the __all__ of a module,
                # but if a user has forgotten the `if __name__ == "__main__":` guard,
                # we might end up executing code that we shouldn't if we try that. So
                # only builtins are imported this way.
                if origin in {"frozen", "built-in"}:
                    module = __import__(node.module)
                    exports = getattr(
                        module, "__all__", [x for x in dir(module) if not x.startswith("_")]
                    )
                    if name in exports:
                        return _TraceResult(core.get_code(node, source), node.lineno, node)

                    continue

                if origin is None:
                    continue

                origin = Path(origin)
                if origin.suffix != ".py":  # We may get .so files for some modules
                    continue

                # For non-builtin modules (unfortunately including much of the stdlib),
                # we try to parse the ast of the module to figure out what __all__ is
                # likely to contain. This is pretty accurate, but not perfect.
                with origin.open("r", encoding="utf-8") as stream:
                    module_source = stream.read()

                if trace_origin(name, module_source, __all__=True):
                    return _TraceResult(core.get_code(node, source), node.lineno, node)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return _TraceResult(core.get_code(node, source), node.lineno, node)

        if isinstance(node, (ast.Assign, ast.AnnAssign)) and any(
            target.id == name for target in parsing.assignment_targets(node)
        ):
            return _TraceResult(core.get_code(node, source), node.lineno, node)

        if isinstance(node, ast.NamedExpr) and core.match_template(node.target, ast.Name(id=name)):
            return _TraceResult(core.get_code(node, source), node.lineno, node)

    return None


def get_defined_names(root: ast.Module) -> Set[str]:
    """Get names defined in scope, excluding imports."""
    return (
        {node.id for node in core.walk(root, ast.Name(ctx=ast.Store))}
        | {node.name for node in core.walk(root, (ast.FunctionDef, ast.AsyncFunctionDef))}
        | {node.name for node in core.walk(root, ast.ClassDef)}
        | {node.arg for node in core.walk(root, ast.arg)}
    )


def _get_referenced_names(root: ast.Module) -> Set[str]:
    return {node.id for node in core.walk(root, ast.Name(ctx=ast.Load))}


def get_undefined_variables(source: str) -> Set[str]:
    root = core.parse(source)
    imported_names = get_imported_names(root)
    defined_names = get_defined_names(root)
    referenced_names = _get_referenced_names(root)

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

    root = core.parse(source)

    template = ast.ImportFrom(names=[ast.alias(name="*")])

    starred_import_name_mapping = collections.defaultdict(set)

    template = tuple(core.filter_nodes(root.body, template))

    if not template:
        return source

    undefined_names = get_undefined_variables(source)
    for name in undefined_names:
        if trace_result := trace_origin(name, source):
            if core.match_template(trace_result.ast, template):
                starred_import_name_mapping[trace_result.ast].add(name)

    for node, names in starred_import_name_mapping.items():
        if names:
            yield node, ast.ImportFrom(
                module=node.module,
                names=[ast.alias(name=name, asname=None) for name in sorted(names)],
                level=0,
            )

    # Remove remaining starred imports
    for node in core.filter_nodes(root.body, template):
        if not core.match_template(node, tuple(starred_import_name_mapping)):
            yield node, None


@processing.fix
def fix_reimported_names(source: str) -> str:
    """Remove reimported names from imports."""

    root = core.parse(source)

    all_template = ast.Assign(
        targets=[ast.Name(id="__all__")], value=ast.List(elts={ast.Constant(value=str)})
    )
    module_from_imports = collections.defaultdict(set)

    import_insert_lineno = min(
        (node.lineno for node in core.walk(root, (ast.ImportFrom, ast.Import))), default=-1
    )
    if import_insert_lineno == -1:
        return source  # No imports, nothing to do

    transaction = 0

    for node in core.walk(root, ast.ImportFrom):
        if node.module in constants.PYTHON_311_STDLIB:
            continue

        if node.module is None:
            continue

        origin = _trace_module_source_file(node.module)
        if origin in {"frozen", "built-in", None}:
            continue

        origin = Path(origin)
        if origin.suffix != ".py":
            continue

        if origin.name == "__init__.py":
            continue

        with origin.open("r", encoding="utf-8") as stream:
            module_source = stream.read()

        module_root = core.parse(module_source)
        if any(core.filter_nodes(module_root.body, all_template)):
            continue

        node_names = []
        for alias in node.names:
            asname = alias.asname
            name = alias.name

            referenced_name = asname if asname else name

            if trace_result := trace_origin(name, module_source, __all__=True):
                *_, module_import_node = trace_result
                if isinstance(module_import_node, ast.ImportFrom):
                    # Remove this alias from node.names
                    # Add this alias to things that should be imported from module_import_node.module
                    if len(module_import_node.names) == 1 and module_import_node.names[0].name == "*":
                        original_name = name
                    else:
                        original_name = next(
                            alias.name
                            for alias in module_import_node.names
                            if alias.asname == name or (alias.asname is None and alias.name == name)
                        )
                    if referenced_name == original_name:
                        new_alias = ast.alias(name=original_name, asname=None)
                    else:
                        new_alias = ast.alias(name=original_name, asname=referenced_name)

                    module_from_imports[module_import_node.module].add(new_alias)

                elif isinstance(module_import_node, ast.Import):
                    # Remove this alias from node.names
                    # Add module_import_node, but with the alias changed from name -> asname if exists, and asname != name
                    original_name = next(
                        alias.name
                        for alias in module_import_node.names
                        if alias.asname == name or (alias.asname is None and alias.name == name)
                    )
                    if referenced_name == original_name:
                        new_alias = ast.alias(name=original_name, asname=None)
                    else:
                        new_alias = ast.alias(name=original_name, asname=referenced_name)

                    yield None, ast.Import(names=[new_alias], lineno=import_insert_lineno), transaction
                else:
                    node_names.append(alias)
            else:
                node_names.append(alias)

        if node_names != node.names:
            if node_names:
                yield node, ast.ImportFrom(module=node.module, names=node_names, level=node.level), transaction
            else:
                yield node, None, transaction

    for module, aliases in module_from_imports.items():
        yield None, ast.ImportFrom(
            module=module,
            names=sorted(aliases, key=lambda alias: alias.name),
            level=0,
            lineno=import_insert_lineno,
        ), transaction
