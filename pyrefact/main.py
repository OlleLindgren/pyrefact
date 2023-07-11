#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import collections
import io
import logging
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Collection, Iterable, Sequence

import rmspace

from pyrefact import abstractions, fixes
from pyrefact import logs as logger
from pyrefact import (
    core,
    formatting,
    object_oriented,
    parsing,
    performance,
    performance_numpy,
    performance_pandas,
    processing,
    symbolic_math,
    tracing,
)

MAX_MODULE_PASSES = 5
MAX_FILE_PASSES = 25


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", help="Paths to refactor", type=Path, nargs="*", default=())
    parser.add_argument(
        "--preserve", "-p", help="Paths to preserve names in", type=Path, nargs="+", default=()
    )
    parser.add_argument("--safe", "-s", help="Don't delete or rename anything", action="store_true")
    parser.add_argument(
        "--from-stdin", help="Recieve input source code from stdin", action="store_true"
    )
    parser.add_argument(
        "--verbose", "-v", help="Set logging threshold to DEBUG", action="store_true"
    )
    return parser.parse_args(args)


def _single_run_fixes(source: str) -> str:
    """Fixes that should run only once.

    Args:
        source (str): Python source code

    Returns:
        str: Modified code
    """
    chain = processing.chain((
        fixes.deinterpolate_logging_args,
        fixes.invalid_escape_sequence,
        tracing.fix_starred_imports,
        tracing.fix_reimported_names,
    ))
    return chain(source)


def _multi_run_fixes(source: str, preserve: Collection[str]) -> str:
    """Fixes that should run over and over until convergence.

    Args:
        source (str): Python source code

    Returns:
        str: Modified code
    """
    source = fixes.delete_commented_code(source)
    source = fixes.remove_dead_ifs(source)
    source = fixes.delete_unreachable_code(source)
    source = fixes.undefine_unused_variables(source, preserve=preserve)
    source = fixes.delete_pointless_statements(source)
    source = fixes.move_before_loop(source)

    source = fixes.delete_unused_functions_and_classes(source, preserve=preserve)

    source = object_oriented.remove_unused_self_cls(source)
    source = object_oriented.move_staticmethod_static_scope(source, preserve=preserve)
    source = fixes.singleton_eq_comparison(source)
    source = fixes.move_imports_to_toplevel(source)
    source = fixes.breakout_common_code_in_ifs(source)
    source = abstractions.simplify_if_control_flow(source)
    source = fixes.breakout_common_code_in_ifs(source)
    source = fixes.swap_if_else(source)
    source = fixes.early_return(source)
    source = fixes.early_continue(source)
    source = fixes.redundant_enumerate(source)
    source = fixes.unused_zip_args(source)
    source = fixes.replace_map_lambda_with_comp(source)
    source = fixes.replace_filter_lambda_with_comp(source)
    source = fixes.replace_with_filter(source)
    source = fixes.merge_chained_comps(source)
    source = fixes.remove_redundant_comprehension_casts(source)
    source = fixes.remove_redundant_chain_casts(source)
    source = fixes.remove_redundant_else(source)
    source = fixes.fix_if_return(source)
    source = fixes.fix_if_assign(source)
    source = fixes.replace_functions_with_literals(source)
    source = fixes.replace_collection_add_update_with_collection_literal(source)
    source = fixes.simplify_collection_unpacks(source)
    source = fixes.remove_duplicate_set_elts(source)
    source = fixes.breakout_starred_args(source)
    source = performance_pandas.replace_loc_at_iloc_iat(source)
    source = performance_pandas.replace_iterrows_index(source)
    source = performance_pandas.replace_iterrows_itertuples(source)
    source = fixes.replace_for_loops_with_set_list_comp(source)
    source = fixes.replace_setcomp_add_with_union(source)
    source = fixes.replace_listcomp_append_with_plus(source)
    source = fixes.replace_for_loops_with_dict_comp(source)
    source = fixes.implicit_dict_keys_values_items(source)
    source = fixes.replace_dict_assign_with_dict_literal(source)
    source = fixes.replace_dict_update_with_dict_literal(source)
    source = fixes.replace_dictcomp_assign_with_dict_literal(source)
    source = fixes.replace_dictcomp_update_with_dict_literal(source)
    source = fixes.simplify_dict_unpacks(source)
    source = fixes.remove_duplicate_dict_keys(source)
    source = performance_numpy.replace_implicit_matmul(source)
    source = performance.replace_subscript_looping(source)
    source = performance_numpy.replace_implicit_dot(source)
    source = fixes.simplify_transposes(source)
    source = performance_numpy.simplify_matmul_transposes(source)
    source = fixes.simplify_transposes(source)
    source = fixes.implicit_defaultdict(source)
    source = fixes.simplify_redundant_lambda(source)
    source = fixes.remove_redundant_comprehensions(source)
    source = symbolic_math.simplify_boolean_expressions(source)
    source = symbolic_math.simplify_constrained_range(source)
    source = symbolic_math.simplify_boolean_expressions_symmath(source)
    source = fixes.inline_math_comprehensions(source)
    source = symbolic_math.simplify_math_iterators(source)
    source = fixes.replace_negated_numeric_comparison(source)
    source = performance.optimize_contains_types(source)
    source = performance.remove_redundant_chained_calls(source)
    source = performance.remove_redundant_iter(source)
    source = performance.replace_sorted_heapq(source)
    source = fixes.missing_context_manager(source)

    source = fixes.remove_duplicate_functions(source, preserve=preserve)
    source = fixes.fix_duplicate_imports(source)
    source = fixes.fix_too_many_blank_lines(source)

    return source


def format_code(
    source: str,
    *,
    preserve: Collection[str] = frozenset(),
    safe: bool = False,
    keep_imports: bool = False,
) -> str:
    if re.findall(r"# pyrefact: skip_file", source):
        return source

    source = source.expandtabs(4)
    source = rmspace.format_str(source)
    source = fixes.fix_too_many_blank_lines(source)

    if not source.strip():
        return source

    if core.is_valid_python(source):
        minimum_indent = 0
    else:
        minimum_indent = formatting.indentation_level(source)
        source = textwrap.dedent(source)

    if not core.is_valid_python(source):
        logger.debug("Result is not valid python.")
        return source

    if safe:
        # Code may not be deleted from module level
        module = core.parse(source)
        def_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        fdef_types = (ast.FunctionDef, ast.AsyncFunctionDef)
        defs = {node.name for node in core.filter_nodes(module.body, def_types)}
        class_funcs = {  # Function definitions directly under a class definition in module scope
            f"{node.name}.{funcdef.name}"
            for node in core.filter_nodes(module.body, ast.ClassDef)
            for funcdef in core.filter_nodes(node.body, fdef_types)
        }
        assignments = {node.id for node in parsing.iter_assignments(module)}
        preserve = set(preserve) | defs | class_funcs | assignments

    if minimum_indent == 0:
        source = fixes.add_missing_imports(source)

    # Fixes for problems that are not expected to ever be regenerated by other steps
    source = _single_run_fixes(source)

    # Remember past versions of source code.
    # This lets us break if it stops making changes, or if it enters a cycle where it returns
    # to a previous version again.
    content_history = {source}

    for _ in range(1, 1 + MAX_FILE_PASSES):
        source = _multi_run_fixes(source, preserve=preserve)
        if source in content_history:
            break

        content_history.add(source)

    source = abstractions.create_abstractions(source)
    source = abstractions.overused_constant(source, root_is_static=minimum_indent == 0)
    source = fixes.simplify_assign_immediate_return(source)

    # If abstractions have added anything, there may be anti-patterns (redundant elif/else
    # usually) in the code, that should be removed.
    if source not in content_history:
        for _ in range(1, 1 + MAX_FILE_PASSES):
            source = _multi_run_fixes(source, preserve=preserve)
            if source in content_history:
                break

            content_history.add(source)

    if minimum_indent == 0:
        source = fixes.align_variable_names_with_convention(source, preserve=preserve)

    if minimum_indent == 0:
        source = fixes.add_missing_imports(source)
        if not keep_imports:
            source = fixes.remove_unused_imports(source)

    source = fixes.sort_imports(source)

    source = fixes.fix_line_lengths(source)
    source = rmspace.format_str(source)

    if minimum_indent > 0:
        source = textwrap.indent(source, " " * minimum_indent)

    return source


def format_file(
    filename: Path, *, preserve: Collection[str] = frozenset(), safe: bool = False
) -> int:
    """Fix a file.

    Args:
        filename (Path): File to fix

    Returns:
        bool: True if any changes were made
    """
    filename = Path(filename).resolve().absolute()
    with open(filename, "r", encoding="utf-8") as stream:
        initial_content = stream.read()

    keep_imports = filename.name == "__init__.py"
    source = format_code(initial_content, preserve=preserve, safe=safe, keep_imports=keep_imports)

    if source != initial_content and (
        core.is_valid_python(source) or not core.is_valid_python(initial_content)
    ):
        with open(filename, "w", encoding="utf-8") as stream:
            stream.write(source)

        return True

    return 0


def _iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        path = path.resolve().absolute()
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from path.rglob("*.py")
        else:
            raise FileNotFoundError(f"Not found: {path}")


def _namespace_name(filename: Path) -> str:
    filename = Path(filename).absolute()
    return str(filename).replace(os.path.sep, ".")


def main(args: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and run pyrefact on provided files.

    Args:
        args (Sequence[str]): sys.argv[1:]

    Returns:
        int: 0 if successful.

    """
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)

    logger.set_level(logging.DEBUG if args.verbose else logging.INFO)

    used_names = collections.defaultdict(set)
    for filename in _iter_python_files(args.preserve):
        with open(filename, "r", encoding="utf-8") as stream:
            source = stream.read()
        ast_root = core.parse(source)
        imported_names = tracing.get_imported_names(ast_root)
        for node in core.walk(ast_root, (ast.Name, ast.Attribute)):
            if isinstance(node, ast.Name) and node.id in imported_names:
                used_names[_namespace_name(filename)].add(node.id)
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in imported_names
            ):
                used_names[_namespace_name(filename)].add(node.attr)
                used_names[_namespace_name(filename)].add(node.value.id)

    if args.from_stdin:
        logger.set_level(100)  # Higher than critical
        source = sys.stdin.read()
        temp_stdout = io.StringIO()
        sys_stdout = sys.stdout
        preserve = set.union(*used_names.values()) if used_names else set()
        try:
            sys.stdout = temp_stdout
            source = format_code(source, preserve=preserve, safe=args.safe)
        finally:
            sys.stdout = sys_stdout
        print(source)
        return 0

    folder_contents = collections.defaultdict(list)
    for filename in _iter_python_files(args.paths):
        filename = filename.absolute()
        folder_contents[filename.parent].append(filename)

    max_passes = 1 if args.safe else MAX_MODULE_PASSES
    for folder, filenames in folder_contents.items():
        module_passes = 0
        for module_passes in range(1, 1 + max_passes):
            changes = False
            for filename in filenames:
                preserve = set()
                for name, variables in used_names.items():
                    if name != _namespace_name(filename):
                        preserve.update(variables)
                logger.info("Analyzing {filename}...", filename=filename)
                changes |= format_file(filename, preserve=frozenset(preserve), safe=args.safe)

            if not changes:
                break

        logger.debug(
            "\nPyrefact made {module_passes} passes on {folder}.\n",
            module_passes=module_passes,
            folder=folder,
        )

    if sum(len(filenames) for filenames in folder_contents.values()) == 0:
        logger.info("No files provided")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
