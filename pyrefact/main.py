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
from typing import Collection, Iterable, Mapping, Sequence

import rmspace

from pyrefact import abstractions, fixes
from pyrefact import logs as logger
import multiprocessing as mp
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


__all__ = ["format_code", "format_file", "format_files", "main"]


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
    parser.add_argument(
        "--n_cores", help="Number of cores to use", type=int, default=mp.cpu_count()
    )
    return parser.parse_args(args)


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
    source = fixes.fix_raise_missing_from(source)
    source = fixes.undefine_unused_variables(source, preserve=preserve)
    source = fixes.delete_pointless_statements(source)
    source = fixes.move_before_loop(source)

    source = object_oriented.fix_unconventional_class_definitions(source)

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
    source = fixes.remove_redundant_boolop_values(source)
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
    if keep_imports:
        single_run_fixes = processing.chain((
            fixes.deinterpolate_logging_args,
            fixes.invalid_escape_sequence,
        ))
    else:
        single_run_fixes = processing.chain((
            fixes.deinterpolate_logging_args,
            fixes.invalid_escape_sequence,
            tracing.fix_starred_imports,
            tracing.fix_reimported_names,
        ))

    source = single_run_fixes(source)

    # Remember past versions of source code.
    # This lets us break if it stops making changes, or if it enters a cycle where it returns
    # to a previous version again.
    content_history = {source}

    for _ in range(1, 1 + MAX_FILE_PASSES):
        source = _multi_run_fixes(source, preserve=preserve)
        if source in content_history:
            break

        content_history.add(source)

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
    filename: Path, preserve: Collection[str] = frozenset(), safe: bool = False
) -> int:
    """Fix a file.

    Args:
        filename (Path): File to fix

    Returns:
        bool: True if any changes were made
    """
    logger.debug("Analyzing {filename}...", filename=filename)
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


def format_files(
    filenames,
    *,
    preserved_filenames: Collection[Path] = frozenset(),
    n_cores: int = mp.cpu_count(),
    max_passes: int = 1,
    safe: bool = False,
) -> bool:
    """Fix lots of files concurrently.

    Args:
        filenames (Path): Files to fix

    Keyword Args:
        preserved_filenames (Collection[Path]): Files that depend on the files being fixed
        n_cores (int): Number of CPU cores to use
        max_passes (int): Maximum number of passes to make over the files
        safe (bool): Don't delete or rename anything at the module level

    Returns:
        bool: True if any changes were made
    """
    used_names = _used_names_in_files(preserved_filenames)

    folder_contents = collections.defaultdict(list)
    for filename in map(Path, sorted(filenames)):
        filename = filename.absolute()
        folder_contents[filename.parent].append(filename)

    module_changes_pass_counts = {
        folder: (True, max_passes)
        for folder in folder_contents
    }
    with mp.Pool(n_cores) as pool:
        for pass_number in range(1, max_passes + 1):
            files_to_format = set()
            for folder, files_in_folder in folder_contents.items():
                changes, passes_left = module_changes_pass_counts[folder]
                if changes and passes_left > 0:
                    files_to_format.update(files_in_folder)

            if not files_to_format:
                break

            files_to_format = sorted(files_to_format)

            filename_preserve = {
                filename: frozenset().union(*(
                    variables
                    for name, variables in used_names.items()
                    if name != _namespace_name(filename)
                ))
                for filename in files_to_format
            }

            results = pool.starmap(
                format_file,
                (
                    (filename, filename_preserve[filename], safe)
                    for filename in files_to_format
                )
            )
            filename_changes = dict(zip(files_to_format, results))
            for folder, files_in_folder in folder_contents.items():
                _, passes_left = module_changes_pass_counts[folder]
                changes = any(filename_changes.get(filename, False) for filename in files_in_folder)

                module_changes_pass_counts[folder] = (changes, passes_left - 1)

            logger.debug(
                "\nPass {pass_number} / {max_passes} completed, {converged_modules} / {total_modules} modules converged.\n",
                pass_number=pass_number,
                max_passes=max_passes,
                converged_modules=sum(1 for changes, _ in module_changes_pass_counts.values() if not changes),
                total_modules=len(folder_contents),
            )

    return any(changes for changes, _ in module_changes_pass_counts.values())


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


def _used_names_in_file(filename: Path) -> Collection[str]:
    with open(filename, "r", encoding="utf-8") as stream:
        source = stream.read()

    ast_root = core.parse(source)
    imported_names = tracing.get_imported_names(ast_root)

    names = []
    for node in core.walk(ast_root, (ast.Name, ast.Attribute)):
        if isinstance(node, ast.Name) and node.id in imported_names:
            names.append(node.id)

        elif isinstance(node, ast.Attribute):
            # Attributes and class methods are hard to trace (it basically requires
            # type checking), so we always add them to preserve.
            names.append(node.attr)
            if isinstance(node.value, ast.Name) and node.value.id in imported_names:
                names.append(node.value.id)

    return frozenset(names)


def _used_names_in_files(filenames: Iterable[Path]) -> Mapping[str, Collection[str]]:
    return {
        _namespace_name(filename): _used_names_in_file(filename)
        for filename in sorted(filenames)
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and run pyrefact on provided files.

    Args:
        args (Sequence[str]): sys.argv[1:]

    Returns:
        int: 0 if successful.

    """
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    logger.set_level(logging.DEBUG if args.verbose else logging.INFO)

    if args.from_stdin:
        logger.set_level(100)  # Higher than critical
        source = sys.stdin.read()
        temp_stdout = io.StringIO()
        sys_stdout = sys.stdout
        used_names = _used_names_in_files(_iter_python_files(args.preserve))
        preserve = set.union(*used_names.values()) if used_names else set()
        try:
            sys.stdout = temp_stdout
            source = format_code(source, preserve=preserve, safe=args.safe)
        finally:
            sys.stdout = sys_stdout
        print(source)
        return 0

    filenames = tuple(_iter_python_files(args.paths))
    preserved_filenames = frozenset(_iter_python_files(args.preserve))

    if not filenames:
        logger.info("No files provided")
        return 1

    format_files(
        filenames,
        preserved_filenames=preserved_filenames,
        n_cores=args.n_cores,
        max_passes=1 if args.safe else MAX_MODULE_PASSES,
        safe=args.safe,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
