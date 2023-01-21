#!/usr/bin/env python3
import argparse
import ast
import collections
import io
import os
import sys
from pathlib import Path
from typing import Collection, Iterable, Sequence

from pyrefact import (
    abstractions,
    completion,
    fixes,
    object_oriented,
    parsing,
    performance,
    performance_numpy,
    symbolic_math,
)

MAX_MODULE_PASSES = 5
MAX_FILE_PASSES = 25


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", help="Paths to refactor", type=Path, nargs="*", default=())
    parser.add_argument(
        "--preserve",
        "-p",
        help="Paths to preserve names in",
        type=Path,
        nargs="+",
        default=(),
    )
    parser.add_argument("--safe", "-s", help="Don't delete or rename anything", action="store_true")
    parser.add_argument(
        "--from-stdin", help="Recieve input source code from stdin", action="store_true"
    )
    return parser.parse_args(args)


def format_code(
    source: str,
    *,
    preserve: Collection[str] = frozenset(),
    safe: bool = False,
    keep_imports: bool = False,
) -> str:
    if not parsing.is_valid_python(source):
        source = completion.autocomplete(source)

    source = fixes.fix_tabs(source)
    source = fixes.fix_rmspace(source)
    source = fixes.fix_too_many_blank_lines(source)

    if not source.strip():
        return source

    if parsing.is_valid_python(source):
        minimum_indent = 0
    else:
        lines = source.splitlines()
        minimum_indent = min(len(line) - len(line.lstrip()) for line in lines if line)
        source = "".join(
            line[minimum_indent:] if line else line for line in source.splitlines(keepends=True)
        )

    if not parsing.is_valid_python(source):
        print("Result is not valid python.")
        return source

    if safe:
        # Code may not be deleted from module level
        module = parsing.parse(source)
        def_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        fdef_types = (ast.FunctionDef, ast.AsyncFunctionDef)
        defs = {node.name for node in parsing.filter_nodes(module.body,def_types)}
        class_funcs = {  # Function definitions directly under a class definition in module scope
            f"{node.name}.{funcdef.name}"
            for node in parsing.filter_nodes(module.body, ast.ClassDef)
            for funcdef in parsing.filter_nodes(node.body, fdef_types)
        }
        assignments = {node.id for node in parsing.iter_assignments(module)}
        preserve = set(preserve) | defs | class_funcs | assignments

    # Remember past versions of source code.
    # This lets us break if it stops making changes, or if it enters a cycle where it returns
    # to a previous version again.
    content_history = {source}

    for _ in range(1, 1 + MAX_FILE_PASSES):

        source = fixes.delete_commented_code(source)
        source = fixes.remove_dead_ifs(source)
        source = fixes.delete_unreachable_code(source)
        source = fixes.undefine_unused_variables(source, preserve=preserve)
        source = fixes.delete_pointless_statements(source)

        source = fixes.delete_unused_functions_and_classes(source, preserve=preserve)

        source = object_oriented.remove_unused_self_cls(source)
        source = object_oriented.move_staticmethod_static_scope(source, preserve=preserve)
        source = fixes.singleton_eq_comparison(source)
        source = fixes.move_imports_to_toplevel(source)
        source = fixes.swap_if_else(source)
        source = fixes.early_return(source)
        source = fixes.early_continue(source)
        source = fixes.replace_with_filter(source)
        source = fixes.remove_redundant_else(source)
        source = fixes.replace_functions_with_literals(source)
        source = fixes.replace_for_loops_with_set_list_comp(source)
        source = fixes.replace_for_loops_with_dict_comp(source)
        source = performance.replace_subscript_looping(source)
        source = performance_numpy.replace_implicit_dot(source)
        source = performance_numpy.replace_implicit_matmul(source)
        source = fixes.simplify_transposes(source)
        source = performance_numpy.simplify_matmul_transposes(source)
        source = fixes.simplify_transposes(source)
        source = fixes.implicit_defaultdict(source)
        source = fixes.simplify_redundant_lambda(source)
        source = fixes.remove_redundant_comprehensions(source)
        source = fixes.inline_math_comprehensions(source)
        source = symbolic_math.simplify_math_iterators(source)
        source = performance.optimize_contains_types(source)
        source = performance.remove_redundant_chained_calls(source)
        source = performance.remove_redundant_iter(source)
        source = performance.replace_sorted_heapq(source)
        source = abstractions.create_abstractions(source)

        source = fixes.remove_duplicate_functions(source, preserve=preserve)
        source = fixes.fix_too_many_blank_lines(source)

        if source in content_history:
            break

        content_history.add(source)

    if minimum_indent == 0:
        source = fixes.align_variable_names_with_convention(source, preserve=preserve)

    if minimum_indent == 0:
        source = fixes.fix_isort(source, line_length=10_000)
        source = fixes.add_missing_imports(source)
        if not keep_imports:
            source = fixes.remove_unused_imports(source)

        source = fixes.fix_isort(source)

    source = fixes.fix_line_lengths(source)
    source = fixes.fix_rmspace(source)

    return "".join(f"{' ' * minimum_indent}{line}" for line in source.splitlines(keepends=True))


def format_file(
    filename: Path,
    *,
    preserve: Collection[str] = frozenset(),
    safe: bool = False,
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
        parsing.is_valid_python(source) or not parsing.is_valid_python(initial_content)
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


def main(args: Sequence[str]) -> int:
    """Parse command-line arguments and run pyrefact on provided files.

    Args:
        args (Sequence[str]): sys.argv[1:]

    Returns:
        int: 0 if successful.

    """
    args = _parse_args(args)

    used_names = collections.defaultdict(set)
    for filename in _iter_python_files(args.preserve):
        with open(filename, "r", encoding="utf-8") as stream:
            source = stream.read()
        ast_root = parsing.parse(source)
        imported_names = parsing.get_imported_names(ast_root)
        for node in parsing.walk(ast_root, (ast.Name, ast.Attribute)):
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

    for folder, filenames in folder_contents.items():

        module_passes = 0
        for module_passes in range(1, 1 + MAX_MODULE_PASSES):
            changes = False
            for filename in filenames:
                preserve = set()
                for name, variables in used_names.items():
                    if name != _namespace_name(filename):
                        preserve.update(variables)
                print(f"Analyzing {filename}...")
                changes |= format_file(filename, preserve=frozenset(preserve), safe=args.safe)

            if not changes:
                break

        print(f"\nPyrefact made {module_passes} passes on {folder}.\n")

    if sum(len(filenames) for filenames in folder_contents.values()) == 0:
        print("No files provided")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
