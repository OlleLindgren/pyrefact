#!/usr/bin/env python3
import argparse
import ast
import collections
import os
import sys
from pathlib import Path
from typing import Collection, Iterable, Sequence

from . import abstractions, completion, constants, fixes, object_oriented, parsing, performance


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", help="Paths to refactor", type=Path, nargs="+", default=())
    parser.add_argument(
        "--preserve",
        "-p",
        help="Paths to preserve names in",
        type=Path,
        nargs="+",
        default=(),
    )
    parser.add_argument("--safe", "-s", help="Don't delete or rename anything", action="store_true")
    return parser.parse_args(args)


def _run_pyrefact(
    filename: Path,
    *,
    preserve: Collection[str] = frozenset(),
    safe: bool = False,
) -> int:
    """Fix a file.

    Args:
        filename (Path): File to fix

    Returns:
        int: 0 if successful
    """
    with open(filename, "r", encoding="utf-8") as stream:
        initial_content = content = stream.read()

    if not parsing.is_valid_python(content):
        content = completion.autocomplete(content)

    content = fixes.fix_rmspace(content)

    if not parsing.is_valid_python(content):
        print("Result is not valid python.")
        return 0

    if safe:
        # Code may not be deleted from module level
        module = parsing.parse(content)
        preserve = set.union(
            set(preserve),
            (node.name for node in module.body if isinstance(node, ast.FunctionDef)),
            (node.name for node in module.body if isinstance(node, ast.AsyncFunctionDef)),
            (node.name for node in module.body if isinstance(node, ast.ClassDef)),
            (  # Function definitions directly under a class definition in module scope
                funcdef.name
                for node in module.body
                if isinstance(node, ast.ClassDef)
                for funcdef in node.body
                if isinstance(funcdef, ast.FunctionDef)
            ),
            (node.id for node in parsing.iter_assignments(module)),
        )

    content = fixes.delete_unreachable_code(content)
    content = fixes.undefine_unused_variables(content, preserve=preserve)
    content = fixes.delete_pointless_statements(content)
    content = fixes.delete_unused_functions_and_classes(content, preserve=preserve)

    if constants.PYTHON_VERSION >= (3, 9):
        content = object_oriented.remove_unused_self_cls(content)
        content = object_oriented.move_staticmethod_static_scope(content, preserve=preserve)
        content = fixes.singleton_eq_comparison(content)
        content = fixes.move_imports_to_toplevel(content)
        content = fixes.swap_if_else(content)
        content = fixes.early_return(content)
        content = fixes.early_continue(content)
        content = fixes.remove_redundant_else(content)
        content = performance.replace_with_sets(content)
        content = performance.remove_redundant_chained_calls(content)
        content = performance.replace_sorted_heapq(content)
        content = abstractions.create_abstractions(content)

    content = fixes.remove_duplicate_functions(content, preserve=preserve)

    content = fixes.align_variable_names_with_convention(content, preserve=preserve)

    content = fixes.fix_black(content)
    content = fixes.fix_isort(content, line_length=10_000)
    content = fixes.add_missing_imports(content)
    content = fixes.remove_unused_imports(content)
    content = fixes.fix_isort(content)
    content = fixes.fix_black(content)
    content = fixes.fix_rmspace(content)

    if content != initial_content and (
        parsing.is_valid_python(content) or not parsing.is_valid_python(initial_content)
    ):
        with open(filename, "w", encoding="utf-8") as stream:
            stream.write(content)

    return 0


def _iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
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

    return_code = 0
    count = 0
    used_names = collections.defaultdict(set)
    for filename in _iter_python_files(args.preserve):
        with open(filename, "r", encoding="utf-8") as stream:
            content = stream.read()
        ast_root = parsing.parse(content)
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

    for filename in _iter_python_files(args.paths):
        count += 1
        preserve = set()
        for name, variables in used_names.items():
            if name != _namespace_name(filename):
                preserve.update(variables)
        print(f"Analyzing {filename}...")
        code = _run_pyrefact(filename, preserve=frozenset(preserve), safe=args.safe)
        if code != 0:
            print(f"pyrefact failed for filename {filename}")
            return_code = max(return_code, code)

    if count == 0:
        print("No files provided")
        return 1

    return return_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
