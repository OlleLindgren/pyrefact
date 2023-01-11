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
    content: str,
    *,
    preserve: Collection[str] = frozenset(),
    safe: bool = False,
    keep_imports: bool = False,
) -> str:
    if not parsing.is_valid_python(content):
        content = completion.autocomplete(content)

    content = fixes.fix_rmspace(content)

    if not content.strip():
        return content

    if parsing.is_valid_python(content):
        minimum_indent = 0
    else:
        lines = content.splitlines()
        minimum_indent = min(len(line) - len(line.lstrip()) for line in lines if line)
        content = "".join(
            line[minimum_indent:] if line else line for line in content.splitlines(keepends=True)
        )

    if not parsing.is_valid_python(content):
        print("Result is not valid python.")
        return content

    if safe:
        # Code may not be deleted from module level
        module = parsing.parse(content)
        preserve = set.union(
            set(preserve),
            (node.name for node in module.body if isinstance(node, ast.FunctionDef)),
            (node.name for node in module.body if isinstance(node, ast.AsyncFunctionDef)),
            (node.name for node in module.body if isinstance(node, ast.ClassDef)),
            (  # Function definitions directly under a class definition in module scope
                f"{node.name}.{funcdef.name}"
                for node in module.body
                if isinstance(node, ast.ClassDef)
                for funcdef in node.body
                if isinstance(funcdef, ast.FunctionDef)
            ),
            (node.id for node in parsing.iter_assignments(module)),
        )

    # Remember past versions of source code.
    # This lets us break if it stops making changes, or if it enters a cycle where it returns
    # to a previous version again.
    content_history = {content}

    for _ in range(1, 1 + MAX_FILE_PASSES):

        content = fixes.delete_commented_code(content)
        content = fixes.remove_dead_ifs(content)
        content = fixes.delete_unreachable_code(content)
        content = fixes.undefine_unused_variables(content, preserve=preserve)
        content = fixes.delete_pointless_statements(content)

        content = fixes.delete_unused_functions_and_classes(content, preserve=preserve)

        content = object_oriented.remove_unused_self_cls(content)
        content = object_oriented.move_staticmethod_static_scope(content, preserve=preserve)
        content = fixes.singleton_eq_comparison(content)
        content = fixes.move_imports_to_toplevel(content)
        content = fixes.swap_if_else(content)
        content = fixes.early_return(content)
        content = fixes.early_continue(content)
        content = fixes.replace_with_filter(content)
        content = fixes.remove_redundant_else(content)
        content = fixes.replace_functions_with_literals(content)
        content = fixes.replace_for_loops_with_set_list_comp(content)
        content = fixes.replace_for_loops_with_dict_comp(content)
        content = performance.replace_subscript_looping(content)
        content = performance_numpy.replace_implicit_dot(content)
        content = performance_numpy.replace_implicit_matmul(content)
        content = fixes.simplify_transposes(content)
        content = performance_numpy.simplify_matmul_transposes(content)
        content = fixes.simplify_transposes(content)
        content = fixes.implicit_defaultdict(content)
        content = fixes.simplify_redundant_lambda(content)
        content = fixes.remove_redundant_comprehensions(content)
        content = fixes.inline_math_comprehensions(content)
        content = symbolic_math.simplify_math_iterators(content)
        content = performance.optimize_contains_types(content)
        content = performance.remove_redundant_chained_calls(content)
        content = performance.remove_redundant_iter(content)
        content = performance.replace_sorted_heapq(content)
        content = abstractions.create_abstractions(content)

        content = fixes.remove_duplicate_functions(content, preserve=preserve)

        if content in content_history:
            break

        content_history.add(content)

    if minimum_indent == 0:
        content = fixes.align_variable_names_with_convention(content, preserve=preserve)

    if minimum_indent == 0:
        content = fixes.fix_isort(content, line_length=10_000)
        content = fixes.add_missing_imports(content)
        if not keep_imports:
            content = fixes.remove_unused_imports(content)

        content = fixes.fix_isort(content)

    content = fixes.fix_line_lengths(content)
    content = fixes.fix_rmspace(content)

    return "".join(f"{' ' * minimum_indent}{line}" for line in content.splitlines(keepends=True))


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
    content = format_code(initial_content, preserve=preserve, safe=safe, keep_imports=keep_imports)

    if content != initial_content and (
        parsing.is_valid_python(content) or not parsing.is_valid_python(initial_content)
    ):
        with open(filename, "w", encoding="utf-8") as stream:
            stream.write(content)

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

    if args.from_stdin:
        content = sys.stdin.read()
        temp_stdout = io.StringIO()
        sys_stdout = sys.stdout
        preserve = set.union(*used_names.values()) if used_names else set()
        try:
            sys.stdout = temp_stdout
            content = format_code(content, preserve=preserve, safe=args.safe)
        finally:
            sys.stdout = sys_stdout
        print(content)
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
