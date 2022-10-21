#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from . import completion, fixes, parsing


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", help="Paths to refactor", type=Path, nargs="+", default=())
    return parser.parse_args(args)


def run_pyrefact(filename: Path) -> None:
    """Fix a file.

    Args:
        filename (Path): File to fix
    """
    with open(filename, "r", encoding="utf-8") as stream:
        content = stream.read()

    try:
        if not parsing.is_valid_python(content):
            content = completion.autocomplete(content)

        content = fixes.capitalize_underscore_statics(content)

        content = fixes.fix_black(content)
        content = fixes.fix_isort(content, line_length=10_000)

        content = fixes.fix_rmspace(content)
        content = fixes.define_undefined_variables(content)
        content = fixes.remove_unused_imports(content)

        content = fixes.fix_isort(content)
        content = fixes.fix_black(content)

    finally:
        if parsing.is_valid_python(content):
            with open(filename, "w", encoding="utf-8") as stream:
                stream.write(content)


def _iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from path.rglob("*.py")
        else:
            raise FileNotFoundError(f"Not found: {path}")


def main(args: Sequence[str]) -> int:
    """Parse command-line arguments and run pyrefact on provided files.

    Args:
        args (Sequence[str]): sys.argv[1:]

    Returns:
        int: 0 if successful.

    """
    args = _parse_args(args)

    count = 0
    for filename in _iter_python_files(args.paths):
        count += 1
        run_pyrefact(filename)

    if count == 0:
        print("No files provided")
        return 1

    return 0
