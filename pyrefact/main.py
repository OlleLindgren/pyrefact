#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Iterable, Sequence

if __package__:
    from . import fixes
else:
    import fixes


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", help="Paths to refactor", type=Path, nargs="+", default=())
    return parser.parse_args(args)


def run_pyrefact(filename: Path) -> None:
    """Fix a file.

    Args:
        filename (Path): File to fix
    """
    fixes.fix_black(filename)
    fixes.fix_isort(filename, line_length=10_000)

    fixes.fix_rmspace(filename)
    fixes.define_undefined_variables(filename)
    fixes.remove_unused_imports(filename)

    fixes.fix_isort(filename)
    fixes.fix_black(filename)


def _iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from path.rglob("*.py")


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
