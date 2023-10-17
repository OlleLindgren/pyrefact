"""Regex-like wrappers for pyrefact's AST-based pattern matching."""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Iterable, Sequence

from pyrefact import core, processing


def finditer(pattern: str | ast.AST, source: str) -> Iterable[core.Match]:
    for rng, _, groups in processing.find_replace(source, pattern, "", yield_match=True):
        yield core.Match(rng, source, groups)


def findall(pattern: str | ast.AST, source: str) -> Sequence[core.Match]:
    return [m.string for m in finditer(pattern, source)]


def sub(pattern: str | ast.AST, repl: str, source: str, count: int = 0) -> str:
    count = count if count > 0 else float("inf")
    @processing.fix
    def fix_func(src: str):
        replacements = 0
        for item in processing.find_replace(src, pattern, repl):
            if replacements >= count:
                break

            yield item
            replacements += 1

    return fix_func(source)


def search(pattern: str | ast.AST, source: str) -> core.Match | None:
    return next(finditer(pattern, source), None)


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pyrefact.pattern_matching.findall")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    # Findall, e.g. python -m pyrefact.pattern_matching findall <pattern> <path>
    findall_parser = subparsers.add_parser("findall")
    findall_parser.add_argument("pattern", type=str)
    findall_parser.add_argument("path", type=Path, nargs="+")

    # Sub, e.g. python -m pyrefact.pattern_matching sub <pattern> <replacement> <path>
    sub_parser = subparsers.add_parser("sub")
    sub_parser.add_argument("pattern", type=str)
    sub_parser.add_argument("replacement", type=str)
    sub_parser.add_argument("path", type=Path, nargs="+")

    return parser.parse_args(args)


def _recursively_find_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from path.rglob("*.py")


def main(args: Sequence[str]) -> int:
    args = _parse_args(args)
    filenames = sorted({
        filename
        for path in args.path
        for filename in _recursively_find_files(path)
    })
    for filename in filenames:
        source = filename.read_text()

        if args.command == "findall":
            for match in finditer(args.pattern, source):
                print(f"{filename}:{match.lineno}:{match.col_offset}: {match.string.splitlines()[0]}")

        elif args.command == "sub":
            print(f"Parsing {filename}...")
            new_source = sub(args.pattern, args.replacement, source)
            if new_source != source:
                filename.write_text(new_source)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
