"""Regex-like wrappers for pyrefact's AST-based pattern matching."""
from __future__ import annotations

import argparse
import ast
import sys
import warnings
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from pyrefact import core, processing


__all__ = ["compile", "findall", "finditer", "search", "sub", "subn", "match", "fullmatch"]

warnings.filterwarnings(
    "ignore",
    message=r".*'pyrefact\.pattern_matching' found in sys\.modules after import of package 'pyrefact', but prior to execution of 'pyrefact\.pattern_matching'.*",
)


def finditer(pattern: str | ast.AST, source: str) -> Iterable[core.Match]:
    for rng, _, groups in processing.find_replace(source, pattern, "", yield_match=True):
        yield core.Match(rng, source, groups)


def findall(pattern: str | ast.AST, source: str) -> Sequence[str]:
    return [m.string for m in finditer(pattern, source)]


def subn(pattern: str | ast.AST, repl: str, source: str, count: int = 0) -> Tuple[str, int]:
    count = count if count > 0 else float("inf")
    replacements = 0

    @processing.fix(max_iter=1)
    def fix_func(src: str):
        nonlocal replacements
        for item in processing.find_replace(src, pattern, repl):
            if replacements >= count:
                break

            yield item
            replacements += 1

    new_source = fix_func(source)

    return new_source, replacements


def sub(pattern: str | ast.AST, repl: str, source: str, count: int = 0) -> str:
    new_source, _ = subn(pattern, repl, source, count=count)
    return new_source


def search(pattern: str | ast.AST, source: str) -> core.Match | None:
    return next(finditer(pattern, source), None)


def match(pattern: str | ast.AST, source: str) -> core.Match | None:
    """Match against the start of the module."""
    root = ast.parse(source)
    if not root.body:
        return None

    for rng, _, groups in processing.find_replace(source, pattern, "", yield_match=True, root=root):
        m = core.Match(rng, source, groups)

        module_body_ranges = [core.get_charnos(node, source) for node in root.body]
        module_body_range = core.Range(
            start=min(rng.start for rng in module_body_ranges),
            end=max(rng.end for rng in module_body_ranges),
        )
        if m.span.start == module_body_range.start:
            return m

    return None


def fullmatch(pattern: str | ast.AST, source: str) -> core.Match | None:
    """Match against the entire module."""
    root = ast.parse(source)
    if not root.body:
        return None

    for rng, _, groups in processing.find_replace(source, pattern, "", yield_match=True, root=root):
        m = core.Match(rng, source, groups)

        module_body_ranges = [core.get_charnos(node, source) for node in root.body]
        module_body_range = core.Range(
            start=min(rng.start for rng in module_body_ranges),
            end=max(rng.end for rng in module_body_ranges),
        )
        if m.span == module_body_range:
            return m

    return None


compile = core.compile_template  # pylint: disable=redefined-builtin,unused-variable


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pyrefact.pattern_matching.findall")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    # Findall, e.g. python -m pyrefact.pattern_matching find <pattern> <path>
    find_parser = subparsers.add_parser("find")
    find_parser.add_argument("pattern", type=str)
    find_parser.add_argument("path", type=Path, nargs="+")

    # Sub, e.g. python -m pyrefact.pattern_matching replace <pattern> <replacement> <path>
    replace_parser = subparsers.add_parser("replace")
    replace_parser.add_argument("pattern", type=str)
    replace_parser.add_argument("replacement", type=str)
    replace_parser.add_argument("path", type=Path, nargs="+")

    return parser.parse_args(args)


def _recursively_find_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from path.rglob("*.py")


def main(args: Sequence[str]) -> int:
    args = _parse_args(args)
    filenames = sorted(
        {filename for path in args.path for filename in _recursively_find_files(path)}
    )
    for filename in filenames:
        source = filename.read_text()

        if args.command == "find":
            for match in finditer(args.pattern, source):
                print(
                    f"{filename}:{match.lineno}:{match.col_offset}: {match.string.splitlines()[0]}"
                )

        elif args.command == "replace":
            print(f"Parsing {filename}...")
            new_source = sub(args.pattern, args.replacement, source)
            if new_source != source:
                filename.write_text(new_source)

        else:
            print(f"Unknown command: {args.command}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
