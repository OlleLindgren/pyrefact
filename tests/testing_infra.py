"""General convenience functions used in tests."""
import itertools
import re
import sys
from pathlib import Path
from typing import Iterable

from pyrefact import formatting


def _remove_multi_whitespace(source: str) -> str:
    source = re.sub(r"(?<![^\n]) *\n", "", f"\n{source}\n")
    source = "".join(line for line in source.splitlines(keepends=True) if line.strip())

    source = re.sub(r" +(?=\n)", "", source)

    return source


def _create_diff_view(processed_content: str, expected_content: str) -> str:
    proc_lines = processed_content.splitlines() or [""]
    exp_lines = expected_content.splitlines() or [""]
    length = max(max(map(len, proc_lines)), max(map(len, exp_lines)))
    diff_view = [
        f"{p.ljust(length, ' ')} {'=' if p==e else '!'} {e.ljust(length, ' ')}\n"
        for p, e in itertools.zip_longest(proc_lines, exp_lines, fillvalue="")
    ]
    return "".join(diff_view)


def check_fixes_equal(
    processed_content: str, expected_abstraction: str, clear_paranthesises=False, clear_whitespace=True
) -> int:
    if clear_whitespace:
        processed_content = _remove_multi_whitespace(processed_content)
        expected_abstraction = _remove_multi_whitespace(expected_abstraction)

    if tuple(sys.version_info) < (3, 9):
        processed_content = formatting.format_with_black(processed_content)
        processed_content = formatting.collapse_trailing_parentheses(processed_content)
        expected_abstraction = formatting.format_with_black(expected_abstraction)
        expected_abstraction = formatting.collapse_trailing_parentheses(expected_abstraction)

    diff_view = _create_diff_view(processed_content, expected_abstraction)

    if tuple(sys.version_info) < (3, 9):
        processed_content = re.sub(r"[()]", "", processed_content)
        expected_abstraction = re.sub(r"[()]", "", expected_abstraction)

    if clear_paranthesises:
        processed_content = re.sub(r"[\(\)]", "", processed_content)
        expected_abstraction = re.sub(r"[\(\)]", "", expected_abstraction)

    if processed_content != expected_abstraction:
        print(diff_view)
        return False

    return True


def iter_unit_tests() -> Iterable[Path]:
    """Iterate over all unit test files"""
    return sorted((Path(__file__).parent / "unit").rglob("test_*.py"))


def iter_integration_tests() -> Iterable[Path]:
    return sorted((Path(__file__).parent / "integration").rglob("test_*.py"))


def ignore_on_version(major: int, minor: int):
    if (major, minor) == sys.version_info[:2]:
        return lambda before, after: ("", "")

    return lambda before, after: (before, after)
