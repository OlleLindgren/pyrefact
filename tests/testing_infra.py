"""General convenience functions used in tests."""
import itertools
import re
import sys


def _remove_multi_whitespace(content: str) -> str:
    content = re.sub(r"(?<![^\n]) *\n", "", f"\n{content}\n")
    content = "".join(line for line in content.splitlines(keepends=True) if line.strip())
    return content


def _create_diff_view(processed_content: str, expected_content: str) -> str:
    proc_lines = processed_content.splitlines() or [""]
    exp_lines = expected_content.splitlines() or [""]
    length = max(max(map(len, proc_lines)), max(map(len, exp_lines)))
    diff_view = [
        f"{p.ljust(length, ' ')} {'=' if p==e else '!'} {e.ljust(length, ' ')}\n"
        for p, e in itertools.zip_longest(proc_lines, exp_lines, fillvalue="")
    ]
    return "".join(diff_view)


def _remove_paranthesis_whitespace(content: str) -> str:
    return content.replace("(", "").replace(")", "").replace(" ", "")


def check_fixes_equal(processed_content: str, expected_abstraction: str) -> int:
    processed_content = _remove_multi_whitespace(processed_content)
    expected_abstraction = _remove_multi_whitespace(expected_abstraction)

    diff_view = _create_diff_view(processed_content, expected_abstraction)

    if tuple(sys.version_info) < (3, 9):
        processed_content = _remove_paranthesis_whitespace(processed_content)
        expected_abstraction = _remove_paranthesis_whitespace(expected_abstraction)

    if processed_content != expected_abstraction:
        print(diff_view)
        return False

    return True
