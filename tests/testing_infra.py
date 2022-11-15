"""General convenience functions used in tests."""
import itertools
import re


def _remove_multi_whitespace(content: str) -> str:
    return re.sub(r"(?<![^\n]) *\n", "", f"\n{content}\n").strip()


def _create_diff_view(processed_content: str, expected_content: str) -> str:
    proc_lines = processed_content.splitlines()
    exp_lines = expected_content.splitlines()
    length = max(max(map(len, proc_lines)), max(map(len, exp_lines)))
    diff_view = [
        f"{p.ljust(length, ' ')} {'=' if p==e else '!'} {e.ljust(length, ' ')}\n"
        for p, e in itertools.zip_longest(proc_lines, exp_lines, fillvalue="")
    ]
    return "".join(diff_view)


def check_fixes_equal(processed_content: str, expected_abstraction: str) -> int:
    processed_content = _remove_multi_whitespace(processed_content)
    expected_abstraction = _remove_multi_whitespace(expected_abstraction)

    if processed_content != expected_abstraction:
        print(_create_diff_view(processed_content, expected_abstraction))
        return False

    return True
