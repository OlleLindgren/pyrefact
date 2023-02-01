"""Code related to formatting"""
import re

import black


def get_indent(source: str) -> int:
    indentation_whitespace = [x.group() for x in re.finditer(r"(?<![^\n]) *(?=[^\n])", source)]
    if indentation_whitespace:
        return min(len(x) for x in indentation_whitespace)

    return 0


def deindent_code(source: str, indent: int) -> str:
    lines = source.splitlines(keepends=True)
    return "".join(line[indent:] if line.strip() else line for line in lines)


def _indent_code(source: str, indent: int) -> str:
    lines = source.splitlines(keepends=True)
    return "".join(" " * indent + line for line in lines)


def _match_wrapping_whitespace(new: str, initial: str) -> str:
    prefix_whitespace = max(re.findall(r"\A^[\s\n]*", initial), key=len)
    suffix_whitespace = max(re.findall(r"[\s\n]*\Z$", initial), key=len)
    return prefix_whitespace + new.strip() + suffix_whitespace

def format_with_black(source: str, *, line_length: int = 100) -> str:
    """Format code with black.

    Args:
        source (str): Python source code

    Returns:
        str: Formatted source code.
    """
    indent = get_indent(source)
    deindented_code = deindent_code(source, indent)
    formatted_deindented_code = black.format_str(
        deindented_code, mode=black.Mode(line_length=max(60, line_length - indent))
    )
    formatted_content = _indent_code(formatted_deindented_code, indent)
    whitespace_adjusted_content = _match_wrapping_whitespace(formatted_content, source)

    return whitespace_adjusted_content
