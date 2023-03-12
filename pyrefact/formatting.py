"""Code related to formatting"""
import re

import black

from pyrefact import logs as logger


def get_indent(source: str) -> int:
    indentation_whitespace = [x.group() for x in re.finditer(r"(?<![^\n]) *(?=[^\n])", source)]
    if indentation_whitespace:
        return min(len(x) for x in indentation_whitespace)

    return 0


def deindent_code(source: str, indent: int) -> str:
    lines = source.splitlines(keepends=True)
    return "".join(line[indent:] if line.strip() else line for line in lines)


def format_with_black(source: str, *, line_length: int = 100) -> str:
    """Format code with black.

    Args:
        source (str): Python source code

    Returns:
        str: Formatted source code.
    """
    try:
        return black.format_str(source, mode=black.Mode(line_length=line_length))
    except (SyntaxError, black.parsing.InvalidInput):
        logger.error("Black raised InvalidInput on code:\n{}", source)
        return source


def collapse_trailing_parentheses(source: str) -> str:
    """Collapse trailing ])} together.

    Args:
        source (str): _description_

    Returns:
        str: _description_
    """
    return re.sub(r"(?<=[\)\}\]])(,?)\s*\n\s*(?=[\)\}\]\n])", r"\1", source)
