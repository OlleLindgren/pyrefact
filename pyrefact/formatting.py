"""Code related to formatting"""
import black
import compactify

from pyrefact import logs as logger


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
    return compactify.format_code(source)
