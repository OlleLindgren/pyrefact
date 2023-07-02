"""Code related to formatting"""

import textwrap

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
    original_source = source
    indent = indentation_level(source)
    if indent > 0:
        source = textwrap.dedent(source)

    try:
        source = black.format_str(
            source, mode=black.Mode(line_length=max(60, line_length - indent))
        )
    except (SyntaxError, black.parsing.InvalidInput):
        logger.error("Black raised InvalidInput on code:\n{}", source)
        return original_source

    if indent > 0:
        source = textwrap.indent(source, " " * indent)

    return source


def collapse_trailing_parentheses(source: str) -> str:
    """Collapse trailing ])} together.

    Args:
        source (str): _description_

    Returns:
        str: _description_
    """
    return compactify.format_code(source)


def _inspect_indentsize(line: str) -> int:
    """Return the indent size, in spaces, at the start of a line of text.

    This function is the same as the undocumented inspect.indentsize() function in the stdlib.
    For stability, we copy the code here rather than depending on the undocumented stdlib function.
    """
    expline = line.expandtabs()
    return len(expline) - len(expline.lstrip())


def indentation_level(source: str) -> int:
    """Return the indentation level of source code."""
    return min(
        (_inspect_indentsize(line) for line in source.splitlines() if line.strip()), default=0
    )
