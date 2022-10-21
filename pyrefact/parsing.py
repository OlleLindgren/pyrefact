import ast
import enum
import json
import re
from pathlib import Path
from typing import Iterable, Sequence, Tuple

with open(Path(__file__).parent / "python_keywords.json", "r", encoding="utf-8") as stream:
    PYTHON_KEYWORDS = frozenset(json.load(stream))


class VariableType(enum.Enum):
    VARIABLE = enum.auto()
    CLASS = enum.auto()
    CALLABLE = enum.auto()


def is_valid_python(content: str) -> bool:
    """Determine if source code is valid python.

    Args:
        content (str): Python source code

    Returns:
        bool: True if content is valid python.
    """
    try:
        ast.parse(content, "")
        return True
    except SyntaxError:
        return False


def get_is_code_mask(content: str) -> Sequence[bool]:
    """Get boolean mask of whether content is code or not.

    Args:
        content (str): Python source code

    Returns:
        Sequence[bool]: True if source code, False if comment or string, for every character.
    """
    singleq = "'''"
    doubleq = '"""'
    s_doubleq = '"'
    s_singleq = "'"
    is_comment = {
        doubleq: False,
        singleq: False,
        s_doubleq: False,
        s_singleq: False,
    }

    mask = []

    buffer = []

    for line in content.splitlines(keepends=True):
        is_hash_comment = False
        for char in line:
            buffer.append(char)
            if len(buffer) > 3:
                del buffer[0]

            if char == "#" and not any(is_comment.values()):
                is_hash_comment = True

            if is_hash_comment:
                mask.append(False)
                continue

            if char == s_singleq:
                is_comment[s_singleq] = not any(is_comment.values())

            if char == s_doubleq:
                is_comment[s_doubleq] = not any(is_comment.values())

            if is_comment[singleq] and buffer == list(singleq):
                is_comment[singleq] = False
                mask.append(False)
                mask[-3:] = False, False, False
                continue

            if is_comment[doubleq] and buffer == list(doubleq):
                is_comment[doubleq] = False
                mask.append(False)
                mask[-3:] = False, False, False
                continue

            if buffer == list(doubleq) and not is_comment[s_singleq] and not is_comment[s_doubleq]:
                is_comment[doubleq] = True
                is_comment[s_doubleq] = False
                mask.append(False)
                continue

            if buffer == list(singleq) and not is_comment[s_singleq] and not is_comment[s_doubleq]:
                is_comment[singleq] = True
                is_comment[s_singleq] = False
                mask.append(False)
                continue

            if any(is_comment.values()) and len(buffer) >= 2 and buffer[-2] in "brf":
                mask[-1] = False

            mask.append(char not in (s_singleq, s_doubleq) and not any(is_comment.values()))

        is_comment[s_singleq] = False
        is_comment[s_doubleq] = False

    return mask


def get_paren_depths(content: str) -> Sequence[int]:
    """Get paranthesis depths of every character in content.

    Args:
        content (str): Python source code

    Returns:
        Sequence[int]: A list of non-negative paranthesis depths, corresponding to every character.
    """
    code_mask = get_is_code_mask(content)
    depth = 0
    depths = []
    for is_code, character in zip(code_mask, content):
        if not is_code:
            depths.append(depth)
            continue
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        depths.append(depth)

    return depths


class FFF:
    prop = 1


def iter_definitions(content: str) -> Iterable[Tuple[str, Sequence[Tuple[str, int]]]]:
    """Iterate over all variables or objects defined in content.

    Args:
        content (str): Python source code

    Yields:
        Tuple[str, Sequence[str]]: name,
    """
    is_code_mask = get_is_code_mask(content)
    assert len(is_code_mask) == len(content), (len(is_code_mask), len(content))

    scopes = []

    yielded_variables = set()

    indent = 0
    parsed_chars = 0
    paren_depth = 0
    for full_line in content.splitlines(keepends=True):
        line = "".join(char for i, char in enumerate(full_line) if is_code_mask[i + parsed_chars])
        parsed_chars += len(full_line)
        line_paren_depth = get_paren_depths(full_line)
        paren_depth += line_paren_depth[-1]
        if not line.strip():
            continue

        indent = len(re.findall(r"^ *", full_line)[0])

        *assignments, _ = line.split("=")

        if line.strip() and paren_depth - line_paren_depth[-1] == 0:
            scopes = [
                (name, start_indent) for (name, start_indent) in scopes if start_indent < indent
            ]

        words = [x.strip() for x in line.split(" ") if x.strip()]
        if words and words[0] in PYTHON_KEYWORDS:
            if words[0] == "class":
                scopes.append(("enum" if "Enum" in words[1] else words[0], indent))
                yield re.sub(r"(\(|:).*", "", words[1]), tuple(scopes[:-1]), VariableType.CLASS
            elif words[0] == "def" or words[0:2] == ["async", "def"]:
                scopes.append(("def", indent))
                yield re.sub(r"(\(|:).*", "", words[1]), tuple(scopes[:-1]), VariableType.CALLABLE

            continue

        for variable in assignments:
            variable = variable.lstrip()
            variable, *_ = variable.split(" ", 1)
            variable = variable.strip()

            if "=" not in line:
                continue

            if not variable:
                continue

            if re.match(r"^[a-zA-Z_]+$", variable):
                yielded_variables.add(variable)
                yield variable, tuple(scopes), VariableType.VARIABLE
