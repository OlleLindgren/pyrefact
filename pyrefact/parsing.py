import ast
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

with open(Path(__file__).parent / "python_keywords.json", "r", encoding="utf-8") as stream:
    _PYTHON_KEYWORDS = frozenset(json.load(stream))


def is_valid_python(content: str) -> bool:
    try:
        ast.parse(content, "")
        return True
    except SyntaxError:
        return False


def get_is_code_mask(content: str) -> Sequence[bool]:
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


def get_static_variables(content: str) -> Iterable[str]:
    is_code_mask = get_is_code_mask(content)
    assert len(is_code_mask) == len(content), (len(is_code_mask), len(content))

    scopes = []

    indent = 0
    parsed_chars = 0
    paren_depth = 0
    for full_line in content.splitlines(keepends=True):
        line = "".join(char for i, char in enumerate(full_line) if is_code_mask[i + parsed_chars])
        parsed_chars += len(full_line)
        line_paren_depth = get_paren_depths(full_line)
        line = "".join(
            char for i, char in enumerate(line) if line_paren_depth[i] + paren_depth == 0
        )
        paren_depth += line_paren_depth[-1]
        if not line.strip():
            continue

        indent = len(re.findall(r"^ *", full_line)[0])

        variable, *_ = line.split("=", 1)
        variable = variable.lstrip()
        variable, *_ = variable.split(" ", 1)
        variable = variable.strip()

        if variable:
            scopes = [
                (name, start_indent) for (name, start_indent) in scopes if start_indent < indent
            ]

        if variable in _PYTHON_KEYWORDS:
            if variable in {"def", "class"} or (variable == "async" and "async def" in line):
                scopes.append((variable, indent))

            continue

        if "=" not in line:
            continue

        if not variable:
            continue

        if scopes:
            continue

        if re.match(r"^[a-zA-Z_]+$", variable):
            yield variable
