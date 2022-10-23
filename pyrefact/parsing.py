import ast
import enum
import json
import re
from pathlib import Path
from typing import Iterable, Sequence, Tuple

WALRUS_RE_PATTERN = r"(?<![<>=!:]):=(?![=])"  # match :=, do not match  =, >=, <=, ==, !=
ASSIGN_RE_PATTERN = r"(?<![<>=!:])=(?![=])"  #  match =,  do not match :=, >=, <=, ==, !=
ASSIGN_OR_WALRUS_RE_PATTERN = r"(?<![<>=!:]):?=(?![=])"
_VARIABLE_RE_PATTERN = r"(?<![a-zA-Z0-9_])[a-zA-Z_]+[a-zA-Z0-9_]*"


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
    regexes = {
        '"""': '"""',
        '"': '"',
        "'''": "'''",
        "'": "'",
        "#": "#",
        "{": r"(?<![{]){(?![{])",
        "}": r"(?<![}])}(?![}])",
        "\n": "\n",
    }

    breakpoints = set()

    for key, regex in regexes.items():
        breakpoints.update(
            (hit.start(), hit.end(), key, content[hit.start() - 1] == "f")
            for hit in re.finditer(regex, content)
        )

    triple_overlapping_singles = set()

    for hit in breakpoints:
        if hit[2] in ("'", '"'):
            triple_matches = [
                candidate
                for candidate in breakpoints
                if candidate[2] == hit[2] * 3 and candidate[0] <= hit[0] <= hit[1] <= candidate[1]
            ]
            if triple_matches:
                triple_overlapping_singles.add(hit)

    for hit in triple_overlapping_singles:
        breakpoints.discard(hit)

    string_types = {
        '"""',
        '"',
        "'''",
        "'",
    }
    inline_comment = "#"
    newline_character = "\n"
    f_escape_start = "{"
    f_escape_end = "}"

    comment_super_ranges = []

    context = []

    while breakpoints:
        start_item = min(breakpoints)
        breakpoints.remove(start_item)
        start, s_end, key, is_f = start_item
        if key in string_types:
            end_key = key
        elif key == inline_comment:
            end_key = newline_character
        elif key == f_escape_start:
            end_key = None
            comment_super_ranges.append([start, None, key, False])
            continue
        elif key == f_escape_end:
            for rng in reversed(comment_super_ranges):
                if rng[2] == f_escape_start and rng[1] is None:
                    rng[1] = s_end
                    break
            else:
                raise ValueError(f"Cannot find corresponding start for {key}")
            continue
        elif key == newline_character:
            continue
        else:
            raise ValueError(f"Unknown delimiter: {key}")

        if not breakpoints:
            break

        end_item = min(item for item in breakpoints if item[2] == end_key)
        breakpoints.remove(end_item)
        _, end, _, _ = end_item
        if is_f:
            context.append(key)
        else:
            breakpoints = {item for item in breakpoints if item[0] >= end}

        comment_super_ranges.append([start, end, key, is_f])

    mask = [True] * len(content)
    for start, end, key, is_f in sorted(comment_super_ranges):
        if key in string_types or key == inline_comment:
            value = False
        elif key == f_escape_start:
            value = True
        else:
            raise ValueError(f"Invalid range start delimiter: {key}")

        if value is False:
            if is_f:
                mask[start - 1 : end] = [value] * (end + 1 - start)
            else:
                mask[start:end] = [value] * (end - start)
        else:
            mask[start + 1 : end - 1] = [value] * (end - start - 2)

    return mask


def get_paren_depths(content: str, code_mask_subset: Sequence[bool]) -> Sequence[int]:
    """Get paranthesis depths of every character in content.

    Args:
        content (str): Python source code

    Returns:
        Sequence[int]: A list of non-negative paranthesis depths, corresponding to every character.
    """
    depth = 0
    depths = []
    for is_code, character in zip(code_mask_subset, content):
        if not is_code:
            depths.append(depth)
            continue
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        depths.append(depth)

    return depths


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

    paren_depths = get_paren_depths(content, is_code_mask)

    indent = 0
    parsed_chars = 0
    for full_line in content.splitlines(keepends=True):
        code_mask_subset = is_code_mask[parsed_chars : parsed_chars + len(full_line)]
        line_paren_depth = paren_depths[parsed_chars : parsed_chars + len(full_line)]
        parsed_chars += len(full_line)

        if not any(code_mask_subset):
            continue

        indent = len(re.findall(r"^ *", full_line)[0])
        if full_line.lstrip(" ").startswith(")"):
            indent += 4

        added_scope = False
        for hit in re.finditer(r"[^\(|\)|\[|\]|\{|\}]+", full_line):
            hit_depths = set(line_paren_depth[hit.start() : hit.end()])
            assert len(hit_depths) == 1
            hit_depth = hit_depths.pop()

            # Do not parse comments and strings
            line = "".join(char for i, char in enumerate(hit.group()) if code_mask_subset[i])

            words = [x.strip() for x in line.split(" ") if x.strip()]

            if words and words[0] in PYTHON_KEYWORDS:
                if words[0] == "class":
                    scopes = [
                        (name, start_indent)
                        for (name, start_indent) in scopes
                        if start_indent < indent
                    ]
                    scopes.append(("enum" if "Enum" in full_line else "class", indent))
                    added_scope = True
                    yield re.sub(r"(\(|:).*", "", words[1]), tuple(scopes[:-1]), VariableType.CLASS
                elif words[0] == "def" or words[0:2] == ["async", "def"]:
                    scopes = [
                        (name, start_indent)
                        for (name, start_indent) in scopes
                        if start_indent < indent
                    ]
                    scopes.append(("def", indent))
                    added_scope = True
                    yield re.sub(r"(\(|:).*", "", words[1]), tuple(
                        scopes[:-1]
                    ), VariableType.CALLABLE

                continue

            *assignments, _ = re.split(
                ASSIGN_OR_WALRUS_RE_PATTERN if hit_depth == 0 else WALRUS_RE_PATTERN, line
            )

            assignments = [
                var
                for part in assignments
                if re.findall(_VARIABLE_RE_PATTERN, part) and part not in PYTHON_KEYWORDS
                for var in re.findall(_VARIABLE_RE_PATTERN, part)
                if var not in PYTHON_KEYWORDS
            ]

            if line.strip() and hit_depth == 0 and not added_scope:
                scopes = [
                    (name, start_indent) for (name, start_indent) in scopes if start_indent < indent
                ]

            for variable in assignments:
                variable = variable.strip("* ")
                if " " in variable:
                    continue

                if not variable:
                    continue

                if (
                    re.match(r"^[a-zA-Z_]+$", variable)
                    and variable not in PYTHON_KEYWORDS
                    and variable not in yielded_variables
                ):
                    yielded_variables.add(variable)
                    yield variable, tuple(scopes), VariableType.VARIABLE
