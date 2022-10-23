import ast
import dataclasses
import enum
import re
from typing import Iterable, Sequence, Tuple

WALRUS_RE_PATTERN = r"(?<![<>=!:]):=(?![=])"  # match :=, do not match  =, >=, <=, ==, !=
ASSIGN_RE_PATTERN = r"(?<![<>=!:])=(?![=])"  #  match =,  do not match :=, >=, <=, ==, !=
ASSIGN_OR_WALRUS_RE_PATTERN = r"(?<![<>=!:]):?=(?![=])"
VARIABLE_RE_PATTERN = r"(?<![a-zA-Z0-9_])[a-zA-Z_]+[a-zA-Z0-9_]*"
STATEMENT_DELIMITER_RE_PATTERN = r"[\(\)\[\]\{\}\n]|(?<![a-zA-Z_])class|async def|def(?![a-zA-Z_])"

from .constants import PYTHON_KEYWORDS


class VariableType(enum.Enum):
    VARIABLE = enum.auto()
    CLASS = enum.auto()
    CALLABLE = enum.auto()


@dataclasses.dataclass()
class Statement:
    start: int
    end: int
    statement_type: VariableType
    indent: int
    paranthesis_depth: int
    statement: str


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

    statement_breaks = set()

    for key, regex in regexes.items():
        statement_breaks.update(
            (hit.start(), hit.end(), key, content[hit.start() - 1] == "f")
            for hit in re.finditer(regex, content)
        )

    triple_overlapping_singles = set()

    for hit in statement_breaks:
        if hit[2] in ("'", '"'):
            triple_matches = [
                candidate
                for candidate in statement_breaks
                if candidate[2] == hit[2] * 3 and candidate[0] <= hit[0] <= hit[1] <= candidate[1]
            ]
            if triple_matches:
                triple_overlapping_singles.add(hit)

    for hit in triple_overlapping_singles:
        statement_breaks.discard(hit)

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

    while statement_breaks:
        start_item = min(statement_breaks)
        statement_breaks.remove(start_item)
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

        if not statement_breaks:
            break

        end_item = min(item for item in statement_breaks if item[2] == end_key)
        statement_breaks.remove(end_item)
        _, end, _, _ = end_item
        if is_f:
            context.append(key)
        else:
            statement_breaks = {item for item in statement_breaks if item[0] >= end}

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
        if character in ")]}":
            depth -= 1
        depths.append(depth)
        if character in "([{":
            depth += 1

    return depths


def _get_line(content: str, charno: int) -> str:
    for hit in re.finditer(".*\n", content):
        if hit.start() <= charno < hit.end():
            return hit.group()

    raise RuntimeError(
        f"Cannot find a line for charno {charno} in content of length {len(content)}"
    )


def _get_indent(line: str) -> int:
    return len(next(re.finditer(r"^ *", line)).group())


def iter_statements(content: str) -> Iterable[Statement]:
    """Find all statements in content

    Args:
        content (str): Python source code

    Returns:
        Iterable[Tuple[str, str, int, int]]: type, name, start, stop
    """
    is_code_mask = get_is_code_mask(content)

    statement_breaks = [(0, "\n")]
    for hit in re.finditer(STATEMENT_DELIMITER_RE_PATTERN, content):
        start = hit.start()
        end = hit.end()
        if not all(is_code_mask[start:end]):
            continue
        value = hit.group()
        if value in {"class", "def", "async def"}:
            statement_breaks.append((start, value))
        elif value in "([{":
            statement_breaks.append((end, value))
        elif value in ")]}":
            statement_breaks.append((start, value))
        elif value == "\n":
            statement_breaks.append((end, value))
        else:
            raise ValueError(f"Invalid brakpoint: {value}")

    paren_depths = get_paren_depths(content, is_code_mask)

    start_end_matches = {"(": ")", "[": "]", "{": "}"}
    end_start_matches = {end: start for start, end in start_end_matches.items()}

    yield Statement(0, len(content), "global", 0, None, content)

    ongoing_statements: Sequence[Statement] = []
    for statement_break, statement_break_type in statement_breaks:
        try:
            depth = paren_depths[statement_break]
        except IndexError:
            for statement in ongoing_statements:
                statement.end = statement_break
                statement.statement = content[statement.start : statement.end]
                yield statement

            break

        line = _get_line(content, statement_break)
        indent = _get_indent(line)

        completed_statements: Sequence[Statement] = []
        if statement_break_type in {"class", "def", "async def", "\n"} and line.strip():
            for statement in reversed(ongoing_statements):
                if (
                    statement.statement_type in {"class", "def", "async def"}
                    and statement.indent >= indent
                    and any(re.finditer(r"\n.*\n", content[statement.start : statement_break]))
                    and not line.startswith(")")
                ):
                    statement.end = statement_break
                    statement.statement = content[statement.start : statement.end]
                    completed_statements.append(statement)
                if statement.statement_type == "\n":
                    statement.end = statement_break
                    statement.statement = content[statement.start : statement.end]
                    completed_statements.append(statement)

        if statement_break_type in {")", "]", "}"}:
            start_match = end_start_matches[statement_break_type]
            for statement in reversed(ongoing_statements):
                if statement.statement_type == start_match:
                    statement.end = statement_break
                    statement.statement = content[statement.start : statement.end]
                    completed_statements.append(statement)
                    break

        for statement in completed_statements:
            ongoing_statements.remove(statement)
            yield statement

        completed_statements.clear()

        if statement_break_type in {"(", "[", "{", "class", "def", "async def", "\n"}:
            ongoing_statements.append(
                Statement(
                    statement_break,
                    None,
                    statement_break_type,
                    indent,
                    depth,
                    None,
                )
            )


def iter_definitions(content: str) -> Iterable[Tuple[str, Sequence[Tuple[str, int]]]]:
    """Iterate over all variables or objects defined in content.

    Args:
        content (str): Python source code

    Yields:
        Tuple[str, Sequence[str]]: name, scopes
    """
    is_code_mask = get_is_code_mask(content)

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
            line = "".join(
                char
                for i, char in enumerate(full_line)
                if code_mask_subset[i] and hit.start() <= i <= hit.end()
            )

            words = [x.strip() for x in line.split(" ") if x.strip()]

            if len(words) >= 2 and words[0] in PYTHON_KEYWORDS:
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
                ASSIGN_OR_WALRUS_RE_PATTERN if hit_depth == 0 else WALRUS_RE_PATTERN,
                line,
            )

            assignments = [
                var
                for part in assignments
                if re.findall(VARIABLE_RE_PATTERN, part) and part not in PYTHON_KEYWORDS
                for var in re.findall(VARIABLE_RE_PATTERN, part)
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
                    re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", variable)
                    and variable not in PYTHON_KEYWORDS
                    and variable not in yielded_variables
                ):
                    yielded_variables.add(variable)
                    yield variable, tuple(scopes), VariableType.VARIABLE


def iter_usages(content: str) -> Iterable[Statement]:
    """Iterate over all names referenced in

    Args:
        content (str): _description_

    Yields:
        Statement: _description_
    """
    code_mask = get_is_code_mask(content)
    paranthesis_depths = get_paren_depths(content, code_mask)
    for hit in re.finditer(VARIABLE_RE_PATTERN, content):
        start = hit.start()
        end = hit.end()
        if not all(code_mask[start:end]):
            continue

        value = hit.group()
        if value in PYTHON_KEYWORDS:
            continue

        # Assignments are not usages
        if set(paranthesis_depths[start:end]) == {0} and re.match(
            value + r"[ a-zA-Z0-9_,\*]*=[^=]", content[start:]
        ):
            continue

        # b, f, r etc in f-strings are not usages of b/f/r
        if re.match(value + r"'(.|\n)*", content[start:]):
            continue
        if re.match(value + r'"(.|\n)*', content[start:]):
            continue

        # Function and class definitions are not usages
        if re.findall(f"(def|class) +{value}$", content[:end]):
            continue

        yield Statement(start, end, None, None, None, value)
