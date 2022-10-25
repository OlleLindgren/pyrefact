import collections
import io
import itertools
import re
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Collection, Iterable, Literal, Sequence, Tuple

import black
import isort
import rmspace
from pylint.lint import Run

from pyrefact import parsing
from pyrefact.constants import ASSUMED_PACKAGES, ASSUMED_SOURCES, PACKAGE_ALIASES, PYTHON_KEYWORDS


def _deconstruct_pylint_warning(error_line: str) -> Tuple[Path, int, int, str, str]:
    filename, lineno, charno, error_code, error_msg = error_line.split(":")

    return filename, lineno, charno, error_code.strip(), error_msg.strip()


def _find_pylint_errors(content: str, error_code: str) -> Iterable[str]:
    original_sys_stdout = sys.stdout
    with tempfile.NamedTemporaryFile(suffix=".py") as temp:
        temp.write(content.encode("utf-8"))
        temp.seek(0)
        stdout = io.StringIO()
        args = [
            "--disable",
            "all",
            "--enable",
            f"{error_code}",
            temp.name,
        ]
        try:
            sys.stdout = stdout
            Run(args, exit=False)
        finally:
            sys.stdout = original_sys_stdout

    re_pattern = re.compile(r".*\.py:\d+:\d+: \w\d+: .* \(" + error_code + r"\)")
    output = stdout.getvalue()
    for line in output.splitlines():
        if re_pattern.match(line):
            yield line


def _get_undefined_variables(content: str) -> Collection[str]:
    variables = set()
    for line in _find_pylint_errors(content, "undefined-variable"):
        try:
            _, *_, error_msg = _deconstruct_pylint_warning(line)
            _, variable_name, _ = error_msg.split("'")
            variables.add(variable_name)
        except ValueError:
            pass

    return variables


def _get_unused_imports(content: str) -> Iterable[Tuple[int, str, str]]:
    for line in _find_pylint_errors(content, "unused-import"):
        try:
            _, lineno, *_, message = _deconstruct_pylint_warning(line)
            if " from " not in message:
                *_, package = message.strip().split(" ")
                yield lineno, package, None
            else:
                _, variable, *_, package, _ = message.strip().split(" ")
                yield lineno, package, variable
        except ValueError:
            pass


def _is_private(variable: str) -> bool:
    return variable.startswith("_")


def _rename_variable(variable: str, *, static: bool, private: bool) -> str:

    renamed_variable = variable.upper() if static else variable.lower()
    renamed_variable = re.sub("_{1,}", "_", renamed_variable)

    if renamed_variable.endswith("_"):
        renamed_variable = renamed_variable[:-1]

    if private and not _is_private(renamed_variable):
        renamed_variable = f"_{renamed_variable}"
    if not private and _is_private(renamed_variable):
        renamed_variable = renamed_variable.lstrip("_")

    if renamed_variable:
        return renamed_variable

    raise RuntimeError(f"Unable to find a replacement name for {variable}")


def _rename_class(name: str, *, private: bool) -> str:
    name = re.sub("_{1,}", "_", name)
    if len(name) == 0:
        raise ValueError("Cannot rename empty name")
    if len(name) == 1:
        name = name.upper()
    else:
        accum = (last := name[0].upper())
        for char in name[1:]:
            if last == "_":
                accum += (last := char.upper())
            else:
                accum += char
            last = char

        name = accum

    if private and not _is_private(name):
        return f"_{name}"
    if not private and _is_private(name):
        return name[1:]

    return name


def _substitute_name(
    variable: str,
    variable_type: parsing.VariableType,
    scope: Literal["class", "def", "enum", None],
) -> str:
    if variable == "_":
        return variable

    if variable_type == parsing.VariableType.CLASS:
        private = _is_private(variable)
        return _rename_class(variable, private=private)

    if variable_type == parsing.VariableType.CALLABLE:
        return _rename_variable(variable, static=False, private=_is_private(variable))

    if variable_type == parsing.VariableType.VARIABLE and scope not in {None, "enum"}:
        private = scope == "class" and _is_private(variable)
        return _rename_variable(variable, static=False, private=private)

    if variable_type == parsing.VariableType.VARIABLE:
        return _rename_variable(variable, static=True, private=_is_private(variable))

    raise NotImplementedError(f"Unknown variable type: {variable_type}")


def _get_variable_name_substitutions(content: str) -> Iterable[str]:
    for variable, scopes, variable_type in parsing.iter_definitions(content):
        substitute = _substitute_name(variable, variable_type, scopes[-1][0] if scopes else None)
        if variable != substitute and substitute not in parsing.PYTHON_KEYWORDS:
            print(f"{variable} should be named {substitute}")
            yield variable, substitute


def _get_variable_re_pattern(variable) -> str:
    return r"(?<![A-Za-z_\.])" + variable + r"(?![A-Za-z_])"


def _fix_variable_names(content: str, renamings: Iterable[Tuple[str, str]]) -> str:
    code_mask = parsing.get_is_code_mask(content)
    paranthesis_map = parsing.get_paren_depths(content, code_mask)

    for variable, substiture in renamings:
        replacements = []
        for match in re.finditer(_get_variable_re_pattern(variable), content):
            start = match.start()
            end = match.end()

            # Ignore string contents or comments
            if not all(code_mask[start:end]):
                continue

            is_in_paranthesis = max(paranthesis_map[start:end]) > 0

            if not is_in_paranthesis:
                replacements.append((start, end))
                continue

            # If inside paranthesis, a = means a keyword argument is being assigned,
            # which should be ignored.
            # The only valid assignment syntax is with the walrus operator :=
            substring = content[end : min(len(content) - 1, end + 3)]
            is_assigned_by_equals = re.match(parsing.ASSIGN_RE_PATTERN, substring) is not None

            if not is_assigned_by_equals:
                replacements.append((start, end))

        if not replacements:
            warnings.warn(f"Unable to find '{variable}' in content")

        for start, end in sorted(replacements, reverse=True):
            content = content[:start] + substiture + content[end:]

    return content


def _fix_undefined_variables(content: str, variables: Collection[str]) -> str:
    variables = set(variables)

    lines = content.splitlines()
    change_count = -len(lines)
    lineno = next(
        i
        for i, line in enumerate(lines)
        if not line.startswith("#")
        and not line.startswith("'''")
        and not line.startswith('"""')
        and not line.startswith("from __future__ import")
    )
    for package, package_variables in ASSUMED_SOURCES.items():
        overlap = variables.intersection(package_variables)
        if overlap:
            fix = f"from {package} import " + ", ".join(sorted(overlap))
            print(f"Inserting '{fix}' at line {lineno}")
            lines.insert(lineno, fix)

    for package in ASSUMED_PACKAGES & variables:
        fix = f"import {package}"
        print(f"Inserting '{fix}' at line {lineno}")
        lines.insert(lineno, fix)

    for alias in PACKAGE_ALIASES.keys() & variables:
        package = PACKAGE_ALIASES[alias]
        fix = f"import {package} as {alias}"
        print(f"Inserting '{fix}' at line {lineno}")
        lines.insert(lineno, fix)

    change_count += len(lines)

    assert change_count >= 0

    if change_count == 0:
        return content

    return "\n".join(lines) + "\n"


def _fix_unused_imports(content: str, problems: Collection[Tuple[int, str, str]]) -> bool:

    lineno_problems = collections.defaultdict(set)
    for lineno, package, variable in problems:
        lineno_problems[int(lineno)].add((package, variable))

    change_count = 0

    new_lines = []
    for i, line in enumerate(content.splitlines(keepends=True)):
        if i + 1 in lineno_problems:
            if re.match(r"from .*? import .*", line):
                packages = {package for package, variable in lineno_problems[i + 1]}
                if len(packages) != 1:
                    raise RuntimeError("Unable to parse unique package")
                bad_variables = {variable for package, variable in lineno_problems[i + 1]}
                _, existing_variables = line.split(" import ")
                existing_variables = set(x.strip() for x in existing_variables.split(","))
                keep_variables = existing_variables - bad_variables
                if keep_variables:
                    fix = f"from {package} import " + ", ".join(sorted(keep_variables)) + "\n"
                    new_lines.append(fix)
                    print(f"Replacing {line.strip()} \nwith      {fix.strip()}")
                    change_count += 1
                    continue

            print(f"Removing '{line.strip()}'")
            change_count += 1
            continue

        new_lines.append(line)

    assert change_count >= 0

    if change_count == 0:
        return content

    return "".join(new_lines)


def define_undefined_variables(content: str) -> str:
    """Attempt to find imports matching all undefined variables.

    Args:
        content (str): Python source code

    Returns:
        str: Source code with added imports
    """
    undefined_variables = _get_undefined_variables(content)
    if undefined_variables:
        return _fix_undefined_variables(content, undefined_variables)

    return content


def remove_unused_imports(content: str) -> str:
    """Remove unused imports from source code.

    Args:
        content (str): Python source code

    Returns:
        str: Source code, with added imports removed
    """
    unused_import_linenos = set(_get_unused_imports(content))
    if unused_import_linenos:
        return _fix_unused_imports(content, unused_import_linenos)

    return content


def fix_rmspace(content: str) -> str:
    """Remove trailing whitespace from source code.

    Args:
        content (str): Python source code

    Returns:
        str: Source code, without trailing whitespace
    """
    return rmspace.format_str(content)


def fix_black(content: str) -> str:
    """Format source code with black.

    Args:
        content (str): Python source code

    Returns:
        str: Source code, formatted with black
    """
    return black.format_str(content, mode=black.Mode(line_length=100))


def fix_isort(content: str, *, line_length: int = 100) -> str:
    """Format source code with isort

    Args:
        content (str): Python source code
        line_length (int, optional): Line length. Defaults to 100.

    Returns:
        str: Source code, formatted with isort
    """
    return isort.code(content, config=isort.Config(profile="black", line_length=line_length))


def align_variable_names_with_convention(content: str) -> str:
    """Align variable names with normal convention

    Class names should have CamelCase names
    Non-static variables and functions should have snake_case names
    Static variables should have UPPERCASE_UNDERSCORED names

    All names defined in global scope may be private and start with a single underscore
    Names outside global scope are never allowed to be private
    __magic__ names may only be defined in global scope

    Args:
        content (str): Python source code

    Returns:
        str: Source code, where all variable names comply with normal convention
    """
    renamings = set(_get_variable_name_substitutions(content))
    if renamings:
        content = _fix_variable_names(content, renamings)

    return content


def undefine_unused_variables(content: str) -> str:
    """Remove definitions of unused variables.

    Args:
        content (str): Python source code

    Returns:
        str: Source code, with no variables pointlessly being set.
    """
    for statement in sorted(
        parsing.iter_statements(content),
        key=lambda stmt: stmt.end - stmt.start,
        reverse=True,
    ):
        if statement.statement_type not in {"def", "async def", "global"}:
            continue

        altered_statement = initial_statement = statement.statement
        defined_variables = set()
        for variable, _, variable_type in parsing.iter_definitions(altered_statement):
            if variable_type is parsing.VariableType.VARIABLE:
                defined_variables.add(variable)

        used_variables = {stmt.statement for stmt in parsing.iter_usages(statement.statement)}

        pointless_variables = defined_variables - used_variables
        if not pointless_variables:
            continue

        # Do not remove non-private static variables without a local use
        pointless_variables = {
            variable
            for variable in pointless_variables
            if not (
                statement.statement_type == "global"
                and not _is_private(variable)
                and variable == variable.upper()
            )
        }

        keep_mask = [True] * len(altered_statement)
        underscore_mask = [False] * len(altered_statement)
        for variable in pointless_variables:
            code_mask = parsing.get_is_code_mask(altered_statement)
            paranthesis_depths = parsing.get_paren_depths(altered_statement, code_mask)
            hits = [
                hit
                for hit in re.finditer(
                    variable + r"[^=](=|(| |,|\/|\+|-|\*)+=|[^a-zA-Z0-9_][ a-zA-Z0-9_,\*]+=)[^=] *",
                    altered_statement,
                )
            ]
            variable_definitions = []
            for hit in hits:
                for true_hit in re.finditer(parsing.VARIABLE_RE_PATTERN, hit.group()):
                    if true_hit.group() != variable:
                        continue

                    start = hit.start() + true_hit.start()
                    end = hit.start() + true_hit.end()

                    if set(paranthesis_depths[start:end]) == {0} and all(code_mask[start:end]):
                        variable_definitions.append((start, end))

            variable_definitions = [
                (start, end)
                for (start, end) in variable_definitions
                if set(paranthesis_depths[start:end]) == {0} and all(code_mask[start:end])
            ]
            if not variable_definitions:
                raise RuntimeError(f"Cannot find any definitions of {variable} in code")

            for start, end in variable_definitions:
                chars_before = re.split("\n", altered_statement[:start])[-1]
                chars_after = re.split(r"[:\+-\/\*]?=(?![=])", altered_statement[end:])[0]
                if "," in chars_before or "," in chars_after:
                    if variable != "_":
                        print(f"{variable} should be replaced with _")
                        underscore_mask[start:end] = [True] * (end - start)
                else:
                    print(f"{variable} should be deleted")
                    end += next(
                        re.finditer(r"[:\+-\/\*]?=(?![=])" + " *", altered_statement[end:])
                    ).end()
                    keep_mask[start:end] = [False] * (end - start)

        altered_statement_chars = []
        for char, keep, underscore in zip(altered_statement, keep_mask, underscore_mask):
            if underscore:
                if altered_statement_chars and altered_statement_chars[-1] != "_":
                    altered_statement_chars.append("_")
            elif keep:
                altered_statement_chars.append(char)

        altered_statement = "".join(altered_statement_chars)

        keep_mask = [True] * len(altered_statement)
        for hit in re.finditer(r"(?<=[\n]) *[ ,_\*]+= *", altered_statement):
            true_hit = next(re.finditer(r"[_\*][ ,_\*]*= *", hit.group()))
            start = hit.start() + true_hit.start()
            end = hit.start() + true_hit.end()
            keep_mask[start:end] = [False] * (end - start)

        altered_statement = "".join(
            char for char, keep in zip(altered_statement, keep_mask) if keep
        )

        if altered_statement != initial_statement:
            print(f"Made replacements under scope: {initial_statement.splitlines()[0]}")
            content = content.replace(initial_statement, altered_statement)

    return content


def _is_docstring(content: str, paren_depths: Sequence[int], value: str, start: int) -> bool:
    """Determine if a string is a docstring.

    Args:
        content (str): Python source code
        paren_depths (Sequence[int]): Paranthesis depths of code
        value (str): Matched string statement
        start (int): Character number in code where the matched string starts

    Returns:
        bool: True if the matched string is a docstring
    """
    code_before_value = content[:start]
    # Function docstrings
    if re.findall(
        r"(?<![a-zA-Z0-9_])(def|class|async def) .*\n?.*\n?.*$",
        "".join(char for char, indent in zip(code_before_value, paren_depths) if indent == 0),
    ):
        return True

    # Module docstrings
    if all(line.startswith("#!") for line in code_before_value.splitlines()) and (
        value.startswith("'''") or value.startswith('"""')
    ):
        return True

    return False


def delete_pointless_statements(content: str) -> str:
    """Define pointless statements.

    Args:
        content (str): Python source code

    Returns:
        str: Source code, with no pointless statements.
    """
    code_mask = parsing.get_is_code_mask(content)
    paren_depths = parsing.get_paren_depths(content, code_mask)

    list(parsing.iter_statements(content))

    keep_mask = [True] * len(content)
    for hit in itertools.chain(
        re.finditer(r'(?<![="])[frb]?"{1} *[^"]*?"{1}(?!["])', content),
        re.finditer(r"(?<![='])[frb]?'{1} *[^']*?'{1}(?!['])", content),
        re.finditer(r'(?<![="])[frb]?"{3} *[^"]*?"{3}(?!["])', content),
        re.finditer(r"(?<![='])[frb]?'{3} *[^']*?'{3}(?!['])", content),
    ):
        start = hit.start()
        end = hit.end()
        if any(code_mask[start:end]):
            continue
        if max(paren_depths[start:end]) > 0:
            continue

        # Assignment
        if re.findall(r"=[ \(\n]*$", content[:start]):
            continue

        value = hit.group()
        if _is_docstring(content, paren_depths, value, start):
            continue

        line = parsing._get_line(content, start)
        if PYTHON_KEYWORDS & set(re.findall(parsing.VARIABLE_RE_PATTERN, line.replace(value, ""))):
            continue

        keep_mask[start:end] = [False] * (end - start)
        print("Removing:")
        print(value)

    content = "".join(char for char, keep in zip(content, keep_mask) if keep)

    return content


def delete_unused_functions_and_classes(content: str) -> str:
    """Delete unused functions and classes.

    Args:
        content (str): Python source code

    Returns:
        str: Code, with unused functions deleted
    """
    defined_functions = [
        statement
        for statement in parsing.iter_statements(content)
        if statement.statement_type in {"def", "async def"}
    ]
    referenced_names = {statement.statement for statement in parsing.iter_usages(content)}

    for statement in sorted(
        defined_functions, key=lambda stmt: (stmt.start, stmt.end), reverse=True
    ):
        name = statement.statement.splitlines()[0].lstrip()
        name = re.sub(" +", " ", name)
        if statement.statement_type == "def":
            _, name = name.split(" ", 1)
        elif statement.statement_type == "async def":
            _, _, name = name.split(" ", 2)
        else:
            raise RuntimeError(f"Cannot parse: {statement.statement_type}")

        name, *_ = re.split(parsing.STATEMENT_DELIMITER_RE_PATTERN, name, 1)

        if name in referenced_names:
            continue

        if not _is_private(name):
            continue

        print(f"Deleting {name}")
        content = content.replace(statement.statement, "")

    return content
