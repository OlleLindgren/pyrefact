import collections
import io
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Collection, Iterable, Literal, Tuple

import black
import isort
import rmspace
from pylint.lint import Run

from . import parsing

with open(Path(__file__).parent / "known_packages.json", "r", encoding="utf-8") as stream:
    _PACKAGE_SOURCES = frozenset(json.load(stream))

with open(Path(__file__).parent / "package_aliases.json", "r", encoding="utf-8") as stream:
    _PACKAGE_ALIASES = json.load(stream)

with open(Path(__file__).parent / "package_variables.json", "r", encoding="utf-8") as stream:
    _ASSUMED_SOURCES = json.load(stream)


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


def _is_uppercase_static_name(variable: str) -> bool:
    return re.match(r"^_[A-Z_]+$", variable) is not None


def _is_magic_variable(variable: str) -> bool:
    return re.match(r"^__[a-z_]+__$", variable) is not None


def _is_regular_variable(variable: str) -> bool:
    if variable.startswith("__"):
        return False
    if variable.endswith("_"):
        return False
    return re.match(r"^[a-z_]+$", variable) is not None


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
    variable: str, variable_type: parsing.VariableType, scope: Literal["class", "def", "enum", None]
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
            yield variable, substitute


def _fix_variable_names(content: str, renamings: Iterable[Tuple[str, str]]) -> str:
    code_mask = parsing.get_is_code_mask(content)
    paranthesis_map = parsing.get_paren_depths(content)

    for variable, substiture in renamings:
        replacements = []
        for match in re.finditer(r"(?<=[^A-Za-z_\.])" + variable + r"(?=[^A-Za-z_])", content):
            replacements.append((match.start(), match.end()))

        if not replacements:
            raise RuntimeError(f"Unable to find '{variable}' in {filename}")

        for start, end in sorted(replacements, reverse=True):
            if not all(code_mask[start:end]):
                continue
            if (
                max(paranthesis_map[start:end]) > 0
                and "=" in content[end : min(len(content) - 1, end + 3)]
            ):
                # kwarg names shouldn't be replaced
                continue

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
    for package, package_variables in _ASSUMED_SOURCES.items():
        overlap = variables.intersection(package_variables)
        if overlap:
            fix = f"from {package} import " + ", ".join(sorted(overlap))
            print(f"Inserting '{fix}' at line {lineno}")
            lines.insert(lineno, fix)

    for package in _PACKAGE_SOURCES & variables:
        fix = f"import {package}"
        print(f"Inserting '{fix}' at line {lineno}")
        lines.insert(lineno, fix)

    for alias in _PACKAGE_ALIASES.keys() & variables:
        package = _PACKAGE_ALIASES[alias]
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

    print(content)

    return content
