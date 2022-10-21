import collections
import io
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Collection, Iterable, Tuple

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


def _get_wrong_name_statics(content: str) -> Iterable[str]:
    for variable in parsing.get_static_variables(content):
        if not _is_uppercase_static_name(variable) and not _is_magic_variable(variable):
            yield variable


def _fix_wrongly_named_statics(content: str, variables: Collection[str]) -> str:
    code_mask = parsing.get_is_code_mask(content)
    paranthesis_map = parsing.get_paren_depths(content)

    vis = ""
    for is_code, c in zip(code_mask, content):
        if c in " \n":
            vis += c
        elif is_code:
            vis += c
        else:
            vis += "#"

    for variable in variables:
        replacements = []
        for match in re.finditer(r"(?<=[^A-Za-z_\.])" + variable + r"(?=[^A-Za-z_])", content):
            replacements.append((match.start(), match.end()))

        if not replacements:
            raise RuntimeError(f"Unable to find '{variable}' in {filename}")

        renamed_variable = re.sub("_{1,}", "_", variable.upper())
        if not renamed_variable.startswith("_"):
            renamed_variable = f"_{renamed_variable}"

        if renamed_variable.endswith("_"):
            renamed_variable = renamed_variable[:-1]

        if not renamed_variable:
            raise RuntimeError(f"Unable to find a replacement name for {variable}")

        for start, end in sorted(replacements, reverse=True):
            if not all(code_mask[start:end]):
                continue
            if (
                max(paranthesis_map[start:end]) > 0
                and "=" in content[end : min(len(content) - 1, end + 3)]
            ):
                # kwarg names shouldn't be replaced
                continue

            content = content[:start] + renamed_variable + content[end:]

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


def define_undefined_variables(content: str) -> bool:
    undefined_variables = _get_undefined_variables(content)
    if undefined_variables:
        return _fix_undefined_variables(content, undefined_variables)

    return content


def remove_unused_imports(content: str) -> bool:
    unused_import_linenos = set(_get_unused_imports(content))
    if unused_import_linenos:
        return _fix_unused_imports(content, unused_import_linenos)

    return content


def fix_rmspace(content: str) -> str:
    return rmspace.format_str(content)


def fix_black(content: str) -> str:
    return black.format_str(content, mode=black.Mode(line_length=100))


def fix_isort(content: str, *, line_length: int = 100) -> str:
    return isort.code(content, config=isort.Config(profile="black", line_length=line_length))


def capitalize_underscore_statics(content: str) -> str:
    wrongly_named_statics = set(_get_wrong_name_statics(content))
    if wrongly_named_statics:
        content = _fix_wrongly_named_statics(content, wrongly_named_statics)

    return content
