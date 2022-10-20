import collections
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Collection, Iterable, Tuple

import rmspace
from pylint.lint import Run

with open(Path(__file__).parent / "known_packages.json", "r", encoding="utf-8") as stream:
    _PACKAGE_SOURCES = frozenset(json.load(stream))

with open(Path(__file__).parent / "package_aliases.json", "r", encoding="utf-8") as stream:
    _PACKAGE_ALIASES = MappingProxyType(json.load(stream))

with open(Path(__file__).parent / "package_variables.json", "r", encoding="utf-8") as stream:
    _ASSUMED_SOURCES = MappingProxyType(
        {package: frozenset(variables) for package, variables in json.load(stream).items()}
    )


def _deconstruct_pylint_warning(error_line: str) -> Tuple[Path, int, int, str, str]:
    filename, lineno, charno, error_code, error_msg = error_line.split(":")

    return filename, lineno, charno, error_code.strip(), error_msg.strip()


def _find_pylint_errors(filename: Path, error_code: str) -> Iterable[str]:
    stdout = io.StringIO()
    args = [
        "--disable",
        "all",
        "--enable",
        f"{error_code}",
        str(filename.absolute()),
    ]
    re_pattern = re.compile(r".*\.py:\d+:\d+: \w\d+: .* \(" + error_code + r"\)")

    original_sys_stdout = sys.stdout
    try:
        sys.stdout = stdout
        Run(args, exit=False)
    finally:
        sys.stdout = original_sys_stdout

    output = stdout.getvalue()
    for line in output.splitlines():
        if re_pattern.match(line):
            yield line


def _get_undefined_variables(filename: Path) -> Collection[str]:
    variables = set()
    for line in _find_pylint_errors(filename, "undefined-variable"):
        try:
            filename, *_, error_msg = _deconstruct_pylint_warning(line)
            _, variable_name, _ = error_msg.split("'")
            variables.add(variable_name)
        except ValueError:
            pass

    return variables


def _get_unused_imports(filename: Path) -> Iterable[Tuple[int, str, str]]:
    for line in _find_pylint_errors(filename, "unused-import"):
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


def _fix_undefined_variables(filename: Path, variables: Collection[str]) -> bool:
    variables = set(variables)

    with open(filename, "r", encoding="utf-8") as stream:
        content = stream.read()

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
        overlap = package_variables & variables
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
        return False

    with open(filename, "w", encoding="utf-8") as stream:
        for line in lines:
            stream.write(line)
            stream.write("\n")

    return change_count > 0


def _fix_unused_imports(filename: Path, problems: Collection[Tuple[int, str, str]]) -> bool:

    lineno_problems = collections.defaultdict(set)
    for lineno, package, variable in problems:
        lineno_problems[int(lineno)].add((package, variable))

    with open(filename, "r", encoding="utf-8") as stream:
        lines = stream.readlines()

    change_count = 0

    new_lines = []
    for i, line in enumerate(lines):
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
        return False

    with open(filename, "w", encoding="utf-8") as stream:
        for line in new_lines:
            stream.write(line)

    return change_count > 0


def define_undefined_variables(filename: Path) -> bool:
    undefined_variables = _get_undefined_variables(filename)
    if undefined_variables:
        return _fix_undefined_variables(filename, undefined_variables)

    return False


def remove_unused_imports(filename: Path) -> bool:
    unused_import_linenos = set(_get_unused_imports(filename))
    if unused_import_linenos:
        return _fix_unused_imports(filename, unused_import_linenos)

    return False


def fix_rmspace(filename: Path) -> None:
    rmspace.main([str(filename)])


def fix_black(filename: Path) -> None:
    cmd = [sys.executable, "-m", "black", "--line-length", "100", filename]
    subprocess.check_call(cmd)


def fix_isort(filename: Path, *, line_length: int = 100) -> None:
    cmd = [
        sys.executable,
        "-m",
        "isort",
        "--line-length",
        str(line_length),
        "--profile",
        "black",
        filename,
    ]
    subprocess.check_call(cmd)
