import argparse
import io
import re
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Collection, Iterable, Sequence

import rmspace
from pylint.lint import Run

_PACKAGE_SOURCES = (
    "argparse",
    "collections",
    "configparser",
    "datetime",
    "Flask",
    "itertools",
    "json",
    "keras",
    "math",
    "matplotlib",
    "numpy",
    "os",
    "pandas",
    "re",
    "requests",
    "scipy",
    "setuptools",
    "shlex",
    "sklearn",
    "subprocess",
    "sys",
    "tensorflow",
    "time",
    "traceback",
    "urllib",
    "warnings",
)

_PACKAGE_ALIASES = {"pd": "pandas", "np": "numpy", "plt": "matplotlib.pyplot"}

_ASSUMED_SOURCES = {
    "typing": (
        "List",
        "Iterable",
        "Tuple",
        "Callable",
        "Sequence",
        "Collection",
        "Literal",
        "Union",
        "Optional",
        "NamedTuple",
    ),
    "pathlib": ("Path",),
    "types": ("MappingProxyType",),
}

_PACKAGE_ALIASES = MappingProxyType(_PACKAGE_ALIASES)
_ASSUMED_SOURCES = MappingProxyType(_ASSUMED_SOURCES)


def _deconstruct_pylint_warning(error_line: str) -> int:
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
        print(line)
        if re_pattern.match(line):
            yield line


def _get_undefined_variables(filename: Path) -> Collection[str]:
    variables = set()
    for line in _find_pylint_errors(filename, "undefined-variable"):
        try:
            filename, lineno, charno, error_code, error_msg = _deconstruct_pylint_warning(line)
            _, variable_name, _ = error_msg.split("'")
            variables.add(variable_name)
        except ValueError:
            pass

    return variables


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
        overlap = set(package_variables) & variables
        if overlap:
            fix = f"from {package} import " + ", ".join(sorted(overlap))
            print(f"Inserting '{fix}' at line {lineno}")
            lines.insert(lineno, fix)

    for package in set(_PACKAGE_SOURCES) & variables:
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


def _fix_undefined_imports(filename: Path) -> bool:
    undefined_variables = _get_undefined_variables(filename)
    if undefined_variables:
        return _fix_undefined_variables(filename, undefined_variables)

    return False


def _fix_rmspace(filename: Path) -> None:
    rmspace.main.main([filename])


def _fix_black(filename: Path) -> None:
    cmd = [sys.executable, "-m", "black", "--line-length", "100", filename]
    subprocess.check_call(cmd)


def _fix_isort(filename: Path) -> None:
    cmd = [sys.executable, "-m", "isort", "--line-length", "100", "--profile", "black", filename]
    subprocess.check_call(cmd)


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="File to refactor", type=Path)
    return parser.parse_args(args)


def main(args: Sequence[str]) -> int:
    args = _parse_args(args)

    filename = args.filename

    _fix_isort(filename)
    _fix_black(filename)

    if _fix_undefined_imports(filename):
        _fix_isort(filename)
