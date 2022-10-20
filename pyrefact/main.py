import argparse
import io
import re
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Collection, Iterable, Sequence, Tuple

import rmspace
from pylint.lint import Run

_PACKAGE_SOURCES = frozenset(
    (
        "argparse",
        "collections",
        "configparser",
        "datetime",
        "Flask",
        "functools",
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
)

_PACKAGE_ALIASES = {"pd": "pandas", "np": "numpy", "plt": "matplotlib.pyplot"}

_ASSUMED_SOURCES = {
    "typing": frozenset(
        (
            "Callable",
            "Collection",
            "Dict",
            "Iterable",
            "List",
            "Literal",
            "Mapping",
            "NamedTuple",
            "Optional",
            "Sequence",
            "Tuple",
            "Union",
        )
    ),
    "pathlib": frozenset(("Path",)),
    "types": frozenset(("MappingProxyType",)),
}

_PACKAGE_ALIASES = MappingProxyType(_PACKAGE_ALIASES)
_ASSUMED_SOURCES = MappingProxyType(_ASSUMED_SOURCES)


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


def _get_unused_imports(filename: Path) -> Iterable[int]:
    for line in _find_pylint_errors(filename, "unused-import"):
        try:
            _, lineno, *_ = _deconstruct_pylint_warning(line)
            yield lineno
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


def _fix_unused_imports(filename: Path, linenos: Collection[int]) -> bool:
    linenos = set(int(lineno) for lineno in linenos)

    with open(filename, "r", encoding="utf-8") as stream:
        lines = stream.readlines()

    new_lines = []
    for i, line in enumerate(lines):
        if i + 1 in linenos:
            print(f"Removing '{line.strip()}'")
        else:
            new_lines.append(line)

    change_count = len(lines) - len(new_lines)

    assert change_count >= 0

    if change_count == 0:
        return False

    with open(filename, "w", encoding="utf-8") as stream:
        for line in new_lines:
            stream.write(line)

    return change_count > 0


def _define_undefined_variables(filename: Path) -> bool:
    undefined_variables = _get_undefined_variables(filename)
    if undefined_variables:
        return _fix_undefined_variables(filename, undefined_variables)

    return False


def _remove_unused_imports(filename: Path) -> bool:
    unused_import_linenos = set(_get_unused_imports(filename))
    if unused_import_linenos:
        return _fix_unused_imports(filename, unused_import_linenos)

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
    parser.add_argument("paths", help="Paths to refactor", type=Path, nargs="+", default=())
    return parser.parse_args(args)


def _default_fixes(filename: Path) -> None:
    _fix_isort(filename)
    _fix_black(filename)


def run_pyrefact(filename: Path) -> None:
    """Fix a file.

    Args:
        filename (Path): File to fix
    """
    _default_fixes(filename)
    _fix_rmspace(filename)
    any_fixes = False
    if _define_undefined_variables(filename):
        any_fixes = True
    if _remove_unused_imports(filename):
        any_fixes = True
    if any_fixes:
        _default_fixes(filename)


def _iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from path.rglob("*.py")


def main(args: Sequence[str]) -> int:
    """Parse command-line arguments and run pyrefact on provided files.

    Args:
        args (Sequence[str]): sys.argv[1:]

    Returns:
        int: 0 if successful.

    """
    args = _parse_args(args)

    count = 0
    for filename in _iter_python_files(args.paths):
        count += 1
        run_pyrefact(filename)

    if count == 0:
        print("No files provided")
        return 1

    return 0
