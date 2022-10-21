import collections
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Collection, Iterable, Sequence, Tuple

import rmspace
from pylint.lint import Run

with open(Path(__file__).parent / "known_packages.json", "r", encoding="utf-8") as stream:
    _PACKAGE_SOURCES = frozenset(json.load(stream))

with open(Path(__file__).parent / "package_aliases.json", "r", encoding="utf-8") as stream:
    _PACKAGE_ALIASES = json.load(stream)

with open(Path(__file__).parent / "package_variables.json", "r", encoding="utf-8") as stream:
    _ASSUMED_SOURCES = json.load(stream)

with open(Path(__file__).parent / "python_keywords.json", "r", encoding="utf-8") as stream:
    _PYTHON_KEYWORDS = frozenset(json.load(stream))


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


def _get_is_code_mask(content: str) -> Sequence[bool]:
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


def _get_paren_depths(content: str) -> Sequence[int]:
    code_mask = _get_is_code_mask(content)
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


def _get_static_variables(content: str) -> Iterable[str]:
    is_code_mask = _get_is_code_mask(content)
    assert len(is_code_mask) == len(content), (len(is_code_mask), len(content))

    scopes = []

    indent = 0
    parsed_chars = 0
    paren_depth = 0
    for full_line in content.splitlines(keepends=True):
        line = "".join(char for i, char in enumerate(full_line) if is_code_mask[i + parsed_chars])
        parsed_chars += len(full_line)
        line_paren_depth = _get_paren_depths(full_line)
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

        if scopes:
            continue

        if not variable:
            continue

        if re.match(r"^[a-zA-Z_]+$", variable):
            yield variable


def _is_uppercase_static_name(variable: str) -> bool:
    return re.match(r"^_[A-Z_]+$", variable) is not None


def _is_magic_variable(variable: str) -> bool:
    return re.match(r"^__[a-z_]+__$", variable) is not None


def _get_wrong_name_statics(filename: Path) -> Iterable[str]:
    with open(filename, "r", encoding="utf-8") as stream:
        content = stream.read()

    for variable in _get_static_variables(content):
        if not _is_uppercase_static_name(variable) and not _is_magic_variable(variable):
            yield variable


def _fix_wrongly_named_statics(filename: Path, variables: Collection[str]) -> None:
    with open(filename, "r", encoding="utf-8") as stream:
        content = stream.read()

    code_mask = _get_is_code_mask(content)
    paranthesis_map = _get_paren_depths(content)

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

    with open(filename, "w", encoding="utf-8") as stream:
        stream.write(content)


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


def capitalize_underscore_statics(filename: Path) -> None:
    wrongly_named_statics = set(_get_wrong_name_statics(filename))
    if wrongly_named_statics:
        _fix_wrongly_named_statics(filename, wrongly_named_statics)

    return False
