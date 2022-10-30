import ast
import collections
import heapq
import io
import itertools
import re
import sys
import tempfile
from pathlib import Path
from typing import Collection, Iterable, List, Mapping, Tuple, Union

import black
import isort
import rmspace
from pylint.lint import Run

from pyrefact import parsing
from pyrefact.constants import ASSUMED_PACKAGES, ASSUMED_SOURCES, PACKAGE_ALIASES

_REDUNDANT_UNDERSCORED_ASSIGN_RE_PATTERN = r"(?<![^\n]) *(\*?_ *,? *)+:?= *(?![=])"


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
    if variable == "_":
        return variable

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


def _iter_ast_nodes(root: ast.AST) -> Iterable[ast.AST]:
    try:
        for child in root.body:
            yield from _iter_ast_nodes(child)
    except AttributeError:
        yield root


def _get_variable_name_substitutions(ast_tree: ast.AST) -> Mapping[ast.AST, str]:
    renamings = collections.defaultdict(set)
    classdefs: List[parsing.Statement] = []
    funcdefs: List[parsing.Statement] = []
    for node in parsing.iter_classdefs(ast_tree):
        name = node.name
        substitute = _rename_class(name, private=_is_private(name))
        classdefs.append(node)
        renamings[node].add(substitute)

    for node in parsing.iter_funcdefs(ast_tree):
        name = node.name
        substitute = _rename_variable(name, private=_is_private(name), static=False)
        funcdefs.append(node)
        renamings[node].add(substitute)

    for node in parsing.iter_assignments(ast_tree):
        substitute = _rename_variable(node.id, private=_is_private(node.id), static=True)
        renamings[node].add(substitute)

    while funcdefs or classdefs:
        for partial_tree in classdefs.copy():
            classdefs.remove(partial_tree)
            for node in parsing.iter_classdefs(partial_tree):
                name = node.name
                classdefs.append(node)
                substitute = _rename_class(name, private=_is_private(name))
                renamings[node].add(substitute)
            for node in parsing.iter_funcdefs(partial_tree):
                name = node.name
                funcdefs.append(node)
                substitute = _rename_variable(name, private=_is_private(name), static=False)
                renamings[node].add(substitute)
            for node in parsing.iter_assignments(partial_tree):
                name = node.id
                substitute = _rename_variable(name, private=_is_private(name), static=False)
                renamings[node].add(substitute)
        for partial_tree in funcdefs.copy():
            funcdefs.remove(partial_tree)
            for node in parsing.iter_classdefs(partial_tree):
                name = node.name
                classdefs.append(node)
                substitute = _rename_class(name, private=False)
                renamings[node].add(substitute)
            for node in parsing.iter_funcdefs(partial_tree):
                name = node.name
                funcdefs.append(node)
                substitute = _rename_variable(name, private=False, static=False)
                renamings[node].add(substitute)
            for node in parsing.iter_assignments(partial_tree):
                name = node.id
                substitute = _rename_variable(name, private=False, static=False)
                renamings[node].add(substitute)

    return renamings


def _get_variable_re_pattern(variable) -> str:
    return r"(?<![A-Za-z_\.])" + variable + r"(?![A-Za-z_])"


def _fix_variable_names(content: str, renamings: Mapping[ast.AST, str], root: ast.AST) -> str:
    replacements = []
    name_nodes = collections.defaultdict(list)
    for node in ast.walk(root):
        if isinstance(node, ast.Name):
            name_nodes[node.id].append(node)
    for node, substitutes in renamings.items():
        if len(substitutes) != 1:
            raise RuntimeError(f"Expected 1 substitute, got {len(substitutes)}: {substitutes}")
        substitute = substitutes.pop()
        if isinstance(node, ast.Name):
            if node.id != substitute:
                for name_node in name_nodes[node.id]:
                    start, end = parsing.get_charnos(name_node, content)
                    replacements.append((start, end, substitute))
            continue

        if node.name == substitute:
            continue

        codeblock = content[start:end]
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            raise TypeError(f"Unknown type: {type(node)}")

        for match in re.finditer(_get_variable_re_pattern(node.name), codeblock):
            assert match.group() == node.name
            end = start + match.end()
            start += match.start()
            break
        else:
            raise RuntimeError(f"Cannot find {node.name} in code block:\n{codeblock}")

        replacements.append((start, end, substitute))  # Name in function definition
        for name_node in name_nodes[node.id]:
            start, end = parsing.get_charnos(name_node, content)
            replacements.append((start, end, substitute))

    for start, end, substitute in sorted(set(replacements), reverse=True):
        print(f"Replacing {content[start:end]} with {substitute}")
        content = content[:start] + substitute + content[end:]

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
                packages = {package for package, _ in lineno_problems[i + 1]}
                if len(packages) != 1:
                    raise RuntimeError("Unable to parse unique package")
                package = packages.pop()
                bad_variables = {variable for _, variable in lineno_problems[i + 1]}
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


def align_variable_names_with_convention(
    content: str, preserve: Collection[str] = frozenset()
) -> str:
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
    ast_tree = ast.parse(content)
    renamings = _get_variable_name_substitutions(ast_tree)

    if renamings:
        content = _fix_variable_names(content, renamings, ast_tree)

    return content


def _iter_defs_recursive(
    ast_root: ast.Module,
) -> Iterable[Union[ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef]]:
    left = list(ast_root.body)
    while left:
        for node in left.copy():
            left.remove(node)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                left.extend(node.body)
                yield node


def _recursive_tuple_unpack(root: ast.Tuple) -> Iterable[ast.Name]:
    for child in root.elts:
        if isinstance(child, ast.Tuple):
            yield from _recursive_tuple_unpack(child)
        elif isinstance(child, ast.Name):
            yield child
        elif isinstance(child, ast.Starred):
            while isinstance(child, ast.Starred):
                child = child.value
            if isinstance(child, ast.Name):
                yield child
            if isinstance(child, ast.Tuple):
                yield from _recursive_tuple_unpack(child)
        else:
            raise ValueError(f"Cannot parse node: {ast.dump(root)}")


def _unique_assignment_targets(
    node: Union[ast.Assign, ast.AnnAssign, ast.AugAssign]
) -> Collection[str]:
    if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        return {node.target}
    if isinstance(node, ast.Assign):
        targets = set()
        for target in node.targets:
            if isinstance(target, ast.Name):
                targets.add(target)
            elif isinstance(target, ast.Tuple):
                targets.update(_recursive_tuple_unpack(target))
            elif isinstance(target, ast.Subscript):
                targets.add(target.value)
        return targets
    raise TypeError(f"Expected Assignment type, got {type(node)}")


def undefine_unused_variables(content: str, preserve: Collection[str] = frozenset()) -> str:
    """Remove definitions of unused variables

    Args:
        content (str): Python source code
        preserve (Collection[str], optional): Variable names to preserve

    Returns:
        str: Python source code, with no definitions of unused variables
    """
    ast_tree = ast.parse(content)
    renamings = collections.defaultdict(set)
    imports = set()
    for node in ast.walk(ast_tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.update(alias.name for alias in node.names)

    for def_node in itertools.chain(
        [ast_tree],
        parsing.iter_funcdefs(ast_tree),
    ):
        reference_nodes = {
            node
            for node in ast.walk(def_node)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        for node in def_node.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                target_nodes = _unique_assignment_targets(node)
                target_names = {
                    x.id if isinstance(x, ast.Name) else x.value.id for x in target_nodes
                }
                referenced_names = set()
                start, end = parsing.get_charnos(node, content)
                for refnode in reference_nodes:
                    n_start, n_end = parsing.get_charnos(refnode, content)
                    if end < n_start or (def_node is ast_tree and n_end < start):
                        referenced_names.add(refnode.id)
                redundant_targets = target_names - referenced_names - imports
                if def_node is ast_tree:
                    redundant_targets = redundant_targets - preserve
                for target_node in target_nodes:
                    if isinstance(target_node, ast.Attribute):
                        target_node = target_node.value
                    if target_node.id in redundant_targets:
                        renamings[target_node].add("_")

    if renamings:
        content = _fix_variable_names(content, renamings, ast_tree)
        ast_tree = ast.parse(content)

    for node in ast.walk(ast_tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            target_names = {x.id if isinstance(x, ast.Name) else x.value.id for x in target_nodes}
            if target_names == {"_"}:
                code = parsing.get_code(node, content)
                changed_code = re.sub(_REDUNDANT_UNDERSCORED_ASSIGN_RE_PATTERN, "", code)
                print(f"Removing redundant assignments in {code}")
                assert code != changed_code
                content = content.replace(code, changed_code)

    return content


def remove_nodes(content: str, nodes: Iterable[ast.AST], root: ast.Module) -> str:
    """Remove ast nodes from code

    Args:
        content (str): Python source code
        nodes (Iterable[ast.AST]): Nodes to delete from code
        root (ast.Module): Complete corresponding module

    Returns:
        str: Code after deleting nodes
    """
    keep_mask = [True] * len(content)
    nodes = list(nodes)
    for node in nodes:
        start, end = parsing.get_charnos(node, content)
        print(f"Removing:\n{content[start:end]}")
        keep_mask[start:end] = [False] * (end - start)
        for decorator_node in getattr(node, "decorator_list", []):
            start, end = parsing.get_charnos(decorator_node, content)
            start -= 1  # The @ is missed otherwise
            print(f"Removing:\n{content[start:end]}")
            keep_mask[start:end] = [False] * (end - start)

    passes = [len(content) + 1]

    for node in ast.walk(root):
        for bodytype in "body", "finalbody", "orelse":
            if body := getattr(node, bodytype, []):
                if isinstance(body, list) and all(child in nodes for child in body):
                    start_charno, _ = parsing.get_charnos(body[0], content)
                    passes.append(start_charno)

    heapq.heapify(passes)

    next_pass = heapq.heappop(passes)
    chars = []
    for i, char, keep in zip(range(len(content)), content, keep_mask):
        if i == next_pass:
            chars.extend("pass")
        elif next_pass < i < next_pass + 3:
            continue
        else:
            if i > next_pass:
                next_pass = heapq.heappop(passes)
            if keep:
                chars.append(char)

    return "".join(chars)


def delete_pointless_statements(content: str) -> str:
    """Delete pointless statements with no side effects from code

    Args:
        content (str): Python source code.

    Returns:
        str: Modified code
    """
    ast_tree = ast.parse(content)
    delete = []
    defined_names = {node.id for node in ast.walk(ast_tree) if isinstance(node, ast.Name)}
    builtin_names = set(dir(__builtins__))
    safe_callables = defined_names - builtin_names - {"print", "exit"}
    for node in itertools.chain([ast_tree], _iter_defs_recursive(ast_tree)):
        for i, child in enumerate(node.body):
            if isinstance(child, ast.Expr) and not parsing.has_side_effect(child, safe_callables):
                if i > 0 or not (
                    isinstance(child.value, ast.Constant) and isinstance(child.value.value, str)
                ):
                    delete.append(child)

    content = remove_nodes(content, delete, ast_tree)

    return content


def delete_unused_functions_and_classes(
    content: str, preserve: Collection[str] = frozenset()
) -> str:
    """Delete unused functions and classes from code.

    Args:
        content (str): Python source code
        preserve (Collection[str], optional): Names to preserve

    Returns:
        str: Python source code, where unused functions and classes have been deleted.
    """
    root = ast.parse(content)

    defs = []
    names = []

    for node in ast.walk(root):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            defs.append(node)
        elif isinstance(node, ast.Name):
            names.append(node)

    delete = []
    for def_node in defs:
        if def_node.name in preserve:
            continue
        usages = [node for node in names if node.id == def_node.name]
        if not usages:
            print(f"{def_node.name} is never used")
            delete.append(def_node)
            continue

        nonrecursive_usages = []
        def_start, def_end = parsing.get_charnos(def_node, content)
        for name in names:
            start, end = parsing.get_charnos(name, content)
            if def_start <= start <= end <= def_end:
                # Recursion is not "real" use
                continue
            nonrecursive_usages.append(name)

        if not nonrecursive_usages:
            print(f"{def_node.name} is never used (recursive)")
            delete.append(def_node)

    content = remove_nodes(content, delete, root)

    return content
