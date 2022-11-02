import ast
import builtins
import collections
import heapq
import io
import itertools
import queue
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

_REDUNDANT_UNDERSCORED_ASSIGN_RE_PATTERN = r"(?<![^\n]) *(\*?_ *,? *)+[\*\+\/\-\|\&:]?= *(?![=])"


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


def _get_uses_of(node: ast.AST, scope: ast.AST, content: str) -> Iterable[ast.Name]:
    name = node.id if isinstance(node, ast.Name) else node.name
    reference_nodes = {
        refnode
        for refnode in ast.walk(scope)
        if isinstance(refnode, ast.Name)
        and isinstance(refnode.ctx, ast.Load)
        and refnode.id == name
    }
    if isinstance(node, ast.Name):
        start = (node.lineno, node.col_offset)
        end = (node.end_lineno, node.end_col_offset)
    elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.FunctionDef)):
        start_charno, end_charno = _get_func_name_start_end(node, content)
        start = (node.lineno, start_charno)
        end = (node.lineno, end_charno)
    else:
        raise NotImplementedError(f"Unknown type: {type(node)}")
    for refnode in reference_nodes:
        n_start = (refnode.lineno, refnode.col_offset)
        n_end = (refnode.end_lineno, refnode.end_col_offset)
        if end < n_start:
            yield refnode
        elif isinstance(node, (ast.Module, ast.ClassDef)) and n_end < start:
            yield refnode


def _get_variable_name_substitutions(ast_tree: ast.AST, content: str) -> Mapping[ast.AST, str]:
    renamings = collections.defaultdict(set)
    classdefs: List[parsing.Statement] = []
    funcdefs: List[parsing.Statement] = []
    for node in parsing.iter_classdefs(ast_tree):
        name = node.name
        substitute = _rename_class(name, private=_is_private(name))
        classdefs.append(node)
        renamings[node].add(substitute)
        for refnode in _get_uses_of(node, ast_tree, content):
            renamings[refnode].add(substitute)

    for node in parsing.iter_funcdefs(ast_tree):
        name = node.name
        substitute = _rename_variable(name, private=_is_private(name), static=False)
        funcdefs.append(node)
        renamings[node].add(substitute)
        for refnode in _get_uses_of(node, ast_tree, content):
            renamings[refnode].add(substitute)

    for node in parsing.iter_assignments(ast_tree):
        substitute = _rename_variable(node.id, private=_is_private(node.id), static=True)
        renamings[node].add(substitute)
        for refnode in _get_uses_of(node, ast_tree, content):
            renamings[refnode].add(substitute)

    while funcdefs or classdefs:
        for partial_tree in classdefs.copy():
            classdefs.remove(partial_tree)
            for node in parsing.iter_classdefs(partial_tree):
                name = node.name
                classdefs.append(node)
                substitute = _rename_class(name, private=_is_private(name))
                renamings[node].add(substitute)
                for refnode in _get_uses_of(node, partial_tree, content):
                    renamings[refnode].add(substitute)
            for node in parsing.iter_funcdefs(partial_tree):
                name = node.name
                funcdefs.append(node)
                substitute = _rename_variable(name, private=_is_private(name), static=False)
                renamings[node].add(substitute)
                for refnode in _get_uses_of(node, partial_tree, content):
                    renamings[refnode].add(substitute)
            for node in parsing.iter_assignments(partial_tree):
                name = node.id
                substitute = _rename_variable(name, private=_is_private(name), static=False)
                renamings[node].add(substitute)
                for refnode in _get_uses_of(node, partial_tree, content):
                    renamings[refnode].add(substitute)
        for partial_tree in funcdefs.copy():
            funcdefs.remove(partial_tree)
            for node in parsing.iter_classdefs(partial_tree):
                name = node.name
                classdefs.append(node)
                substitute = _rename_class(name, private=False)
                renamings[node].add(substitute)
                for refnode in _get_uses_of(node, partial_tree, content):
                    renamings[refnode].add(substitute)
            for node in parsing.iter_funcdefs(partial_tree):
                name = node.name
                funcdefs.append(node)
                substitute = _rename_variable(name, private=False, static=False)
                renamings[node].add(substitute)
                for refnode in _get_uses_of(node, partial_tree, content):
                    renamings[refnode].add(substitute)
            for node in parsing.iter_assignments(partial_tree):
                name = node.id
                substitute = _rename_variable(name, private=False, static=False)
                renamings[node].add(substitute)
                for refnode in _get_uses_of(node, partial_tree, content):
                    renamings[refnode].add(substitute)

    return renamings


def _get_variable_re_pattern(variable) -> str:
    return r"(?<![A-Za-z_\.])" + variable + r"(?![A-Za-z_])"


def _get_func_name_start_end(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef], content: str
) -> Tuple[int, int]:
    start, end = parsing.get_charnos(node, content)
    codeblock = content[start:end]
    for match in re.finditer(_get_variable_re_pattern(node.name), codeblock):
        if match.group() == node.name:
            end = start + match.end()
            start += match.start()
            return start, end

    raise RuntimeError(f"Cannot find {node.name} in code block:\n{codeblock}")


def _fix_variable_names(
    content: str,
    renamings: Mapping[ast.AST, str],
    preserve: Collection[str] = frozenset(),
) -> str:
    replacements = []
    for node, substitutes in renamings.items():
        if len(substitutes) != 1:
            raise RuntimeError(
                f"Expected 1 substitute, got {len(substitutes)}: {substitutes}\nCode:\n{ast.dump(node, indent=2)}"
            )
        substitute = substitutes.pop()
        if isinstance(node, ast.Name):
            if node.id != substitute and node.id not in preserve:
                start, end = parsing.get_charnos(node, content)
                replacements.append((start, end, substitute))
            continue

        if node.name == substitute or node.name in preserve:
            continue

        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            raise TypeError(f"Unknown type: {type(node)}")

        start, end = _get_func_name_start_end(node, content)

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


def _recursive_attribute_name(attribute: ast.Attribute) -> str:
    if isinstance(attribute.value, ast.Attribute):
        return f"{_recursive_attribute_name(attribute.value)}.{attribute.attr}"
    return f"{attribute.value.id}.{attribute.attr}"


def _get_unused_imports(ast_tree: ast.Module) -> str:

    names = set()
    attributes = set()
    imports = set()
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node)
        elif isinstance(node, ast.Attribute):
            attributes.add(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            for alias in node.names:
                imports.add(alias.name if alias.asname is None else alias.asname)

    used_names = {name.id for name in names}
    for attribute in attributes:
        try:
            full_name = _recursive_attribute_name(attribute)
        except AttributeError:
            continue

        used_names.add(full_name)
        while "." in full_name:
            full_name = re.sub(r"\.[^\.]*$", "", full_name)
            used_names.add(full_name)

    return imports - used_names


def _get_unused_imports_split(
    ast_tree: ast.Module, unused_imports: Collection[str]
) -> Tuple[
    Collection[Union[ast.Import, ast.ImportFrom]],
    Collection[Union[ast.Import, ast.ImportFrom]],
]:
    import_unused_aliases = collections.defaultdict(set)
    for node in ast.walk(ast_tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                used_name = alias.name if alias.asname is None else alias.asname
                if used_name in unused_imports:
                    import_unused_aliases[node].add(alias)

    partially_unused_imports = set()
    completely_unused_imports = set()

    for node, unused_aliases in import_unused_aliases.items():
        if set(node.names) - unused_aliases:
            partially_unused_imports.add(node)
        else:
            completely_unused_imports.add(node)

    return completely_unused_imports, partially_unused_imports


def _construct_import_statement(
    node: Union[ast.Import, ast.ImportFrom], unused_imports: Collection[str]
) -> str:
    statement = "import " + ", ".join(
        sorted(
            alias.name if alias.asname is None else f"{alias.name} as {alias.asname}"
            for alias in node.names
            if (alias.name if alias.asname is None else alias.asname) not in unused_imports
        )
    )
    if isinstance(node, ast.Import):
        return statement

    return f"from {node.module} {statement}"


def _remove_unused_imports(
    ast_tree: ast.Module, content: str, unused_imports: Collection[str]
) -> str:
    completely_unused_imports, partially_unused_imports = _get_unused_imports_split(
        ast_tree, unused_imports
    )
    if completely_unused_imports:
        content = remove_nodes(content, completely_unused_imports, ast_tree)
        if not partially_unused_imports:
            return content
        ast_tree = ast.parse(content)
        completely_unused_imports, partially_unused_imports = _get_unused_imports_split(
            ast_tree, unused_imports
        )

    if completely_unused_imports:
        raise RuntimeError("Failed to remove unused imports")

    # For every import, construct what we would like it to look like with redundant stuff removed, find the old
    # version of it, and replace it.

    # Iterate from bottom to top of file, so we don't have to re-calculate the linenos etc.
    for node in sorted(
        partially_unused_imports,
        key=lambda n: (n.lineno, n.col_offset, n.end_lineno, n.end_col_offset),
        reverse=True,
    ):
        start, end = parsing.get_charnos(node, content)
        code = content[start:end]
        replacement = _construct_import_statement(node, unused_imports)
        print(f"Replacing:\n{code}\nWith:\n{replacement}")
        content = content[:start] + replacement + content[end:]

    return content


def remove_unused_imports(content: str) -> str:
    """Remove unused imports from source code.

    Args:
        content (str): Python source code

    Returns:
        str: Source code, with added imports removed
    """
    ast_tree = ast.parse(content)
    unused_imports = _get_unused_imports(ast_tree)
    if unused_imports:
        content = _remove_unused_imports(ast_tree, content, unused_imports)

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
    renamings = _get_variable_name_substitutions(ast_tree, content)

    if renamings:
        content = _fix_variable_names(content, renamings, preserve)

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


def _unique_assignment_targets(
    node: Union[ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For]
) -> Collection[ast.Name]:
    targets = set()
    if isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For)):
        for subtarget in ast.walk(node.target):
            if isinstance(subtarget, ast.Name) and isinstance(subtarget.ctx, ast.Store):
                targets.add(subtarget)
        return targets
    if isinstance(node, ast.Assign):
        for target in node.targets:
            for subtarget in ast.walk(target):
                if isinstance(subtarget, ast.Name) and isinstance(subtarget.ctx, ast.Store):
                    targets.add(subtarget)
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
        body = queue.PriorityQueue()
        for node in def_node.body:
            body.put((node.lineno, node))
        while not body.empty():
            _, node = body.get()
            if isinstance(node, ast.For):
                for subnode in reversed(node.body):
                    body.put((subnode.lineno, subnode))
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For)):
                target_nodes = _unique_assignment_targets(node)
                if not target_nodes:
                    continue
                target_names = {x.id for x in target_nodes}
                referenced_names = set()
                starts = []
                ends = []
                for target_node in target_nodes:
                    s, e = parsing.get_charnos(target_node, content)
                    starts.append(s)
                    ends.append(e)
                start = min(starts)
                end = max(ends)
                for refnode in reference_nodes:
                    n_start, n_end = parsing.get_charnos(refnode, content)
                    if (
                        end < n_start
                        or (isinstance(def_node, (ast.ClassDef, ast.Module)) and n_end < start)
                        or isinstance(def_node, ast.For)
                    ):
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
        content = _fix_variable_names(content, renamings, preserve)
        ast_tree = ast.parse(content)

    for node in ast.walk(ast_tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            target_nodes = _unique_assignment_targets(node)
            target_names = {x.id for x in target_nodes}
            if target_names == {"_"}:
                code = parsing.get_code(node, content)
                changed_code = re.sub(_REDUNDANT_UNDERSCORED_ASSIGN_RE_PATTERN, "", code)
                if code != changed_code:
                    print(f"Removing redundant assignments in {code}")
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
                    print(f"Found empty {bodytype}")
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


def _compute_safe_funcdef_calls(root: ast.Module) -> Collection[str]:
    """Compute what functions can safely be called without having a side effect.

    This is also to compute the inverse, i.e. what function calls may be removed
    without breaking something.

    Args:
        root (ast.Module): Module to find function definitions in

    Returns:
        Collection[str]: Names of all functions that have no side effect when called.
    """
    defined_names = {
        node.id
        for node in ast.walk(root)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    function_defs = {
        node for node in ast.walk(root) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    builtin_names = set(dir(builtins))
    safe_callables = builtin_names - {"print", "exit"}
    changes = True
    while changes:
        changes = False
        for node in function_defs:
            if node.name in defined_names:
                continue
            nonreturn_children = [
                child
                for child in node.body
                if not isinstance(
                    child,
                    (ast.Return, ast.Yield, ast.YieldFrom, ast.Continue, ast.Break),
                )
            ]
            if not any(
                parsing.has_side_effect(child, safe_callables) for child in nonreturn_children
            ):
                safe_callables.add(node.name)
                changes = True
        function_defs = {node for node in function_defs if node.name not in safe_callables}
    return safe_callables


def delete_pointless_statements(content: str) -> str:
    """Delete pointless statements with no side effects from code

    Args:
        content (str): Python source code.

    Returns:
        str: Modified code
    """
    ast_tree = ast.parse(content)
    delete = []
    safe_callables = _compute_safe_funcdef_calls(ast_tree)
    for node in itertools.chain([ast_tree], _iter_defs_recursive(ast_tree)):
        for i, child in enumerate(node.body):
            if not parsing.has_side_effect(child, safe_callables):
                if i > 0 or not (
                    isinstance(child, ast.Expr)
                    and isinstance(child.value, ast.Constant)
                    and isinstance(child.value.value, str)
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
        usages = {node for node in names if node.id == def_node.name}
        recursive_usages = {
            node
            for node in ast.walk(def_node)
            if isinstance(node, ast.Name) and node.id == def_node.name
        }
        if not usages - recursive_usages:
            print(f"{def_node.name} is never used")
            delete.append(def_node)
            continue

    content = remove_nodes(content, delete, root)

    return content
