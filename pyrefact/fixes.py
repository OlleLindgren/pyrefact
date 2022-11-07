import ast
import builtins
import collections
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

from pyrefact import parsing, processing
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
    classdefs: List[ast.ClassDef] = []
    funcdefs: List[ast.FunctionDef] = []
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
                if name.startswith("__") and name.endswith("__"):
                    continue
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
    ast_tree = ast.parse(content)
    blacklisted_names = parsing.get_imported_names(ast_tree) | set(dir(builtins))
    for node, substitutes in renamings.items():
        if len(substitutes) != 1:
            raise RuntimeError(
                f"Expected 1 substitute, got {len(substitutes)}: {substitutes}\nCode:\n{ast.dump(node, indent=2)}"
            )
        substitute = substitutes.pop()
        if substitute in blacklisted_names:
            continue
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


def _get_unused_imports(ast_tree: ast.Module) -> Collection[str]:
    """Get names that are imported in ast tree but never used.

    Args:
        ast_tree (ast.Module): Ast tree to search

    Returns:
        Collection[str]: A collection of names that are imported but never used.
    """
    imports = parsing.get_imported_names(ast_tree)

    names = {
        node.id
        for node in ast.walk(ast_tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.Attribute):
            try:
                full_name = _recursive_attribute_name(node)
            except AttributeError:
                continue

            names.add(full_name)
            while "." in full_name:
                full_name = re.sub(r"\.[^\.]*$", "", full_name)
                names.add(full_name)

    return imports - names


def _get_unused_imports_split(
    ast_tree: ast.Module, unused_imports: Collection[str]
) -> Tuple[
    Collection[Union[ast.Import, ast.ImportFrom]],
    Collection[Union[ast.Import, ast.ImportFrom]],
]:
    """Split unused imports into completely and partially unused imports.

    Args:
        ast_tree (ast.Module): Ast tree to search
        unused_imports (Collection[str]): Names that are imported but never used.

    Returns:
        Tuple: completely_unused_imports, partially_unused_imports
    """
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
    names = ", ".join(
        sorted(
            alias.name if alias.asname is None else f"{alias.name} as {alias.asname}"
            for alias in node.names
            if (alias.name if alias.asname is None else alias.asname) not in unused_imports
        )
    )
    if isinstance(node, ast.Import):
        return f"import {names}"

    return f"from {node.module} import {names}"


def _remove_unused_imports(
    ast_tree: ast.Module, content: str, unused_imports: Collection[str]
) -> str:
    completely_unused_imports, partially_unused_imports = _get_unused_imports_split(
        ast_tree, unused_imports
    )
    if completely_unused_imports:
        content = processing.remove_nodes(content, completely_unused_imports, ast_tree)
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
            body.put((node.lineno, node, None))
        while not body.empty():
            _, node, containing_loop_node = body.get()
            if isinstance(node, (ast.For, ast.While)):
                for subnode in reversed(node.body):
                    body.put((subnode.lineno, subnode, containing_loop_node or node))
            elif isinstance(node, ast.If):
                for subnode in node.body:
                    body.put((subnode.lineno, subnode, containing_loop_node))
                for subnode in node.orelse:
                    body.put((subnode.lineno, subnode, containing_loop_node))
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For)):
                target_nodes = _unique_assignment_targets(node)
                if not target_nodes:
                    continue
                target_names = {x.id for x in target_nodes}
                referenced_names = set()
                starts = []
                ends = []
                if containing_loop_node is not None:
                    loop_start, loop_end = parsing.get_charnos(containing_loop_node, content)
                else:
                    loop_start = loop_end = -1
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
                        or loop_start <= n_start <= n_end <= loop_end
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


def delete_pointless_statements(content: str) -> str:
    """Delete pointless statements with no side effects from code

    Args:
        content (str): Python source code.

    Returns:
        str: Modified code
    """
    ast_tree = ast.parse(content)
    delete = []
    safe_callables = parsing.safe_callable_names(ast_tree)
    for node in itertools.chain([ast_tree], parsing.iter_bodies_recursive(ast_tree)):
        for i, child in enumerate(node.body):
            if not parsing.has_side_effect(child, safe_callables):
                if i > 0 or not (
                    isinstance(child, ast.Expr)
                    and isinstance(child.value, ast.Constant)
                    and isinstance(child.value.value, str)
                ):
                    delete.append(child)

    content = processing.remove_nodes(content, delete, ast_tree)

    return content


def _get_unused_functions_classes(root: ast.AST, preserve: Collection[str]) -> Iterable[ast.AST]:
    funcdefs = []
    classdefs = []
    names = []

    for node in ast.walk(root):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcdefs.append(node)
        elif isinstance(node, ast.ClassDef):
            classdefs.append(node)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.append(node)

    class_magics = collections.defaultdict(set)
    for node in classdefs:
        for child in node.body:
            if (
                isinstance(child, ast.FunctionDef)
                and child.name.startswith("__")
                and child.name.endswith("__")
            ):
                class_magics[node].add(child)

    magic_source_classes = {}
    for classdef, magics in class_magics.items():
        for magic in magics:
            magic_source_classes[magic] = classdef

    for def_node in funcdefs:
        if def_node.name in preserve:
            continue
        usages = {node for node in names if node.id == def_node.name}
        if parent_class := magic_source_classes.get(def_node):
            usages.update(node for node in names if node.id == parent_class.name)
        recursive_usages = {
            node
            for node in ast.walk(def_node)
            if isinstance(node, ast.Name) and node.id == def_node.name
        }
        if not usages - recursive_usages:
            print(f"{def_node.name} is never used")
            yield def_node

    for def_node in classdefs:
        if def_node.name in preserve:
            continue
        usages = {node for node in names if node.id == def_node.name}
        internal_usages = {
            node
            for node in ast.walk(def_node)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in (def_node.name, "self", "cls")
        }
        if not usages - internal_usages:
            print(f"{def_node.name} is never used")
            yield def_node


def _iter_unreachable_nodes(body: Iterable[ast.AST]) -> Iterable[ast.AST]:
    after_block = False
    for node in body:
        if after_block:
            yield node
            continue
        if parsing.is_blocking(node):
            after_block = True


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

    delete = set(_get_unused_functions_classes(root, preserve))

    content = processing.remove_nodes(content, delete, root)

    return content


def delete_unreachable_code(content: str) -> str:
    """Find and delete dead code.

    Args:
        content (str): Python source code

    Returns:
        str: Source code with dead code deleted
    """
    root = ast.parse(content)

    delete = set()
    for node in parsing.iter_bodies_recursive(root):
        if isinstance(node, (ast.If, ast.While)):
            try:
                test_value = parsing.literal_value(node.test)
            except ValueError:
                pass
            else:
                if isinstance(node, ast.If):
                    if test_value and node.body:
                        delete.update(node.orelse)
                    elif not test_value and node.orelse:
                        delete.update(node.body)
                    else:
                        delete.add(node)
                elif not test_value:
                    delete.add(node)
        else:
            delete.update(_iter_unreachable_nodes(node.body))

    content = processing.remove_nodes(content, delete, root)

    return content


def _can_be_evaluated(node: ast.AST, safe_callables: Collection[str]) -> bool:
    """Determine if a node can be evaluated.

    Args:
        node (ast.AST): Node to check

    Raises:
        ValueError: If the node has a side effect

    Returns:
        bool: True if the node can be evaluated
    """
    safe_callables = parsing.safe_callable_names(node)
    if parsing.has_side_effect(node, safe_callables):
        raise ValueError("Cannot evaluate node with side effects.")
    try:
        eval(ast.unparse(node))  # pylint: disable=eval-used
    except Exception:  # pylint: disable=broad-except
        return False

    return True


def _is_contains_comparison(node) -> bool:
    if not isinstance(node, ast.Compare):
        return False
    if len(node.ops) != 1:
        return False
    if not isinstance(node.ops[0], ast.In):
        return False
    return True


def replace_with_sets(content: str) -> str:
    """Replace inlined lists with sets.

    Args:
        content (str): Python source code

    Returns:
        str: Modified python source code
    """
    root = ast.parse(content)
    safe_callables = parsing.safe_callable_names(root)

    replacements = {}

    for node in ast.walk(root):
        if not _is_contains_comparison(node):
            continue

        for comp in node.comparators:
            if isinstance(comp, (ast.ListComp, ast.GeneratorExp)):
                replacement = ast.SetComp(elt=comp.elt, generators=comp.generators)
            elif isinstance(comp, ast.DictComp):
                replacement = ast.SetComp(elt=comp.key, generators=comp.generators)
            elif isinstance(comp, (ast.List, ast.Tuple)):
                replacement = ast.Set(elts=comp.elts)
            elif (
                isinstance(comp, ast.Call)
                and isinstance(comp.func, ast.Name)
                and isinstance(comp.func.ctx, ast.Load)
                and comp.func.id in {"sorted", "list", "tuple"}
            ):
                replacement = ast.Call(
                    func=ast.Name(id="set", ctx=ast.Load()), args=comp.args, keywords=comp.keywords
                )
            else:
                continue

            if (
                not parsing.has_side_effect(comp, safe_callables)
                and not parsing.has_side_effect(replacement, safe_callables)
                and _can_be_evaluated(replacement, safe_callables)
            ):
                replacements[comp] = replacement

    if replacements:
        content = processing.replace_nodes(content, replacements)

    return content


def remove_redundant_chained_calls(content: str) -> str:
    root = ast.parse(content)

    function_chain_redundancy_mapping = {
        "sorted": {"list", "sorted", "tuple"},
        "list": {"list", "tuple"},
        "set": {"set", "list", "sorted", "tuple"},
    }

    replacements = {}
    touched_linenos = set()

    for node in ast.walk(root):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
            node_lineno_range = set(range(node.lineno, node.end_lineno + 1))
            if node_lineno_range & touched_linenos:
                continue
            redundant_call_names = function_chain_redundancy_mapping.get(node.func.id)
            if not redundant_call_names:
                continue
            modified_node = node
            while (
                isinstance(modified_node.args[0], ast.Call)
                and isinstance(modified_node.args[0].func, ast.Name)
                and modified_node.args[0].func.id in redundant_call_names
            ):
                modified_node = replacements[node] = ast.Call(
                    func=node.func, args=modified_node.args[0].args, keywords=[]
                )
                touched_linenos.update(node_lineno_range)

    if replacements:
        content = processing.replace_nodes(content, replacements)

    return content


def _get_package_names(node: Union[ast.Import, ast.ImportFrom]):
    if isinstance(node, ast.ImportFrom):
        return [node.module]

    return [alias.name for alias in node.names]


def move_imports_to_toplevel(content: str) -> str:
    root = ast.parse(content)
    toplevel_imports = {
        node for node in root.body if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    all_imports = {
        node for node in ast.walk(root) if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    toplevel_packages = set()
    for node in toplevel_imports:
        toplevel_packages.update(_get_package_names(node))

    builtin_packages = {name.lstrip("_") for name in sys.builtin_module_names}
    builtin_packages |= {
        "typing",
        "math",
        "pathlib",
        "datetime",
        "enum",
        "dataclasses",
        "heapq",
        "queue",
        "re",
    }
    # TODO implement a better way to get the builtin packages, there are many missing

    safe_toplevel_packages = builtin_packages | toplevel_packages

    imports_movable_to_toplevel = {
        node
        for node in all_imports - toplevel_imports
        if all(name in safe_toplevel_packages for name in _get_package_names(node))
    }

    defs = {
        node
        for node in root.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    if defs:
        first_def_lineno = min(node.lineno for node in defs)
        imports_movable_to_toplevel.update(
            node for node in toplevel_imports if node.lineno > first_def_lineno
        )

    for i, node in enumerate(root.body):
        if i > 0 and not isinstance(node, (ast.Import, ast.ImportFrom)):
            lineno = node.lineno - 1
            break
        elif (
            i == 0
            and not isinstance(node, (ast.Import, ast.ImportFrom))
            and not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            )
        ):
            lineno = node.lineno - 1
            break
    else:
        if root.body:
            lineno = root.body[-1].end_lineno + 1
        else:
            lineno = 1

    additions = []
    removals = []
    for node in imports_movable_to_toplevel:
        removals.append(node)
        if isinstance(node, ast.Import):
            new_node = ast.Import(names=node.names, lineno=lineno)
        else:
            new_node = ast.ImportFrom(
                module=node.module, names=node.names, level=node.level, lineno=lineno
            )
        additions.append(new_node)

    if removals:
        content = processing.remove_nodes(content, removals, root)
    if additions:
        content = processing.insert_nodes(content, additions)

    # Isort will remove redundant imports

    return content
