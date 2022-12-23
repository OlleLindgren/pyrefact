import ast
import collections
import itertools
import queue
import re
from typing import Collection, Iterable, List, Mapping, Sequence, Tuple, Union

import black
import isort
import rmspace

from pyrefact import abstractions, constants, parsing, processing

_REDUNDANT_UNDERSCORED_ASSIGN_RE_PATTERN = r"(?<![^\n]) *(\*?_ *,? *)+[\*\+\/\-\|\&:]?= *(?![=])"


def _get_undefined_variables(content: str) -> Collection[str]:
    root = parsing.parse(content)
    imported_names = parsing.get_imported_names(root)
    defined_names = set()
    referenced_names = set()
    for node in parsing.walk(root, ast.Name):
        if isinstance(node.ctx, ast.Load):
            referenced_names.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            defined_names.add(node.id)
    for node in parsing.walk(root, ast.arg):
        defined_names.add(node.arg)

    return (
        referenced_names
        - defined_names
        - imported_names
        - {name.split(".")[0] for name in imported_names}
        - constants.BUILTIN_FUNCTIONS
    )


def _rename_variable(variable: str, *, static: bool, private: bool) -> str:
    if variable == "_":
        return variable

    renamed_variable = _make_snakecase(variable, uppercase=static)

    if private and not parsing.is_private(renamed_variable):
        renamed_variable = f"_{renamed_variable}"
    if not private and parsing.is_private(renamed_variable):
        renamed_variable = renamed_variable.lstrip("_")

    if renamed_variable:
        return renamed_variable

    raise RuntimeError(f"Unable to find a replacement name for {variable}")


def _list_words(name: str) -> Sequence[str]:
    return [
        match.group()
        for match in re.finditer(r"([A-Z]{2,}(?![a-z])|[A-Z]?[a-z]*)\d*", name)
        if match.end() > match.start()
    ]


def _make_snakecase(name: str, *, uppercase: bool = False) -> str:
    return "_".join(word.upper() if uppercase else word.lower() for word in _list_words(name))


def _make_camelcase(name: str) -> str:
    return "".join(word[0].upper() + word[1:].lower() for word in _list_words(name))


def _rename_class(name: str, *, private: bool) -> str:
    name = re.sub("_{1,}", "_", name)
    if len(name) == 0:
        raise ValueError("Cannot rename empty name")

    name = _make_camelcase(name)

    if private and not parsing.is_private(name):
        return f"_{name}"
    if not private and parsing.is_private(name):
        return name[1:]

    return name


def _get_uses_of(node: ast.AST, scope: ast.AST, content: str) -> Iterable[ast.Name]:
    name = node.id if isinstance(node, ast.Name) else node.name
    if isinstance(node, ast.Name):
        start = (node.lineno, node.col_offset)
        end = (node.end_lineno, node.end_col_offset)
    elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.FunctionDef)):
        start_charno, end_charno = _get_func_name_start_end(node, content)
        start = (node.lineno, start_charno)
        end = (node.lineno, end_charno)
    else:
        raise NotImplementedError(f"Unknown type: {type(node)}")
    is_maybe_unordered_scope = isinstance(scope, (ast.Module, ast.ClassDef, ast.While, ast.For))

    # Prevent renaming variables in function scopes
    blacklisted_names = set()
    for funcdef in parsing.walk(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node in parsing.walk(funcdef, type(node)):
            continue
        if any(arg.arg == name for arg in parsing.walk(funcdef.args, ast.arg)):
            blacklisted_names.update(parsing.walk(funcdef, ast.Name))
        for node in parsing.walk(funcdef, ast.Name):
            if isinstance(node.ctx, ast.Store) and node.id == name:
                blacklisted_names.update(parsing.walk(funcdef, ast.Name))

    for refnode in parsing.walk(scope, ast.Name):
        if refnode in blacklisted_names:
            continue
        if isinstance(refnode.ctx, ast.Load) and refnode.id == name:
            n_start = (refnode.lineno, refnode.col_offset)
            n_end = (refnode.end_lineno, refnode.end_col_offset)
            if end < n_start:
                yield refnode
            elif is_maybe_unordered_scope and n_end < start:
                yield refnode


def _get_variable_name_substitutions(
    ast_tree: ast.AST, content: str, preserve: Collection[str]
) -> Mapping[ast.AST, str]:
    renamings = collections.defaultdict(set)
    classdefs: List[ast.ClassDef] = []
    funcdefs: List[ast.FunctionDef] = []
    for node in parsing.iter_classdefs(ast_tree):
        name = node.name
        substitute = _rename_class(name, private=parsing.is_private(name) or name not in preserve)
        classdefs.append(node)
        renamings[node].add(substitute)
        for refnode in _get_uses_of(node, ast_tree, content):
            renamings[refnode].add(substitute)

    typevars = set()
    for node in parsing.iter_typedefs(ast_tree):
        assert len(node.targets) == 1
        target = node.targets[0]
        assert isinstance(target, (ast.Name, ast.Attribute))
        typevars.add(target)
        for refnode in _get_uses_of(target, ast_tree, content):
            typevars.add(refnode)

    for node in parsing.iter_funcdefs(ast_tree):
        name = node.name
        substitute = _rename_variable(
            name, private=parsing.is_private(name) or name not in preserve, static=False
        )
        funcdefs.append(node)
        renamings[node].add(substitute)
        for refnode in _get_uses_of(node, ast_tree, content):
            renamings[refnode].add(substitute)

    for node in parsing.iter_assignments(ast_tree):
        if node in typevars:
            substitute = _rename_class(node.id, private=parsing.is_private(node.id))
        else:
            substitute = _rename_variable(node.id, private=parsing.is_private(node.id), static=True)
        renamings[node].add(substitute)
        for refnode in _get_uses_of(node, ast_tree, content):
            renamings[refnode].add(substitute)

    while funcdefs or classdefs:
        for partial_tree in classdefs.copy():
            classdefs.remove(partial_tree)
            for node in parsing.iter_classdefs(partial_tree):
                name = node.name
                classdefs.append(node)
                substitute = _rename_class(name, private=parsing.is_private(name))
                renamings[node].add(substitute)
                for refnode in _get_uses_of(node, partial_tree, content):
                    renamings[refnode].add(substitute)
            for node in parsing.iter_funcdefs(partial_tree):
                if parsing.is_magic_method(node):
                    continue
                name = node.name
                funcdefs.append(node)
                substitute = _rename_variable(name, private=parsing.is_private(name), static=False)
                renamings[node].add(substitute)
                for refnode in _get_uses_of(node, partial_tree, content):
                    renamings[refnode].add(substitute)
            for node in parsing.iter_assignments(partial_tree):
                name = node.id
                substitute = _rename_variable(name, private=parsing.is_private(name), static=False)
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
    ast_tree = parsing.parse(content)
    blacklisted_names = parsing.get_imported_names(ast_tree) | constants.BUILTIN_FUNCTIONS
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
    for package, package_variables in constants.ASSUMED_SOURCES.items():
        overlap = variables.intersection(package_variables)
        if overlap:
            fix = f"from {package} import " + ", ".join(sorted(overlap))
            print(f"Inserting '{fix}' at line {lineno}")
            lines.insert(lineno, fix)

    for package in (constants.ASSUMED_PACKAGES | constants.PYTHON_311_STDLIB) & variables:
        fix = f"import {package}"
        print(f"Inserting '{fix}' at line {lineno}")
        lines.insert(lineno, fix)

    for alias in constants.PACKAGE_ALIASES.keys() & variables:
        package = constants.PACKAGE_ALIASES[alias]
        fix = f"import {package} as {alias}"
        print(f"Inserting '{fix}' at line {lineno}")
        lines.insert(lineno, fix)

    change_count += len(lines)

    assert change_count >= 0

    if change_count == 0:
        return content

    return "\n".join(lines) + "\n"


def add_missing_imports(content: str) -> str:
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

    names = {node.id for node in parsing.walk(ast_tree, ast.Name) if isinstance(node.ctx, ast.Load)}
    for node in parsing.walk(ast_tree, ast.Attribute):
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
    for node in parsing.walk(ast_tree, (ast.Import, ast.ImportFrom)):
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
        print("Removing unused imports")
        content = processing.remove_nodes(content, completely_unused_imports, ast_tree)
        if not partially_unused_imports:
            return content
        ast_tree = parsing.parse(content)
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
    ast_tree = parsing.parse(content)
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
    ast_tree = parsing.parse(content)
    renamings = _get_variable_name_substitutions(ast_tree, content, preserve)

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
    ast_tree = parsing.parse(content)
    renamings = collections.defaultdict(set)
    imports = set()
    for node in parsing.walk(ast_tree, (ast.Import, ast.ImportFrom)):
        imports.update(alias.name for alias in node.names)

    for def_node in itertools.chain(
        [ast_tree],
        parsing.iter_funcdefs(ast_tree),
    ):
        reference_nodes = {
            node for node in parsing.walk(def_node, ast.Name) if isinstance(node.ctx, ast.Load)
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
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For)):
                continue

            target_nodes = _unique_assignment_targets(node)
            if not target_nodes:
                continue
            target_names = {x.id for x in target_nodes}
            referenced_names = set()
            starts = []
            ends = []
            if containing_loop_node is not None:
                (loop_start, loop_end) = parsing.get_charnos(containing_loop_node, content)
            else:
                loop_start = loop_end = -1
            for target_node in target_nodes:
                (s, e) = parsing.get_charnos(target_node, content)
                starts.append(s)
                ends.append(e)
            start = min(starts)
            end = max(ends)
            for refnode in reference_nodes:
                (n_start, n_end) = parsing.get_charnos(refnode, content)
                if (
                    end < n_start
                    or (isinstance(def_node, (ast.ClassDef, ast.Module)) and n_end < start)
                    or isinstance(def_node, ast.For)
                    or (loop_start <= n_start <= n_end <= loop_end)
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
        ast_tree = parsing.parse(content)

    for node in parsing.walk(ast_tree, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        target_nodes = _unique_assignment_targets(node)
        target_names = {x.id for x in target_nodes}
        if target_names == {"_"}:
            code = parsing.get_code(node, content)
            changed_code = re.sub(_REDUNDANT_UNDERSCORED_ASSIGN_RE_PATTERN, "", code)
            if code != changed_code:
                print(f"Removing redundant assignments in {code}")
                content = content.replace(code, changed_code)

    return content


def _is_pointless_string(node: ast.AST) -> bool:
    """Check if an AST is a pointless string statement.

    This is useful for figuring out if a node is a docstring.

    Args:
        node (ast.AST): AST to check

    Returns:
        bool: True if the node is a pointless string statement.
    """
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def delete_pointless_statements(content: str) -> str:
    """Delete pointless statements with no side effects from code

    Args:
        content (str): Python source code.

    Returns:
        str: Modified code
    """
    ast_tree = parsing.parse(content)
    delete = []
    safe_callables = parsing.safe_callable_names(ast_tree)
    for node in itertools.chain([ast_tree], parsing.iter_bodies_recursive(ast_tree)):
        for i, child in enumerate(node.body):
            if not parsing.has_side_effect(child, safe_callables):
                if i > 0 or not _is_pointless_string(child):  # Docstring
                    delete.append(child)

    if delete:
        print("Removing pointless statements")
        content = processing.remove_nodes(content, delete, ast_tree)

    return content


def _get_unused_functions_classes(root: ast.AST, preserve: Collection[str]) -> Iterable[ast.AST]:
    funcdefs = []
    classdefs = []
    name_usages = collections.defaultdict(set)

    for node in parsing.walk(root, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.name not in preserve:
            funcdefs.append(node)
    for node in parsing.walk(root, ast.ClassDef):
        if node.name not in preserve:
            classdefs.append(node)
    for node in parsing.walk(root, ast.Name):
        if isinstance(node.ctx, ast.Load):
            name_usages[node.id].add(node)

    constructors = collections.defaultdict(set)
    for node in classdefs:
        for child in node.body:
            if parsing.is_magic_method(child):
                constructors[node].add(child)

    constructor_classes = {}
    for classdef, magics in constructors.items():
        for magic in magics:
            constructor_classes[magic] = classdef

    for def_node in funcdefs:
        usages = name_usages[def_node.name]
        if parent_class := constructor_classes.get(def_node):
            constructor_usages = name_usages[parent_class.name]
        else:
            constructor_usages = set()
        recursive_usages = {
            node
            for node in ast.walk(def_node)
            if isinstance(node, ast.Name) and node.id == def_node.name
        }
        if not (usages | constructor_usages) - recursive_usages:
            print(f"{def_node.name} is never used")
            yield def_node

    for def_node in classdefs:
        usages = name_usages[def_node.name]
        internal_usages = {
            node
            for node in ast.walk(def_node)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in {def_node.name, "self", "cls"}
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
    root = parsing.parse(content)

    delete = set(_get_unused_functions_classes(root, preserve))

    if delete:
        print("Removing unused functions and classes")
        content = processing.remove_nodes(content, delete, root)

    return content


def delete_unreachable_code(content: str) -> str:
    """Find and delete dead code.

    Args:
        content (str): Python source code

    Returns:
        str: Source code with dead code deleted
    """
    root = parsing.parse(content)

    replacements = {}
    for node in parsing.walk(root, ast.IfExp):
        try:
            test_value = parsing.literal_value(node.test)
        except ValueError:
            pass
        else:
            replacements[node] = node.body if test_value else node.orelse

    if replacements:
        content = processing.replace_nodes(content, replacements)
        root = parsing.parse(content)

    delete = set()
    for node in parsing.iter_bodies_recursive(root):
        if not isinstance(node, (ast.If, ast.While)):
            delete.update(_iter_unreachable_nodes(node.body))
            continue

        try:
            test_value = parsing.literal_value(node.test)
        except ValueError:
            pass
        else:
            if isinstance(node, ast.While) and not test_value:
                delete.add(node)
                continue
            if not isinstance(node, ast.If):
                if test_value and node.body:
                    delete.update(node.orelse)
                elif not test_value and node.orelse:
                    delete.update(node.body)
                else:
                    delete.add(node)

    if delete:
        print("Removing unreachable code")
        content = processing.remove_nodes(content, delete, root)

    return content


def _get_package_names(node: Union[ast.Import, ast.ImportFrom]):
    if isinstance(node, ast.ImportFrom):
        return [node.module]

    return [alias.name for alias in node.names]


def move_imports_to_toplevel(content: str) -> str:
    root = parsing.parse(content)
    toplevel_imports = {
        node for node in root.body if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    all_imports = {node for node in parsing.walk(root, (ast.Import, ast.ImportFrom))}
    toplevel_packages = set()
    for node in toplevel_imports:
        toplevel_packages.update(_get_package_names(node))

    imports_movable_to_toplevel = {
        node
        for node in all_imports - toplevel_imports
        if all(
            name in constants.PYTHON_311_STDLIB or name in toplevel_packages
            for name in _get_package_names(node)
        )
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
        if (
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
            if node.module in constants.PYTHON_311_STDLIB:
                safe_position_lineno = lineno
            else:
                module_import_linenos = [
                    candidate.lineno
                    for candidate in toplevel_imports
                    if node.module in _get_package_names(candidate)
                ]
                if not module_import_linenos:
                    continue
                safe_position_lineno = min(module_import_linenos)
            new_node = ast.ImportFrom(
                module=node.module,
                names=node.names,
                level=node.level,
                lineno=safe_position_lineno,
            )
        additions.append(new_node)

    if removals:
        print("Removing imports outside of toplevel")
        content = processing.remove_nodes(content, removals, root)
    if additions:
        print("Adding imports to toplevel")
        content = processing.insert_nodes(content, additions)

    # Isort will remove redundant imports

    return content


def remove_duplicate_functions(content: str, preserve: Collection[str]) -> str:
    """Remove duplicate function definitions.

    Args:
        content (str): Python source code
        preserve (Collection[str]): Names to preserve

    Returns:
        str: Modified code
    """
    root = parsing.parse(content)
    function_defs = collections.defaultdict(set)

    for node in root.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        function_defs[abstractions.hash_node(node, preserve)].add(node)

    delete = set()
    renamings = {}

    for funcdefs in function_defs.values():
        if len(funcdefs) == 1:
            continue
        print(", ".join(node.name for node in funcdefs) + " are equivalent")
        preserved_nodes = {node for node in funcdefs if node.name in preserve}
        if preserved_nodes:
            replacement = min(preserved_nodes, key=lambda node: node.lineno)
        else:
            replacement = min(funcdefs, key=lambda node: node.lineno)
            preserved_nodes = {replacement}

        for node in funcdefs - preserved_nodes:
            delete.add(node)
            renamings[node.name] = replacement.name

    if not delete and not renamings:
        return content

    names = collections.defaultdict(list)
    for node in parsing.walk(root, ast.Name):
        names[node.id].append(node)

    node_renamings = collections.defaultdict(set)
    for name, substitute in renamings.items():
        for node in names[name]:
            node_renamings[node].add(substitute)

    if node_renamings:
        content = _fix_variable_names(content, node_renamings, preserve)
    if delete:
        content = processing.remove_nodes(content, delete, root)

    return content


def remove_redundant_else(content: str) -> str:
    """Remove redundante else and elif statements in code.

    Args:
        content (str): Python source code

    Returns:
        str: Code with no redundant else/elifs.
    """
    changes = True
    while changes:
        changes = False
        root = parsing.parse(content)
        for node in parsing.walk(root, ast.If):
            if not node.orelse:
                continue
            if not parsing.get_code(node, content).startswith("if"):  # Otherwise we get FPs on elif
                continue
            if not any((parsing.is_blocking(child) for child in node.body)):
                continue

            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                (start, end) = parsing.get_charnos(node.orelse[0], content)
                orelse = content[start:end]
                if orelse.startswith("elif"):  # Regular elif
                    modified_orelse = re.sub("^elif", "if", orelse)
                    print("Found redundant elif:")
                    print(parsing.get_code(node, content))
                    content = content[:start] + modified_orelse + content[end:]
                    continue

                # Otherwise it's an else: if:, which is handled below

            # else:
            ranges = [parsing.get_charnos(child, content) for child in node.orelse]
            start = min((s for (s, _) in ranges))
            end = max((e for (_, e) in ranges))
            last_else = list(re.finditer("(?<![^\\n]) *else: *\\n?", content[:start]))[-1]
            indent = len(re.findall("^ *", last_else.group())[0])
            modified_pre_else = content[: last_else.start()].rstrip() + "\n\n"
            modified_orelse = (
                " " * indent + re.sub("(?<![^\\n])    ", "", content[start:end]).lstrip()
            )
            print("Found redundant else:")
            print(parsing.get_code(node, content))
            content = modified_pre_else + modified_orelse + content[end:]

    return content


def singleton_eq_comparison(content: str) -> str:
    """Replace singleton comparisons using "==" with "is".

    Args:
        content (str): Python source code

    Returns:
        str: Fixed code
    """
    root = parsing.parse(content)

    replacements = {}
    for node in parsing.walk(root, ast.Compare):
        changes = False
        operators = []
        for comparator, node_operator in zip(node.comparators, node.ops):
            is_comparator_singleton = isinstance(comparator, ast.Constant) and (
                comparator.value is None or comparator.value is True or comparator.value is False
            )
            if is_comparator_singleton and isinstance(node_operator, ast.Eq):
                operators.append(ast.Is())
                changes = True
            elif is_comparator_singleton and isinstance(node_operator, ast.NotEq):
                operators.append(ast.IsNot())
                changes = True
            else:
                operators.append(node_operator)

        if changes:
            replacements[node] = ast.Compare(
                left=node.left,
                ops=operators,
                comparators=node.comparators,
            )

    if replacements:
        content = processing.replace_nodes(content, replacements)

    return content


def _negate_condition(node: ast.AST) -> ast.AST:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return node.operand

    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == len(node.comparators) == 1
        and type(node.ops[0]) in constants.REVERSE_OPERATOR_MAPPING
    ):
        opposite_operator_type = constants.REVERSE_OPERATOR_MAPPING[type(node.ops[0])]
        return ast.Compare(
            left=node.left, ops=[opposite_operator_type()], comparators=node.comparators
        )

    return ast.UnaryOp(op=ast.Not(), operand=node)


def swap_if_else(content: str) -> str:

    replacements = {}

    root = parsing.parse(content)
    loops = list(parsing.walk(root, (ast.For, ast.While)))

    for stmt in parsing.walk(root, ast.If):
        if (
            stmt.orelse
            and any(parsing.is_blocking(node) for node in stmt.body)
            and not any(parsing.is_blocking(node) for node in stmt.orelse)
        ):
            continue  # Redundant else
        if parsing.get_code(stmt, content).startswith("elif"):
            continue
        body_lines = stmt.body[-1].end_lineno - stmt.body[0].lineno
        orelse_lines = stmt.orelse[-1].end_lineno - stmt.orelse[0].lineno if stmt.orelse else 0
        if all((isinstance(node, ast.Pass) for node in stmt.body)) or body_lines > 2 * orelse_lines:
            if stmt.orelse:
                replacements[stmt] = ast.If(
                    test=_negate_condition(stmt.test),
                    body=stmt.orelse,
                    orelse=[node for node in stmt.body if not isinstance(node, ast.Pass)],
                    lineno=stmt.lineno,
                )
            elif len(stmt.body) > 3 and any(stmt is loop.body[-1] for loop in loops):
                replacements[stmt] = ast.If(
                    test=_negate_condition(stmt.test),
                    body=[ast.Continue()],
                    orelse=[node for node in stmt.body if not isinstance(node, ast.Pass)],
                    lineno=stmt.lineno,
                )

    last_end = -1
    for node, _ in sorted(replacements.items(), key=lambda t: (t[0].lineno, t[0].end_lineno)):
        if node.lineno <= last_end:
            del replacements[node]
        else:
            last_end = node.end_lineno

    content = processing.replace_nodes(content, replacements)

    return content


def early_return(content: str) -> str:

    replacements = {}
    removals = []

    root = parsing.parse(content)
    for funcdef in parsing.iter_funcdefs(root):
        if not (
            len(funcdef.body) >= 2
            and isinstance(funcdef.body[-1], ast.Return)
            and isinstance(funcdef.body[-2], ast.If)
        ):
            continue

        ret_stmt = funcdef.body[-1]
        if_stmt = funcdef.body[-2]
        if not isinstance(ret_stmt.value, ast.Name):
            continue
        retval = ret_stmt.value.id
        recursive_last_if_nodes = [if_stmt]
        recursive_last_nonif_nodes = []
        while recursive_last_if_nodes:
            node = recursive_last_if_nodes.pop()
            last_body = node.body[-1] if node.body else None
            last_orelse = node.orelse[-1] if node.orelse else None
            if isinstance(last_body, ast.If):
                recursive_last_if_nodes.append(last_body)
            else:
                recursive_last_nonif_nodes.append(last_body)
            if isinstance(last_orelse, ast.If):
                recursive_last_if_nodes.append(last_orelse)
            else:
                recursive_last_nonif_nodes.append(last_orelse)
        if all(
            (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and (node.targets[0].id == retval)
                for node in recursive_last_nonif_nodes
            )
        ):
            for node in recursive_last_nonif_nodes:
                replacements[node] = ast.Return(value=node.value, lineno=node.lineno)
            removals.append(ret_stmt)

    content = processing.alter_code(
        content,
        root,
        replacements=replacements,
        removals=removals,
    )

    return content


def early_continue(content: str) -> str:

    additions = []

    root = parsing.parse(content)
    for loop in parsing.walk(root, ast.For):
        stmt = loop.body[-1]
        if isinstance(stmt, ast.If) and not isinstance(stmt.body[-1], ast.Continue):
            recursive_ifs = [stmt]
            for child in stmt.orelse:
                recursive_ifs.extend(parsing.walk(child, ast.If))
            if any(len(node.orelse) > 2 for node in recursive_ifs):
                additions.append(
                    ast.Continue(
                        lineno=stmt.body[-1].end_lineno,
                        col_offset=stmt.body[-1].col_offset,
                    )
                )

    content = processing.alter_code(
        content,
        root,
        additions=additions,
    )

    return content
