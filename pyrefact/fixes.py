import ast
import collections
import copy
import itertools
import queue
import re
from typing import Collection, Iterable, List, Literal, Mapping, Sequence, Tuple, Union

import isort
import rmspace

from pyrefact import abstractions, constants, parsing, processing, style, formatting

_REDUNDANT_UNDERSCORED_ASSIGN_RE_PATTERN = r"(?<![^\n]) *(\*?_ *,? *)+[\*\+\/\-\|\&:]?= *(?![=])"


def _get_undefined_variables(source: str) -> Collection[str]:
    root = parsing.parse(source)
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


def _get_uses_of(node: ast.AST, scope: ast.AST, source: str) -> Iterable[ast.Name]:
    if isinstance(node, ast.Name):
        name = node.id
        start = (node.lineno, node.col_offset)
        end = (node.end_lineno, node.end_col_offset)
    elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.FunctionDef)):
        name = node.name
        start_charno, end_charno = _get_func_name_start_end(node, source)
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
        if any(parsing.walk(funcdef.args, ast.arg(arg=name))):
            blacklisted_names.update(parsing.walk(funcdef, ast.Name))
        for child in parsing.walk(funcdef, ast.Name(ctx=ast.Store, id=name)):
            blacklisted_names.update(parsing.walk(child, ast.Name))

    augass_candidates = {
        target
        for augass in parsing.walk(scope, ast.AugAssign)
        for target in parsing.walk(augass, ast.Name(id=name))
    }

    ctx_load_candidates = {
        refnode
        for refnode in parsing.walk(scope, ast.Name(ctx=ast.Load, id=name))
        if refnode not in blacklisted_names
    }

    for refnode in augass_candidates | ctx_load_candidates:
        n_start = (refnode.lineno, refnode.col_offset)
        n_end = (refnode.end_lineno, refnode.end_col_offset)
        if end < n_start:
            yield refnode
        elif is_maybe_unordered_scope and n_end < start:
            yield refnode


def _get_variable_name_substitutions(
    ast_tree: ast.AST, source: str, preserve: Collection[str]
) -> Mapping[ast.AST, str]:
    renamings = collections.defaultdict(set)
    classdefs: List[ast.ClassDef] = []
    funcdefs: List[ast.FunctionDef] = []
    for node in parsing.iter_classdefs(ast_tree):
        name = node.name
        substitute = style.rename_class(
            name, private=parsing.is_private(name) or name not in preserve
        )
        classdefs.append(node)
        renamings[node].add(substitute)
        for refnode in _get_uses_of(node, ast_tree, source):
            renamings[refnode].add(substitute)

    typevars = set()
    for node in parsing.iter_typedefs(ast_tree):
        assert len(node.targets) == 1
        target = node.targets[0]
        assert isinstance(target, (ast.Name, ast.Attribute))
        typevars.add(target)
        for refnode in _get_uses_of(target, ast_tree, source):
            typevars.add(refnode)

    for node in parsing.iter_funcdefs(ast_tree):
        name = node.name
        substitute = style.rename_variable(
            name, private=parsing.is_private(name) or name not in preserve, static=False
        )
        funcdefs.append(node)
        renamings[node].add(substitute)
        for refnode in _get_uses_of(node, ast_tree, source):
            renamings[refnode].add(substitute)

    for node in parsing.iter_assignments(ast_tree):
        if node in typevars:
            substitute = style.rename_class(node.id, private=parsing.is_private(node.id))
        else:
            substitute = style.rename_variable(
                node.id, private=parsing.is_private(node.id), static=True
            )
        renamings[node].add(substitute)
        for refnode in _get_uses_of(node, ast_tree, source):
            renamings[refnode].add(substitute)

    while funcdefs or classdefs:
        for partial_tree in classdefs.copy():
            classdefs.remove(partial_tree)
            for node in parsing.iter_classdefs(partial_tree):
                name = node.name
                classdefs.append(node)
                substitute = style.rename_class(name, private=parsing.is_private(name))
                renamings[node].add(substitute)
                for refnode in _get_uses_of(node, partial_tree, source):
                    renamings[refnode].add(substitute)
            for node in parsing.iter_funcdefs(partial_tree):
                if parsing.is_magic_method(node):
                    continue
                name = node.name
                funcdefs.append(node)
                substitute = style.rename_variable(
                    name, private=parsing.is_private(name), static=False
                )
                renamings[node].add(substitute)
                for refnode in _get_uses_of(node, partial_tree, source):
                    renamings[refnode].add(substitute)
            for node in parsing.iter_assignments(partial_tree):
                name = node.id
                substitute = style.rename_variable(
                    name, private=parsing.is_private(name), static=False
                )
                renamings[node].add(substitute)
                for refnode in _get_uses_of(node, partial_tree, source):
                    renamings[refnode].add(substitute)
        for partial_tree in funcdefs.copy():
            funcdefs.remove(partial_tree)
            for node in parsing.iter_classdefs(partial_tree):
                name = node.name
                classdefs.append(node)
                substitute = style.rename_class(name, private=False)
                renamings[node].add(substitute)
                for refnode in _get_uses_of(node, partial_tree, source):
                    renamings[refnode].add(substitute)
            for node in parsing.iter_funcdefs(partial_tree):
                name = node.name
                funcdefs.append(node)
                substitute = style.rename_variable(name, private=False, static=False)
                renamings[node].add(substitute)
                for refnode in _get_uses_of(node, partial_tree, source):
                    renamings[refnode].add(substitute)
            for node in parsing.iter_assignments(partial_tree):
                name = node.id
                substitute = style.rename_variable(name, private=False, static=False)
                renamings[node].add(substitute)
                for refnode in _get_uses_of(node, partial_tree, source):
                    renamings[refnode].add(substitute)

    return renamings


def _get_variable_re_pattern(variable) -> str:
    return r"(?<![A-Za-z_\.])" + variable + r"(?![A-Za-z_])"


def _get_func_name_start_end(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef], source: str
) -> Tuple[int, int]:
    start, end = parsing.get_charnos(node, source)
    codeblock = source[start:end]
    for match in re.finditer(_get_variable_re_pattern(node.name), codeblock):
        if match.group() == node.name:
            end = start + match.end()
            start += match.start()
            return start, end

    raise RuntimeError(f"Cannot find {node.name} in code block:\n{codeblock}")


def _fix_variable_names(
    source: str,
    renamings: Mapping[ast.AST, str],
    preserve: Collection[str] = frozenset(),
) -> str:
    replacements = []
    ast_tree = parsing.parse(source)
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
                start, end = parsing.get_charnos(node, source)
                replacements.append((start, end, substitute))
            continue

        if node.name == substitute or node.name in preserve:
            continue

        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            raise TypeError(f"Unknown type: {type(node)}")

        start, end = _get_func_name_start_end(node, source)

        replacements.append((start, end, substitute))

    for start, end, substitute in sorted(set(replacements), reverse=True):
        print(f"Replacing {source[start:end]} with {substitute}")
        source = source[:start] + substitute + source[end:]

    return source


def _fix_undefined_variables(source: str, variables: Collection[str]) -> str:
    variables = set(variables)

    lines = source.splitlines()
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
        return source

    return "\n".join(lines) + "\n"


def add_missing_imports(source: str) -> str:
    """Attempt to find imports matching all undefined variables.

    Args:
        source (str): Python source code

    Returns:
        str: Source code with added imports
    """
    undefined_variables = _get_undefined_variables(source)
    if undefined_variables:
        return _fix_undefined_variables(source, undefined_variables)

    return source


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

    names = {node.id for node in parsing.walk(ast_tree, ast.Name(ctx=ast.Load))}
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
    ast_tree: ast.Module, source: str, unused_imports: Collection[str]
) -> str:
    completely_unused_imports, partially_unused_imports = _get_unused_imports_split(
        ast_tree, unused_imports
    )
    if completely_unused_imports:
        print("Removing unused imports")
        source = processing.remove_nodes(source, completely_unused_imports, ast_tree)
        if not partially_unused_imports:
            return source
        ast_tree = parsing.parse(source)
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
        start, end = parsing.get_charnos(node, source)
        code = source[start:end]
        replacement = _construct_import_statement(node, unused_imports)
        print(f"Replacing:\n{code}\nWith:\n{replacement}")
        source = source[:start] + replacement + source[end:]

    return source


def remove_unused_imports(source: str) -> str:
    """Remove unused imports from source code.

    Args:
        source (str): Python source code

    Returns:
        str: Source code, with added imports removed
    """
    ast_tree = parsing.parse(source)
    unused_imports = _get_unused_imports(ast_tree)
    if unused_imports:
        source = _remove_unused_imports(ast_tree, source, unused_imports)

    return source


def fix_tabs(source: str) -> str:
    """Replace tabs with 4 spaces in source code

    Args:
        source (str): Python source code

    Returns:
        str: Formatted source code
    """
    return re.sub(r"\t", " " * 4, source)


def fix_too_many_blank_lines(source: str) -> str:
    source = re.sub(r"\n{4,}", "\n" * 3, source)
    source = re.sub(r"\n{2,}\Z", "\n", source)

    return source


def fix_rmspace(source: str) -> str:
    """Remove trailing whitespace from source code.

    Args:
        source (str): Python source code

    Returns:
        str: Source code, without trailing whitespace
    """
    return rmspace.format_str(source)


def fix_isort(source: str, *, line_length: int = 100) -> str:
    """Format source code with isort

    Args:
        source (str): Python source code
        line_length (int, optional): Line length. Defaults to 100.

    Returns:
        str: Source code, formatted with isort
    """
    return isort.code(source, config=isort.Config(profile="black", line_length=line_length))


@processing.fix
def fix_line_lengths(source: str, *, max_line_length: int = 100) -> str:

    root = parsing.parse(source)

    formatted_nodes = set()
    formatted_ranges = set()

    subscopes = []

    for scope in parsing.walk(
        root, (ast.AST(body=list), ast.AST(orelse=list), ast.AST(finalbody=list))
    ):
        subscopes.append(getattr(scope, "body", []))
        subscopes.append(getattr(scope, "orelse", []))
        subscopes.append(getattr(scope, "finalbody", []))

    for node in itertools.chain.from_iterable(subscopes):
        max_node_line_length = max(
            child.end_col_offset for child in parsing.walk(node, ast.AST(end_col_offset=int))
        )
        if node in formatted_nodes or max_node_line_length <= max_line_length:
            continue

        start, end = parsing.get_charnos(node, source, keep_first_indent=True)

        current_code = source[start:end]
        elif_pattern = r"\A[\s\n]*(elif)"
        if_pattern = r"\A[\s\n]*(if)"
        elif_matches = list(re.finditer(elif_pattern, current_code))
        if elif_matches:
            re_match = elif_matches[0]
            current_code = re.sub(
                elif_pattern, re_match.group().replace("elif", "if"), current_code, 1
            )
            new_code = formatting.format_with_black(
                current_code, line_length=max(60, max_line_length)
            )
            if_match = next(re.finditer(if_pattern, new_code))
            new_code = re.sub(if_pattern, if_match.group().replace("if", "elif"), new_code, 1)
        else:
            new_code = formatting.format_with_black(
                current_code, line_length=max(60, max_line_length)
            )

        if new_code != current_code and (
            not any((e >= start and s <= end for s, e in formatted_ranges))
        ):
            yield node, new_code
            formatted_ranges.add((start, end))


def align_variable_names_with_convention(
    source: str, preserve: Collection[str] = frozenset()
) -> str:
    """Align variable names with normal convention

    Class names should have CamelCase names
    Non-static variables and functions should have snake_case names
    Static variables should have UPPERCASE_UNDERSCORED names

    All names defined in global scope may be private and start with a single underscore
    Names outside global scope are never allowed to be private
    __magic__ names may only be defined in global scope

    Args:
        source (str): Python source code

    Returns:
        str: Source code, where all variable names comply with normal convention
    """
    ast_tree = parsing.parse(source)
    renamings = _get_variable_name_substitutions(ast_tree, source, preserve)

    if renamings:
        source = _fix_variable_names(source, renamings, preserve)

    return source


def _unique_assignment_targets(
    node: Union[ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For]
) -> Collection[ast.Name]:
    targets = set()
    if isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For)):
        return set(parsing.walk(node.target, ast.Name(ctx=ast.Store)))
    if isinstance(node, ast.Assign):
        for target in node.targets:
            targets.update(parsing.walk(target, ast.Name(ctx=ast.Store)))
        return targets
    raise TypeError(f"Expected Assignment type, got {type(node)}")


def undefine_unused_variables(source: str, preserve: Collection[str] = frozenset()) -> str:
    """Remove definitions of unused variables

    Args:
        source (str): Python source code
        preserve (Collection[str], optional): Variable names to preserve

    Returns:
        str: Python source code, with no definitions of unused variables
    """
    ast_tree = parsing.parse(source)
    renamings = collections.defaultdict(set)
    imports = set()
    for node in parsing.walk(ast_tree, (ast.Import, ast.ImportFrom)):
        imports.update(alias.name for alias in node.names)

    for def_node in parsing.walk(ast_tree, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
        reference_nodes = set(parsing.walk(def_node, ast.Name(ctx=ast.Load)))
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
            if containing_loop_node is None:
                loop_start = loop_end = -1
            else:
                (loop_start, loop_end) = parsing.get_charnos(containing_loop_node, source)
            for target_node in target_nodes:
                (s, e) = parsing.get_charnos(target_node, source)
                starts.append(s)
                ends.append(e)
            start = min(starts)
            end = max(ends)
            for refnode in reference_nodes:
                (n_start, n_end) = parsing.get_charnos(refnode, source)
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
        source = _fix_variable_names(source, renamings, preserve)
        ast_tree = parsing.parse(source)

    for node in parsing.walk(ast_tree, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        target_nodes = _unique_assignment_targets(node)
        target_names = {x.id for x in target_nodes}
        if target_names != {"_"}:
            continue

        (start_charno, end_charno) = parsing.get_charnos(node, source)
        code = parsing.get_code(node, source)
        changed_code = re.sub(_REDUNDANT_UNDERSCORED_ASSIGN_RE_PATTERN, "", code)
        if code != changed_code:
            print(f"Removing redundant assignments in {code}")
            source = source[:start_charno] + changed_code + source[end_charno:]

    return source


def _is_pointless_string(node: ast.AST) -> bool:
    """Check if an AST is a pointless string statement.

    This is useful for figuring out if a node is a docstring.

    Args:
        node (ast.AST): AST to check

    Returns:
        bool: True if the node is a pointless string statement.
    """
    return parsing.match_template(node, ast.Expr(value=ast.Constant(value=str)))


def delete_pointless_statements(source: str) -> str:
    """Delete pointless statements with no side effects from code

    Args:
        source (str): Python source code.

    Returns:
        str: Modified code
    """
    ast_tree = parsing.parse(source)
    delete = []
    safe_callables = parsing.safe_callable_names(ast_tree)
    for node in itertools.chain([ast_tree], parsing.iter_bodies_recursive(ast_tree)):
        for i, child in enumerate(node.body):
            if not parsing.has_side_effect(child, safe_callables):
                if i > 0 or not _is_pointless_string(child):  # Docstring
                    delete.append(child)

    if delete:
        print("Removing pointless statements")
        source = processing.remove_nodes(source, delete, ast_tree)

    return source


def _get_unused_functions_classes(root: ast.AST, preserve: Collection[str]) -> Iterable[ast.AST]:
    funcdefs = []
    classdefs = []
    name_usages = collections.defaultdict(set)

    preserved_class_funcdefs = {
        funcdef
        for node in parsing.walk(root, ast.ClassDef)
        for funcdef in parsing.filter_nodes(node.body, (ast.FunctionDef, ast.AsyncFunctionDef))
        if f"{node.name}.{funcdef.name}" in preserve
    }

    for node in parsing.walk(root, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.name not in preserve and node not in preserved_class_funcdefs:
            funcdefs.append(node)

    for node in parsing.walk(root, ast.ClassDef):
        if node.name not in preserve:
            classdefs.append(node)

    for node in parsing.walk(root, ast.Name(ctx=ast.Load)):
        name_usages[node.id].add(node)

    for node in parsing.walk(root, ast.Attribute):
        name_usages[node.attr].add(node)
        for name in parsing.walk(node, ast.Name):
            name_usages[name.id].add(node)

    constructors = collections.defaultdict(set)
    for node in classdefs:
        for child in filter(parsing.is_magic_method, node.body):

            constructors[node].add(child)

    constructor_classes = {
        magic: classdef for (classdef, magics) in constructors.items() for magic in magics
    }

    for def_node in funcdefs:
        usages = name_usages[def_node.name]
        if parent_class := constructor_classes.get(def_node):
            constructor_usages = name_usages[parent_class.name]
        else:
            constructor_usages = set()
        recursive_usages = set(parsing.walk(def_node, ast.Name(id=def_node.name)))
        if not (usages | constructor_usages) - recursive_usages:
            yield def_node

    for def_node in classdefs:
        usages = name_usages[def_node.name]
        internal_usages = set(
            parsing.walk(def_node, ast.Name(ctx=ast.Load, id=(def_node.name, "self", "cls")))
        )
        if not usages - internal_usages:
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
    source: str, preserve: Collection[str] = frozenset()
) -> str:
    """Delete unused functions and classes from code.

    Args:
        source (str): Python source code
        preserve (Collection[str], optional): Names to preserve

    Returns:
        str: Python source code, where unused functions and classes have been deleted.
    """
    root = parsing.parse(source)

    delete = set(_get_unused_functions_classes(root, preserve))

    if delete:
        print("Removing unused functions and classes")
        source = processing.remove_nodes(source, delete, root)

    return source


def delete_unreachable_code(source: str) -> str:
    """Find and delete dead code.

    Args:
        source (str): Python source code

    Returns:
        str: Source code with dead code deleted
    """
    root = parsing.parse(source)

    delete = set()
    for node in parsing.iter_bodies_recursive(root):
        if not isinstance(node, (ast.If, ast.While)):
            delete.update(_iter_unreachable_nodes(node.body))
            continue

        try:
            test_value = parsing.literal_value(node.test)
        except ValueError:
            continue

        if isinstance(node, ast.While) and not test_value:
            delete.add(node)
            continue

        if isinstance(node, ast.If):
            if test_value and node.body:
                delete.update(node.orelse)
            elif not test_value and node.orelse:
                delete.update(node.body)
            else:
                delete.add(node)

    if delete:
        print("Removing unreachable code")
        source = processing.remove_nodes(source, delete, root)

    return source


def _get_package_names(node: Union[ast.Import, ast.ImportFrom]):
    if isinstance(node, ast.ImportFrom):
        return [node.module]

    return [alias.name for alias in node.names]


def move_imports_to_toplevel(source: str) -> str:
    root = parsing.parse(source)
    toplevel_imports = set(parsing.filter_nodes(root.body, (ast.Import, ast.ImportFrom)))
    all_imports = set(parsing.walk(root, (ast.Import, ast.ImportFrom)))
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

    if defs := set(
        parsing.filter_nodes(root.body, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ):
        first_def_lineno = min(node.lineno - len(node.decorator_list) for node in defs)
        imports_movable_to_toplevel.update(
            node for node in toplevel_imports if node.lineno > first_def_lineno
        )

    for i, node in enumerate(root.body):
        if i > 0 and not isinstance(node, (ast.Import, ast.ImportFrom)):
            lineno = min(x.lineno for x in parsing.walk(node, ast.AST(lineno=int))) - 1
            break
        if i == 0 and not parsing.match_template(
            node, (ast.Import, ast.ImportFrom, ast.Expr(value=ast.Constant(value=str)))
        ):
            lineno = min(x.lineno for x in parsing.walk(node, ast.AST(lineno=int))) - 1
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
            additions.append(new_node)
            continue

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

    if removals or additions:
        print("Moving imports to toplevel")
        source = processing.alter_code(source, root, removals=removals, additions=additions)

    # Isort will remove redundant imports

    return source


def remove_duplicate_functions(source: str, preserve: Collection[str]) -> str:
    """Remove duplicate function definitions.

    Args:
        source (str): Python source code
        preserve (Collection[str]): Names to preserve

    Returns:
        str: Modified code
    """
    root = parsing.parse(source)
    function_defs = collections.defaultdict(set)

    for node in parsing.filter_nodes(root.body, ast.FunctionDef):
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
        return source

    names = collections.defaultdict(list)
    for node in parsing.walk(root, ast.Name):
        names[node.id].append(node)

    node_renamings = collections.defaultdict(set)
    for name, substitute in renamings.items():
        for node in names[name]:
            node_renamings[node].add(substitute)

    if node_renamings:
        source = _fix_variable_names(source, node_renamings, preserve)
    if delete:
        source = processing.remove_nodes(source, delete, root)

    return source


@processing.fix(restart_on_replace=True)
def remove_redundant_else(source: str) -> str:
    """Remove redundante else and elif statements in code.

    Args:
        source (str): Python source code

    Returns:
        str: Code with no redundant else/elifs.
    """
    root = parsing.parse(source)
    for node in parsing.walk(root, ast.If):
        if not node.orelse:
            continue
        if not parsing.get_code(node, source).startswith("if"):  # Otherwise we get FPs on elif
            continue
        if not any((parsing.is_blocking(child) for child in node.body)):
            continue

        if parsing.match_template(node.orelse, [ast.If]):
            (start, end) = parsing.get_charnos(node.orelse[0], source)
            orelse = source[start:end]
            if orelse.startswith("elif"):  # Regular elif
                modified_orelse = re.sub("^elif", "if", orelse)

                source = source[:start] + modified_orelse + source[end:]
                yield processing.Range(start, end), modified_orelse

            # Otherwise it's an else: if:, which is handled below

        # else

        ranges = [parsing.get_charnos(child, source) for child in node.orelse]
        start = min((s for (s, _) in ranges))
        end = max((e for (_, e) in ranges))
        last_else = list(re.finditer("(?<![^\\n]) *else: *\\n?", source[:start]))[-1]
        indent = len(re.findall("^ *", last_else.group())[0])
        modified_orelse = " " * indent + re.sub("(?<![^\\n])    ", "", source[start:end]).lstrip()

        pre_else = source[: last_else.start()]
        start_offset = len(pre_else) - len(pre_else.rstrip())

        yield processing.Range(last_else.start() - start_offset, end), "\n\n" + modified_orelse


@processing.fix
def singleton_eq_comparison(source: str) -> str:
    """Replace singleton comparisons using "==" with "is".

    Args:
        source (str): Python source code

    Returns:
        str: Fixed code
    """
    root = parsing.parse(source)

    for node in parsing.walk(root, ast.Compare):
        changes = False
        operators = []
        for comparator, node_operator in zip(node.comparators, node.ops):
            is_comparator_singleton = parsing.match_template(
                comparator, ast.Constant(value=(None, True, False))
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
            yield node, ast.Compare(
                left=node.left,
                ops=operators,
                comparators=node.comparators,
            )


def _negate_condition(node: ast.AST) -> ast.AST:
    if parsing.match_template(node, ast.UnaryOp(op=ast.Not)):
        return node.operand

    if parsing.match_template(
        node, ast.Compare(ops=[tuple(constants.REVERSE_OPERATOR_MAPPING)], comparators=[object])
    ):
        opposite_operator_type = constants.REVERSE_OPERATOR_MAPPING[type(node.ops[0])]
        return ast.Compare(
            left=node.left, ops=[opposite_operator_type()], comparators=node.comparators
        )

    if parsing.match_template(node, ast.BoolOp(op=ast.And)):
        return ast.BoolOp(op=ast.Or(), values=[_negate_condition(child) for child in node.values])

    if parsing.match_template(node, ast.BoolOp(op=ast.Or)):
        return ast.BoolOp(op=ast.And(), values=[_negate_condition(child) for child in node.values])

    return ast.UnaryOp(op=ast.Not(), operand=node)


def _iter_implicit_if_elses(
    root: ast.AST,
) -> Iterable[Tuple[ast.If, Sequence[ast.AST], Sequence[ast.AST]]]:
    for (condition,), *implicit_orelse in parsing.walk_sequence(
        root, ast.If, ast.AST, expand_last=True
    ):
        implicit_orelse = [x[0] for x in implicit_orelse]
        if any(parsing.is_blocking(child) for child in condition.body) and not condition.orelse:
            yield condition, condition.body, implicit_orelse


def _iter_explicit_if_elses(
    root: ast.AST,
) -> Iterable[Tuple[ast.If, Sequence[ast.AST], Sequence[ast.AST]]]:
    for condition in parsing.walk(root, ast.If):
        if condition.body and condition.orelse:
            yield condition, condition.body, condition.orelse


def _count_children(node: ast.AST, child_type: ast.AST) -> int:
    return sum(1 for _ in parsing.walk(node, child_type))


def _count_branches(nodes: Sequence[ast.AST]) -> int:
    return 1 + sum(_count_children(node, ast.If) for node in nodes)


def _orelse_preferred_as_body(body: Sequence[ast.AST], orelse: Sequence[ast.AST]) -> bool:
    if all(isinstance(node, ast.Pass) for node in body):
        return True
    if all(isinstance(node, ast.Pass) for node in orelse):
        return False

    body_blocking = any(parsing.is_blocking(node) for node in body)
    orelse_blocking = any(parsing.is_blocking(node) for node in orelse)
    if body_blocking and not orelse_blocking:
        return False
    if orelse_blocking and not body_blocking:
        return True
    body_branches = _count_branches(body)
    orelse_branches = _count_branches(orelse)
    if orelse_blocking and body_blocking and body_branches >= 2 * orelse_branches:
        return True
    if isinstance(orelse[0], (ast.Return, ast.Continue, ast.Break)) and len(body) > 3:
        return True

    return False


def _sequential_similar_ifs(source: str, root: ast.AST) -> Collection[ast.If]:
    return set.union(
        set(),
        *map(set, parsing.iter_similar_nodes(root, source, ast.If, count=2, length=15)),
        *map(set, parsing.iter_similar_nodes(root, source, ast.If, count=3, length=10)),
    )


@processing.fix(restart_on_replace=True)
def _swap_explicit_if_else(source: str) -> str:

    root = parsing.parse(source)
    sequential_similar_ifs = _sequential_similar_ifs(source, root)

    for stmt, body, orelse in _iter_explicit_if_elses(root):
        if isinstance(stmt.test, ast.NamedExpr):
            continue
        if stmt in sequential_similar_ifs:
            continue
        if (
            orelse
            and any(parsing.is_blocking(node) for node in body)
            and not any(parsing.is_blocking(node) for node in orelse)
        ):
            continue  # Redundant else
        if parsing.get_code(stmt, source).startswith("elif"):
            continue
        if _orelse_preferred_as_body(body, orelse):
            if orelse:
                yield stmt, ast.If(
                    test=_negate_condition(stmt.test),
                    body=orelse,
                    orelse=[node for node in body if not isinstance(node, ast.Pass)],
                    lineno=stmt.lineno,
                )


def _swap_implicit_if_else(source: str) -> str:
    replacements = {}
    removals = set()

    root = parsing.parse(source)
    sequential_similar_ifs = _sequential_similar_ifs(source, root)

    for stmt, body, orelse in _iter_implicit_if_elses(root):
        if stmt in sequential_similar_ifs:
            continue
        if isinstance(stmt.test, ast.NamedExpr):
            continue
        if (
            orelse
            and any(parsing.is_blocking(node) for node in body)
            and not any(parsing.is_blocking(node) for node in orelse)
        ):
            continue  # body is blocking but orelse is not
        if parsing.get_code(stmt, source).startswith("elif"):
            continue
        if _orelse_preferred_as_body(body, orelse):
            if orelse:
                replacements[stmt] = ast.If(
                    test=_negate_condition(stmt.test),
                    body=orelse,
                    orelse=[node for node in body if not isinstance(node, ast.Pass)],
                    lineno=stmt.lineno,
                )
                removals.update(orelse)
                break

    if replacements or removals:
        source = processing.alter_code(source, root, replacements=replacements, removals=removals)
        return _swap_explicit_if_else(source)

    return source


def swap_if_else(source: str) -> str:
    source = _swap_implicit_if_else(source)
    source = _swap_explicit_if_else(source)

    return source


@processing.fix
def early_return(source: str) -> str:

    root = parsing.parse(source)
    for funcdef in parsing.iter_funcdefs(root):
        if not parsing.match_template(funcdef.body[-2:], [ast.If, ast.Return(value=ast.Name)]):
            continue

        ret_stmt = funcdef.body[-1]
        if_stmt = funcdef.body[-2]

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
                parsing.match_template(node, ast.Assign(targets=[ast.Name(id=retval)]))
                for node in recursive_last_nonif_nodes
            )
        ):
            for node in recursive_last_nonif_nodes:
                yield node, ast.Return(value=node.value, lineno=node.lineno)

            yield ret_stmt, None


def _total_linenos(nodes: Iterable[ast.AST]) -> int:
    start_lineno = 1000_000
    end_lineno = 0
    for node in nodes:
        for child in parsing.walk(node, ast.AST(lineno=int, end_lineno=int)):
            start_lineno = min(start_lineno, child.lineno)
            end_lineno = max(end_lineno, child.end_lineno)

    return max(end_lineno - start_lineno, 0)


def early_continue(source: str) -> str:

    additions = []
    replacements = {}

    root = parsing.parse(source)
    blacklisted_ifs = _sequential_similar_ifs(source, root)

    for loop in parsing.walk(root, ast.For):
        stmt = loop.body[-1]
        if (
            isinstance(stmt, ast.If)
            and not isinstance(stmt.body[-1], ast.Continue)
            and stmt not in blacklisted_ifs
        ):
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
            elif (
                sum(
                    len(x.body) - 1
                    for x in itertools.chain([stmt], parsing.iter_bodies_recursive(stmt))
                )
                >= 3
                and _total_linenos(stmt.body) >= 5
                and not stmt.orelse
            ):
                replacements[stmt] = ast.If(
                    body=[ast.Continue()],
                    orelse=stmt.body,
                    test=_negate_condition(stmt.test),
                )

    source = processing.alter_code(
        source,
        root,
        additions=additions,
        replacements=replacements,
    )
    return source


@processing.fix
def remove_redundant_comprehensions(source: str) -> str:
    root = parsing.parse(source)

    for node in parsing.walk(root, (ast.DictComp, ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        wrapper = parsing.get_comp_wrapper_func_equivalent(node)

        if not isinstance(node, ast.DictComp):
            elt_generator_equal = (
                len(node.generators) == 1
                and ast.dump(node.elt).replace("Load", "__ctx__")
                == ast.dump(node.generators[0].target).replace("Store", "__ctx__")
                and (not node.generators[0].ifs)
            )
        else:
            elt = ast.Tuple(elts=[node.key, node.value], ctx=ast.Load())
            elt_generator_equal = (
                len(node.generators) == 1
                and ast.dump(elt).replace("Load", "__ctx__")
                == ast.dump(node.generators[0].target).replace("Store", "__ctx__")
                and (not node.generators[0].ifs)
            )

        if elt_generator_equal:
            yield node, ast.Call(
                func=ast.Name(id=wrapper, ctx=ast.Load()),
                args=[node.generators[0].iter],
                keywords=[],
            )


@processing.fix
def replace_functions_with_literals(source: str) -> str:

    root = parsing.parse(source)

    func_literal_template = ast.Call(
        func=ast.Name(id=parsing.Wildcard("func", ("list", "tuple", "dict"))), args=[], keywords=[]
    )
    for node, func in parsing.walk_wildcard(root, func_literal_template):
        if func == "list":
            yield node, ast.List(elts=[], ctx=ast.Load())
        elif func == "tuple":
            yield node, ast.Tuple(elts=[], ctx=ast.Load())
        elif func == "dict":
            yield node, ast.Dict(keys=[], values=[], ctx=ast.Load())

    func_comprehension_template = ast.Call(
        func=ast.Name(id=parsing.Wildcard("func", ("list", "tuple", "set", "iter"))),
        args=[
            parsing.Wildcard(
                "arg",
                (ast.List, ast.ListComp, ast.Tuple, ast.Set, ast.SetComp, ast.GeneratorExp),
            )
        ],
        keywords=[],
    )
    for node, arg, func in parsing.walk_wildcard(root, func_comprehension_template):
        if func == "list":
            if isinstance(arg, (ast.List, ast.ListComp)):
                yield node, arg
            elif isinstance(arg, ast.Tuple):
                yield node, ast.List(elts=arg.elts, ctx=arg.ctx)

        elif func == "tuple":
            if isinstance(arg, ast.Tuple):
                yield node, arg
            elif isinstance(arg, ast.List):
                yield node, ast.Tuple(elts=arg.elts, ctx=arg.ctx)

        elif func == "set":
            if isinstance(arg, (ast.Set, ast.SetComp)):
                yield node, arg
            elif isinstance(arg, (ast.Tuple, ast.List)):
                yield node, ast.Set(elts=arg.elts, ctx=arg.ctx)
            elif isinstance(arg, ast.GeneratorExp):
                yield node, ast.SetComp(elt=arg.elt, generators=arg.generators)

        elif func == "iter":
            if isinstance(arg, ast.GeneratorExp):
                yield node, arg


@processing.fix
def replace_for_loops_with_dict_comp(source: str) -> str:

    assign_template = ast.Assign(
        value=parsing.Wildcard("value", (ast.Dict, ast.DictComp)),
        targets=[ast.Name(id=parsing.Wildcard("target", str))],
    )

    root = parsing.parse(source)
    for (_, target, value), (n2,) in parsing.walk_sequence(
        root,
        assign_template,
        ast.For,
    ):
        body_node = n2
        generators = []

        while parsing.match_template(
            body_node, (ast.For(body=[object]), ast.If(body=[object], orelse=[]))
        ):
            if isinstance(body_node, ast.If):
                generators[-1].ifs.append(body_node.test)
            elif isinstance(body_node, ast.For):
                generators.append(
                    ast.comprehension(
                        target=body_node.target,
                        iter=body_node.iter,
                        ifs=[],
                        is_async=0,
                    )
                )
            else:
                raise RuntimeError(f"Unexpected type of node: {type(body_node)}")

            body_node = body_node.body[0]

        for comprehension in generators:
            if len(comprehension.ifs) > 1:
                comprehension.ifs = [ast.BoolOp(op=ast.And(), values=comprehension.ifs)]

        if not parsing.match_template(
            body_node, ast.Assign(targets=[ast.Subscript(value=ast.Name(id=target))])
        ):
            continue

        comp = ast.DictComp(
            key=body_node.targets[0].slice, value=body_node.value, generators=generators
        )
        if parsing.match_template(value, ast.Dict(values=[], keys=[])):
            yield value, comp
            yield n2, None
        elif parsing.match_template(value, ast.Dict(values=list, keys={None})):
            yield value, ast.Dict(keys=value.keys + [None], values=value.values + [comp])
            yield n2, None
        elif parsing.match_template(value, ast.Dict(values=list, keys=list)):
            yield value, ast.Dict(keys=[None, None], values=[value, comp])
            yield n2, None
        elif isinstance(value, ast.DictComp):
            yield value, ast.Dict(keys=[None, None], values=[value, comp])
            yield n2, None


@processing.fix
def replace_for_loops_with_set_list_comp(source: str) -> str:

    assign_template = ast.Assign(
        value=parsing.Wildcard("value", object),
        targets=[ast.Name(id=parsing.Wildcard("target", str))],
    )
    for_template = ast.For(body=[object])
    if_template = ast.If(body=[object], orelse=[])

    set_init_template = ast.Call(func=ast.Name(id="set"), args=[], keywords=[])
    list_init_template = ast.List(elts=[])  # list() should have been replaced by [] elsewhere.

    root = parsing.parse(source)
    for (_, target, value), (n2,) in parsing.walk_sequence(root, assign_template, for_template):
        body_node = n2
        generators = []

        while parsing.match_template(body_node, (for_template, if_template)):
            if isinstance(body_node, ast.If):
                generators[-1].ifs.append(body_node.test)
            elif isinstance(body_node, ast.For):
                generators.append(
                    ast.comprehension(
                        target=body_node.target,
                        iter=body_node.iter,
                        ifs=[],
                        is_async=0,
                    )
                )
            else:
                raise RuntimeError(f"Unexpected type of node: {type(body_node)}")

            body_node = body_node.body[0]

        for comprehension in generators:
            if len(comprehension.ifs) > 1:
                comprehension.ifs = [ast.BoolOp(op=ast.And(), values=comprehension.ifs)]

        target_alter_template = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=target), attr=parsing.Wildcard("attr", ("add", "append"))
                ),
                args=[object],
            )
        )

        augass_template = ast.AugAssign(op=(ast.Add, ast.Sub), target=ast.Name(id=target))

        if template_match := parsing.match_template(body_node, target_alter_template):
            if parsing.match_template(value, list_init_template) and (
                template_match.attr == "append"
            ):
                comp_type = ast.ListComp
            elif parsing.match_template(value, set_init_template) and (
                template_match.attr == "add"
            ):
                comp_type = ast.SetComp
            else:
                continue

            yield value, comp_type(
                elt=body_node.value.args[0],
                generators=generators,
            )
            yield n2, None

        elif parsing.match_template(body_node, augass_template):
            if isinstance(value, ast.List):
                replacement = ast.ListComp(
                    elt=body_node.value,
                    generators=generators,
                )
            else:
                comprehension = ast.GeneratorExp(
                    elt=body_node.value,
                    generators=generators,
                )
                replacement = ast.Call(func=ast.Name(id="sum"), args=[comprehension], keywords=[])

            try:
                if not parsing.literal_value(value):
                    if isinstance(body_node.op, ast.Sub):
                        replacement = ast.UnaryOp(op=body_node.op, operand=replacement)
                    yield value, replacement
                    yield n2, None
                    continue

            except ValueError:
                pass

            replacement = ast.BinOp(left=value, op=body_node.op, right=replacement)
            yield value, replacement
            yield n2, None


def inline_math_comprehensions(source: str) -> str:
    root = parsing.parse(source)

    replacements = {}
    blacklist = set()

    assign_template = ast.Assign(targets=[parsing.Wildcard("target", ast.Name)])
    augassign_template = ast.AugAssign(target=parsing.Wildcard("target", ast.Name))
    annassign_template = ast.AnnAssign(target=parsing.Wildcard("target", ast.Name))

    comprehension_assignments = [
        (assignment, target, assignment.value)
        for (assignment, target) in parsing.walk_wildcard(
            root, (assign_template, augassign_template, annassign_template)
        )
        if isinstance(assignment.value, (ast.GeneratorExp, ast.ListComp, ast.SetComp))
        or (
            isinstance(assignment.value, ast.Call)
            and isinstance(assignment.value.func, ast.Name)
            and (assignment.value.func.id in constants.ITERATOR_FUNCTIONS)
        )
    ]

    scope_types = (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
    for scope in parsing.walk(root, scope_types):
        for assignment, target, value in comprehension_assignments:
            uses = list(_get_uses_of(target, scope, source))
            if len(uses) != 1:
                blacklist.add(assignment)
                continue

            use = uses.pop()

            _, set_end_charno = parsing.get_charnos(value, source)
            use_start_charno, _ = parsing.get_charnos(use, source)

            # May be in a loop and the below dependency check won't be reliable.
            if use_start_charno < set_end_charno:
                blacklist.add(use)
                break

            # Check for references to any of the iterator's dependencies between set and use.
            # Perhaps some of these could be skipped, but I'm not sure that's a good idea.
            value_dependencies = tuple({node.id for node in parsing.walk(value, ast.Name)})
            for node in parsing.walk(scope, ast.Name(id=value_dependencies)):
                start, end = parsing.get_charnos(node, source)
                if set_end_charno < start <= end < use_start_charno:
                    blacklist.add(use)
                    break

            if use in blacklist:
                break

            for call in parsing.walk(
                scope, ast.Call(func=ast.Name(id=tuple(constants.MATH_FUNCTIONS)), args=[type(use)])
            ):
                if call.args[0] is use:
                    if use in replacements:
                        blacklist.add(use)
                    else:
                        replacements[use] = value
                    break

    for assignment in blacklist:
        if assignment in replacements:
            del replacements[assignment]

    source = processing.replace_nodes(source, replacements)

    return source


@processing.fix(restart_on_replace=True)
def simplify_transposes(source: str) -> str:
    root = parsing.parse(source)

    calls = parsing.walk(root, ast.Call)
    attributes = parsing.walk(root, ast.Attribute)

    for node in filter(
        parsing.is_transpose_operation,
        itertools.chain(calls, attributes),
    ):
        first_transpose_target = parsing.transpose_target(node)
        if parsing.is_transpose_operation(first_transpose_target):
            second_transpose_target = parsing.transpose_target(first_transpose_target)
            yield node, second_transpose_target


@processing.fix(restart_on_replace=True)
def remove_dead_ifs(source: str) -> str:
    root = parsing.parse(source)

    removals = set()
    replacements = {}

    for node in parsing.walk(root, (ast.If, ast.While, ast.IfExp)):
        try:
            value = parsing.literal_value(node.test)
        except ValueError:
            continue

        if isinstance(node, ast.While) and not value:
            yield node, None

        if isinstance(node, ast.IfExp):
            yield node, node.body if value else node.orelse

        if isinstance(node, ast.If):

            # Both body and orelse are dead => node is dead
            if value and not node.body:
                yield node, None

            if not value and not node.orelse:
                yield node, None

            if value and node.body:
                # Replace node with node.body, node.orelse is dead if exists
                remove = node.body

            elif not value and node.orelse:
                # Replace node with node.orelse, node.body is dead
                remove = node.orelse

            else:
                continue

            ranges = [parsing.get_charnos(child, source) for child in remove]
            start = min((s for (s, _) in ranges))
            end = max((e for (_, e) in ranges))
            indent = node.col_offset
            node_start, node_end = parsing.get_charnos(node, source)
            modified_body = " " * indent + re.sub("(?<![^\\n])    ", "", source[start:end]).lstrip()

            pre_else = source[:node_start]
            start_offset = len(pre_else) - len(pre_else.rstrip())

            yield processing.Range(
                node_start - start_offset, node_end
            ), "\n\n" + modified_body + "\n\n"


@processing.fix(restart_on_replace=True)
def delete_commented_code(source: str) -> str:
    matches = list(re.finditer(r"(?<![^\n])(\s*(#.*))+", source))
    root = parsing.parse(source)
    code_string_ranges = {
        parsing.get_charnos(node, source)
        for node in parsing.walk(root, (ast.Constant(value=str), ast.JoinedStr))
    }
    for commented_block in matches:
        start = commented_block.start()
        end = commented_block.end()
        start_offset = 0
        end_offset = 0

        line_lengths = [len(line) for line in commented_block.group().splitlines(keepends=True)]

        for si in range(len(line_lengths)):
            for se in range(len(line_lengths)):
                if si + se >= len(line_lengths):
                    continue

                start_offset = sum(line_lengths[:si]) if si > 0 else 0
                end_offset = sum(line_lengths[-se:]) if se > 0 else 0

                selection_end = end - end_offset
                selection_start = start + start_offset

                if any(
                    node_start <= selection_end and selection_start <= node_end
                    for node_start, node_end in code_string_ranges
                ):
                    continue

                uncommented_block = re.sub(
                    r"(?<![^\n])(\s*#)", "", source[start + start_offset : end - end_offset]
                )
                indentation_lengths = [
                    x.end() - x.start() for x in re.finditer("(?<![^\n]) +", uncommented_block)
                ]
                indent = min(indentation_lengths or [0])
                uncommented_block = re.sub(
                    r"(?<![^\n]) {" + str(indent) + "}", "", uncommented_block
                )

                if not (uncommented_block.strip() and parsing.is_valid_python(uncommented_block)):
                    continue

                parsed_content = parsing.parse(uncommented_block)
                if (
                    parsing.match_template(parsed_content.body, [ast.Expr])
                    and len(uncommented_block) < 20
                    and not isinstance(parsed_content.body[0].value, ast.Call)
                ):
                    continue

                if parsing.match_template(parsed_content.body, [ast.Name]):
                    continue

                # Magic comments should not be removed
                if any(
                    parsing.filter_nodes(parsed_content.body, ast.Expr(value=ast.Name(id="noqa")))
                ):
                    continue
                if any(
                    name.id in {"pylint", "mypy", "flake8", "noqa", "type"}
                    for annassign in parsing.walk(parsed_content, ast.AnnAssign)
                    for name in parsing.walk(annassign, ast.Name)
                ):
                    continue

                yield processing.Range(start + start_offset, end - end_offset), None


@processing.fix
def replace_with_filter(source: str) -> str:
    root = parsing.parse(source)

    for node in parsing.walk(root, ast.For):
        if parsing.is_call(node.iter, ("filter", "filterfalse", "itertools.filterfalse")):
            continue
        if parsing.match_template(node.body, [ast.If]):
            test = node.body[0].test
            negative = isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
            if negative:
                test = test.operand
                if not parsing.match_template(node.body[0].body, {ast.Continue}):
                    continue
            else:
                if not parsing.match_template(node.body[0].orelse, {ast.Continue}):
                    continue
            if isinstance(node.target, ast.Name) and parsing.match_template(
                test, ast.Call(args=[ast.Name(id=node.target.id)], keywords=[])
            ):
                yield node.iter, ast.Call(
                    func=ast.Name(id="filter"),
                    args=[test.func, node.iter],
                    keywords=[],
                )
                yield node.body[0].test, ast.Constant(value=not negative, kind=None)
            elif isinstance(test, ast.Name) and parsing.match_template(
                node.target, ast.Name(id=test.id)
            ):
                yield node.iter, ast.Call(
                    func=ast.Name(id="filter"),
                    args=[ast.Constant(value=None, kind=None), node.iter],
                    keywords=[],
                )
                yield node.body[0].test, ast.Constant(value=not negative, kind=None)
            continue
        if len(node.body) < 2:
            continue
        first_node, second_node, *_ = node.body
        if parsing.match_template(first_node, ast.If(orelse=[])) and isinstance(
            first_node.body[0], ast.Continue
        ):
            test = first_node.test

            if (
                isinstance(node.target, ast.Name)
                and len(node.body) >= 2
                and parsing.match_template(
                    test,
                    ast.UnaryOp(
                        op=ast.Not,
                        operand=ast.Call(keywords=[], args=[ast.Name(id=node.target.id)]),
                    ),
                )
                and not parsing.match_template(
                    second_node,
                    ast.If(body=[ast.Continue], test=ast.UnaryOp(op=ast.Not, operand=ast.Call)),
                )
            ):
                yield node.iter, ast.Call(
                    func=ast.Name(id="filter"),
                    args=[test.operand.func, node.iter],
                    keywords=[],
                )
                yield first_node, None


def _get_contains_args(node: ast.Compare) -> Tuple[str, str, bool]:
    negative = isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
    if negative:
        node = node.operand

    template = ast.Compare(
        left=ast.Name(id=parsing.Wildcard(name="key", template=str)),
        ops=[(ast.In, ast.NotIn)],
        comparators=[ast.Name(id=parsing.Wildcard(name="value", template=str))],
    )

    if template_match := parsing.match_template(node, template):
        _, key, value = template_match
        if isinstance(node.ops[0], ast.In):
            return key, value, negative
        if isinstance(node.ops[0], ast.NotIn):
            return key, value, not negative

    raise ValueError(f"Node is not a pure compare node: {node}")


def _get_subscript_functions(node: ast.Expr) -> Tuple[str, str, str, str]:
    slice_value = (
        ast.Name(id=parsing.Wildcard("key", str))
        if constants.PYTHON_VERSION >= (3, 9)
        else ast.Index(value=ast.Name(id=parsing.Wildcard("key", str)))
    )
    template = ast.Expr(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Subscript(
                    value=ast.Name(id=parsing.Wildcard("obj", str)),
                    slice=slice_value,
                ),
                attr=parsing.Wildcard("call", str),
            ),
            args=[parsing.Wildcard("value", object)],
        )
    )
    if template_match := parsing.match_template(node, template):
        _, call, key, obj, value = template_match
        return obj, call, key, value

    raise ValueError(f"Node {node} is not a subscript call")


def _get_assign_functions(node: ast.Expr) -> Tuple[str, str]:
    slice_value = (
        ast.Name(id=parsing.Wildcard("key", str))
        if constants.PYTHON_VERSION >= (3, 9)
        else ast.Index(value=ast.Name(id=parsing.Wildcard("key", str)))
    )
    template = ast.Assign(
        targets=[ast.Subscript(slice=slice_value, value=ast.Name(id=parsing.Wildcard("obj", str)))]
    )
    if template_match := parsing.match_template(node, template):
        _, key, obj = template_match
        value = node.value
        return obj, key, value

    raise ValueError(f"Node {node} is not a subscript assignment")


def _preferred_comprehension_type(node: ast.AST) -> Union[ast.AST, ast.SetComp, ast.GeneratorExp]:
    if isinstance(node, ast.ListComp):
        return ast.GeneratorExp(elt=node.elt, generators=node.generators)

    return node


@processing.fix
def implicit_defaultdict(source: str) -> str:
    replacements = {}
    removals = set()

    assign_template = ast.Assign(
        targets=[parsing.Wildcard("target", ast.Name)],
        value=parsing.Wildcard("value", ast.Dict(keys=[], values=[])),
    )
    if_template = ast.If(body=[object], orelse=[])

    root = parsing.parse(source)
    for (_, target, value), (n2,) in parsing.walk_sequence(root, assign_template, ast.For):
        loop_replacements = {}
        loop_removals = set()
        subscript_calls = set()
        consistent = True

        for (condition,), (append,) in parsing.walk_sequence(n2, if_template, ast.Expr):
            try:
                (key, obj, negative) = _get_contains_args(condition.test)
                (f_obj, f_key, f_value) = _get_assign_functions(condition.body[0])
                (t_obj, t_call, t_key, _) = _get_subscript_functions(append)
            except ValueError:
                continue
            if obj != target.id:
                continue
            if not negative:
                continue
            if not (t_obj == f_obj == obj and t_key == f_key == key):
                continue

            subscript_calls.add(t_call)
            if parsing.match_template(f_value, ast.List(elts=[])) and (
                t_call in {"append", "extend"}
            ):
                loop_removals.add(condition)
                continue
            if parsing.match_template(f_value, ast.Call(func=ast.Name(id="set"), args=[])) and (
                t_call in {"add", "update"}
            ):
                loop_removals.add(condition)
                continue
            consistent = False
            break

        if_orelse_template = ast.If(body=[object], orelse=[object])
        for condition in parsing.walk(ast.Module(body=n2.body), if_orelse_template):
            if condition in loop_replacements:
                continue

            try:
                key, obj, negative = _get_contains_args(condition.test)
            except ValueError:
                continue
            if obj != target.id:
                continue
            on_true = condition.body[0]
            on_false = condition.orelse[0]
            if negative:
                on_true, on_false = (on_false, on_true)
            try:
                t_obj, t_call, t_key, t_value = _get_subscript_functions(on_true)
                f_obj, f_key, f_value = _get_assign_functions(on_false)
            except ValueError:
                continue
            if not t_obj == f_obj == obj or not t_key == f_key == key:
                continue

            subscript_calls.add(t_call)
            if (
                t_call in {"add", "append"}
                and parsing.match_template(
                    f_value, (ast.List(elts=[object]), ast.Set(elts=[object]))
                )
                and (parsing.unparse(t_value) == parsing.unparse(f_value.elts[0]))
            ):
                if isinstance(f_value, ast.List) == (t_call == "append"):
                    loop_replacements[condition] = on_true
                    continue
                consistent = False
                break
            t_value_preferred = _preferred_comprehension_type(t_value)
            f_value_preferred = _preferred_comprehension_type(f_value)
            if parsing.unparse(t_value_preferred) == parsing.unparse(
                f_value_preferred
            ) and t_call in {"update", "extend"}:
                loop_replacements[condition] = on_true
                continue

        if not consistent:
            continue

        if subscript_calls and subscript_calls <= {"add", "update"}:
            yield from loop_replacements.items()
            yield from zip(loop_removals, itertools.repeat(None))
            yield value, ast.Call(
                func=ast.Attribute(value=ast.Name(id="collections"), attr="defaultdict"),
                args=[ast.Name(id="set")],
                keywords=[],
            )

        if subscript_calls and subscript_calls <= {"append", "extend"}:
            yield from loop_replacements.items()
            yield from zip(loop_removals, itertools.repeat(None))
            yield value, ast.Call(
                func=ast.Attribute(value=ast.Name(id="collections"), attr="defaultdict"),
                args=[ast.Name(id="list")],
                keywords=[],
            )


@processing.fix
def simplify_redundant_lambda(source: str) -> str:
    root = parsing.parse(source)

    template = ast.Lambda(
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=(
            ast.Call(args=[], keywords=[]),
            ast.List(elts=[]),
            ast.Tuple(elts=[]),
            ast.Dict(keys=[], values=[]),
        ),
    )

    for node in parsing.walk(root, template):
        if isinstance(node.body, ast.Call):
            yield node, node.body.func
        elif isinstance(node.body, ast.List):
            yield node, ast.Name(id="list")
        elif isinstance(node.body, ast.Tuple):
            yield node, ast.Name(id="tuple")
        elif isinstance(node.body, ast.Dict):
            yield node, ast.Name(id="dict")


def _is_same_code(*nodes: ast.AST) -> bool:
    return len({parsing.unparse(node) for node in nodes}) == 1


def _all_branches(
    *starting_nodes: ast.AST, expand_ifs_on: Literal["start", "end"]
) -> Iterable[ast.AST]:
    if not starting_nodes:
        raise ValueError("At least one node must be provided.")
    if expand_ifs_on == "start":
        index = 0
    elif expand_ifs_on == "end":
        index = -1
    else:
        raise ValueError(f"Expected 'start' or 'end', not {expand_ifs_on!r}.")
    for node in starting_nodes:
        if isinstance(node, ast.If):
            # If either is empty, this generates an IndexError
            yield from _all_branches(node.body[index], expand_ifs_on=expand_ifs_on)
            yield from _all_branches(node.orelse[index], expand_ifs_on=expand_ifs_on)

        else:
            yield node


def _move_before_scope(
    scope: ast.AST,
    nodes: Iterable[ast.AST],
) -> Tuple[Collection[ast.AST], Collection[ast.AST]]:
    removals = set(nodes)
    first_node = min(removals, key=lambda node: node.lineno)
    replacement = copy.copy(first_node)
    replacement.lineno = scope.lineno - 1
    replacement.col_offset = scope.col_offset
    additions = {replacement}

    return additions, removals


def _move_after_scope(
    scope: ast.AST,
    nodes: Iterable[ast.AST],
) -> Tuple[Collection[ast.AST], Collection[ast.AST]]:
    removals = set(nodes)
    last_node = max(removals, key=lambda node: node.lineno)
    replacement = copy.copy(last_node)
    replacement.col_offset = scope.col_offset
    additions = {replacement}

    return additions, removals


def breakout_common_code_in_ifs(source: str) -> str:

    root = parsing.parse(source)

    for node in parsing.walk(root, ast.If):
        if parsing.get_code(node, source).startswith("elif"):
            continue

        if node.body and node.orelse:
            first_body = node.body[0]
            first_orelse = node.orelse[0]
            last_body = node.body[-1]
            last_orelse = node.orelse[-1]

            removals = set()
            additions = set()

            # If := is used, it may (this is quite probable, even) set a variable that is needed in the
            # body or orelse.
            # I suppose there could be some other side effect going on in the test, but in that case I
            # would consider 99.9% to be robust enough.
            has_namedexpr = any(parsing.walk(node.test, ast.NamedExpr))

            start_branches = [node.body[0], node.orelse[0]]
            end_branches = [node.body[-1], node.orelse[-1]]

            if not has_namedexpr and _is_same_code(*start_branches):
                additions, removals = _move_before_scope(node, start_branches)

            elif _is_same_code(*end_branches):
                additions, removals = _move_after_scope(node, end_branches)

            try:
                start_branches = list(
                    _all_branches(node.body[0], node.orelse[0], expand_ifs_on="start")
                )
                end_branches = list(
                    _all_branches(node.body[-1], node.orelse[-1], expand_ifs_on="end")
                )
            except (ValueError, IndexError):
                pass
            else:
                if not has_namedexpr and _is_same_code(*start_branches):
                    additions, removals = _move_before_scope(node, start_branches)

                elif _is_same_code(*end_branches):
                    additions, removals = _move_after_scope(node, end_branches)

                else:
                    end_nonblocking_branches = [
                        branch for branch in end_branches if not parsing.is_blocking(branch)
                    ]
                    count = len(end_nonblocking_branches)
                    if count >= 2 and _is_same_code(*end_nonblocking_branches):
                        additions, removals = _move_after_scope(node, end_nonblocking_branches)

            if parsing.match_template(list(additions), [ast.Pass]):
                continue

            if additions and removals:
                source = processing.alter_code(source, root, additions=additions, removals=removals)
                return breakout_common_code_in_ifs(source)

    return source


@processing.fix
def invalid_escape_sequence(source: str) -> str:
    """Prepend 'r' to invalid escape sequences

    Args:
        source (str): Python source code

    Returns:
        str: Modified source code
    """
    # Recognized esc sequences from python.org documentation, Jan 2023
    # https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals
    valid_escape_sequences = (
        r"\\",
        r"\'",
        r"\"",
        r"\a",
        r"\b",
        r"\f",
        r"\n",
        r"\r",
        r"\t",
        r"\v",
        r"\ooo",
        r"\xhh",
        r"\N",
        r"\u",
        r"\U",
    )

    root = parsing.parse(source)

    for node in parsing.walk(root, ast.Constant(value=str)):
        code = parsing.get_code(node, source)
        # Normal string containing backslash but no valid escape sequences
        if (
            code[0] in "'\""
            and "\\" in code
            and not any(sequence in code for sequence in valid_escape_sequences)
        ):
            yield node, "r" + code


@processing.fix
def replace_filter_lambda_with_comp(source: str) -> str:
    """Replace filter(lambda ..., iterable) with equivalent list comprehension

    Args:
        source (str): Python source code

    Returns:
        str: Modified source code
    """
    root = parsing.parse(source)

    filter_template = ast.Name(id="filter")
    filterfalse_template = (
        ast.Name(id="filterfalse"),
        ast.Attribute(
            value=ast.Name(id="itertools"),
            attr="filterfalse",
        ),
    )

    template = ast.Call(
        func=parsing.Wildcard("func", (filter_template, filterfalse_template)),
        args=[
            ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=parsing.Wildcard("args", list),
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=parsing.Wildcard("condition", object),
            ),
            parsing.Wildcard("iterable", object),
        ],
        keywords=[],
    )

    blacklist = {
        iterator
        for _, iterator in parsing.walk_wildcard(
            root, ast.For(iter=parsing.Wildcard("iterator", object))
        )
    }

    for node, args, condition, func, iterable in parsing.walk_wildcard(root, template):
        if node in blacklist:
            continue
        if not args:
            continue
        if parsing.match_template(func, filterfalse_template):
            condition = _negate_condition(condition)
        args = ast.Tuple(elts=args) if len(args) > 1 else args[0]
        replacement_node = ast.GeneratorExp(
            elt=args,
            generators=[ast.comprehension(target=args, iter=iterable, ifs=[condition], is_async=0)],
        )

        yield node, replacement_node


@processing.fix
def replace_map_lambda_with_comp(source: str) -> str:
    """Replace map(lambda ..., iterable) with equivalent list comprehension

    Args:
        source (str): Python source code

    Returns:
        str: Modified source code
    """
    root = parsing.parse(source)

    template = ast.Call(
        func=ast.Name(id="map"),
        args=[
            ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=parsing.Wildcard("args", list),
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=parsing.Wildcard("body", object),
            ),
            parsing.Wildcard("iterable", object),
        ],
        keywords=[],
    )

    blacklist = {
        iterator
        for _, iterator in parsing.walk_wildcard(
            root, ast.For(iter=parsing.Wildcard("iterator", object))
        )
    }

    for node, args, body, iterable in parsing.walk_wildcard(root, template):
        if node in blacklist:
            continue
        if not args:
            continue
        args = ast.Tuple(elts=args) if len(args) > 1 else args[0]
        replacement_node = ast.GeneratorExp(
            elt=body, generators=[ast.comprehension(target=args, iter=iterable, ifs=[], is_async=0)]
        )

        yield node, replacement_node

@processing.fix(restart_on_replace=True)
def merge_chained_comps(source: str) -> str:

    root = parsing.parse(source)

    template = ast.AST(
        elt=object,
        generators=[
            ast.comprehension(
                target=parsing.Wildcard("common_target", object),
                iter=ast.AST(
                    elt=parsing.Wildcard("common_target", object),
                    generators=[
                        ast.comprehension(
                        target=parsing.Wildcard("common_target", object),
                        iter=parsing.Wildcard("iter_inner", object),
                        ifs=parsing.Wildcard("ifs_inner", list),
                        is_async=0)
                    ]
                ),
                ifs=parsing.Wildcard("ifs_outer", list),
                is_async=0)
        ])

    for template_match in parsing.walk_wildcard(root, template):
        if type(template_match.root) is not type(template_match.root.generators[0].iter):
            continue
        if not isinstance(template_match.root, (ast.SetComp, ast.GeneratorExp, ast.ListComp)):
            continue

        replacement = type(template_match.root)(
            elt=template_match.root.elt,
            generators=[
                ast.comprehension(
                    target=template_match.common_target,
                    iter=template_match.iter_inner,
                    ifs=template_match.ifs_inner + template_match.ifs_outer,
                    is_async=0
                )
            ])

        yield template_match.root, replacement


@processing.fix(restart_on_replace=True)
def remove_redundant_comprehension_casts(source: str) -> str:
    root = parsing.parse(source)

    template = ast.Call(
        func=ast.Name(id=parsing.Wildcard("func", ("list", "set", "iter"))),
        args=[parsing.Wildcard("comp", (ast.GeneratorExp, ast.ListComp, ast.SetComp))],
        keywords=[]
    )

    for node, comp, func in parsing.walk_wildcard(root, template):
        if func == "set":
            yield node, ast.SetComp(comp.elt, comp.generators)
        if func == "list" and not isinstance(comp, ast.SetComp):
            yield node, ast.ListComp(comp.elt, comp.generators)
        if func == "iter" and isinstance(comp, ast.GeneratorExp):
            yield node, comp
        if func == "iter" and isinstance(comp, ast.ListComp):
            yield ast.GeneratorExp(comp.elt, comp.generators)

    template = ast.Call(
        func=ast.Name(id=parsing.Wildcard("func", ("list", "set", "iter", "dict"))),
        args=[parsing.Wildcard("comp", (ast.DictComp))],
        keywords=[]
    )

    for node, comp, func in parsing.walk_wildcard(root, template):
        equivalent_setcomp = ast.SetComp(comp.key, comp.generators)
        if func == "dict":
            yield node, comp
        if func == "set":
            yield node, equivalent_setcomp
        if func in ("list", "iter"):
            yield node, ast.Call(
                func=ast.Name(id=func),
                args=[equivalent_setcomp],
                keywords=[]
            )
