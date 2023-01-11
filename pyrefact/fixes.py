import ast
import collections
import itertools
import queue
import re
from typing import Collection, Iterable, List, Mapping, Sequence, Tuple, Union

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

    if variable.startswith("__") and variable.endswith("__"):
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
    if isinstance(node, ast.Name):
        name = node.id
        start = (node.lineno, node.col_offset)
        end = (node.end_lineno, node.end_col_offset)
    elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.FunctionDef)):
        name = node.name
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


def fix_isort(content: str, *, line_length: int = 100) -> str:
    """Format source code with isort

    Args:
        content (str): Python source code
        line_length (int, optional): Line length. Defaults to 100.

    Returns:
        str: Source code, formatted with isort
    """
    return isort.code(content, config=isort.Config(profile="black", line_length=line_length))


def fix_line_lengths(content: str, *, max_line_length: int = 100) -> str:

    root = parsing.parse(content)

    replacements = {}
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
            child.end_col_offset
            for child in parsing.walk(node, ast.AST(end_col_offset=int))
        )
        if node in formatted_nodes or max_node_line_length <= max_line_length:
            continue

        start, end = parsing.get_charnos(node, content, keep_first_indent=True)

        current_code = content[start:end]
        new_code = processing.format_with_black(current_code, line_length=max_line_length)

        if new_code != current_code and (
            not any((e >= start and s <= end for s, e in formatted_ranges))
        ):
            replacements[node] = new_code
            formatted_ranges.add((start, end))

    content = processing.replace_nodes(content, replacements)

    return content


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
        return set(parsing.walk(node.target, ast.Name(ctx=ast.Store)))
    if isinstance(node, ast.Assign):
        for target in node.targets:
            targets.update(parsing.walk(target, ast.Name(ctx=ast.Store)))
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
                (loop_start, loop_end) = parsing.get_charnos(containing_loop_node, content)
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
        if target_names != {"_"}:
            continue

        (start_charno, end_charno) = parsing.get_charnos(node, content)
        code = parsing.get_code(node, content)
        changed_code = re.sub(_REDUNDANT_UNDERSCORED_ASSIGN_RE_PATTERN, "", code)
        if code != changed_code:
            print(f"Removing redundant assignments in {code}")
            content = content[:start_charno] + changed_code + content[end_charno:]

    return content


def _is_pointless_string(node: ast.AST) -> bool:
    """Check if an AST is a pointless string statement.

    This is useful for figuring out if a node is a docstring.

    Args:
        node (ast.AST): AST to check

    Returns:
        bool: True if the node is a pointless string statement.
    """
    return parsing.match_template(node, ast.Expr(value=ast.Constant(value=str)))


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
            print(f"{def_node.name} is never used")
            yield def_node

    for def_node in classdefs:
        usages = name_usages[def_node.name]
        internal_usages = set(
            parsing.walk(def_node, ast.Name(ctx=ast.Load, id=(def_node.name, "self", "cls")))
        )
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
        content = processing.remove_nodes(content, delete, root)

    return content


def _get_package_names(node: Union[ast.Import, ast.ImportFrom]):
    if isinstance(node, ast.ImportFrom):
        return [node.module]

    return [alias.name for alias in node.names]


def move_imports_to_toplevel(content: str) -> str:
    root = parsing.parse(content)
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
        content = processing.alter_code(content, root, removals=removals, additions=additions)

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


def _de_indent_from_else(content: str, orelse: Sequence[ast.AST]) -> str:
    ranges = [parsing.get_charnos(child, content) for child in orelse]
    start = min((s for (s, _) in ranges))
    end = max((e for (_, e) in ranges))
    last_else = list(re.finditer("(?<![^\\n]) *else: *\\n?", content[:start]))[-1]
    indent = len(re.findall("^ *", last_else.group())[0])
    modified_pre_else = content[: last_else.start()].rstrip() + "\n\n"
    modified_orelse = " " * indent + re.sub("(?<![^\\n])    ", "", content[start:end]).lstrip()
    content = modified_pre_else + modified_orelse + content[end:]

    return content


def _de_indent_body(content: str, node: ast.AST, body: Sequence[ast.AST]) -> str:
    ranges = [parsing.get_charnos(child, content) for child in body]
    start = min((s for (s, _) in ranges))
    end = max((e for (_, e) in ranges))
    indent = node.col_offset
    node_start, node_end = parsing.get_charnos(node, content)
    modified_pre = content[:node_start].rstrip() + "\n\n"
    modified_body = " " * indent + re.sub("(?<![^\\n])    ", "", content[start:end]).lstrip()
    modified_post = "\n\n" + content[node_end:]
    content = modified_pre + modified_body + modified_post

    return content


def remove_redundant_else(content: str) -> str:
    """Remove redundante else and elif statements in code.

    Args:
        content (str): Python source code

    Returns:
        str: Code with no redundant else/elifs.
    """
    root = parsing.parse(content)
    for node in parsing.walk(root, ast.If):
        if not node.orelse:
            continue
        if not parsing.get_code(node, content).startswith("if"):  # Otherwise we get FPs on elif
            continue
        if not any((parsing.is_blocking(child) for child in node.body)):
            continue

        if parsing.match_template(node.orelse, [ast.If]):
            (start, end) = parsing.get_charnos(node.orelse[0], content)
            orelse = content[start:end]
            if orelse.startswith("elif"):  # Regular elif
                modified_orelse = re.sub("^elif", "if", orelse)
                print("Found redundant elif:")
                print(parsing.get_code(node, content))
                content = content[:start] + modified_orelse + content[end:]
                continue

            # Otherwise it's an else: if:, which is handled below

        # else
        print("Found redundant else:")
        print(parsing.get_code(node, content))

        content = _de_indent_from_else(content, node.orelse)
        return remove_redundant_else(content)

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
            replacements[node] = ast.Compare(
                left=node.left,
                ops=operators,
                comparators=node.comparators,
            )

    if replacements:
        content = processing.replace_nodes(content, replacements)

    return content


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
    for (condition, ), *implicit_orelse in parsing.walk_sequence(
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


def _sequential_similar_ifs(content: str, root: ast.AST) -> Collection[ast.If]:
    return set.union(
        set(),
        *map(set, parsing.iter_similar_nodes(root, content, ast.If, count=2, length=15)),
        *map(set, parsing.iter_similar_nodes(root, content, ast.If, count=3, length=10)),
    )


def _swap_explicit_if_else(content: str) -> str:
    replacements = {}

    root = parsing.parse(content)
    sequential_similar_ifs = _sequential_similar_ifs(content, root)

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
        if parsing.get_code(stmt, content).startswith("elif"):
            continue
        if _orelse_preferred_as_body(body, orelse):
            if orelse:
                replacements[stmt] = ast.If(
                    test=_negate_condition(stmt.test),
                    body=orelse,
                    orelse=[node for node in body if not isinstance(node, ast.Pass)],
                    lineno=stmt.lineno,
                )
                break

    if replacements:
        content = processing.replace_nodes(content, replacements)
        return _swap_explicit_if_else(content)

    return content


def _swap_implicit_if_else(content: str) -> str:
    replacements = {}
    removals = set()

    root = parsing.parse(content)
    sequential_similar_ifs = _sequential_similar_ifs(content, root)

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
        if parsing.get_code(stmt, content).startswith("elif"):
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
        content = processing.alter_code(content, root, replacements=replacements, removals=removals)
        return _swap_explicit_if_else(content)

    return content


def swap_if_else(content: str) -> str:
    content = _swap_implicit_if_else(content)
    content = _swap_explicit_if_else(content)

    return content


def early_return(content: str) -> str:

    replacements = {}
    removals = []

    root = parsing.parse(content)
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
                replacements[node] = ast.Return(value=node.value, lineno=node.lineno)
            removals.append(ret_stmt)

    content = processing.alter_code(
        content,
        root,
        replacements=replacements,
        removals=removals,
    )

    return content


def _total_linenos(nodes: Iterable[ast.AST]) -> int:
    start_lineno = 1000_000
    end_lineno = 0
    for node in nodes:
        for child in parsing.walk(node, ast.AST(lineno=int, end_lineno=int)):
            start_lineno = min(start_lineno, child.lineno)
            end_lineno = max(end_lineno, child.end_lineno)

    return max(end_lineno - start_lineno, 0)


def early_continue(content: str) -> str:

    additions = []
    replacements = {}

    root = parsing.parse(content)
    blacklisted_ifs = _sequential_similar_ifs(content, root)

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

    content = processing.alter_code(
        content,
        root,
        additions=additions,
        replacements=replacements,
    )

    return content


def remove_redundant_comprehensions(content: str) -> str:
    root = parsing.parse(content)

    replacements = {}

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
            replacements[node] = ast.Call(
                func=ast.Name(id=wrapper, ctx=ast.Load()),
                args=[node.generators[0].iter],
                keywords=[],
            )

    content = processing.replace_nodes(content, replacements)

    return content


def replace_functions_with_literals(content: str) -> str:

    root = parsing.parse(content)
    replacements = {}

    func_literal_template = ast.Call(
        func=ast.Name(id=parsing.Wildcard("func", ("list", "tuple", "dict"))), args=[], keywords=[]
    )
    for node, func in parsing.walk_wildcard(root, func_literal_template):
        if func == "list":
            replacements[node] = ast.List(elts=[], ctx=ast.Load())
        elif func == "tuple":
            replacements[node] = ast.Tuple(elts=[], ctx=ast.Load())
        elif func == "dict":
            replacements[node] = ast.Dict(keys=[], values=[], ctx=ast.Load())

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
                replacements[node] = arg
            elif isinstance(arg, ast.Tuple):
                replacements[node] = ast.List(elts=arg.elts, ctx=arg.ctx)

        elif func == "tuple":
            if isinstance(arg, ast.Tuple):
                replacements[node] = arg
            elif isinstance(arg, ast.List):
                replacements[node] = ast.Tuple(elts=arg.elts, ctx=arg.ctx)

        elif func == "set":
            if isinstance(arg, (ast.Set, ast.SetComp)):
                replacements[node] = arg
            elif isinstance(arg, (ast.Tuple, ast.List)):
                replacements[node] = ast.Set(elts=arg.elts, ctx=arg.ctx)
            elif isinstance(arg, ast.GeneratorExp):
                replacements[node] = ast.SetComp(elt=arg.elt, generators=arg.generators)

        elif func == "iter":
            if isinstance(arg, ast.GeneratorExp):
                replacements[node] = arg

    content = processing.replace_nodes(content, replacements)

    return content


def replace_for_loops_with_dict_comp(content: str) -> str:
    replacements = {}
    removals = set()

    assign_template = ast.Assign(
        value=parsing.Wildcard("value", (ast.Dict, ast.DictComp)),
        targets=[ast.Name(id=parsing.Wildcard("target", str))]
    )

    root = parsing.parse(content)
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
            replacements[value] = comp
            removals.add(n2)
        elif parsing.match_template(value, ast.Dict(values=list, keys={None})):
            replacements[value] = ast.Dict(
                keys=value.keys + [None], values=value.values + [comp]
            )
            removals.add(n2)
        elif parsing.match_template(value, ast.Dict(values=list, keys=list)):
            replacements[value] = ast.Dict(keys=[None, None], values=[value, comp])
            removals.add(n2)
        elif isinstance(value, ast.DictComp):
            replacements[value] = ast.Dict(keys=[None, None], values=[value, comp])
            removals.add(n2)

    content = processing.alter_code(content, root, removals=removals, replacements=replacements)

    return content


def replace_for_loops_with_set_list_comp(content: str) -> str:
    replacements = {}
    removals = set()

    assign_template = ast.Assign(
        value=parsing.Wildcard("value", object),
        targets=[ast.Name(id=parsing.Wildcard("target", str))]
    )
    for_template = ast.For(body=[object])
    if_template = ast.If(body=[object], orelse=[])

    set_init_template = ast.Call(func=ast.Name(id="set"), args=[], keywords=[])
    list_init_template = ast.List(elts=[])  # list() should have been replaced by [] elsewhere.

    root = parsing.parse(content)
    for (n1, target, value), (n2,) in parsing.walk_sequence(root, assign_template, for_template):
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
                func=ast.Attribute(value=ast.Name(id=target), attr=parsing.Wildcard("attr", ("add", "append"))),
                args=[object],
            )
        )

        augass_template = ast.AugAssign(op=(ast.Add, ast.Sub), target=ast.Name(id=target))

        if template_match := parsing.match_template(body_node, target_alter_template):
            if parsing.match_template(value, list_init_template) and (template_match.attr == "append"):
                comp_type = ast.ListComp
            elif parsing.match_template(value, set_init_template) and (template_match.attr == "add"):
                comp_type = ast.SetComp
            else:
                continue

            replacements[value] = comp_type(
                elt=body_node.value.args[0],
                generators=generators,
            )
            removals.add(n2)

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
                    replacements[value] = replacement
                    removals.add(n2)
                    continue

            except ValueError:
                pass

            replacement = ast.BinOp(left=value, op=body_node.op, right=replacement)
            replacements[value] = replacement
            removals.add(n2)

    content = processing.alter_code(content, root, removals=removals, replacements=replacements)

    return content


def inline_math_comprehensions(content: str) -> str:
    root = parsing.parse(content)

    replacements = {}
    blacklist = set()

    assign_template = ast.Assign(targets=[parsing.Wildcard("target", ast.Name)])
    augassign_template = ast.AugAssign(target=parsing.Wildcard("target", ast.Name))
    annassign_template = ast.AnnAssign(target=parsing.Wildcard("target", ast.Name))

    comprehension_assignments = []
    for assignment, target in parsing.walk_wildcard(root, (assign_template, augassign_template, annassign_template)):
        if isinstance(assignment.value, (ast.GeneratorExp, ast.ListComp, ast.SetComp)) or (
            isinstance(assignment.value, ast.Call)
            and isinstance(assignment.value.func, ast.Name)
            and assignment.value.func.id in constants.ITERATOR_FUNCTIONS
        ):
            comprehension_assignments.append((assignment, target, assignment.value))

    scope_types = (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
    for scope in parsing.walk(root, scope_types):
        for assignment, target, value in comprehension_assignments:
            uses = list(_get_uses_of(target, scope, content))
            if len(uses) != 1:
                blacklist.add(assignment)
                continue

            use = uses.pop()

            _, set_end_charno = parsing.get_charnos(value, content)
            use_start_charno, _ = parsing.get_charnos(use, content)

            # May be in a loop and the below dependency check won't be reliable.
            if use_start_charno < set_end_charno:
                blacklist.add(use)
                break

            # Check for references to any of the iterator's dependencies between set and use.
            # Perhaps some of these could be skipped, but I'm not sure that's a good idea.
            value_dependencies = tuple({node.id for node in parsing.walk(value, ast.Name)})
            for node in parsing.walk(scope, ast.Name(id=value_dependencies)):
                start, end = parsing.get_charnos(node, content)
                if set_end_charno < start <= end < use_start_charno:
                    blacklist.add(use)
                    break

            if use in blacklist:
                break

            for call in parsing.walk(scope, ast.Call(func=ast.Name(id=tuple(constants.MATH_FUNCTIONS)), args=[type(use)])):
                if call.args[0] is use:
                    if use in replacements:
                        blacklist.add(use)
                    else:
                        replacements[use] = value
                    break

    for assignment in blacklist:
        if assignment in replacements:
            del replacements[assignment]

    content = processing.replace_nodes(content, replacements)

    return content


def simplify_transposes(content: str) -> str:
    root = parsing.parse(content)

    replacements = {}

    calls = parsing.walk(root, ast.Call)
    attributes = parsing.walk(root, ast.Attribute)

    for node in filter(
        parsing.is_transpose_operation,
        itertools.chain(calls, attributes),
    ):
        first_transpose_target = parsing.transpose_target(node)
        if parsing.is_transpose_operation(first_transpose_target):
            second_transpose_target = parsing.transpose_target(first_transpose_target)
            replacements[node] = second_transpose_target
            break

    content = processing.replace_nodes(content, replacements)
    if replacements:
        return simplify_transposes(content)

    return content


def remove_dead_ifs(content: str) -> str:
    root = parsing.parse(content)

    removals = set()
    replacements = {}

    for node in parsing.walk(root, (ast.If, ast.While, ast.IfExp)):
        try:
            value = parsing.literal_value(node.test)
        except ValueError:
            continue

        if isinstance(node, ast.While) and not value:
            removals.add(node)

        if isinstance(node, ast.IfExp):
            replacements[node] = node.body if value else node.orelse

        if isinstance(node, ast.If):
            if value and node.body:
                # Replace node with node.body, node.orelse is dead if exists
                content = _de_indent_body(content, node, node.body)
                return remove_dead_ifs(content)
            if not value and node.orelse:
                # Replace node with node.orelse, node.body is dead
                content = _de_indent_body(content, node, node.orelse)
                return remove_dead_ifs(content)

            # Both body and orelse are dead => node is dead
            if value and not node.body:
                removals.add(node)
            elif not value and not node.orelse:
                removals.add(node)

    content = processing.alter_code(content, root, replacements=replacements, removals=removals)

    return content


def delete_commented_code(content: str) -> str:
    matches = list(re.finditer(r"(?<![^\n])(\s*(#.*))+", content))
    root = parsing.parse(content)
    code_string_ranges = {
        parsing.get_charnos(node, content)
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
                    r"(?<![^\n])(\s*#)", "", content[start + start_offset : end - end_offset]
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

                print("Deleting commented code")
                print(content[start + start_offset : end - end_offset])
                content = content[: start + start_offset] + "\n" + content[end - end_offset :]

                # Recursion due to likely race conditions
                return delete_commented_code(content)

    return content


def replace_with_filter(content: str) -> str:
    root = parsing.parse(content)

    replacements = {}
    removals = set()

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
                replacements[node.iter] = ast.Call(
                    func=ast.Name(id="filter"),
                    args=[test.func, node.iter],
                    keywords=[],
                )
                replacements[node.body[0].test] = ast.Constant(value=not negative, kind=None)
            elif isinstance(test, ast.Name) and parsing.match_template(
                node.target, ast.Name(id=test.id)
            ):
                replacements[node.iter] = ast.Call(
                    func=ast.Name(id="filter"),
                    args=[ast.Constant(value=None, kind=None), node.iter],
                    keywords=[],
                )
                replacements[node.body[0].test] = ast.Constant(value=not negative, kind=None)
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
                replacements[node.iter] = ast.Call(
                    func=ast.Name(id="filter"),
                    args=[test.operand.func, node.iter],
                    keywords=[],
                )
                removals.add(first_node)

    content = processing.alter_code(content, root, replacements=replacements, removals=removals)

    return content


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
                attr=parsing.Wildcard("call", str)
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


def implicit_defaultdict(content: str) -> str:
    replacements = {}
    removals = set()

    assign_template = ast.Assign(
        targets=[parsing.Wildcard("target", ast.Name)],
        value=parsing.Wildcard("value", ast.Dict(keys=[], values=[])),
    )
    if_template = ast.If(body=[object], orelse=[])

    root = parsing.parse(content)
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
                and (processing.unparse(t_value) == processing.unparse(f_value.elts[0]))
            ):
                if isinstance(f_value, ast.List) == (t_call == "append"):
                    loop_replacements[condition] = on_true
                    continue
                consistent = False
                break
            t_value_preferred = _preferred_comprehension_type(t_value)
            f_value_preferred = _preferred_comprehension_type(f_value)
            if processing.unparse(t_value_preferred) == processing.unparse(
                f_value_preferred
            ) and t_call in {"update", "extend"}:
                loop_replacements[condition] = on_true
                continue

        if not consistent:
            continue

        if subscript_calls and subscript_calls <= {"add", "update"}:
            replacements.update(loop_replacements)
            removals.update(loop_removals)
            replacements[value] = ast.Call(
                func=ast.Attribute(value=ast.Name(id="collections"), attr="defaultdict"),
                args=[ast.Name(id="set")],
                keywords=[],
            )

        if subscript_calls and subscript_calls <= {"append", "extend"}:
            replacements.update(loop_replacements)
            removals.update(loop_removals)
            replacements[value] = ast.Call(
                func=ast.Attribute(value=ast.Name(id="collections"), attr="defaultdict"),
                args=[ast.Name(id="list")],
                keywords=[],
            )

    content = processing.alter_code(content, root, replacements=replacements, removals=removals)

    return content


def simplify_redundant_lambda(content: str) -> str:
    root = parsing.parse(content)

    replacements = {}

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
            replacements[node] = node.body.func
        elif isinstance(node.body, ast.List):
            replacements[node] = ast.Name(id="list")
        elif isinstance(node.body, ast.Tuple):
            replacements[node] = ast.Name(id="tuple")
        elif isinstance(node.body, ast.Dict):
            replacements[node] = ast.Name(id="dict")

    content = processing.replace_nodes(content, replacements)

    return content
