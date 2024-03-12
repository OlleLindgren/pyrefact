from __future__ import annotations

import ast
import collections
import copy
import itertools
import re
import textwrap
from pathlib import Path
from typing import Collection, Iterable, List, Literal, Mapping, Sequence, Tuple

from pyrefact import (
    abstractions,
    constants,
    core,
    formatting,
    logs as logger,
    parsing,
    processing,
    style,
    tracing,
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

    if all(usage is node for usage in core.walk(scope, ast.Name(id=name))):
        return

    is_maybe_unordered_scope = isinstance(scope, (ast.Module, ast.ClassDef, ast.While, ast.For))

    # Prevent renaming variables in function scopes
    blacklisted_names = set()
    for funcdef in core.walk(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node in core.walk(funcdef, type(node)):
            continue
        if any(core.walk(funcdef.args, ast.arg(arg=name))):
            blacklisted_names.update(core.walk(funcdef, ast.Name))
        for child in core.walk(funcdef, ast.Name(ctx=ast.Store, id=name)):
            blacklisted_names.update(core.walk(child, ast.Name))

    augass_candidates = {
        target
        for augass in core.walk(scope, ast.AugAssign)
        for target in core.walk(augass, ast.Name(id=name))
    }

    ctx_load_candidates = {
        refnode
        for refnode in core.walk(scope, ast.Name(ctx=ast.Load, id=name))
        if refnode not in blacklisted_names
    }

    for refnode in augass_candidates | ctx_load_candidates:
        n_start = (refnode.lineno, refnode.col_offset)
        n_end = (refnode.end_lineno, refnode.end_col_offset)
        if end < n_start:
            yield refnode
        elif is_maybe_unordered_scope and n_end < start:
            yield refnode


def _get_variable_re_pattern(variable) -> str:
    return r"(?<![A-Za-z_\.])" + variable + r"(?![A-Za-z_])"


def _get_func_name_start_end(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, source: str
) -> Tuple[int, int]:
    start, end = core.get_charnos(node, source)
    codeblock = source[start:end]
    for match in re.finditer(_get_variable_re_pattern(node.name), codeblock):
        if match.group() == node.name:
            end = start + match.end()
            start += match.start()
            return start, end

    raise RuntimeError(f"Cannot find {node.name} in code block:\n{codeblock}")


def _fix_variable_names(
    source: str, renamings: Mapping[ast.AST, str], preserve: Collection[str] = frozenset()
) -> str:
    replacements = []
    ast_tree = core.parse(source)
    blacklisted_names = (
        tracing.get_imported_names(ast_tree)
        | constants.BUILTIN_FUNCTIONS
        | constants.PYTHON_KEYWORDS
    )
    for node, substitutes in renamings.items():
        if len(substitutes) != 1:
            continue
        substitute = substitutes.pop()
        if substitute in blacklisted_names:
            continue
        if isinstance(node, ast.Name):
            if node.id != substitute and node.id not in preserve:
                start, end = core.get_charnos(node, source)
                replacements.append((start, end, substitute))
            continue

        if node.name == substitute or node.name in preserve:
            continue

        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            raise TypeError(f"Unknown type: {type(node)}")

        start, end = _get_func_name_start_end(node, source)

        replacements.append((start, end, substitute))

    for start, end, substitute in sorted(set(replacements), reverse=True):
        logger.debug("Replacing {old} with {new}", old=source[start:end], new=substitute)
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
            logger.debug("Inserting '{fix}' at line {lineno}", fix=fix, lineno=lineno)
            lines.insert(lineno, fix)

    for package in (constants.ASSUMED_PACKAGES | constants.PYTHON_311_STDLIB) & variables:
        fix = f"import {package}"
        logger.debug("Inserting '{fix}' at line {lineno}", fix=fix, lineno=lineno)
        lines.insert(lineno, fix)

    for alias in constants.PACKAGE_ALIASES.keys() & variables:
        package = constants.PACKAGE_ALIASES[alias]
        fix = f"import {package} as {alias}"
        logger.debug("Inserting '{fix}' at line {lineno}", fix=fix, lineno=lineno)
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
    undefined_variables = tracing.get_undefined_variables(source)
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
    imports = tracing.get_imported_names(ast_tree)

    names = {node.id for node in core.walk(ast_tree, ast.Name(ctx=ast.Load))}
    for node in core.walk(ast_tree, ast.Attribute):
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
) -> Tuple[Collection[ast.Import | ast.ImportFrom], Collection[ast.Import | ast.ImportFrom]]:
    """Split unused imports into completely and partially unused imports.

    Args:
        ast_tree (ast.Module): Ast tree to search
        unused_imports (Collection[str]): Names that are imported but never used.

    Returns:
        Tuple: completely_unused_imports, partially_unused_imports
    """
    import_unused_aliases = collections.defaultdict(set)
    for node in core.walk(ast_tree, (ast.Import, ast.ImportFrom)):
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
    node: ast.Import | ast.ImportFrom, unused_imports: Collection[str]
) -> str:
    names = ", ".join(
        sorted(
            alias.name if alias.asname is None else f"{alias.name} as {alias.asname}"
            for alias in node.names
            if (alias.name if alias.asname is None else alias.asname) not in unused_imports
    ))
    if isinstance(node, ast.Import):
        return f"import {names}"

    return f"from {'.' * node.level}{node.module or ''} import {names}"


@processing.fix
def remove_unused_imports(source: str) -> str:
    """Remove unused imports from source code.

    Args:
        source (str): Python source code

    Returns:
        str: Source code, with added imports removed
    """
    root = core.parse(source)
    unused_imports = _get_unused_imports(root)
    completely_unused_imports, partially_unused_imports = _get_unused_imports_split(
        root, unused_imports
    )

    for node in completely_unused_imports:
        yield node, None

    # For every import, construct what we would like it to look like with redundant stuff removed, find the old
    # version of it, and replace it.

    # Iterate from bottom to top of file, so we don't have to re-calculate the linenos etc.
    for node in partially_unused_imports:
        yield node, _construct_import_statement(node, unused_imports)

    return source


def fix_too_many_blank_lines(source: str) -> str:
    # At module level, remove all above 2 blank lines
    source = re.sub(r"(\n\s*){3,}\n", "\n" * 3, source)

    # At EOF, remove all newlines and whitespace above 1
    source = re.sub(r"(\n\s*){2,}\Z", "\n", source)

    # At non-module (any indented) level, remove all newlines above 1, preserve indent
    source = re.sub(r"(\n\s*){2,}(\n\s+)(?=[^\n\s])", r"\n\g<2>", source)

    return source


@processing.fix(max_iter=1)
def fix_line_lengths(source: str, *, max_line_length: int = 100) -> str:
    root = core.parse(source)

    formatted_nodes = set()
    formatted_ranges = set()

    subscopes = []

    for scope in core.walk(
        root, (ast.AST(body=list), ast.AST(orelse=list), ast.AST(finalbody=list))
    ):
        subscopes.append(getattr(scope, "body", []))
        subscopes.append(getattr(scope, "orelse", []))
        subscopes.append(getattr(scope, "finalbody", []))

    for node in itertools.chain.from_iterable(subscopes):
        if node in formatted_nodes:
            continue

        source_range = core.get_charnos(node, source, keep_first_indent=True)
        if any(source_range & r for r in formatted_ranges):
            continue

        current_code = source[source_range.start : source_range.end]

        indent = formatting.indentation_level(current_code)
        if indent > 0:
            current_code = textwrap.dedent(current_code)

        elif_pattern = r"(\A[\s\n]*)(el)(if)"
        if_pattern = r"(\A[\s\n]*)(if)"
        elif_matches = list(re.finditer(elif_pattern, current_code))
        if elif_matches:
            # Convert elif to if
            current_code = re.sub(elif_pattern, r"\g<1>\g<3>", current_code, 1)
            new_code = formatting.format_with_black(
                current_code, line_length=max(60, max_line_length - indent)
            )
            # Convert if to elif
            new_code = re.sub(if_pattern, r"\g<1>el\g<2>", new_code, 1)
        else:
            new_code = formatting.format_with_black(
                current_code, line_length=max(60, max_line_length - indent)
            )

        if indent > 0:
            new_code = textwrap.indent(new_code, " " * indent)

        new_code = formatting.collapse_trailing_parentheses(new_code)
        if new_code != formatting.collapse_trailing_parentheses(current_code):
            yield source_range, new_code
            formatted_ranges.add(source_range)


@processing.fix
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
    ast_tree = core.parse(source)
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
                name = node.name
                # Don't rename magic members, don't rename if there is inheritance.
                if partial_tree.bases or parsing.is_magic_method(node):
                    renamings[node] = {name}
                funcdefs.append(node)
                substitute = style.rename_variable(
                    name, private=parsing.is_private(name), static=False
                )
                renamings[node].add(substitute)
                for refnode in _get_uses_of(node, partial_tree, source):
                    renamings[refnode].add(substitute)
            for node in parsing.iter_assignments(partial_tree):
                name = node.id
                # Don't rename magic members, don't rename if there is inheritance.
                if partial_tree.bases or (name.startswith("__") and name.endswith("__")):
                    renamings[node] = {name}
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

    blacklisted_names = (
        tracing.get_imported_names(ast_tree)
        | tracing.get_defined_names(ast_tree)
        | constants.BUILTIN_FUNCTIONS
        | constants.PYTHON_KEYWORDS
    )
    renamings = {
        node: list(substitutes)[0]
        for node, substitutes in renamings.items()
        if len(substitutes) == 1 and blacklisted_names.isdisjoint(substitutes)
    }
    substitute_node_renamings = collections.defaultdict(set)
    for node, substitute in renamings.items():
        substitute_node_renamings[substitute].add(node)

    transaction = 0
    for substitute, nodes in substitute_node_renamings.items():
        replacements = []
        for node in nodes:
            if isinstance(node, ast.Name):
                if node.id == substitute:
                    continue
                if node.id in preserve:
                    continue
                replacement = ast.Name(id=substitute)
            elif isinstance(node, ast.FunctionDef):
                if node.name == substitute:
                    continue
                if node.name in preserve:
                    continue
                replacement = ast.FunctionDef(
                    name=substitute,
                    args=node.args,
                    body=node.body,
                    decorator_list=node.decorator_list,
                    returns=node.returns,
                    type_comment=node.type_comment,
                )
            elif isinstance(node, ast.AsyncFunctionDef):
                if node.name == substitute:
                    continue
                if node.name in preserve:
                    continue
                replacement = ast.AsyncFunctionDef(
                    name=substitute,
                    args=node.args,
                    body=node.body,
                    decorator_list=node.decorator_list,
                    returns=node.returns,
                    type_comment=node.type_comment,
                )
            elif isinstance(node, ast.ClassDef):
                if node.name == substitute:
                    continue
                if node.name in preserve:
                    continue
                replacement = ast.ClassDef(
                    name=substitute,
                    bases=node.bases,
                    keywords=node.keywords,
                    body=node.body,
                    decorator_list=node.decorator_list,
                )
            else:
                logger.error("Renaming not implemented for node {} of type {}", node, type(node))
                replacements.clear()
                break

            ast.fix_missing_locations(replacement)

            yield node, replacement, transaction

        transaction += 1


@processing.fix
def undefine_unused_variables(source: str, preserve: Collection[str] = frozenset()) -> str:
    """Remove definitions of unused variables

    Args:
        source (str): Python source code
        preserve (Collection[str], optional): Variable names to preserve

    Returns:
        str: Python source code, with no definitions of unused variables
    """
    root = core.parse(source)

    # It's sketchy to figure out if class properties and stuff are used. Will not
    # support this for the time being.
    class_body_blacklist = set()
    for scope in core.walk(root, ast.ClassDef):
        for node in core.filter_nodes(scope.body, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            class_body_blacklist.update(parsing.assignment_targets(node))

    yielded = set()
    for name in _iter_unused_names(root):
        if name.id not in preserve and name.id != "_" and name not in class_body_blacklist and name not in yielded:
            yield name, ast.Name(id="_")
            yielded.add(name)

    for node in core.walk(
        root,
        (
            ast.Assign(
                targets={
                    ast.Name(id="_"),
                    ast.Starred(value=ast.Name(id="_")),
                    ast.Tuple(elts={ast.Name(id="_"), ast.Starred(value=ast.Name(id="_"))}),
            }),
            ast.AnnAssign(target=ast.Name(id="_")),
            ast.AugAssign(target=ast.Name(id="_")),
    ),):
        if node not in class_body_blacklist:
            yield node, node.value


def _iter_unused_names(
    scope: ast.AST, preserve: Collection[str] = frozenset()
) -> Iterable[ast.Name]:
    # preserve presumably contains everything that any outer scope could be interested
    # in. So any name that is set but never accessed, and that is not in preserve, can
    # immediately be deleted.
    names_in_scope = {name.id for name in core.walk(scope, ast.Name)}
    for name in names_in_scope - preserve:
        if not any(core.walk(scope, ast.Name(id=name, ctx=(ast.Load)))):
            for node in core.walk(scope, ast.Name(id=name)):
                yield node

    # For everything else, the algorithm is like this:
    # Iterate over all subset sequences of nodes in every scope, such that no node outside
    # of that sequence touch some particular name.

    # (2) Find all names defined that are ever stored anywhere in it,
    bodies = []
    if core.match_template(scope, ast.AST(body=list)):
        bodies.append(scope.body)
    if core.match_template(scope, ast.AST(orelse=list)):
        bodies.append(scope.orelse)
    if core.match_template(scope, ast.AST(finalbody=list)):
        bodies.append(scope.finalbody)
    if isinstance(scope, (ast.For, ast.While)):
        *_, required_names = tracing.code_dependencies_outputs([scope])
        preserve = preserve | required_names

    for body in filter(None, bodies):
        names_defined_in_scope = {
            target.id
            for node in body
            for assign in core.walk(node, (ast.Assign, ast.AnnAssign))
            for target in parsing.assignment_targets(assign)
        }
        name_mentions = collections.defaultdict(set)
        for node in body:
            _, created_names, required_names = tracing.code_dependencies_outputs([node])
            for name in itertools.chain(created_names, required_names):
                name_mentions[name].add(node)
        # (3) And group the code in the smallest possible sequences that will contain
        #     (directly or recursively) all references (set and get) of that name.
        name_node_sequences = {
            name: sorted(mentions, key=lambda node: node.lineno)
            for name, mentions in name_mentions.items()
            if name in names_defined_in_scope
        }
        # (4) For every (name, node_sequence) in that grouping,
        for name, sequence in name_node_sequences.items():
            _, created_names, required_names = tracing.code_dependencies_outputs(sequence)
            # (7) For every (node) at position (i) in the sequence,
            for i, node in enumerate(sequence):
                remainder = sequence[i + 1 :]
                if isinstance(node, (ast.For, ast.While)):
                    remainder.extend(sequence[:i])
                _, node_created, _ = tracing.code_dependencies_outputs([node])
                subsequent_created, _, subsequent_required = tracing.code_dependencies_outputs(
                    remainder
                )
                # (8) If (name) is in its outputs, but (name) is not in the dependencies of
                # node_sequence[i:],
                if name in node_created:
                    # (9) then (name) is being redundantly defined in node (i).

                    # If node (i) is an assign node, we can just un-assign it.
                    if (
                        isinstance(node, (ast.Assign, ast.AnnAssign))
                        and name not in subsequent_required
                        and (
                            # And (name) is either not in preserve (so nothing upstream cares about
                            # it), or (name) will surely be defined by a subsequent node
                            name not in preserve
                            # TODO this seems impossible, figure out what is intended
                            or name in subsequent_created
                    )):
                        for creation_node in core.filter_nodes(
                            parsing.assignment_targets(node), ast.Name(id=name, ctx=ast.Store)
                        ):
                            yield creation_node
                    else:
                        # If node (i) is something more complicated (like a loop or something), it
                        # may be that (name) is defined and then used in node (i). But definitions
                        # of (name) that (node) considers unused are still surely unused.
                        yield from core.filter_nodes(
                            _iter_unused_names(node, preserve=preserve | subsequent_required),
                            ast.Name(id=name),
                        )


def move_before_loop(source: str) -> str:
    root = core.parse(source)

    for scope in core.walk(root, (ast.For, ast.While)):
        header_scope = [scope.target, scope.iter] if isinstance(scope, ast.For) else [scope.test]
        for i, node in enumerate(scope.body):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            if core.has_side_effect(node.value):
                continue

            # If targets are likely to be mutated in the loop, keep them in the loop.
            targets = tuple(name.id for name in parsing.assignment_targets(node))
            if any(core.walk(scope, ast.Call(func=ast.Attribute(value=ast.Name(id=targets))))):
                continue  # i.e. x.append(y)
            if any(
                core.walk(
                    scope, ast.Subscript(value=ast.Name(id=targets), ctx=(ast.Store, ast.Del))
            )):
                continue  # i.e. x[3] = 2
            if any(core.walk(scope, ast.AugAssign(target=ast.Name(id=targets)))):
                continue  # i.e. x += 1

            remainder = scope.body[i + 1 :] + scope.body[:i]
            definite_created_names, maybe_created_names, _ = tracing.code_dependencies_outputs(
                remainder
            )
            _, node_created_names, node_required_names = tracing.code_dependencies_outputs([node])

            before = scope.body[:i]
            recursive_before_children = itertools.chain(
                before, *(ast.walk(child) for child in before)
            )

            if any(core.is_blocking(child) for child in recursive_before_children):
                continue

            _, before_created, before_required = tracing.code_dependencies_outputs(before)

            # If the loop may create names that the node depends on, keep it in the loop
            if maybe_created_names & node_required_names:
                continue

            # If the node creates names that only sometimes are overwritten by the loop, keep it in the loop
            if (maybe_created_names - definite_created_names) & node_created_names:
                continue

            if node_created_names & (before_required | before_created):
                continue

            _, header_created, header_required = tracing.code_dependencies_outputs(header_scope)

            if header_created & node_required_names:
                continue

            if header_required & node_created_names:
                continue

            new_node = copy.copy(node)
            new_node.lineno = scope.lineno - 1
            new_node.col_offset = scope.col_offset

            source = processing.alter_code(source, root, additions=[new_node], removals=[node])
            return move_before_loop(source)

    return source


def _is_pointless_string(node: ast.AST) -> bool:
    """Check if an AST is a pointless string statement.

    This is useful for figuring out if a node is a docstring.

    Args:
        node (ast.AST): AST to check

    Returns:
        bool: True if the node is a pointless string statement.
    """
    return core.match_template(node, ast.Expr(value=ast.Constant(value=str)))


@processing.fix
def delete_pointless_statements(source: str) -> str:
    """Delete pointless statements with no side effects from code

    Args:
        source (str): Python source code.

    Returns:
        str: Modified code
    """
    ast_tree = core.parse(source)
    safe_callables = parsing.safe_callable_names(ast_tree)
    for node in itertools.chain([ast_tree], parsing.iter_bodies_recursive(ast_tree)):
        for i, child in enumerate(node.body):
            if not core.has_side_effect(child, safe_callables):
                if i > 0 or not _is_pointless_string(child):  # Docstring
                    yield child, None


def _iter_unreachable_nodes(body: Iterable[ast.AST]) -> Iterable[ast.AST]:
    after_block = False
    for node in body:
        if after_block:
            yield node
            continue
        if core.is_blocking(node):
            after_block = True


@processing.fix
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
    root = core.parse(source)

    funcdefs = []
    classdefs = []
    name_usages = collections.defaultdict(set)

    preserved_class_funcdefs = {
        funcdef
        for node in core.walk(root, ast.ClassDef)
        for funcdef in core.filter_nodes(node.body, (ast.FunctionDef, ast.AsyncFunctionDef))
        if f"{node.name}.{funcdef.name}" in preserve or node.bases
    }

    for node in core.walk(root, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.name not in preserve and node not in preserved_class_funcdefs:
            funcdefs.append(node)

    for node in core.walk(root, ast.ClassDef):
        if node.name not in preserve:
            classdefs.append(node)

    for node in core.walk(root, ast.Name(ctx=ast.Load)):
        name_usages[node.id].add(node)

    for node in core.walk(root, ast.Attribute):
        name_usages[node.attr].add(node)
        for name in core.walk(node, ast.Name):
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
        recursive_usages = set(core.walk(def_node, ast.Name(id=def_node.name)))
        if not (usages | constructor_usages) - recursive_usages:
            yield def_node, None

    for def_node in classdefs:
        usages = name_usages[def_node.name]
        internal_usages = set(
            core.walk(def_node, ast.Name(ctx=ast.Load, id=(def_node.name, "self", "cls")))
        )
        if not usages - internal_usages:
            yield def_node, None


@processing.fix
def delete_unreachable_code(source: str) -> str:
    """Find and delete dead code.

    Args:
        source (str): Python source code

    Returns:
        str: Source code with dead code deleted
    """
    root = core.parse(source)

    transaction = 0
    for node in parsing.iter_bodies_recursive(root):
        if not isinstance(node, (ast.If, ast.While)):
            for unreachable_node in _iter_unreachable_nodes(node.body):
                yield unreachable_node, None, transaction

            transaction += 1
            continue

        try:
            test_value = core.literal_value(node.test)
        except ValueError:
            continue

        if isinstance(node, ast.While) and not test_value:
            yield node, None, transaction
            continue

        if isinstance(node, ast.If):
            if test_value and node.body:
                for _ in node.orelse:
                    yield node, None, transaction
            elif not test_value and node.orelse:
                for _ in node.body:
                    yield node, None, transaction
            else:
                yield node, None, transaction

            transaction += 1


def _get_package_names(node: ast.Import | ast.ImportFrom):
    if isinstance(node, ast.ImportFrom):
        return [node.module]

    return [alias.name for alias in node.names]


def move_imports_to_toplevel(source: str) -> str:
    root = core.parse(source)
    toplevel_imports = set(core.filter_nodes(root.body, (ast.Import, ast.ImportFrom)))
    all_imports = set(core.walk(root, (ast.Import, ast.ImportFrom)))
    toplevel_packages = set()
    for node in toplevel_imports:
        toplevel_packages.update(_get_package_names(node))

    imports_movable_to_toplevel = {
        node
        for node in all_imports - toplevel_imports
        if all(
            name in constants.PYTHON_311_STDLIB or name in toplevel_packages
            for name in _get_package_names(node)
    )}

    if defs := set(
        core.filter_nodes(root.body, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ):
        first_def_lineno = min(node.lineno - len(node.decorator_list) for node in defs)
        imports_movable_to_toplevel.update(
            node for node in toplevel_imports if node.lineno > first_def_lineno
        )

    for i, node in enumerate(root.body):
        if i > 0 and not isinstance(node, (ast.Import, ast.ImportFrom)):
            lineno = min(x.lineno for x in core.walk(node, ast.AST(lineno=int))) - 1
            break
        if i == 0 and not core.match_template(
            node, (ast.Import, ast.ImportFrom, ast.Expr(value=ast.Constant(value=str)))
        ):
            lineno = min(x.lineno for x in core.walk(node, ast.AST(lineno=int))) - 1
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

        source_lines = source.splitlines()
        while safe_position_lineno > 1 and re.findall(r"^\s+", source_lines[safe_position_lineno]):
            safe_position_lineno -= 1

        new_node = ast.ImportFrom(
            module=node.module, names=node.names, level=node.level, lineno=safe_position_lineno
        )
        additions.append(new_node)

    # Remove duplicates
    additions = {core.unparse(x): x for x in additions}.values()

    if removals or additions:
        logger.debug("Moving imports to toplevel")
        source = processing.alter_code(source, root, removals=removals, additions=additions)

    return source


def remove_duplicate_functions(source: str, preserve: Collection[str]) -> str:
    """Remove duplicate function definitions.

    Args:
        source (str): Python source code
        preserve (Collection[str]): Names to preserve

    Returns:
        str: Modified code
    """
    root = core.parse(source)
    function_defs = collections.defaultdict(set)

    for node in core.filter_nodes(root.body, ast.FunctionDef):
        function_defs[abstractions.hash_node(node, preserve)].add(node)

    delete = set()
    renamings = {}

    for funcdefs in function_defs.values():
        if len(funcdefs) == 1:
            continue
        logger.debug(", ".join(node.name for node in funcdefs) + " are equivalent")
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
    for node in core.walk(root, ast.Name):
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


@processing.fix
def remove_redundant_else(source: str) -> str:
    """Remove redundante else and elif statements in code.

    Args:
        source (str): Python source code

    Returns:
        str: Code with no redundant else/elifs.
    """
    root = core.parse(source)
    for node in core.walk(root, ast.If):
        if not node.orelse:
            continue
        if not core.get_code(node, source).startswith("if"):  # Otherwise we get FPs on elif
            continue
        if not any((core.is_blocking(child) for child in node.body)):
            continue

        if core.match_template(node.orelse, [ast.If]):
            (start, end) = core.get_charnos(node.orelse[0], source)
            orelse = source[start:end]
            if orelse.startswith("elif"):  # Regular elif
                modified_orelse = re.sub("^elif", "if", orelse)

                yield core.Range(start, end), modified_orelse
                continue

            # Otherwise it's an else: if:, which is handled below

        # else

        ranges = [core.get_charnos(child, source) for child in node.orelse]
        start = min((s for (s, _) in ranges))
        end = max((e for (_, e) in ranges))
        last_else = list(re.finditer("(?<![^\\n]) *else: *\\n?", source[:start]))[-1]
        indent = len(re.findall("^ *", last_else.group())[0])
        modified_orelse = " " * indent + re.sub("(?<![^\\n])    ", "", source[start:end]).lstrip()

        pre_else = source[: last_else.start()]
        start_offset = len(pre_else) - len(pre_else.rstrip())

        yield core.Range(last_else.start() - start_offset, end), "\n\n" + modified_orelse


@processing.fix
def singleton_eq_comparison(source: str) -> str:
    """Replace singleton comparisons using "==" with "is".

    Args:
        source (str): Python source code

    Returns:
        str: Fixed code
    """
    root = core.parse(source)

    for node in core.walk(root, ast.Compare):
        changes = False
        operators = []
        for comparator, node_operator in zip(node.comparators, node.ops):
            is_comparator_singleton = core.match_template(
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
            yield node, ast.Compare(left=node.left, ops=operators, comparators=node.comparators)


def _negate_condition(node: ast.AST) -> ast.AST:
    if core.match_template(node, ast.UnaryOp(op=ast.Not)):
        return node.operand

    if core.match_template(
        node, ast.Compare(ops=[tuple(constants.REVERSE_OPERATOR_MAPPING)], comparators=[object])
    ):
        opposite_operator_type = constants.REVERSE_OPERATOR_MAPPING[type(node.ops[0])]
        return ast.Compare(
            left=node.left, ops=[opposite_operator_type()], comparators=node.comparators
        )

    if core.match_template(node, ast.BoolOp(op=ast.And)):
        return ast.BoolOp(op=ast.Or(), values=[_negate_condition(child) for child in node.values])

    if core.match_template(node, ast.BoolOp(op=ast.Or)):
        return ast.BoolOp(op=ast.And(), values=[_negate_condition(child) for child in node.values])

    return ast.UnaryOp(op=ast.Not(), operand=node)


def _iter_implicit_if_elses(
    root: ast.AST,
) -> Iterable[Tuple[ast.If, Sequence[ast.AST], Sequence[ast.AST]]]:
    for (condition,), *implicit_orelse in core.walk_sequence(
        root, ast.If, ast.AST, expand_last=True
    ):
        implicit_orelse = [x[0] for x in implicit_orelse]
        if any(core.is_blocking(child) for child in condition.body) and not condition.orelse:
            yield condition, condition.body, implicit_orelse


def _iter_explicit_if_elses(
    root: ast.AST,
) -> Iterable[Tuple[ast.If, Sequence[ast.AST], Sequence[ast.AST]]]:
    for condition in core.walk(root, ast.If):
        if condition.body and condition.orelse:
            yield condition, condition.body, condition.orelse


def _count_children(node: ast.AST, child_type: ast.AST) -> int:
    return sum(1 for _ in core.walk(node, child_type))


def _count_branches(nodes: Sequence[ast.AST]) -> int:
    return 1 + sum(_count_children(node, ast.If) for node in nodes)


def _orelse_preferred_as_body(body: Sequence[ast.AST], orelse: Sequence[ast.AST]) -> bool:
    if all(isinstance(node, ast.Pass) for node in body):
        return True
    if all(isinstance(node, ast.Pass) for node in orelse):
        return False

    body_blocking = any(core.is_blocking(node) for node in body)
    orelse_blocking = any(core.is_blocking(node) for node in orelse)
    if body_blocking and not orelse_blocking:
        return False
    if orelse_blocking and not body_blocking:
        return True
    body_branches = _count_branches(body)
    orelse_branches = _count_branches(orelse)
    if orelse_blocking and body_blocking and body_branches >= 2 * orelse_branches:
        return True

    return isinstance(orelse[0], (ast.Return, ast.Continue, ast.Break)) and len(body) > 3


def _sequential_similar_ifs(source: str, root: ast.AST) -> Collection[ast.If]:
    return set.union(
        set(),
        *map(set, parsing.iter_similar_nodes(root, source, ast.If, count=2, length=15)),
        *map(set, parsing.iter_similar_nodes(root, source, ast.If, count=3, length=10)),
    )


@processing.fix
def _swap_explicit_if_else(source: str) -> str:
    root = core.parse(source)
    sequential_similar_ifs = _sequential_similar_ifs(source, root)

    for stmt, body, orelse in _iter_explicit_if_elses(root):
        if isinstance(stmt.test, ast.NamedExpr):
            continue
        if stmt in sequential_similar_ifs:
            continue
        if (
            orelse
            and any(core.is_blocking(node) for node in body)
            and not any(core.is_blocking(node) for node in orelse)
        ):
            continue  # Redundant else
        if core.get_code(stmt, source).startswith("elif"):
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

    root = core.parse(source)
    sequential_similar_ifs = _sequential_similar_ifs(source, root)

    for stmt, body, orelse in _iter_implicit_if_elses(root):
        if stmt in sequential_similar_ifs:
            continue
        if isinstance(stmt.test, ast.NamedExpr):
            continue
        if (
            orelse
            and any(core.is_blocking(node) for node in body)
            and not any(core.is_blocking(node) for node in orelse)
        ):
            continue  # body is blocking but orelse is not
        if core.get_code(stmt, source).startswith("elif"):
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
    root = core.parse(source)
    for funcdef in parsing.iter_funcdefs(root):
        if not core.match_template(funcdef.body[-2:], [ast.If, ast.Return(value=ast.Name)]):
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
        if all((
            core.match_template(node, ast.Assign(targets=[ast.Name(id=retval)]))
            for node in recursive_last_nonif_nodes
        )):
            for node in recursive_last_nonif_nodes:
                yield node, ast.Return(value=node.value, lineno=node.lineno)

            yield ret_stmt, None


def _total_linenos(nodes: Iterable[ast.AST]) -> int:
    start_lineno = 1000_000
    end_lineno = 0
    for node in nodes:
        for child in core.walk(node, ast.AST(lineno=int, end_lineno=int)):
            start_lineno = min(start_lineno, child.lineno)
            end_lineno = max(end_lineno, child.end_lineno)

    return max(end_lineno - start_lineno, 0)


def early_continue(source: str) -> str:
    additions = []
    replacements = {}

    root = core.parse(source)
    blacklisted_ifs = _sequential_similar_ifs(source, root)

    for loop in core.walk(root, ast.For):
        stmt = loop.body[-1]
        if (
            isinstance(stmt, ast.If)
            and not isinstance(stmt.body[-1], ast.Continue)
            and stmt not in blacklisted_ifs
        ):
            recursive_ifs = [stmt]
            for child in stmt.orelse:
                recursive_ifs.extend(core.walk(child, ast.If))
            if any(len(node.orelse) > 2 for node in recursive_ifs):
                additions.append(
                    ast.Continue(
                        lineno=stmt.body[-1].end_lineno, col_offset=stmt.body[-1].col_offset
                ))
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
                    body=[ast.Continue()], orelse=stmt.body, test=_negate_condition(stmt.test)
                )

    return processing.alter_code(source, root, additions=additions, replacements=replacements)


@processing.fix
def remove_redundant_comprehensions(source: str) -> str:
    comprehension_wrapper_funcs = {
        ast.DictComp: "dict",
        ast.ListComp: "list",
        ast.SetComp: "set",
        ast.GeneratorExp: "iter",
        ast.Dict: "dict",
        ast.List: "list",
        ast.Set: "set",
        ast.Tuple: "tuple",
    }
    find = core.compile_template((
        "[{{target}} for {{target}} in {{iterable}}]",
        "{{{target}} for {{target}} in {{iterable}}}",
        "({{target}} for {{target}} in {{iterable}})",
        "{{{key}}: {{value}} for {{key}}, {{value}} in {{iterable}}}",
    ))
    replace = "{{funcname(root)}}({{iterable}})"
    funcname = lambda template_match_tuple: comprehension_wrapper_funcs[type(template_match_tuple)]

    yield from processing.find_replace(source, find, replace, funcname=funcname)


@processing.fix
def replace_functions_with_literals(source: str) -> str:
    root = core.parse(source)

    yield from processing.find_replace(source, "list()", "[]")
    yield from processing.find_replace(source, "tuple()", "()")
    yield from processing.find_replace(source, "dict()", "{}")

    template = core.compile_template(
        "{{func}}({{arg}})",
        func=ast.Name(id=("list", "tuple", "set", "iter")),
        arg=(ast.List, ast.ListComp, ast.Tuple, ast.Set, ast.SetComp, ast.GeneratorExp),
    )
    for node, arg, func in core.walk_wildcard(root, template):
        if func.id == "list":
            if isinstance(arg, (ast.List, ast.ListComp)):
                yield node, arg
            elif isinstance(arg, ast.Tuple):
                yield node, ast.List(elts=arg.elts, ctx=arg.ctx)

        elif func.id == "tuple":
            if isinstance(arg, ast.Tuple):
                yield node, arg
            elif isinstance(arg, ast.List):
                yield node, ast.Tuple(elts=arg.elts, ctx=arg.ctx)

        elif func.id == "set":
            if isinstance(arg, (ast.Set, ast.SetComp)):
                yield node, arg
            elif isinstance(arg, (ast.Tuple, ast.List)):
                yield node, ast.Set(elts=arg.elts, ctx=arg.ctx)
            elif isinstance(arg, ast.GeneratorExp):
                yield node, ast.SetComp(elt=arg.elt, generators=arg.generators)

        elif func.id == "iter":
            if isinstance(arg, ast.GeneratorExp):
                yield node, arg


@processing.fix
def replace_for_loops_with_dict_comp(source: str) -> str:
    assign_template = ast.Assign(
        value=core.Wildcard("value", (ast.Dict, ast.DictComp)),
        targets=[ast.Name(id=core.Wildcard("target", str))],
    )

    transaction = 0
    root = core.parse(source)
    for (_, target, value), (n2,) in core.walk_sequence(root, assign_template, ast.For):
        body_node = n2
        generators = []
        transaction += 1

        while core.match_template(
            body_node, (ast.For(body=[object]), ast.If(body=[object], orelse=[]))
        ):
            if isinstance(body_node, ast.If):
                generators[-1].ifs.append(body_node.test)
            elif isinstance(body_node, ast.For):
                generators.append(
                    ast.comprehension(
                        target=body_node.target, iter=body_node.iter, ifs=[], is_async=0
                ))
            else:
                raise RuntimeError(f"Unexpected type of node: {type(body_node)}")

            body_node = body_node.body[0]

        for comprehension in generators:
            if len(comprehension.ifs) > 1:
                comprehension.ifs = [ast.BoolOp(op=ast.And(), values=comprehension.ifs)]

        if not core.match_template(
            body_node, ast.Assign(targets=[ast.Subscript(value=ast.Name(id=target))])
        ):
            continue

        comp = ast.DictComp(
            key=body_node.targets[0].slice, value=body_node.value, generators=generators
        )
        if core.match_template(value, ast.Dict(values=[], keys=[])):
            yield value, comp, transaction
            yield n2, None, transaction
        elif core.match_template(value, ast.Dict(values=list, keys={None})):
            yield value, ast.Dict(
                keys=value.keys + [None], values=value.values + [comp]
            ), transaction
            yield n2, None, transaction
        elif core.match_template(value, ast.Dict(values=list, keys=list)):
            yield value, ast.Dict(keys=[None, None], values=[value, comp]), transaction
            yield n2, None, transaction
        elif isinstance(value, ast.DictComp):
            yield value, ast.Dict(keys=[None, None], values=[value, comp]), transaction
            yield n2, None, transaction


@processing.fix
def replace_for_loops_with_set_list_comp(source: str) -> str:
    assign_template = ast.Assign(
        value=core.Wildcard("value", object), targets=[ast.Name(id=core.Wildcard("target", str))]
    )
    for_template = ast.For(body=[object])
    if_template = ast.If(body=[object], orelse=[])

    set_init_template = ast.Call(func=ast.Name(id="set"), args=[], keywords=[])
    list_init_template = ast.List(elts=[])  # list() should have been replaced by [] elsewhere.

    transaction = 0
    root = core.parse(source)
    for (_, target, value), (n2,) in core.walk_sequence(root, assign_template, for_template):
        body_node = n2
        generators = []
        transaction += 1

        while core.match_template(body_node, (for_template, if_template)):
            if isinstance(body_node, ast.If):
                generators[-1].ifs.append(body_node.test)
            elif isinstance(body_node, ast.For):
                generators.append(
                    ast.comprehension(
                        target=body_node.target, iter=body_node.iter, ifs=[], is_async=0
                ))
            else:
                raise RuntimeError(f"Unexpected type of node: {type(body_node)}")

            body_node = body_node.body[0]

        for comprehension in generators:
            if len(comprehension.ifs) > 1:
                comprehension.ifs = [ast.BoolOp(op=ast.And(), values=comprehension.ifs)]

        target_alter_template = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=target), attr=core.Wildcard("attr", ("add", "append"))
                ),
                args=[object],
        ))

        augass_template = ast.AugAssign(op=(ast.Add, ast.Sub), target=ast.Name(id=target))

        if template_match := core.match_template(body_node, target_alter_template):
            if core.match_template(value, list_init_template) and (template_match.attr == "append"):
                comp_type = ast.ListComp
            elif core.match_template(value, set_init_template) and (template_match.attr == "add"):
                comp_type = ast.SetComp
            else:
                continue

            yield value, comp_type(elt=body_node.value.args[0], generators=generators), transaction
            yield n2, None, transaction

        elif core.match_template(body_node, augass_template):
            if isinstance(value, ast.List):
                replacement = ast.ListComp(elt=body_node.value, generators=generators)
            else:
                comprehension = ast.GeneratorExp(elt=body_node.value, generators=generators)
                replacement = ast.Call(func=ast.Name(id="sum"), args=[comprehension], keywords=[])

            try:
                if not core.literal_value(value):
                    if isinstance(body_node.op, ast.Sub):
                        replacement = ast.UnaryOp(op=body_node.op, operand=replacement)
                    yield value, replacement, transaction
                    yield n2, None, transaction
                    continue

            except ValueError:
                pass

            replacement = ast.BinOp(left=value, op=body_node.op, right=replacement)
            yield value, replacement, transaction
            yield n2, None, transaction


@processing.fix
def inline_math_comprehensions(source: str) -> str:
    root = core.parse(source)

    replacements = {}
    blacklist = set()

    assign_template = ast.Assign(targets=[core.Wildcard("target", ast.Name)])
    augassign_template = ast.AugAssign(target=core.Wildcard("target", ast.Name))
    annassign_template = ast.AnnAssign(target=core.Wildcard("target", ast.Name))

    comprehension_assignments = [
        (assignment, target, assignment.value)
        for (assignment, target) in core.walk_wildcard(
            root, (assign_template, augassign_template, annassign_template)
        )
        if isinstance(assignment.value, (ast.GeneratorExp, ast.ListComp, ast.SetComp))
        or (
            isinstance(assignment.value, ast.Call)
            and isinstance(assignment.value.func, ast.Name)
            and (assignment.value.func.id in constants.ITERATOR_FUNCTIONS)
    )]

    scope_types = (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
    for scope in core.walk(root, scope_types):
        for assignment, target, value in comprehension_assignments:
            uses = list(_get_uses_of(target, scope, source))
            if len(uses) != 1:
                blacklist.add(assignment)
                continue

            use = uses.pop()

            _, set_end_charno = core.get_charnos(value, source)
            use_start_charno, _ = core.get_charnos(use, source)

            # May be in a loop and the below dependency check won't be reliable.
            if use_start_charno < set_end_charno:
                blacklist.add(use)
                break

            # Check for references to any of the iterator's dependencies between set and use.
            # Perhaps some of these could be skipped, but I'm not sure that's a good idea.
            value_dependencies = tuple({node.id for node in core.walk(value, ast.Name)})
            for node in core.walk(scope, ast.Name(id=value_dependencies)):
                start, end = core.get_charnos(node, source)
                if set_end_charno < start <= end < use_start_charno:
                    blacklist.add(use)
                    break

            if use in blacklist:
                break

            for call in core.walk(
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

    yield from replacements.items()


@processing.fix
def simplify_transposes(source: str) -> str:
    find = "zip(*zip(*{{value}}))"
    replace = "{{value}}"
    yield from processing.find_replace(source, find, replace)

    find = "zip(*{{value}}.T)"
    replace = "{{value}}"
    yield from processing.find_replace(source, find, replace)

    find = "zip(*{{value}}).T"
    replace = "{{value}}"
    yield from processing.find_replace(source, find, replace)

    find = "{{value}}.T.T"
    replace = "{{value}}"
    yield from processing.find_replace(source, find, replace)

    find = "np.array({{value}}.T).T"
    replace = "{{value}}"
    yield from processing.find_replace(source, find, replace)

    find = "np.array(np.matmul({{left}}, {{right}}))"
    replace = "np.matmul({{left}}, {{right}})"
    yield from processing.find_replace(source, find, replace)

    find = "np.array(np.matmul({{left}}, {{right}}).T)"
    replace = "np.matmul({{left}}, {{right}}).T"
    yield from processing.find_replace(source, find, replace)

    find = "np.matmul({{left}}.T, {{right}}.T).T"
    replace = "np.matmul({{right}}, {{left}})"
    yield from processing.find_replace(source, find, replace)

    root = core.parse(source)

    calls = core.walk(root, ast.Call)
    attributes = core.walk(root, ast.Attribute)

    for node in filter(parsing.is_transpose_operation, itertools.chain(calls, attributes)):
        first_transpose_target = parsing.transpose_target(node)
        if parsing.is_transpose_operation(first_transpose_target):
            second_transpose_target = parsing.transpose_target(first_transpose_target)
            yield node, second_transpose_target


@processing.fix
def remove_dead_ifs(source: str) -> str:
    root = core.parse(source)

    for node in core.walk(root, (ast.If, ast.While, ast.IfExp)):
        try:
            value = core.literal_value(node.test)
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

            ranges = [core.get_charnos(child, source) for child in remove]
            start = min((s for (s, _) in ranges))
            end = max((e for (_, e) in ranges))
            indent = node.col_offset
            node_start, node_end = core.get_charnos(node, source)
            modified_body = " " * indent + re.sub("(?<![^\\n])    ", "", source[start:end]).lstrip()

            pre_else = source[:node_start]
            start_offset = len(pre_else) - len(pre_else.rstrip())

            yield core.Range(node_start - start_offset, node_end), "\n\n" + modified_body + "\n\n"

    for node in core.walk(root, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        generators = []
        any_comprehension_modified = False
        for comprehension in node.generators:
            ifs = []
            any_if_always_false = False
            for if_ in comprehension.ifs:
                try:
                    value = core.literal_value(if_)
                except ValueError:
                    ifs.append(if_)
                    continue

                if not value:
                    # Condition is always False, so the whole comprehension is dead
                    any_if_always_false = True
                    break

                # (else): Condition is always True, so the condition can be removed.
                # We skip adding it to ifs, so that will be the result.

            if any_if_always_false:
                any_comprehension_modified = True
                continue

            if len(ifs) < len(comprehension.ifs):
                replacement = ast.comprehension(
                    target=comprehension.target,
                    iter=comprehension.iter,
                    ifs=ifs,
                    is_async=comprehension.is_async,
                )
                generators.append(replacement)
                any_comprehension_modified = True
            else:
                generators.append(comprehension)

        if not any_comprehension_modified:
            continue

        if generators:
            yield (node, type(node)(**{**node.__dict__, "generators": generators}))
            continue

        # If all generators are dead, replace the comprehension with an empty container
        # of the same type.

        if isinstance(node, ast.ListComp):
            yield (node, ast.List(elts=[]))
            continue

        if isinstance(node, ast.SetComp):
            yield (node, ast.Call(func=ast.Name(id="set"), args=[], keywords=[]))
            continue

        # Although an empty generator would be more correctly replaced with iter([]) or
        # some similar construct, I think that will just confuse people, so we replace
        # it with a tuple instead, which is semantically equivalent and more readable.
        if isinstance(node, ast.GeneratorExp):
            yield (node, ast.Tuple(elts=[]))
            continue

        if isinstance(node, ast.DictComp):
            yield (node, ast.Dict(keys=[], values=[]))
            continue


@processing.fix
def delete_commented_code(source: str) -> str:
    matches = list(re.finditer(r"(?<![^\n])(\s*(#.*))+", source))
    root = core.parse(source)
    code_ranges = [
        core.get_charnos(node, source)
        for node in core.walk(root, (ast.Constant(value=str), ast.JoinedStr))
    ]
    removed_ranges = []
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

                removed_range = core.Range(start + start_offset, end - end_offset)

                if any(removed_range & other for other in removed_ranges):
                    continue
                if any(removed_range & other for other in code_ranges):
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

                if not (uncommented_block.strip() and core.is_valid_python(uncommented_block)):
                    continue
                any_line_is_a_path = False
                for line in filter(None, map(str.strip, uncommented_block.splitlines())):
                    try:
                        any_line_is_a_path |= Path(line).exists()
                    except OSError:
                        # Raised if a ling was too long. Which also means it's not a path.
                        pass

                if any_line_is_a_path:
                    continue

                parsed_content = core.parse(uncommented_block)
                if (
                    core.match_template(parsed_content.body, [ast.Expr])
                    and len(uncommented_block) < 20
                    and not isinstance(parsed_content.body[0].value, ast.Call)
                ):
                    continue

                if core.match_template(parsed_content.body, [ast.Name]):
                    continue

                # Magic comments should not be removed
                if any(core.filter_nodes(parsed_content.body, ast.Expr(value=ast.Name))):
                    continue
                if any(core.filter_nodes(parsed_content.body, ast.NamedExpr(target=ast.Name))):
                    continue

                if any(
                    name.id in {"pylint", "mypy", "flake8", "noqa", "type"}
                    for annassign in core.walk(parsed_content, ast.AnnAssign)
                    for name in core.walk(annassign, ast.Name)
                ):
                    continue

                yield removed_range, None
                removed_ranges.append(removed_range)


@processing.fix
def replace_with_filter(source: str) -> str:
    find_positive = """
    for {{target}} in {{iter}}:
        if {{target}}:
            {{body}}
    """
    find_negative = """
    for {{target}} in {{iter}}:
        if not {{target}}:
            continue
        {{body}}
    """
    replace = """
    for {{target}} in filter(None, {{iter}}):
        {{body}}
    """
    template = core.compile_template((find_positive, find_negative), expand="body")
    iterator1 = processing.find_replace(source, template, replace, yield_match=True)

    find_positive = """
    for {{target}} in {{iter}}:
        if {{test}}({{target}}):
            {{body}}
    """
    find_negative = """
    for {{target}} in {{iter}}:
        if not {{test}}({{target}}):
            continue
        {{body}}
    """
    replace = """
    for {{target}} in filter({{test}}, {{iter}}):
        {{body}}
    """
    template = core.compile_template((find_positive, find_negative), expand="body")
    iterator2 = processing.find_replace(source, template, replace, yield_match=True)

    filter_derivative_template = ast.Call(
        func=core.compile_template(("filter", "filterfalse", "itertools.filterfalse"))
    )

    for range_, replacement, template_match in itertools.chain(iterator1, iterator2):
        if not core.match_template(template_match.iter, filter_derivative_template):
            yield range_, replacement


def _get_contains_args(node: ast.Compare) -> Tuple[str, str, bool]:
    template = core.compile_template((
        "{{key}} in {{value}}",
        "{{key}} not in {{value}}",
        "not {{key}} in {{value}}",
        "not {{key}} not in {{value}}",
        ),
        key=ast.Name,
        value=ast.Name,
    )

    if template_match := core.match_template(node, template):
        if negative := isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            node = node.operand

        _, key, value = template_match
        if isinstance(node.ops[0], ast.In):
            return key.id, value.id, negative
        if isinstance(node.ops[0], ast.NotIn):
            return key.id, value.id, not negative

    raise ValueError(f"Node is not a pure compare node: {node}")


def _get_subscript_functions(node: ast.Expr) -> Tuple[str, str, str, str]:
    template = core.compile_template(
        "{{obj}}[{{key}}].{{call}}({{value}})", keep_expr=True, key=ast.Name, obj=ast.Name
    )
    if template_match := core.match_template(node, template):
        _, call, key, obj, value = template_match
        return obj.id, call, key.id, value

    raise ValueError(f"Node {node} is not a subscript call")


def _get_assign_functions(node: ast.Expr) -> Tuple[str, str]:
    template = core.compile_template("{{obj}}[{{key}}] = {{value}}", key=ast.Name, obj=ast.Name)
    if template_match := core.match_template(node, template):
        _, key, obj, _ = template_match
        value = node.value
        return obj.id, key.id, value

    raise ValueError(f"Node {node} is not a subscript assignment")


def _preferred_comprehension_type(node: ast.AST) -> ast.AST | ast.SetComp | ast.GeneratorExp:
    if isinstance(node, ast.ListComp):
        return ast.GeneratorExp(elt=node.elt, generators=node.generators)

    return node


@processing.fix
def implicit_defaultdict(source: str) -> str:
    assign_template = ast.Assign(
        targets=[core.Wildcard("target", ast.Name)],
        value=core.Wildcard("value", ast.Dict(keys=[], values=[])),
    )
    if_template = ast.If(body=[object], orelse=[])

    transaction = 0
    root = core.parse(source)
    for (_, target, value), (n2,) in core.walk_sequence(root, assign_template, ast.For):
        loop_replacements = {}
        loop_removals = set()
        subscript_calls = set()
        consistent = True
        transaction += 1

        for (condition,), (append,) in core.walk_sequence(n2, if_template, ast.Expr):
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
            if core.match_template(f_value, ast.List(elts=[])) and (t_call in {"append", "extend"}):
                loop_removals.add(condition)
                continue
            if core.match_template(f_value, ast.Call(func=ast.Name(id="set"), args=[])) and (
                t_call in {"add", "update"}
            ):
                loop_removals.add(condition)
                continue
            consistent = False
            break

        if_orelse_template = ast.If(body=[object], orelse=[object])
        for condition in core.walk(ast.Module(body=n2.body), if_orelse_template):
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
                and core.match_template(f_value, (ast.List(elts=[object]), ast.Set(elts=[object])))
                and (core.unparse(t_value) == core.unparse(f_value.elts[0]))
            ):
                if isinstance(f_value, ast.List) == (t_call == "append"):
                    loop_replacements[condition] = on_true
                    continue
                consistent = False
                break
            t_value_preferred = _preferred_comprehension_type(t_value)
            f_value_preferred = _preferred_comprehension_type(f_value)
            if core.unparse(t_value_preferred) == core.unparse(f_value_preferred) and t_call in {
                "update",
                "extend",
            }:
                loop_replacements[condition] = on_true
                continue

        if not consistent:
            continue

        replacements = []

        if subscript_calls and subscript_calls <= {"add", "update"}:
            replacements.extend(loop_replacements.items())
            replacements.extend(zip(loop_removals, itertools.repeat(None)))
            replacements.append((
                value,
                ast.Call(
                    func=ast.Attribute(value=ast.Name(id="collections"), attr="defaultdict"),
                    args=[ast.Name(id="set")],
                    keywords=[],
            ),))

        if subscript_calls and subscript_calls <= {"append", "extend"}:
            replacements.extend(loop_replacements.items())
            replacements.extend(zip(loop_removals, itertools.repeat(None)))
            replacements.append((
                value,
                ast.Call(
                    func=ast.Attribute(value=ast.Name(id="collections"), attr="defaultdict"),
                    args=[ast.Name(id="list")],
                    keywords=[],
            ),))

        for before, after in replacements:
            yield before, after, transaction


@processing.fix
def _replace_lambda_with_literal(source: str) -> str:
    for find, replace in (
        ("lambda: []", "list"),
        ("lambda: {}", "dict"),
        ("lambda: ()", "tuple"),
        ("lambda {{args}}: [*{{args}}]", "list"),
        ("lambda {{args}}: {*{{args}}}", "set"),
        ("lambda {{args}}: (*{{args}},)", "tuple"),
        ("lambda {{args}}, /: [*{{args}}]", "list"),
        ("lambda {{args}}, /: {*{{args}}}", "set"),
        ("lambda {{args}}, /: (*{{args}},)", "tuple"),
        ("lambda: {{func}}()", "{{func}}"),
    ):
        yield from processing.find_replace(source, find, replace)


@processing.fix
def _replace_lambda_with_function(source: str) -> str:
    find = ast.Lambda(
        args=core.Wildcard("sign_args", ast.arguments),
        body=ast.Call(
            func=core.Wildcard("func", ast.Name),
            args=core.Wildcard("call_args"),
            keywords=core.Wildcard("call_keywords"),
    ),)
    replace = "{{func}}"
    for replacement_range, replacement, template_match in processing.find_replace(
        source, find, replace, yield_match=True
    ):
        _, call_args, call_keywords, _, sign_args = template_match
        if sign_args.kw_defaults:
            continue

        expected_call_args = [
            ast.Name(id=arg.arg) for arg in sign_args.posonlyargs + sign_args.args
        ]
        if sign_args.vararg:
            expected_call_args.append(ast.Starred(value=ast.Name(id=sign_args.vararg.arg)))

        if not core.match_template(call_args, expected_call_args):
            continue

        expected_call_keywords = {
            ast.keyword(arg=arg, value=ast.Name(id=arg)) for arg in sign_args.kwonlyargs
        }
        if sign_args.kwarg:
            expected_call_keywords.add(ast.keyword(value=ast.Name(id=sign_args.kwarg.arg)))

        if not core.match_template(call_keywords, expected_call_keywords):
            continue

        yield replacement_range, replacement


@processing.fix
def simplify_redundant_lambda(source: str) -> str:
    yield from _replace_lambda_with_literal._fix_func(source)
    yield from _replace_lambda_with_function._fix_func(source)


def _is_same_code(*nodes: ast.AST) -> bool:
    return len({core.unparse(node) for node in nodes}) == 1


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
    scope: ast.AST, nodes: Iterable[ast.AST]
) -> Tuple[Collection[ast.AST], Collection[ast.AST]]:
    removals = set(nodes)
    first_node = min(removals, key=lambda node: node.lineno)
    replacement = copy.copy(first_node)
    replacement.lineno = scope.lineno - 1
    replacement.col_offset = scope.col_offset
    additions = {replacement}

    return additions, removals


def _move_after_scope(
    scope: ast.AST, nodes: Iterable[ast.AST]
) -> Tuple[Collection[ast.AST], Collection[ast.AST]]:
    removals = set(nodes)
    last_node = max(removals, key=lambda node: node.lineno)
    replacement = copy.copy(last_node)
    replacement.col_offset = scope.col_offset
    replacement.lineno = max(x.lineno for x in core.walk(scope, ast.AST(lineno=int))) + 1
    additions = {replacement}

    return additions, removals


def breakout_common_code_in_ifs(source: str) -> str:
    root = core.parse(source)
    for node, body, orelse in _iter_explicit_if_elses(root):
        if core.get_code(node, source).startswith("elif"):
            continue

        if not body or not orelse:
            continue

        removals = set()
        additions = set()
        has_namedexpr = any(core.walk(node.test, ast.NamedExpr))
        start_branches = [body[0], orelse[0]]
        end_branches = [body[-1], orelse[-1]]
        if not has_namedexpr and _is_same_code(*start_branches):
            additions, removals = _move_before_scope(node, start_branches)
        elif _is_same_code(*end_branches):
            additions, removals = _move_after_scope(node, end_branches)
        try:
            start_branches = list(_all_branches(body[0], orelse[0], expand_ifs_on="start"))
            end_branches = list(_all_branches(body[-1], orelse[-1], expand_ifs_on="end"))
        except (ValueError, IndexError):
            pass
        else:
            if not has_namedexpr and _is_same_code(*start_branches):
                additions, removals = _move_before_scope(node, start_branches)
            elif _is_same_code(*end_branches):
                additions, removals = _move_after_scope(node, end_branches)
            else:
                end_nonblocking_branches = [
                    branch for branch in end_branches if not core.is_blocking(branch)
                ]
                count = len(end_nonblocking_branches)
                if count >= 2 and _is_same_code(*end_nonblocking_branches):
                    additions, removals = _move_after_scope(node, end_nonblocking_branches)
        if core.match_template(list(additions), [ast.Pass]):
            continue
        if additions and removals:
            source = processing.alter_code(source, root, additions=additions, removals=removals)
            return breakout_common_code_in_ifs(source)
    for node, body, orelse in _iter_implicit_if_elses(root):
        if core.get_code(node, source).startswith("elif"):
            continue

        if not body or not orelse:
            continue

        removals = set()
        additions = set()
        has_namedexpr = any(core.walk(node.test, ast.NamedExpr))
        start_branches = [body[0], orelse[0]]
        if not has_namedexpr and _is_same_code(*start_branches):
            additions, removals = _move_before_scope(node, start_branches)
        try:
            start_branches = list(_all_branches(body[0], orelse[0], expand_ifs_on="start"))
        except (ValueError, IndexError):
            pass
        else:
            if not has_namedexpr and _is_same_code(*start_branches):
                additions, removals = _move_before_scope(node, start_branches)
        if core.match_template(list(additions), [ast.Pass]):
            continue
        if additions or removals:
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

    root = core.parse(source)

    for node in core.walk(root, ast.Constant(value=str)):
        code = core.get_code(node, source)
        # Normal string containing backslash but no valid escape sequences
        if (
            code
            and code[0] in "'\""
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
    # Prevent replacement of map() calls where the map() call is the iterated value of a for loop
    root = core.parse(source)
    for_ranges = {core.get_charnos(node.iter, source) for node in core.walk(root, ast.For())}

    find = "filter(lambda {{arg}}: {{body}}, {{iterable}})"
    replace = "({{arg}} for {{arg}} in {{iterable}} if {{body}})"
    for replacement_range, replacement in processing.find_replace(source, find, replace):
        if any(replacement_range & for_range for for_range in for_ranges):
            continue

        yield replacement_range, replacement

    find = "filterfalse(lambda {{arg}}: {{body}}, {{iterable}})"
    replace = "({{arg}} for {{arg}} in {{iterable}} if not {{body}})"
    for replacement_range, replacement in processing.find_replace(
        source, (find, "itertools." + find), replace
    ):
        if any(replacement_range & for_range for for_range in for_ranges):
            continue

        yield replacement_range, replacement


@processing.fix
def replace_map_lambda_with_comp(source: str) -> str:
    """Replace map(lambda ..., iterable) with equivalent list comprehension

    Args:
        source (str): Python source code

    Returns:
        str: Modified source code
    """
    find = "map(lambda {{arg}}: {{body}}, {{iterable}})"
    replace = "({{body}} for {{arg}} in {{iterable}})"

    # Prevent replacement of map() calls where the map() call is the iterated value of a for loop
    root = core.parse(source)
    for_ranges = {core.get_charnos(node.iter, source) for node in core.walk(root, ast.For())}
    for replacement_range, replacement in processing.find_replace(source, find, replace):
        if any(replacement_range & for_range for for_range in for_ranges):
            continue

        yield replacement_range, replacement


@processing.fix
def replace_negated_numeric_comparison(source: str) -> str:
    root = core.parse(source)
    template = core.compile_template("not {{compare}}", compare=ast.Compare(comparators=[object]))

    numeric_template = ast.Constant(value=(int, float))
    numeric_template = (
        numeric_template,
        ast.UnaryOp(op=ast.USub, operand=numeric_template),
        ast.BinOp(left=numeric_template),
        ast.BinOp(right=numeric_template),
    )
    safe_reversible_set_operations = (ast.Eq, ast.NotEq, ast.Is, ast.IsNot, ast.In, ast.NotIn)
    for template_match in core.walk_wildcard(root, template):
        node, compare, *_ = template_match
        if core.match_template(compare.ops[0], safe_reversible_set_operations) or (
            core.match_template(compare.ops[0], tuple(constants.REVERSE_OPERATOR_MAPPING))
            and (
                core.match_template(compare.left, numeric_template)
                or core.match_template(compare.comparators[0], numeric_template)
        )):
            yield node, ast.Compare(
                left=compare.left,
                ops=[constants.REVERSE_OPERATOR_MAPPING[type(compare.ops[0])]()],
                comparators=compare.comparators,
            )


@processing.fix
def merge_chained_comps(source: str) -> str:
    root = core.parse(source)

    template = ast.AST(
        elt=object,
        generators=[
            ast.comprehension(
                target=core.Wildcard("common_target", object),
                iter=ast.AST(
                    elt=core.Wildcard("common_target", object),
                    generators=[
                        ast.comprehension(
                            target=core.Wildcard("common_target", object),
                            iter=core.Wildcard("iter_inner", object),
                            ifs=core.Wildcard("ifs_inner", list),
                            is_async=0,
                )],),
                ifs=core.Wildcard("ifs_outer", list),
                is_async=0,
    )],)

    for template_match in core.walk_wildcard(root, template):
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
                    is_async=0,
        )],)

        yield template_match.root, replacement


@processing.fix
def remove_redundant_comprehension_casts(source: str) -> str:
    root = core.parse(source)

    template = ast.Call(
        func=ast.Name(id=core.Wildcard("func", ("list", "set", "iter"))),
        args=[core.Wildcard("comp", (ast.GeneratorExp, ast.ListComp, ast.SetComp))],
        keywords=[],
    )

    for node, comp, func in core.walk_wildcard(root, template):
        if func == "set":
            yield node, ast.SetComp(comp.elt, comp.generators)
        if func == "list" and not isinstance(comp, ast.SetComp):
            yield node, ast.ListComp(comp.elt, comp.generators)
        if func == "iter" and isinstance(comp, ast.GeneratorExp):
            yield node, comp
        if func == "iter" and isinstance(comp, ast.ListComp):
            yield ast.GeneratorExp(comp.elt, comp.generators)

    template = ast.Call(
        func=ast.Name(id=core.Wildcard("func", ("list", "set", "iter", "dict"))),
        args=[core.Wildcard("comp", (ast.DictComp))],
        keywords=[],
    )

    for node, comp, func in core.walk_wildcard(root, template):
        equivalent_setcomp = ast.SetComp(comp.key, comp.generators)
        if func == "dict":
            yield node, comp
        if func == "set":
            yield node, equivalent_setcomp
        if func in {"list", "iter"}:
            yield node, ast.Call(func=ast.Name(id=func), args=[equivalent_setcomp], keywords=[])


@processing.fix
def remove_redundant_chain_casts(source: str) -> str:
    root = core.parse(source)

    template = ast.Call(
        func=ast.Name(id=core.Wildcard("func_outer", ("list", "set", "iter", "tuple"))),
        args=[
            ast.Call(
                func=ast.Attribute(value=ast.Name(id="itertools"), attr="chain"),
                args=core.Wildcard("args", list),
                keywords=[],
        )],
        keywords=[],
    )

    for node, args, func_outer in core.walk_wildcard(root, template):
        if func_outer == "iter" and len(args) >= 1:
            yield node, node.args[0]
        if func_outer == "iter" and not args:
            yield node, ast.Call(func=ast.Name(id="iter"), args=[], keywords=[])
        elts = [ast.Starred(value=arg) for arg in args]
        if func_outer == "set" and elts:
            yield node, ast.Set(elts=elts)
        if func_outer == "set" and not elts:
            yield node, ast.Call(func=ast.Name(id="set"), args=[], keywords=[])
        if func_outer == "list":
            yield node, ast.List(elts=elts)
        if func_outer == "tuple":
            yield node, ast.Tuple(elts=elts)


@processing.fix
def replace_dict_assign_with_dict_literal(source: str) -> str:
    root = core.parse(source)

    target_template = core.Wildcard("target", ast.Name(id=str))
    value_template = ast.Dict(
        keys=core.Wildcard("keys", list), values=core.Wildcard("values", list)
    )
    template = [
        ast.Assign(targets=[target_template], value=value_template),
        ast.Assign(
            targets=[
                ast.Subscript(
                    value=target_template, slice=core.Wildcard("key", object, common=False)
            )],
            value=core.Wildcard("value", object, common=False),
    ),]

    for transaction, (first, *matches) in enumerate(
        core.walk_sequence(root, *template, expand_last=True)
    ):
        replacement = ast.Assign(
            targets=[first.target],
            value=ast.Dict(
                keys=first.keys + [m.key for m in matches],
                values=first.values + [m.value for m in matches],
            ),
            lineno=first.target.lineno,
        )
        yield first.root, replacement, transaction
        for m in matches:
            yield m.root, None, transaction


@processing.fix
def replace_dict_update_with_dict_literal(source: str) -> str:
    root = core.parse(source)

    target_template = core.Wildcard("target", ast.Name(id=str))
    value_template = ast.Dict(
        keys=core.Wildcard("keys", list, common=False),
        values=core.Wildcard("values", list, common=False),
    )
    template = [
        ast.Assign(targets=[target_template], value=value_template),
        ast.Expr(
            value=ast.Call(
                func=ast.Attribute(value=target_template),
                args=[core.Wildcard("other", object, common=False)],
    )),]

    for transaction, (first, *matches) in enumerate(
        core.walk_sequence(root, *template, expand_last=True)
    ):
        replacement = ast.Assign(
            targets=[first.target],
            value=ast.Dict(
                keys=first.keys + [None] * len(matches),
                values=first.values + [m.other for m in matches],
            ),
            lineno=first.target.lineno,
        )
        yield first.root, replacement, transaction
        for m in matches:
            yield m.root, None, transaction


@processing.fix
def replace_dictcomp_assign_with_dict_literal(source: str) -> str:
    root = core.parse(source)

    target_template = core.Wildcard("target", ast.Name(id=str))
    template = [
        ast.Assign(targets=[target_template], value=ast.DictComp),
        ast.Assign(
            targets=[
                ast.Subscript(
                    value=target_template, slice=core.Wildcard("key", object, common=False)
            )],
            value=core.Wildcard("value", object, common=False),
    ),]

    for transaction, (first, *matches) in enumerate(
        core.walk_sequence(root, *template, expand_last=True)
    ):
        replacement = ast.Assign(
            targets=[first.target],
            value=ast.Dict(
                keys=[None] + [m.key for m in matches],
                values=[first.root.value] + [m.value for m in matches],
            ),
            lineno=first.target.lineno,
        )
        yield first.root, replacement, transaction
        for m in matches:
            yield m.root, None, transaction


@processing.fix
def replace_dictcomp_update_with_dict_literal(source: str) -> str:
    root = core.parse(source)

    target_template = core.Wildcard("target", ast.Name(id=str))
    template = [
        ast.Assign(targets=[target_template], value=ast.DictComp),
        ast.Expr(
            value=ast.Call(
                func=ast.Attribute(value=target_template),
                args=[core.Wildcard("other", object, common=False)],
    )),]

    for transaction, (first, *matches) in enumerate(
        core.walk_sequence(root, *template, expand_last=True)
    ):
        replacement = ast.Assign(
            targets=[first.target],
            value=ast.Dict(
                keys=[None] * (1 + len(matches)),
                values=[first.root.value] + [m.other for m in matches],
            ),
            lineno=first.target.lineno,
        )
        yield first.root, replacement, transaction
        for m in matches:
            yield m.root, None, transaction


def _is_recursive_binop_chain(node: ast.AST, op: ast.AST) -> bool:
    if isinstance(node, ast.BinOp):
        return isinstance(node.op, op) and _is_recursive_binop_chain(node.right, op)

    return True


@processing.fix
def replace_setcomp_add_with_union(source: str) -> str:
    find = """
    {{variable}} = {{something}}
    for {{target}} in {{iterable}}:
        {{variable}}.add({{something_else}})
    """
    replace = """
    {{variable}} = {{something}} | {{{something_else}} for {{target}} in {{iterable}}}
    """
    find = core.compile_template(find, something=(ast.SetComp, ast.Set, ast.BinOp(op=ast.BitOr)))
    for before, after, template_match in processing.find_replace(
        source, find, replace, yield_match=True
    ):
        if isinstance(template_match.root, ast.BinOp):
            if _is_recursive_binop_chain(template_match.root, ast.BitOr):
                yield before, after
        else:
            yield before, after

    find = """
    {{variable}} = {{something}}
    {{variable}}.update({{something_else}})
    """
    replace = """
    {{variable}} = {{something}} | set({{something_else}})
    """
    find = core.compile_template(find, something=(ast.SetComp, ast.Set, ast.BinOp(op=ast.BitOr)))
    for before, after, template_match in processing.find_replace(
        source, find, replace, yield_match=True
    ):
        if isinstance(template_match.root, ast.BinOp):
            if _is_recursive_binop_chain(template_match.root, ast.BitOr):
                yield before, after
        else:
            yield before, after


@processing.fix
def replace_listcomp_append_with_plus(source: str) -> str:
    find = """
    {{variable}} = {{something}}
    for {{target}} in {{iterable}}:
        {{variable}}.append({{something_else}})
    """
    replace = """
    {{variable}} = {{something}} + [{{something_else}} for {{target}} in {{iterable}}]
    """
    find = core.compile_template(find, something=(ast.ListComp, ast.List, ast.BinOp(op=ast.Add)))
    for before, after, template_match in processing.find_replace(
        source, find, replace, yield_match=True
    ):
        if isinstance(template_match.root, ast.BinOp):
            if _is_recursive_binop_chain(template_match.root, ast.Add):
                yield before, after
        else:
            yield before, after

    find = """
    {{variable}} = {{something}}
    {{variable}}.extend({{something_else}})
    """
    replace = """
    {{variable}} = {{something}} + list({{something_else}})
    """
    find = core.compile_template(find, something=(ast.ListComp, ast.List, ast.BinOp(op=ast.Add)))
    for before, after, template_match in processing.find_replace(
        source, find, replace, yield_match=True
    ):
        if isinstance(template_match.root, ast.BinOp):
            if _is_recursive_binop_chain(template_match.root, ast.Add):
                yield before, after
        else:
            yield before, after


@processing.fix
def simplify_dict_unpacks(source: str) -> str:
    root = core.parse(source)

    for node in core.walk(root, ast.Dict):
        if not any((k is None and isinstance(v, ast.Dict) for k, v in zip(node.keys, node.values))):
            continue

        values = []
        keys = []
        for k, v in zip(node.keys, node.values):
            if k is None and isinstance(v, ast.Dict):
                keys.extend(v.keys)
                values.extend(v.values)
            else:
                keys.append(k)
                values.append(v)
        yield (node, ast.Dict(keys=keys, values=values))


@processing.fix
def simplify_collection_unpacks(source: str) -> str:
    root = core.parse(source)

    for node in core.walk(root, (ast.List, ast.Set, ast.Tuple)):
        replacements = False
        if not any((
            core.match_template(
                elt, ast.Starred(value=(ast.List, ast.Set, ast.Tuple, ast.Dict))
            )
            for elt in node.elts
        )):
            continue

        elts = []
        for elt in node.elts:
            if (
                core.match_template(elt, ast.Starred(value=(ast.List, ast.Tuple)))
                or core.match_template([node, elt], [ast.Set, ast.Starred(value=ast.Set)])
                or (
                    core.match_template(elt, ast.Starred(value=ast.Set))
                    and len(elt.value.elts) <= 1
            )):
                elts.extend(elt.value.elts)
                replacements = True
            elif core.match_template(  # Can't have a dict in a set, but you can have a dict's keys
                elt, ast.Starred(value=(ast.Dict))
            ) and (
                (isinstance(node, ast.Set) and None not in elt.value.keys)
                or len(elt.value.values) <= 1
            ):
                elts.extend(elt.value.keys)
                replacements = True
            else:
                elts.append(elt)
        if replacements:
            if isinstance(node, ast.Set) and (not elts):
                yield (node, ast.Call(func=ast.Name(id="set"), args=[], keywords=[]))
            else:
                yield (node, type(node)(elts=elts))


@processing.fix
def remove_duplicate_dict_keys(source: str) -> str:
    root = core.parse(source)

    for node in core.walk(root, ast.Dict):
        key_occurences = collections.defaultdict(set)
        for i, key in enumerate(node.keys):
            if isinstance(key, ast.Constant):
                key_occurences[key.value].add(i)

        keys = []
        values = []
        for i, (key, value) in enumerate(zip(node.keys, node.values)):
            if not isinstance(key, ast.Constant) or i == max(key_occurences[key.value]):
                keys.append(key)
                values.append(value)

        if len(keys) < len(node.keys):
            yield node, ast.Dict(keys=keys, values=values)


@processing.fix
def remove_duplicate_set_elts(source: str) -> str:
    root = core.parse(source)
    for node in core.walk(root, ast.Set):
        elt_occurences = collections.defaultdict(set)
        for i, elt in enumerate(node.elts):
            if isinstance(elt, ast.Constant):
                elt_occurences[elt.value].add(i)
        elts = [
            elt
            for i, elt in enumerate(node.elts)
            if not isinstance(elt, ast.Constant) or i == min(elt_occurences[elt.value])
        ]
        if len(elts) < len(node.elts):
            yield node, ast.Set(elts=elts)


@processing.fix
def replace_collection_add_update_with_collection_literal(source: str) -> str:
    root = core.parse(source)

    target_template = core.Wildcard("common_target", ast.Name(id=str), common=True)
    assign_template = core.compile_template(
        "{{common_target}} = {{other}}",
        other=(ast.List, ast.Set, ast.ListComp, ast.SetComp),
        common_target=target_template,
    )
    modify_template = ast.Expr(
        value=ast.Call(
            func=ast.Attribute(value=target_template, attr=("add", "update", "append", "extend")),
            keywords=[],
    ))
    template = [assign_template, modify_template]
    for transaction, (node, *matches) in enumerate(
        core.walk_sequence(root, *template, expand_last=True)
    ):
        assigned_value = node.root.value
        other_elts = []
        for m in matches:
            if m[0].value.func.attr in {"append", "add"}:
                other_elts.append(m[0].value.args[0])
            elif m[0].value.func.attr in {"extend", "update"}:
                for arg in m[0].value.args:
                    if isinstance(arg, (ast.List, ast.Tuple)):
                        other_elts.extend(arg.elts)
                    else:
                        other_elts.append(ast.Starred(value=arg))
            else:
                raise RuntimeError(f"Unexpected match type found: {type(m[0].value.func.attr)}")
        if isinstance(assigned_value, (ast.List, ast.Set)):
            elts = assigned_value.elts + other_elts

            if isinstance(assigned_value, ast.List):
                replacement = ast.List(elts=elts)
            else:
                replacement = ast.Set(elts=elts)

            yield assigned_value, replacement, transaction
            for m in matches:
                yield m.root, None, transaction

        elif isinstance(assigned_value, (ast.ListComp, ast.SetComp)):
            elts = [ast.Starred(value=assigned_value)] + other_elts

            if isinstance(assigned_value, ast.ListComp):
                replacement = ast.List(elts=elts)
            else:
                replacement = ast.Set(elts=elts)

            yield assigned_value, replacement, transaction
            for m in matches:
                yield m.root, None, transaction


@processing.fix
def breakout_starred_args(source: str) -> str:
    root = core.parse(source)

    # One element is unique, more than 1 may not be.
    # So, a 1-length set can safely be unpacked, but not a 2-length set.
    starred_arg_template = ast.Starred(value=(ast.List, ast.Tuple, ast.Set(elts=[object])))
    for node in core.walk(root, ast.Call):
        matched = False
        args = []
        for arg in node.args:
            if core.match_template(arg, starred_arg_template):
                args.extend(arg.value.elts)
                matched = True
            else:
                args.append(arg)

        if matched:
            yield node, ast.Call(func=node.func, args=args, keywords=node.keywords)


def _convert_to_string_formatting(fstring: ast.JoinedStr) -> Tuple[str, Sequence[ast.AST]]:
    fstring_template = ast.JoinedStr(
        values={
            ast.Constant(value=str),
            ast.FormattedValue(format_spec=(None, ast.JoinedStr(values=[ast.Constant(value=str)]))),
    })
    if not core.match_template(fstring, fstring_template):
        raise ValueError(f"Invalid input: {ast.dump(fstring)}")

    format_string_entries = []
    format_args = []
    for entry in fstring.values:
        if isinstance(entry, ast.Constant):
            format_string_entries.append(entry.value)
        elif isinstance(entry, ast.FormattedValue):
            if entry.format_spec:
                format_spec = entry.format_spec.values[0].value
            else:
                format_spec = ""
            format_string_entries.append("{" + format_spec + "}")  # This is ironic, isn't it
            format_args.append(entry.value)

    format_string = ast.Constant(value="".join(format_string_entries), kind=None)

    return format_string, format_args


@processing.fix
def deinterpolate_logging_args(source: str) -> str:
    """De-interpolate logging arguments.

    Interpolated logging arguments are bad-practice, since converting whatever args are sent to
    strings may be expensive.
    Pylint complains about this:
    https://pylint.pycqa.org/en/latest/user_guide/messages/warning/logging-format-interpolation.html
    There are also security problems (that this fix doesn't solve, but perhaps it may raise awareness
    of the badness of this practice):
    https://bugs.python.org/issue46200

    Args:
        source (str): Python source code

    Returns:
        str: Modified code
    """
    root = core.parse(source)
    logging_functions = ("info", "debug", "warning", "error", "critical", "exception", "log")
    logging_module = ast.Name(id="logging")
    logger_object = ast.Name(id=("log", "logger"))
    template = ast.Call(
        func=ast.Attribute(
            value=(logging_module, logger_object),
            attr=core.Wildcard("function_name", logging_functions),
        ),
        args=list,
        keywords=[],
    )
    fstring_template = ast.JoinedStr(
        values={
            ast.Constant(value=str),
            ast.FormattedValue(format_spec=(None, ast.JoinedStr(values=[ast.Constant(value=str)]))),
    })
    fmtstring_template = ast.Call(func=ast.Attribute(value=ast.Constant(value=str), attr="format"))
    for node, function_name in core.walk_wildcard(root, template):
        if function_name == "log" and core.match_template(node.args, [object, fmtstring_template]):
            yield node, ast.Call(
                func=node.func,
                args=[node.args[0], node.args[1].func.value] + node.args[1].args,
                keywords=node.keywords + node.args[1].keywords,
            )
        if function_name != "log" and core.match_template(node.args, [fmtstring_template]):
            yield node, ast.Call(
                func=node.func,
                args=[node.args[0].func.value] + node.args[0].args,
                keywords=node.keywords + node.args[0].keywords,
            )
        if function_name == "log" and core.match_template(node.args, [object, fstring_template]):
            format_string, format_args = _convert_to_string_formatting(node.args[1])
            yield node, ast.Call(
                func=node.func,
                args=[node.args[0], format_string] + format_args,
                keywords=node.keywords,
            )
        if function_name != "log" and core.match_template(node.args, [fstring_template]):
            format_string, format_args = _convert_to_string_formatting(node.args[0])
            yield node, ast.Call(
                func=node.func, args=[format_string] + format_args, keywords=node.keywords
            )


@processing.fix
def _keys_to_items(source: str) -> Iterable[Tuple[ast.AST, ast.AST]]:
    root = core.parse(source)
    comprehension_template = ast.comprehension(
        target=core.Wildcard("target", object),
        iter=ast.Call(
            func=ast.Attribute(value=core.Wildcard("value", object), attr="keys"),
            args=[],
            keywords=[],
    ),)
    template = (
        ast.SetComp(generators=[comprehension_template]),
        ast.ListComp(generators=[comprehension_template]),
        ast.GeneratorExp(generators=[comprehension_template]),
        ast.DictComp(generators=[comprehension_template]),
    )

    for transaction, (node, target, value) in enumerate(core.walk_wildcard(root, template)):
        subscript_template = core.compile_template(
            "{{value}}[{{target}}]", value=value, target=target
        )
        value_target_subscripts = list(
            core.walk(
                node,
                subscript_template,
                ignore=("ctx", "lineno", "end_lineno", "col_offset", "end_col_offset"),
        ))

        if not value_target_subscripts:
            continue

        node_target_name = f"{core.unparse(value)}_{core.unparse(target)}"
        node_target_name = re.sub("[^a-zA-Z]", "_", node_target_name)
        yield (
            node.generators[0].iter,
            ast.Call(func=ast.Attribute(value=value, attr="items"), args=[], keywords=[]),
            transaction,
        )

        yield target, ast.Tuple(elts=[target, ast.Name(id=node_target_name)]), transaction
        for subscript_use in value_target_subscripts:
            yield subscript_use, ast.Name(id=node_target_name), transaction


@processing.fix
def _items_to_keys(source: str) -> Iterable[Tuple[ast.AST, ast.AST]]:
    root = core.parse(source)
    template = ast.comprehension(
        target=ast.Tuple(elts=[core.Wildcard("target", object), ast.Name(id="_")]),
        iter=ast.Call(
            func=ast.Attribute(value=core.Wildcard("value", object), attr="items"),
            args=[],
            keywords=[],
        ),
        ifs=core.Wildcard("ifs", list),
        is_async=core.Wildcard("is_async", int),
    )
    for transaction, (node, _, _, target, value) in enumerate(core.walk_wildcard(root, template)):
        yield node.target, target, transaction
        yield node.iter, ast.Call(
            func=ast.Attribute(value=value, attr="keys"), args=[], keywords=[]
        ), transaction


@processing.fix
def _items_to_values(source: str) -> Iterable[Tuple[ast.AST, ast.AST]]:
    root = core.parse(source)
    template = ast.comprehension(
        target=ast.Tuple(elts=[ast.Name(id="_"), core.Wildcard("target", object)]),
        iter=ast.Call(
            func=ast.Attribute(value=core.Wildcard("value", object), attr="items"),
            args=[],
            keywords=[],
        ),
        ifs=core.Wildcard("ifs", list),
        is_async=core.Wildcard("is_async", int),
    )
    for transaction, (node, _, _, target, value) in enumerate(core.walk_wildcard(root, template)):
        yield node.target, target, transaction
        yield node.iter, ast.Call(
            func=ast.Attribute(value=value, attr="values"), args=[], keywords=[]
        ), transaction


@processing.fix
def _for_keys_to_items(source: str) -> Iterable[Tuple[ast.AST, ast.AST]]:
    root = core.parse(source)
    template = ast.For(
        target=core.Wildcard("target", object),
        iter=ast.Call(
            func=ast.Attribute(value=core.Wildcard("value", object), attr="keys"),
            args=[],
            keywords=[],
    ),)
    for transaction, (node, target, value) in enumerate(core.walk_wildcard(root, template)):
        subscript_template = core.compile_template(
            "{{value}}[{{target}}]", value=value, target=target
        )
        value_target_subscripts = list(
            core.walk(
                node,
                subscript_template,
                ignore=("ctx", "lineno", "end_lineno", "col_offset", "end_col_offset"),
        ))

        if not value_target_subscripts:
            continue

        node_target_name = f"{core.unparse(value)}_{core.unparse(target)}"
        node_target_name = re.sub("[^a-zA-Z]", "_", node_target_name)
        yield (
            node.iter,
            ast.Call(func=ast.Attribute(value=value, attr="items"), args=[], keywords=[]),
            transaction,
        )

        yield target, ast.Tuple(elts=[target, ast.Name(id=node_target_name)]), transaction
        for subscript_use in value_target_subscripts:
            yield subscript_use, ast.Name(id=node_target_name), transaction


@processing.fix
def _for_items_to_keys(source: str) -> Iterable[Tuple[ast.AST, ast.AST]]:
    root = core.parse(source)
    template = ast.For(
        target=ast.Tuple(elts=[core.Wildcard("target", object), ast.Name(id="_")]),
        iter=ast.Call(
            func=ast.Attribute(value=core.Wildcard("value", object), attr="items"),
            args=[],
            keywords=[],
    ),)

    for transaction, (node, target, value) in enumerate(core.walk_wildcard(root, template)):
        yield node.target, target, transaction
        yield node.iter, ast.Call(
            func=ast.Attribute(value=value, attr="keys"), args=[], keywords=[]
        ), transaction


@processing.fix
def _for_items_to_values(source: str) -> Iterable[Tuple[ast.AST, ast.AST]]:
    root = core.parse(source)
    template = ast.For(
        target=ast.Tuple(elts=[ast.Name(id="_"), core.Wildcard("target", object)]),
        iter=ast.Call(
            func=ast.Attribute(value=core.Wildcard("value", object), attr="items"),
            args=[],
            keywords=[],
    ),)

    for transaction, (node, target, value) in enumerate(core.walk_wildcard(root, template)):
        yield node.target, target, transaction
        yield node.iter, ast.Call(
            func=ast.Attribute(value=value, attr="values"), args=[], keywords=[]
        ), transaction


def implicit_dict_keys_values_items(source: str) -> str:
    source = _keys_to_items(source)
    source = _items_to_keys(source)
    source = _items_to_values(source)
    source = _for_keys_to_items(source)
    source = _for_items_to_keys(source)
    source = _for_items_to_values(source)
    return source


@processing.fix
def redundant_enumerate(source: str) -> str:
    root = core.parse(source)
    iter_template = ast.Call(
        func=ast.Name(id="enumerate"), args=[core.Wildcard("iter", object)], keywords=[]
    )
    target_template = ast.Tuple(elts=[ast.Name(id="_"), core.Wildcard("target", object)])

    template = (
        ast.comprehension(iter=iter_template, target=target_template),
        ast.For(iter=iter_template, target=target_template),
    )

    for transaction, (node, node_iter, target) in enumerate(core.walk_wildcard(root, template)):
        yield node.iter, node_iter, transaction
        yield node.target, target, transaction


@processing.fix
def unused_zip_args(source: str) -> str:
    root = core.parse(source)
    iter_template = ast.Call(
        func=core.Wildcard(
            "func",
            (
                ast.Name(id="zip"),
                ast.Name(id="zip_longest"),
                ast.Attribute(value=ast.Name(id="itertools"), attr="zip_longest"),
        ),),
        args=core.Wildcard("iter", list),
        keywords=[],
    )
    target_template = ast.Tuple(elts=core.Wildcard("elts", list))

    template = (
        ast.comprehension(iter=iter_template, target=target_template),
        ast.For(iter=iter_template, target=target_template),
    )

    safe_callables = parsing.safe_callable_names(root)
    for transaction, (node, elts, func, node_iter) in enumerate(core.walk_wildcard(root, template)):
        new_elts = []
        iters = []
        changes = False

        # Starred messed with the order of things in slightly complicated ways
        # TODO handle starred args
        if any(isinstance(arg, ast.Starred) for arg in elts):
            continue
        if any(isinstance(arg, ast.Starred) for arg in node_iter):
            continue

        for elt, arg in zip(elts, node_iter):
            if core.match_template(elt, ast.Name(id="_")) and not core.has_side_effect(
                arg, safe_callables
            ):
                changes = True
            else:
                new_elts.append(elt)
                iters.append(arg)

        if not changes:
            continue

        if len(new_elts) == 0:  # 0 args used => replace zip with first arg
            yield node.target, elts[0], transaction
            yield node.iter, node_iter[0], transaction
        elif len(new_elts) == 1:  # 1 arg used => replace zip with that arg
            yield node.target, new_elts[0], transaction
            yield node.iter, iters[0], transaction
        elif len(new_elts) > 1:  # > 1 args used => remove unused ones, keep zip
            yield node.target, ast.Tuple(elts=new_elts), transaction
            yield node.iter, ast.Call(func=func, args=iters, keywords=[]), transaction


@processing.fix
def simplify_assign_immediate_return(source: str) -> str:
    find = """
    {{name}} = {{value}}
    return {{name}}
    """
    replace = "return {{value}}"

    root = core.parse(source)
    for scope in core.walk(root, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # For every variable, how many times is it assigned in this scope?
        name_assign_counts = collections.Counter(
            target.id
            for assignment in core.walk(scope, (ast.Assign, ast.AnnAssign, ast.AugAssign))
            for target in core.filter_nodes(
                parsing.assignment_targets(assignment), ast.Name(id=str)
        ))

        names_assigned_only_once = tuple(
            name for name, count in name_assign_counts.items() if count == 1
        )
        name_template = ast.Name(id=names_assigned_only_once)

        yield from processing.find_replace(
            source,
            core.compile_template(find, value=object, name=name_template),
            replace=replace,
            root=scope,
        )


def missing_context_manager(source: str) -> str:
    root = core.parse(source)

    func_template = core.compile_template((
        "open",
        "requests.Session",
        "sqlite3.connect",
        "tempfile.TemporaryFile",
        "tempfile.NamedTemporaryFile",
        "tempfile.TemporaryDirectory",
        "zipfile.ZipFile",
        "tarfile.TarFile",
        "gzip.GzipFile",
        "bz2.BZ2File",
        "lzma.LZMAFile",
        "socket.socket",
        "multiprocessing.Pool",
        "subprocess.Popen",
        "ftplib.FTP",
        "ftplib.FTP_TLS",
        "smtplib.SMTP",
        "smtplib.SMTP_SSL",
        "imaplib.IMAP4",
        "imaplib.IMAP4_SSL",
        "poplib.POP3",
        "poplib.POP3_SSL",
        "ssl.wrap_socket",
        "psycopg2.connect",
        "pymysql.connect",
        "pyodbc.connect",
        "connection.cursor",
        "con.cursor",
    ))

    template = ast.Assign(
        targets=[core.Wildcard("target", ast.Name(id=str))],
        value=core.Wildcard("value", ast.Call(func=func_template)),
    )

    removals = []
    replacements = {}
    for (asmt, target, value), *nodes in core.walk_sequence(
        root, template, object, expand_last=True
    ):
        target_template = ast.Name(id=target.id)
        if any(
            isinstance(node, (ast.Yield, ast.Return)) and core.walk(node, target_template)
            for node in nodes
        ):
            continue

        if any(core.filter_nodes(nodes, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))):
            continue

        nodes = [tup[0] for tup in nodes]
        while nodes:
            if core.walk(nodes[-1], target_template):
                break

            nodes.pop()

        for i, node in enumerate(nodes):
            if core.match_template(
                node,
                (
                    ast.Call(func=ast.Attribute(value=target_template, attr="close")),
                    ast.Expr(
                        value=ast.Call(func=ast.Attribute(value=target_template, attr="close"))
            ),),):
                nodes = nodes[:i]  # prevent the close() going into the with statement
                removals.append(node)
                break

        removals.extend(nodes)

        replacements[asmt] = ast.With(
            items=[ast.withitem(context_expr=value, optional_vars=target)],
            body=[parsing.with_added_indent(node, 4) for node in nodes],
            lineno=asmt.lineno,
            col_offset=asmt.col_offset,
        )

        break

    if replacements:
        source = processing.alter_code(source, root, replacements=replacements, removals=removals)
        return missing_context_manager(source)

    return source


def _group_statements_of_type(root: ast.AST, template: core.Template) -> Sequence[Sequence[ast.AST]]:
    """Get unique groups of imports, such that they're as long as possible, and don't overlap."""
    groups = [
        [m[0] for m in matches]
        for matches in core.walk_sequence(root, template, expand_first=True, expand_last=True)
    ]
    node_groups = collections.defaultdict(list)
    for group in groups:
        for node in group:
            node_groups[node].append(group)
    node_groups = {node: max(node_groups, key=len) for node, node_groups in node_groups.items()}
    groups = {id(group): group for group in node_groups.values()}.values()
    groups = sorted(groups, key=lambda group: min(n.lineno for n in group))

    return groups


def _fix_duplicate_from_imports(source: str) -> str:
    """Remove duplicate from-style imports from the same module."""
    root = core.parse(source)

    replacements = {}
    removals = set()
    for group in _group_statements_of_type(root, ast.ImportFrom):
        module_import_aliases = collections.defaultdict(set)
        module_import_nodes = collections.defaultdict(list)

        for node in group:
            module_import_aliases[node.module].update(
                (alias.name, alias.asname if alias.asname != alias.name else None)
                for alias in node.names
            )
            module_import_nodes[node.module].append(node)

        for module, import_nodes in module_import_nodes.items():
            if len(import_nodes) > 1:
                replacements[import_nodes[0]] = ast.ImportFrom(
                    module=module,
                    names=[
                        ast.alias(name=name, asname=asname)
                        for name, asname in sorted(
                            module_import_aliases[module],
                            key=lambda t: (t[0], t[1] is not None, t[1]),
                    )],
                    level=import_nodes[0].level,
                )
                removals.update(import_nodes[1:])

    if replacements or removals:
        source = processing.alter_code(source, root, replacements=replacements, removals=removals)
        return _fix_duplicate_regular_imports(source)

    return source


def _fix_duplicate_regular_imports(source: str) -> str:
    """Remove duplicate plain imports from the same module."""
    root = core.parse(source)

    import_aliases = collections.defaultdict(set)
    import_nodes = collections.defaultdict(list)

    for node in core.walk(root, ast.Import):
        for alias in node.names:
            asname = (
                alias.asname
                if alias.asname != alias.name and alias.asname is not None
                else alias.name
            )
            name = alias.name

            import_nodes[asname].append(node)
            import_aliases[name].add(asname)

    replacements = {}
    removals = set()

    for asname, nodes in import_nodes.items():
        if len(nodes) > 1:
            for node in nodes[1:]:
                new_aliases = {
                    (alias.name, alias.asname if alias.asname != alias.name else None)
                    for alias in node.names
                    if (alias.asname or alias.name) != asname
                }
                new_names = [
                    ast.alias(name=name, asname=asname)
                    for name, asname in sorted(
                        new_aliases, key=lambda t: (t[0], t[1] is not None, t[1])
                )]
                if new_names:
                    replacements[node] = ast.Import(names=new_names)
                else:
                    removals.add(node)

    if replacements or removals:
        source = processing.alter_code(source, root, replacements=replacements, removals=removals)
        return _fix_duplicate_regular_imports(source)

    return source


def _breakout_stacked_imports(source: str) -> str:
    """Breakout stacked imports in the same statement onto separate lines."""
    root = core.parse(source)

    replacements = {}
    additions = set()

    for node in core.walk(root, ast.Import):
        if len(node.names) <= 1:
            continue

        names = sorted(
            {(alias.name, alias.asname) for alias in node.names},
            key=lambda t: (t[0], t[1] is not None, t[1]),
        )
        names = [
            ast.alias(name=name, asname=asname if asname != name else None)
            for name, asname in names
        ]
        replacements[node] = ast.Import(names=[names[0]])
        for name in names[1:]:
            additions.add(ast.Import(names=[name], lineno=node.lineno, col_offset=node.col_offset))

    if replacements or additions:
        source = processing.alter_code(source, root, replacements=replacements, additions=additions)

    return source


@processing.fix
def _fix_imported_attr_as_self(source: str) -> str:
    root = core.parse(source)

    template = ast.Import(
        names=[
            ast.alias(
                name=core.Wildcard("name", str),
                asname=core.Wildcard("asname", str)
    )])
    for node, asname, name in core.walk_wildcard(root, template):
        *names, last = name.split(".")
        if last == asname:
            if names:
                replacement = ast.ImportFrom(
                    module=".".join(names),
                    names=[ast.alias(name=last, asname=None)],
                    level=0,
                )
            else:
                replacement = ast.Import(names=[ast.alias(name=last, asname=None)])

            yield node, replacement


def _fix_imported_as_self_or_unsorted(source: str) -> str:
    root = core.parse(source)

    replacements = {}
    for node in core.walk(root, ast.Import):
        names = [(alias.name, alias.asname) for alias in node.names]
        expected_names = sorted(
            [(name, asname if asname != name else None) for name, asname in names],
            key=lambda t: (t[0], t[1] is not None, t[1]),
        )
        if names != expected_names:
            replacements[node] = ast.Import(
                names=[ast.alias(name=name, asname=asname) for name, asname in expected_names]
            )

    for node in core.walk(root, ast.ImportFrom):
        names = [(alias.name, alias.asname) for alias in node.names]
        expected_names = sorted(
            [(name, asname if asname != name else None) for name, asname in names],
            key=lambda t: (t[0], t[1] is not None, t[1]),
        )
        if names != expected_names:
            replacements[node] = ast.ImportFrom(
                module=node.module,
                names=[ast.alias(name=name, asname=asname) for name, asname in expected_names],
                level=node.level,
            )

    if replacements:
        source = processing.alter_code(source, root, replacements=replacements)

    return source


def fix_duplicate_imports(source: str) -> str:
    """Fix duplicate imports in the same statement."""

    source = _fix_duplicate_from_imports(source)
    source = _fix_duplicate_regular_imports(source)
    source = _breakout_stacked_imports(source)
    source = _fix_imported_attr_as_self(source)

    return source


def _is_stdlib(node: ast.Import | ast.ImportFrom) -> bool:
    """Determine if all modules in an import statement are from the standard library."""
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0] in constants.PYTHON_311_STDLIB

    if isinstance(node, ast.Import):
        return {alias.name.split(".")[0] for alias in node.names} <= constants.PYTHON_311_STDLIB

    raise ValueError(f"Expected Import or ImportFrom, got {type(node)}")


def _is_future(node: ast.Import | ast.ImportFrom) -> bool:
    if isinstance(node, ast.ImportFrom):
        return node.module == "__future__"

    return False


def _import_group_key(node: ast.Import | ast.ImportFrom) -> Tuple[int, int]:
    return (
        not (isinstance(node, ast.ImportFrom) and node.module == "__future__"),
        not _is_stdlib(node),
        node.level != 0 if isinstance(node, ast.ImportFrom) else False,
        -node.level if isinstance(node, ast.ImportFrom) else 0,
        isinstance(node, ast.ImportFrom) and node.module is not None,
        isinstance(node, ast.ImportFrom),
        (node.module or "") if isinstance(node, ast.ImportFrom) else "",
        tuple(sorted(alias.name for alias in node.names)),
        tuple(sorted(alias.asname or alias.name for alias in node.names)),
    )


def _sort_import_statements(source: str) -> str:
    root = core.parse(source)
    replacements = {}
    for nodes in _group_statements_of_type(root, (ast.Import, ast.ImportFrom)):
        if len(nodes) < 2 or set(nodes) & replacements.keys():
            continue

        sorted_nodes = sorted(nodes, key=_import_group_key)
        for node, sorted_node in zip(nodes, sorted_nodes):
            if node is not sorted_node:
                replacements[node] = sorted_node

    if replacements:
        source = processing.alter_code(source, root, replacements=replacements)

    return source


def fix_import_spacing(source: str) -> str:
    root = core.parse(source)

    template = (ast.Import, ast.ImportFrom)
    replacements = {}
    for (i1, *_), (i2, *_) in core.walk_sequence(root, ast.AST, ast.AST):
        _, i1_end = core.get_charnos(i1, source)
        i2_start, i2_end = core.get_charnos(i2, source)
        whitespace_between = source[i1_end:i2_start]

        if set(whitespace_between) - set("\n "):
            continue

        if isinstance(i1, template) and isinstance(i2, template):
            if _is_stdlib(i1) == _is_stdlib(i2) and _is_future(i1) == _is_future(i2):
                correct_newline_count = 1
            else:
                correct_newline_count = 2

        elif isinstance(i1, template) or isinstance(i2, template):
            if isinstance(i2, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                correct_newline_count = 3
            else:
                correct_newline_count = 2

        else:
            continue

        indentation_level = formatting.indentation_level(
            whitespace_between + source[i2_start:i2_end]
        )
        spacing = "\n" * correct_newline_count + " " * indentation_level
        spacing = re.sub(r"\n +\n", "\n\n", spacing)
        replacement_range = core.Range(i1_end, i2_start)

        current_newline_count = whitespace_between.count("\n")
        if correct_newline_count == 1 and current_newline_count > 1:
            replacements[replacement_range] = spacing
        elif correct_newline_count != current_newline_count > 0:
            replacements[replacement_range] = spacing

    new_source = source
    for replacement_range in sorted(replacements, reverse=True):
        new_source = (
            new_source[: replacement_range.start]
            + replacements[replacement_range]
            + new_source[replacement_range.end :]
        )

    if core.is_valid_python(new_source):
        return new_source

    return source


def sort_imports(source: str) -> str:
    """Sort imports in alphabetic order. Respect existing import groups."""

    source = _sort_import_statements(source)
    source = _fix_imported_as_self_or_unsorted(source)
    source = fix_import_spacing(source)

    return source


@processing.fix
def fix_if_return(source: str) -> str:
    find = """
    if {{condition}}:
        return True
    return False
    """
    replace = "return {{condition}}"

    yield from processing.find_replace(source, find, replace, transaction=0)

    find = """
    if {{condition}}:
        return False
    return True
    """
    replace = "return not ({{condition}})"

    yield from processing.find_replace(source, find, replace, condition=ast.BoolOp, transaction=1)

    find = """
    if {{condition}}:
        return False
    return True
    """
    replace = "return not {{condition}}"

    yield from processing.find_replace(source, find, replace, transaction=2)


@processing.fix
def fix_if_assign(source: str) -> str:
    find = """
    if {{condition}}:
        {{variable}} = True
    else:
        {{variable}} = False
    """
    replace = "{{variable}} = {{condition}}"

    yield from processing.find_replace(source, find, replace, transaction=0)

    find = """
    if {{condition}}:
        {{variable}} = False
    else:
        {{variable}} = True
    """
    replace = "{{variable}} = not ({{condition}})"

    yield from processing.find_replace(source, find, replace, condition=ast.BoolOp, transaction=1)

    find = """
    if {{condition}}:
        {{variable}} = False
    else:
        {{variable}} = True
    """
    replace = "{{variable}} = not {{condition}}"

    yield from processing.find_replace(source, find, replace, transaction=2)


@processing.fix
def fix_raise_missing_from(source: str) -> str:
    find = """
    try:
        {{stuff}}
    except {{exception}}:
        raise {{something}}
    """
    replace = """
    try:
        {{stuff}}
    except {{exception}} as error:
        raise {{something}} from error
    """
    yield from processing.find_replace(source, find, replace)


@processing.fix
def remove_redundant_boolop_values(source: str) -> str:

    root = core.parse(source)

    unknown = object()
    falsy = object()
    truthy = object()

    for node in core.walk(root, ast.BoolOp(op=(ast.Or, ast.And))):
        mask = []
        for value in node.values:
            try:
                deterministic_value = core.literal_value(value)
            except ValueError:
                mask.append(unknown)
            else:
                mask.append(truthy if deterministic_value else falsy)

        redundant = [False for _ in mask]

        for i, (truthyness, next_truthyness) in enumerate(zip(mask[:-1], mask[1:])):
            if truthyness is next_truthyness is truthy:
                if isinstance(node.op, ast.Or):
                    redundant[i + 1] = True
                if isinstance(node.op, ast.And):
                    redundant[i] = True

            if truthyness is next_truthyness is falsy:
                if isinstance(node.op, ast.Or):
                    redundant[i] = True
                if isinstance(node.op, ast.And):
                    redundant[i + 1] = True

            if truthyness is falsy and isinstance(node.op, ast.And):
                redundant[i + 1:] = [True] * (len(mask) - i - 1)
                break

            if truthyness is falsy and next_truthyness is not falsy and isinstance(node.op, ast.Or):
                redundant[i] = True

            if truthyness is truthy and next_truthyness is not truthy and isinstance(node.op, ast.And):
                redundant[i] = True

        values = [value for value, is_redundant in zip(node.values, redundant) if not is_redundant]
        if len(values) == 1:
            yield node, values[0]
        elif len(values) < len(node.values):
            new_node = ast.BoolOp(op=node.op, values=values)
            new_node = ast.copy_location(new_node, node)
            yield node, new_node
