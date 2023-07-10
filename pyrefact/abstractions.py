import ast
import collections
import itertools
import re
from typing import Collection, Iterable, Sequence, Tuple

from pyrefact import constants, core, parsing, processing, style, tracing


class _EverythingContainer:
    """Object that contains everything."""

    @staticmethod
    def __contains__(_) -> bool:
        return True


def _scoped_dependencies(node: ast.AST):
    return {name.id for name in parsing.iter_assignments(node)}


def hash_node(
    node: ast.AST, preserved_callable_names: Collection[str] = _EverythingContainer()
) -> int:
    """Compute a hash for a node, such that equivalent nodes should get the same hash.

    For example, the expressions (lambda x: x) and (lambda y: y) should get the same hash.

    Args:
        node (ast.AST): AST to hash.
        preserved_callable_names (Collection[str], optional): Names to preserve. By default all names.

    Returns:
        int: Hash of node.
    """
    name_increment = 0
    name_hashes = {}

    things_to_hash = []
    for child in ast.walk(node):
        things_to_hash.append(type(child))
        names = []
        if isinstance(child, ast.Name):
            names = [child.id]
        elif isinstance(child, ast.arg):
            names = [child.arg]
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [child.name]
        else:
            things_to_hash.extend(
                (key, value)
                for key, value in child.__dict__.items()
                if isinstance(value, (str, int))
                if key not in {"lineno", "end_lineno", "col_offset", "end_col_offset"}
            )
        for name in names:
            if name in preserved_callable_names:
                things_to_hash.append(name)

                continue

            if name in name_hashes:
                things_to_hash.append(name_hashes[name])

                continue

            name_increment += 1
            name_hashes[name] = name_increment
            things_to_hash.append(name_increment)

    return hash(tuple(things_to_hash))


def _possible_external_effects(node: ast.AST, safe_callables: Collection[str]) -> Iterable[ast.AST]:
    """Things that could possibly happen when going into a node

    Args:
        node (ast.AST): Ast node to enter
        safe_callables (Collection[str]): Function names that are safe to call

    Yields:
        ast.AST: Node that could potentially be encountered, and that may have some effect.
    """
    comprehension_local_vars = {comp.target for comp in core.walk(node, ast.comprehension)}
    for child in core.walk(
        node, (ast.Name(ctx=ast.Store), ast.Subscript(ctx=ast.Store, value=ast.Name))
    ):
        if child not in comprehension_local_vars:
            yield child
    halting_types = (ast.Yield, ast.YieldFrom, ast.Continue, ast.Break, ast.Return)
    for child in core.walk(node, halting_types):
        yield child
    for child in core.walk(node, ast.Call):
        if not (isinstance(child.func, ast.Name) and child.func.id in safe_callables):
            yield child


def _definite_external_effects(
    node: ast.AST, safe_callables: Collection[str]
) -> Iterable[Sequence[ast.AST]]:
    """Things that will surely happen when going into a node

    Args:
        node (ast.AST): Ast node to enter
        safe_callables (Collection[str]): Function names that are safe to call

    Yields:
        ast.AST: Node that will be encountered, and that may have some effect.
    """
    if isinstance(node, (ast.Break, ast.Continue)):
        yield node
        return
    if not isinstance(node, (ast.If, ast.IfExp)):
        yield from _possible_external_effects(node, safe_callables)
        return

    body_effects = {}
    for child in node.body:
        body_effects.update({
            hash_node(n, safe_callables): n
            for n in _definite_external_effects(child, safe_callables)
        })
        if core.is_blocking(node):
            break
    orelse_effects = {}
    for child in node.body:
        orelse_effects.update({
            hash_node(n, safe_callables): n
            for n in _definite_external_effects(child, safe_callables)
        })
        if core.is_blocking(node):
            break
    for key in body_effects.keys() & orelse_effects:
        yield body_effects[key]


def _definite_stored_names(node: ast.AST) -> Iterable[str]:
    for child in _definite_external_effects(node, _EverythingContainer()):
        if core.match_template(child, ast.Name(ctx=ast.Store)):
            yield child.id
        elif core.match_template(child, ast.Subscript(ctx=ast.Store, value=ast.Name)):
            yield child.value.id


def _hashable_node_purpose_type(node: ast.AST) -> Tuple[str]:
    if isinstance(node, (ast.Continue, ast.Break)):
        return (type(node),)
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr, ast.If)):
        side_effects = [
            child.value if isinstance(child, ast.Subscript) else child
            for child in _possible_external_effects(node, _EverythingContainer())
        ]
        if all(isinstance(child, ast.Name) for child in side_effects):
            stored_names = set(_definite_stored_names(node))
            if all(effect.id in stored_names for effect in side_effects):
                return tuple([ast.Assign] + sorted(stored_names))
        elif all(isinstance(node, ast.Break) for node in side_effects):
            return (ast.Break,)
        elif all(isinstance(node, ast.Continue) for node in side_effects):
            return (ast.Continue,)

        raise ValueError(f"Node {node} of type {type(node)} has no singular purpose")

    raise NotImplementedError(f"Cannot determine hashable node type of node of type {type(node)}")


def _group_nodes_by_purpose(body: Sequence[ast.AST]) -> Iterable[Sequence[ast.AST]]:
    """Group nodes by common purpose

    Args:
        body (Sequence[ast.AST]): Sequence of ast nodes to group ifs in.
        preserved_callable_names (Collection[str]): Callable names that should be preserved.

    Yields:
        Sequence[ast.AST]: A number of ifs that occur in sequence, and that have the same body.
    """
    if len(body) <= 1:
        return

    hashes = []
    for child in body:
        try:
            hashes.append(_hashable_node_purpose_type(child))
        except (NotImplementedError, ValueError):
            hashes.append(None)

    nodes = []
    iterator = iter(range(len(body)))
    i = next(iterator)
    if hashes[i] is not None:
        nodes.append(body[i])

    for i in iterator:
        use = hashes[i] is not None
        append = hashes[i] == hashes[i - 1]

        if use and append:
            nodes.append(body[i])
        elif use:
            if nodes:
                yield nodes
            nodes = [body[i]]

    if nodes:
        yield nodes


def _build_function_body(
    nodes: Sequence[ast.AST], return_injection_type: ast.AST
) -> ast.FunctionDef:
    """Build function from nodes.

    All effects should assign the same variable, or call the same function.

    Args:
        node_effects (Sequence[Tuple[ast.AST, ast.AST]]): Tuples of (test, effect)

    Raises:
        NotImplementedError: If all effects are not of type (Assign, Break, Continue, Call)

    Returns:
        FunctionDef: Function created from effects
    """
    body = []
    for node in nodes:
        if return_injection_type is not None and isinstance(node, return_injection_type):
            if return_injection_type == ast.Continue:
                body.append(ast.Return(value=ast.Constant(value=False, kind=None)))
            elif return_injection_type == ast.Break:
                body.append(ast.Return(value=ast.Constant(value=True, kind=None)))
            elif return_injection_type == ast.Assign:
                # The purpose of the function is just to assign some variable, so we return
                # whatever it would have been assigned to.
                body.append(ast.Return(value=node.value))
            else:
                raise NotImplementedError(f"Unknown injection type: {return_injection_type}")
        elif isinstance(node, ast.If):
            body.append(
                ast.If(
                    test=node.test,
                    body=_build_function_body(node.body, return_injection_type),
                    orelse=_build_function_body(node.orelse, return_injection_type),
            ))
        elif isinstance(node, ast.With):
            body.append(
                ast.With(
                    items=node.items, body=_build_function_body(node.body, return_injection_type)
            ))
        elif isinstance(node, ast.For):
            body.append(
                ast.For(
                    target=node.target,
                    iter=node.iter,
                    body=node.body,
                    orelse=_build_function_body(node.orelse, return_injection_type),
            ))
        elif isinstance(node, ast.While):
            body.append(
                ast.While(
                    test=node.test,
                    body=node.body,
                    orelse=_build_function_body(node.orelse, return_injection_type),
            ))
        else:
            body.append(node)

    return body


def _get_function_insertion_lineno(
    containing_node: ast.AST, function_def_linenos: Collection[int], import_linenos: Collection[int]
) -> int:
    """Get line number to insert new function at.

    Args:
        containing_node (ast.AST): Node that originally contained the logic in the function.
        function_def_linenos (Collection[int]): Line numbers of function defs.
        import_linenos (Collection[int]): Line numbers of imports.

    Returns:
        int: Line number to insert function definition at
    """
    if isinstance(containing_node, ast.Module):
        if function_def_linenos:
            return min(function_def_linenos) - 1
        if import_linenos:
            return max(import_linenos) + 1

        return 1

    if containing_node.lineno in function_def_linenos:
        return containing_node.lineno - 1

    if function_def_linenos:
        if all((lineno > containing_node.lineno for lineno in function_def_linenos)):
            return max(import_linenos) if import_linenos else containing_node.lineno - 1

        return (
            max((lineno for lineno in function_def_linenos if lineno <= containing_node.lineno)) - 1
        )

    if import_linenos:
        return max(import_linenos) + 1

    return containing_node.lineno - 1


def _get_constant_insertion_lineno(scope: ast.AST) -> int:
    import_types = (ast.Import, ast.ImportFrom)
    imports = [node for node in scope.body if not isinstance(node, import_types)]
    return min((node.lineno for node in imports)) - 1


def create_abstractions(source: str) -> str:
    root = core.parse(source)
    global_names = (
        _scoped_dependencies(root)
        | tracing.get_imported_names(root)
        | constants.BUILTIN_FUNCTIONS
        | {node.name for node in parsing.iter_funcdefs(root)}
        | {node.name for node in parsing.iter_classdefs(root)}
    )

    replacements = {}
    additions = []
    removals = []
    abstraction_count = 0

    function_def_linenos = []
    import_linenos = []

    for node in core.walk(root, (ast.AsyncFunctionDef, ast.FunctionDef)):
        candidates = [node.lineno] + [x.lineno for x in node.decorator_list]
        function_def_linenos.append(min(candidates))
    for node in core.walk(root, (ast.Import, ast.ImportFrom)):
        import_linenos.append(node.lineno)

    for node in itertools.chain([root], parsing.iter_bodies_recursive(root)):
        for nodes in _group_nodes_by_purpose(node.body):
            if len(nodes) > len(node.body) - 2:
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and all(
                core.match_template(child, (ast.Return, ast.Expr(value=ast.Constant)))
                or child in nodes
                for child in node.body
            ):
                continue

            purposes = {_hashable_node_purpose_type(child) for child in nodes}
            assert len(purposes) == 1
            purpose = purposes.pop()

            if core.match_template(nodes, [(ast.Assign, ast.AnnAssign)]):
                continue

            children_with_purpose = sum(
                isinstance(grandchild, purpose[0])
                for child in nodes
                for grandchild in ast.walk(child)
            )
            if children_with_purpose <= 2:
                continue

            created_names, _, required_names = tracing.code_dependencies_outputs(nodes)
            if len(created_names) > 1:
                continue

            abstraction_count += 1
            function_name = f"_pyrefact_abstraction_{abstraction_count}"
            while function_name in global_names:
                abstraction_count += 1
                function_name = f"_pyrefact_abstraction_{abstraction_count}"

            args = sorted(
                required_names
                - (global_names - _scoped_dependencies(root) if node is root else global_names)
            )
            # Abstractions with too many arguments are bad
            if len(args) > 4:
                continue

            call_args = [ast.Name(id=arg, ctx=ast.Load()) for arg in args]
            signature_args = ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=arg, annotation=None) for arg in args],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=None,
                kwarg=None,
                defaults=[],
            )
            return_args = sorted(set(created_names))

            if set(args) & {"self", "cls"}:  # Not implemented
                continue

            call = ast.Call(
                func=ast.Name(id=function_name, ctx=ast.Load()),
                args=call_args,
                keywords=[],
                starargs=[],
                kwargs=[],
            )

            if purpose[0] in (ast.Continue, ast.Break):
                if purpose[0] == ast.Continue:
                    call = ast.UnaryOp(op=ast.Not(), operand=call)
                function_call = ast.If(
                    test=call, body=[purpose[0](col_offset=nodes[0].col_offset + 4)], orelse=[]
                )
                return_value = purpose[0] == ast.Continue
                function_body = _build_function_body(nodes, purpose[0]) + [
                    ast.Return(value=ast.Constant(value=return_value, kind=None))
                ]
                returns = ast.Name(id="bool", ctx=ast.Load())

            elif purpose[0] == ast.Assign:
                assign_targets = [ast.Name(id=arg, ctx=ast.Store()) for arg in return_args]
                return_targets = [ast.Name(id=arg, ctx=ast.Load()) for arg in return_args]
                if len(assign_targets) > 1:
                    assign_targets = [ast.Tuple(elts=assign_targets)]
                    return_targets = [ast.Tuple(elts=return_targets)]
                function_call = ast.Assign(
                    targets=assign_targets, value=call, lineno=nodes[0].lineno
                )
                ifs = [c for n in nodes for c in core.walk(n, ast.If)]

                pure_nested_if = len(nodes) == 1 and all(
                    len(n.body) == len(n.orelse) == 1 for n in ifs
                )
                function_body = _build_function_body(nodes, purpose[0] if pure_nested_if else None)
                if not pure_nested_if:
                    if not isinstance(nodes[0], (ast.Assign, ast.AnnAssign)) and not all(
                        len(n.body) == len(n.orelse) == 1 for n in core.walk(nodes[0], ast.If)
                    ):
                        continue
                    function_body.append(ast.Return(value=return_targets))
                returns = None

            else:
                continue

            insertion_lineno = _get_function_insertion_lineno(
                node, function_def_linenos, import_linenos
            )

            function_def = ast.FunctionDef(
                name=function_name,
                args=signature_args,
                body=function_body,
                decorator_list=[],
                type_params=[],
                returns=returns,
                lineno=insertion_lineno,
                end_lineno=insertion_lineno + len(function_body),  # This isn't necessarily accurate
                col_offset=0,  # May be inaccurate
                end_col_offset=0,  # Definitely inaccurate
            )

            if isinstance(function_call, ast.Assign) and not return_args:
                continue  # Probably a property of an input arg being modified, I don't particularly like these anyways

            if isinstance(function_call, (ast.Continue, ast.Break)) and len(return_args) != 1:
                raise RuntimeError("Found bool abstraction without return")

            after_nodes = False
            nodes_after_abstraction = []
            for child in node.body:
                if after_nodes:
                    nodes_after_abstraction.append(child)
                elif child is nodes[-1]:
                    after_nodes = True

            is_singular_return_reassignment = (
                isinstance(function_call, ast.Assign)
                and len(return_args) == 1
                and core.match_template(
                    nodes_after_abstraction,
                    [(
                        ast.Return(value=ast.Name),
                        ast.Expr(
                            value=(ast.Yield(value=ast.Name), ast.YieldFrom(value=ast.Name))
            ),)],))

            if is_singular_return_reassignment and isinstance(
                nodes_after_abstraction[0], ast.Return
            ):
                col_offset = nodes[0].col_offset
                lineno = nodes[0].lineno
                for i, child in enumerate(function_body):
                    child.lineno = lineno + i
                    child.col_offset = col_offset
                removals.extend(nodes)
                additions.extend(function_body)
                continue

            if not is_singular_return_reassignment:
                replacements[nodes[0]] = function_call
                removals.extend(nodes[1:])
                additions.append(function_def)
                continue

            if isinstance(nodes_after_abstraction[0], ast.Return):
                inlined_return_call = ast.Return(
                    value=function_call.value, lineno=function_call.lineno
                )
            else:
                inlined_return_call = ast.Expr(
                    type(nodes_after_abstraction[0].value)(value=function_call.value),
                    lineno=function_call.lineno,
                )
            replacements[nodes_after_abstraction[0]] = inlined_return_call
            removals.extend(nodes)
            additions.append(function_def)

    return processing.alter_code(
        source, root, additions=additions, removals=removals, replacements=replacements
    )


def overused_constant(source: str, *, root_is_static: bool) -> str:
    """Create variables for overused constants

    Args:
        source (str): Python source code
        root_is_static (bool): True if the outermost scope is module-level

    Returns:
        str: Modified source code
    """
    root = core.parse(source)

    template = (
        ast.Constant,
        ast.Dict(keys={ast.Constant}, values={ast.Constant}),
        ast.Set(elts={ast.Constant}),
        ast.Tuple(elts={ast.Constant}),
        ast.List(elts={ast.Constant}),
    )
    blacklisted_names = (
        tracing.get_imported_names(root)
        | tracing.get_defined_names(root)
        | constants.BUILTIN_FUNCTIONS
        | constants.PYTHON_KEYWORDS
    )
    blacklisted_names |= {name.upper() for name in blacklisted_names}
    blacklisted_names |= {name.lower() for name in blacklisted_names}

    candidates = set(core.walk(root, template))

    for fstring in core.walk(root, ast.JoinedStr):
        for node in core.walk(fstring, ast.AST):
            candidates.discard(node)

    # For every node, all scopes it can be found in
    scope_node_definitions = collections.defaultdict(set)
    for scope in itertools.chain([root], core.walk(root, (ast.FunctionDef, ast.AsyncFunctionDef))):
        for node in core.walk(scope, ast.AST):
            scope_node_definitions[node].add(scope)

    for scope in itertools.chain([root], core.walk(root, ast.AST(body=list))):
        if scope.body and core.match_template(scope.body[0], ast.Expr(value=ast.Constant)):
            candidates.discard(scope.body[0].value)

    code_node_mapping = collections.defaultdict(set)
    for node in candidates:
        code_node_mapping[core.unparse(node)].add(node)

    replacements = {}
    additions = set()

    i = 0
    while f"pyrefact_overused_constant_{i}" in blacklisted_names and i < 10:
        i += 1

    if f"pyrefact_overused_constant_{i}" in blacklisted_names:
        return source

    for code, nodes in sorted(code_node_mapping.items(), key=lambda t: t[0]):
        if len(nodes) < 5:
            continue
        if len(re.sub(r"\s", "", code)) < 20:
            continue

        common_scopes = set.intersection(*(scope_node_definitions[node] for node in nodes))

        # root is a Module and has no lineno
        best_common_scope = max(
            common_scopes, key=lambda node: getattr(node, "lineno", 1), default=root
        )
        nodes = list(nodes)
        if (
            core.match_template(nodes[0], ast.Constant(value=str))
            and re.match(r"[a-zA-Z_]\w*", nodes[0].value)
            and re.sub(r"[^a-zA-Z0-9_]", "", nodes[0].value)
        ):
            variable_name = nodes[0].value
        else:
            variable_name = f"pyrefact_overused_constant_{i}"
            i += 1

        variable_name = style.rename_variable(
            variable_name, static=best_common_scope is root and root_is_static, private=False
        )

        name = ast.Name(id=variable_name)
        assign = core.parse(f"{variable_name} = {code}").body[0]
        assign.lineno = _get_constant_insertion_lineno(best_common_scope)
        assign.col_offset = best_common_scope.body[0].col_offset
        additions.add(assign)
        replacements.update({node: name for node in nodes})

    return processing.alter_code(source, root, additions=additions, replacements=replacements)


def simplify_if_control_flow(source: str) -> str:
    root = core.parse(source)

    # This should run after the first run of breakout_common_code_in_ifs(), so that code that
    # can be broken out without this having run first can be broken out first. That way, this
    # doesn't trigger unless it enables breakout_common_code_in_ifs() to do work it wouldn't
    # otherwise have done.

    for node in core.walk(root, ast.If):
        if not node.orelse:
            continue

        # Assignments make things more complicated. For example, the variables fed into the
        # if body/orelse blocks may be mutated and then used later on, or they may be changed
        # inside the node, etc. I won't deal with these cases for now.
        if any(core.walk(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr))):
            continue

        node_body_names = sorted(
            (name for n in node.body for name in core.walk(n, ast.Name)),
            key=lambda n: (n.lineno, n.col_offset),
        )
        node_orelse_names = sorted(
            (name for n in node.orelse for name in core.walk(n, ast.Name)),
            key=lambda n: (n.lineno, n.col_offset),
        )
        if len(node_body_names) != len(node_orelse_names):
            continue

        differing_names_index_name_mapping = {
            i: names
            for i, names in enumerate(zip(node_body_names, node_orelse_names))
            if len({n.id for n in names}) > 1
        }
        something = collections.defaultdict(list)
        for i, names in differing_names_index_name_mapping.items():
            something[tuple(name.id for name in names)].append(i)

        # Add definitions of new variables at start of if statements
        # Replace all references to differing variables with new ones
        # Then, breakout_common_code_in_ifs() should trigger on the
        # identical code at the end of both branches and move it after
        # the branches.

        bodies = [node.body, node.orelse]

        body_equivalent_function_srcs = [[
            core.unparse(
                ast.FunctionDef(
                    name="func",
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        vararg=None,
                        kwonlyargs=[],
                        kw_defaults=[],
                        kwarg=None,
                        defaults=[],
                    ),
                    body=body,
                    decorator_list=[],
                    type_params=[],
                    returns=None,
                    lineno=0,
                    end_lineno=0 + len(body),
                    col_offset=0,
                    end_col_offset=0,
            ))]  # Put as single element in list to emulate mutable string
            for body in bodies
        ]

        additions = set()
        replacements = {}
        for new_variable_number, (names, indexes) in enumerate(something.items()):
            new_variable = ast.Name(id=f"var_{new_variable_number + 1}")
            for src_list, body, name in zip(body_equivalent_function_srcs, bodies, names):
                assign = ast.Assign(targets=[new_variable], value=ast.Name(id=name))
                ast.copy_location(assign, body[0])
                assign.lineno -= 1
                additions.add(assign)

                src_list[0] = src_list[0].replace(name, new_variable.id)

            for index in indexes:
                for name in differing_names_index_name_mapping[index]:
                    replacements[name] = new_variable

        # Check if replacing all names really made the code equivalent.
        # This will be prone to false-negatives, in all kinds of ways, which is better
        # than the opposite.
        unique_srcs = {src_list[0] for src_list in body_equivalent_function_srcs}
        if len(unique_srcs) != 1:
            continue

        added_code_chars = sum(len(core.unparse(addition)) for addition in additions)
        removed_code_chars = len(unique_srcs.pop())

        if added_code_chars > removed_code_chars / 2:
            continue

        if additions and replacements:
            source = processing.alter_code(
                source,
                root,
                replacements=replacements,
                additions=additions,
                priority=("additions", "replacements"),
            )

            return simplify_if_control_flow(source)

    return source
