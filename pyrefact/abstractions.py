import ast
import itertools
import re
from typing import Collection, Iterable, Sequence, Tuple

from . import constants, parsing, processing


class _EverythingContainer:
    """Object that contains everything."""

    @staticmethod
    def __contains__(_) -> bool:
        return True


def _scoped_dependencies(node: ast.AST):
    return set(name.id for name in parsing.iter_assignments(node))


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
    comprehension_local_vars = {comp.target for comp in parsing.walk(node, ast.comprehension)}
    types = (
        ast.Name,
        ast.Subscript,
        ast.Call,
        ast.Yield,
        ast.YieldFrom,
        ast.Continue,
        ast.Break,
        ast.Return,
    )
    for child in parsing.walk(node, types):
        if (
            isinstance(child, ast.Name)
            and isinstance(child.ctx, ast.Store)
            and child not in comprehension_local_vars
        ):
            yield child
        elif (
            isinstance(child, ast.Subscript)
            and isinstance(child.ctx, ast.Store)
            and child not in comprehension_local_vars
            and isinstance(child.value, ast.Name)
        ):
            yield child
        elif isinstance(child, (ast.Yield, ast.YieldFrom, ast.Continue, ast.Break, ast.Return)):
            yield child
        elif isinstance(child, ast.Call) and not (
            isinstance(child.func, ast.Name) and child.func.id in safe_callables
        ):
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
    if not isinstance(node, (ast.If, ast.IfExp)):
        yield from _possible_external_effects(node, safe_callables)
        if isinstance(node, (ast.Break, ast.Continue)):
            yield node
        return

    body_effects = {}
    for child in node.body:
        body_effects.update(
            {
                hash_node(n, safe_callables): n
                for n in _definite_external_effects(child, safe_callables)
            }
        )
        if parsing.is_blocking(node):
            break
    orelse_effects = {}
    for child in node.body:
        orelse_effects.update(
            {
                hash_node(n, safe_callables): n
                for n in _definite_external_effects(child, safe_callables)
            }
        )
        if parsing.is_blocking(node):
            break
    for key in body_effects.keys() & orelse_effects:
        yield body_effects[key]
    if isinstance(node, (ast.Break, ast.Continue)):
        yield node


def _definite_stored_names(node: ast.AST) -> Iterable[str]:
    for child in _definite_external_effects(node, _EverythingContainer()):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            yield child.id
        elif (
            isinstance(child, ast.Subscript)
            and isinstance(child.ctx, ast.Store)
            and isinstance(child.value, ast.Name)
        ):
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
    nodes: Sequence[ast.AST],
    return_injection_type: ast.AST,
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
                body.append(ast.Return(value=ast.Constant(value=False)))
            elif return_injection_type == ast.Break:
                body.append(ast.Return(value=ast.Constant(value=True)))
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
                )
            )
        elif isinstance(node, ast.With):
            body.append(
                ast.With(
                    items=node.items,
                    body=_build_function_body(node.body, return_injection_type),
                )
            )
        elif isinstance(node, ast.For):
            body.append(
                ast.For(
                    target=node.target,
                    iter=node.iter,
                    body=node.body,
                    orelse=_build_function_body(node.orelse, return_injection_type),
                )
            )
        elif isinstance(node, ast.While):
            body.append(
                ast.While(
                    test=node.test,
                    body=node.body,
                    orelse=_build_function_body(node.orelse, return_injection_type),
                )
            )
        else:
            body.append(node)

    return body


def _code_dependencies_outputs(
    code: Sequence[ast.AST],
) -> Tuple[Collection[ast.Name], Collection[ast.Name]]:
    required_names = set()
    created_names = set()
    for node in code:
        temp_children = []
        children = []
        if isinstance(node, (ast.While, ast.For, ast.If)):
            temp_children = [node.test]
            children = []
            for subset in (node.body, node.orelse):
                if not any(parsing.is_blocking(child) for child in subset):
                    children.append(subset)
        elif isinstance(node, ast.With):
            temp_children = [node.items]
            children = [node.body]
        elif isinstance(node, (ast.Try, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(
                "Dependency mapping is not implemented for code with exception handling."
            )
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            created_names.update(
                alias.name if alias.asname is None else alias.asname for alias in node.names
            )
            continue
        else:
            node_created = set()
            node_needed = set()
            generator_internal_names = set()
            for child in ast.walk(node):
                if isinstance(child, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                    comp_created = {comp.target.id for comp in child.generators}
                    for grandchild in ast.walk(child):
                        if isinstance(grandchild, ast.Name) and grandchild.id in comp_created:
                            generator_internal_names.add(grandchild)

            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Load):
                    for n in ast.walk(child):
                        if isinstance(n, ast.Name) and n not in generator_internal_names:
                            node_needed.add(n.id)

                if (
                    isinstance(child, ast.Name)
                    and child.id not in node_needed
                    and child not in generator_internal_names
                ):
                    if isinstance(child.ctx, ast.Load):
                        node_needed.add(child.id)
                    elif isinstance(child.ctx, ast.Store):
                        node_created.add(child.id)
                    else:
                        # Del
                        node_created.discard(child.id)
                        created_names.discard(child.id)

            created_names.update(node_created)
            required_names.update(node_needed)
            continue

        temp_created, temp_needed = _code_dependencies_outputs(temp_children)
        created = []
        needed = []
        for nodes in children:
            c_created, c_needed = _code_dependencies_outputs(nodes)
            created.append(c_created)
            needed.append(c_needed - temp_created)

        node_created = set.intersection(*created)
        node_needed = set.union(*needed)

        node_needed -= created_names
        node_needed -= temp_created
        node_needed |= temp_needed
        created_names.update(node_created)
        required_names.update(node_needed)

    return created_names, required_names


def _code_complexity_length(node: ast.AST) -> int:
    node_unparse_length = len(re.sub(" *", "", ast.unparse(node)))
    node_string_length = len(
        re.sub(
            " *",
            "",
            "".join(
                node.value
                for node in ast.walk(node)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            ),
        )
    )
    return node_unparse_length - node_string_length


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
        elif import_linenos:
            return max(import_linenos) + 1
        else:
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


def create_abstractions(content: str) -> str:
    root = parsing.parse(content)
    global_names = (
        _scoped_dependencies(root) | parsing.get_imported_names(root) | constants.BUILTIN_FUNCTIONS
    )
    for node in parsing.iter_funcdefs(root):
        global_names.add(node.name)
    for node in parsing.iter_classdefs(root):
        global_names.add(node.name)
    replacements = {}
    additions = []
    removals = []
    abstraction_count = 0

    function_def_linenos = []
    import_linenos = []

    for node in parsing.walk(root, (ast.AsyncFunctionDef, ast.FunctionDef)):
        candidates = [node.lineno] + [x.lineno for x in node.decorator_list]
        function_def_linenos.append(min(candidates))
    for node in parsing.walk(root, (ast.Import, ast.ImportFrom)):
        import_linenos.append(node.lineno)

    for node in itertools.chain([root], parsing.iter_bodies_recursive(root)):
        for nodes in _group_nodes_by_purpose(node.body):
            if len(nodes) > len(node.body) - 2:
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and all(
                isinstance(child, ast.Return)
                or (isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant))
                or child in nodes
                for child in node.body
            ):
                continue

            purposes = {_hashable_node_purpose_type(child) for child in nodes}
            assert len(purposes) == 1
            purpose = purposes.pop()

            if isinstance(nodes[0], (ast.Assign, ast.AnnAssign)) and len(nodes) == 1:
                continue

            children_with_purpose = sum(
                isinstance(grandchild, purpose[0])
                for child in nodes
                for grandchild in ast.walk(child)
            )
            if (
                sum(_code_complexity_length(node) for node in nodes) < 100
                and children_with_purpose < 3
            ):
                continue
            if children_with_purpose <= 2:
                continue

            created_names, required_names = _code_dependencies_outputs(nodes)
            if len(created_names) > 1:
                continue

            abstraction_count += 1
            function_name = f"_pyrefact_abstraction_{abstraction_count}"
            while function_name in global_names:
                abstraction_count += 1
                function_name = f"_pyrefact_abstraction_{abstraction_count}"

            args = sorted(required_names - (global_names - _scoped_dependencies(root) if node is root else global_names))
            call_args = [ast.Name(id=arg, ctx=ast.Load()) for arg in args]
            signature_args = ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=arg) for arg in args],
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
                    test=call,
                    body=[purpose[0](col_offset=nodes[0].col_offset + 4)],
                    orelse=[],
                )
                return_value = purpose[0] == ast.Continue
                function_body = _build_function_body(nodes, purpose[0]) + [
                    ast.Return(value=ast.Constant(value=return_value))
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
                ifs = []
                for n in nodes:
                    for c in ast.walk(n):
                        if isinstance(c, ast.If):
                            ifs.append(c)
                pure_nested_if = len(nodes) == 1 and all(
                    len(n.body) == len(n.orelse) == 1 for n in ifs
                )
                function_body = _build_function_body(
                    nodes,
                    purpose[0] if pure_nested_if else None,
                )
                if not pure_nested_if:
                    ifs = []
                    for c in ast.walk(nodes[0]):
                        if isinstance(c, ast.If):
                            ifs.append(c)
                    if not isinstance(nodes[0], (ast.Assign, ast.AnnAssign)) and not all(
                        len(n.body) == len(n.orelse) == 1 for n in ifs
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
                and len(nodes_after_abstraction) == 1
                and (
                    (
                        isinstance(nodes_after_abstraction[0], ast.Return)
                        and isinstance(nodes_after_abstraction[0].value, ast.Name)
                    )
                    or (
                        isinstance(nodes_after_abstraction[0], ast.Expr)
                        and isinstance(nodes_after_abstraction[0].value, (ast.Yield, ast.YieldFrom))
                        and isinstance(nodes_after_abstraction[0].value.value, ast.Name)
                    )
                )
            )

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
                    value=function_call.value,
                    lineno=function_call.lineno,
                )
            else:
                inlined_return_call = ast.Expr(
                    type(nodes_after_abstraction[0].value)(value=function_call.value),
                    lineno=function_call.lineno,
                )
            replacements[nodes_after_abstraction[0]] = inlined_return_call
            removals.extend(nodes)
            additions.append(function_def)

    content = processing.alter_code(
        content, root, additions=additions, removals=removals, replacements=replacements
    )

    return content
