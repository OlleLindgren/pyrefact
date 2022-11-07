import ast
import builtins
import re
from typing import Collection, Iterable, Sequence, Tuple

from . import parsing, processing


class EverythingContainer:
    """Object that contains everything."""

    @staticmethod
    def __contains__(_) -> bool:
        return True


def _scoped_dependencies(node: ast.AST):
    return set(name.id for name in parsing.iter_assignments(node))


def _hash_node(
    node: ast.AST, preserved_callable_names: Collection[str] = EverythingContainer()
) -> int:
    name_increment = 0
    name_hashes = {}

    things_to_hash = []
    for child in ast.walk(node):
        things_to_hash.append(type(child))
        things_to_hash.extend(
            (key, value) for key, value in child.__dict__.items() if isinstance(value, (str, int))
        )
        if isinstance(node, ast.Name):
            if node.id in preserved_callable_names:
                things_to_hash.append(node.id)
            elif node.id in name_hashes:
                things_to_hash.append(name_hashes[node.id])
            else:
                name_increment += 1
                name_hashes[node.id] = name_increment
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
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
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
    if isinstance(node, (ast.If, ast.IfExp)):
        body_effects = {}
        for child in node.body:
            body_effects.update(
                {
                    _hash_node(n, safe_callables): n
                    for n in _definite_external_effects(child, safe_callables)
                }
            )
            if parsing.is_blocking(node):
                break
        orelse_effects = {}
        for child in node.body:
            orelse_effects.update(
                {
                    _hash_node(n, safe_callables): n
                    for n in _definite_external_effects(child, safe_callables)
                }
            )
            if parsing.is_blocking(node):
                break
        for key in body_effects.keys() & orelse_effects:
            yield body_effects[key]
    else:
        yield from _possible_external_effects(node, safe_callables)
    if isinstance(node, (ast.Break, ast.Continue)):
        yield node


def _definite_stored_names(node: ast.AST) -> Iterable[str]:
    for child in _definite_external_effects(node, EverythingContainer()):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            yield child.id


def _hashable_node_purpose_type(node: ast.AST) -> Tuple[str]:
    if isinstance(node, (ast.Continue, ast.Break)):
        return (type(node),)
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr, ast.If)):
        side_effects = [child for child in _possible_external_effects(node, EverythingContainer())]
        if all(isinstance(node, ast.Name) for node in side_effects):
            stored_names = set(_definite_stored_names(node))
            return tuple([type(node)] + sorted(stored_names))
        elif all(isinstance(node, ast.Break) for node in side_effects):
            return (ast.Break,)
        elif all(isinstance(node, ast.Continue) for node in side_effects):
            return (ast.Continue,)
        else:
            raise ValueError(f"Node {node} of type {type(node)} has no singular purpose")

    raise NotImplementedError(f"Cannot determine hashable node type of node of type {type(node)}")


def group_nodes_by_purpose(body: Sequence[ast.AST]) -> Iterable[Sequence[ast.AST]]:
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


def _build_bool_body(
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
        if isinstance(node, return_injection_type):
            body.append(ast.Return(value=ast.Constant(value=True)))
        elif isinstance(node, ast.If):
            body.append(
                ast.If(
                    test=node.test,
                    body=_build_bool_body(node.body, return_injection_type),
                    orelse=_build_bool_body(node.orelse, return_injection_type),
                )
            )
        elif isinstance(node, ast.With):
            body.append(
                ast.With(
                    items=node.items,
                    body=_build_bool_body(node.body, return_injection_type),
                )
            )
        elif isinstance(node, ast.For):
            body.append(
                ast.For(
                    target=node.target,
                    iter=node.iter,
                    body=node.body,
                    orelse=_build_bool_body(node.orelse, return_injection_type),
                )
            )
        elif isinstance(node, ast.While):
            body.append(
                ast.While(
                    test=node.test,
                    body=node.body,
                    orelse=_build_bool_body(node.orelse, return_injection_type),
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
            children = [node.body, node.orelse]
        elif isinstance(node, ast.With):
            temp_children = [node.items]
            children = [node.body]
        elif isinstance(node, (ast.Try, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(
                "Dependency mapping is not implemented for code with exception handling."
            )
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            children = [comp.iter for comp in node.generators]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            created_names.update(
                alias.name if alias.asname is None else alias.asname for alias in node.names
            )
            continue
        else:
            node_created = set()
            node_needed = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Load):
                    for n in ast.walk(child):
                        if isinstance(n, ast.Name):
                            node_needed.add(n.id)

                if isinstance(child, ast.Name) and child.id not in node_needed:
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
        for child in children:
            c_created, c_needed = _code_dependencies_outputs(node.body)
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


def create_abstractions(content: str) -> str:
    root = ast.parse(content)
    global_names = (
        _scoped_dependencies(root) | parsing.get_imported_names(root) | set(dir(builtins))
    )
    for node in parsing.iter_funcdefs(root):
        global_names.add(node.name)
    for node in parsing.iter_classdefs(root):
        global_names.add(node.name)
    replacements = {}
    additions = []
    abstraction_count = 0

    function_def_linenos = []
    import_linenos = []

    for node in ast.walk(root):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            function_def_linenos.append(node.lineno)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            import_linenos.append(node.lineno)

    for node in parsing.iter_bodies_recursive(root):
        if _code_complexity_length(node) < 100:
            continue

        for nodes in group_nodes_by_purpose(node.body):
            if sum(_code_complexity_length(node) for node in nodes) < 100:
                continue

            purposes = {_hashable_node_purpose_type(child) for child in nodes}
            assert len(purposes) == 1
            purpose = purposes.pop()

            if len(nodes) == 1 and isinstance(nodes[0], purpose):
                continue

            created_names, required_names = _code_dependencies_outputs(nodes)
            if len(created_names) > 1:
                continue

            abstraction_count += 1
            function_name = f"_pyrefact_abstraction_{abstraction_count}"
            args = sorted(required_names - global_names)
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

            if args and args[0] in {"self", "cls"}:  # Not implemented
                continue

            call = ast.Call(
                func=ast.Name(id=function_name, ctx=ast.Load()),
                args=call_args,
                keywords=[],
                starargs=[],
                kwargs=[],
            )

            if purpose[0] in (ast.Continue, ast.Break):
                function_call = ast.If(
                    test=call,
                    body=[purpose[0](col_offset=nodes[0].col_offset + 4)],
                    orelse=[],
                )
                function_body = _build_bool_body(nodes, purpose[0]) + [
                    ast.Return(value=ast.Constant(value=False))
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
                function_body = nodes + [ast.Return(value=return_targets)]
                returns = None

            else:
                continue

            if node.lineno in function_def_linenos:
                insertion_lineno = node.lineno - 1
            elif function_def_linenos:
                if all(lineno > node.lineno for lineno in function_def_linenos):
                    insertion_lineno = max(import_linenos) if import_linenos else node.lineno - 1
                else:
                    insertion_lineno = (
                        max(lineno for lineno in function_def_linenos if lineno <= node.lineno) - 1
                    )
            else:
                if import_linenos:
                    insertion_lineno = max(import_linenos) + 1
                else:
                    insertion_lineno = node.lineno - 1

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
                raise RuntimeError("Found assignment without return")

            if isinstance(function_call, (ast.Continue, ast.Break)) and len(return_args) != 1:
                raise RuntimeError("Found bool abstraction without return")

            replacements[nodes[0]] = function_call
            for child in nodes[1:]:
                replacements[child] = ast.Pass()
            additions.append(function_def)

    if replacements or additions:
        content = processing.replace_nodes(content, replacements)
        content = processing.insert_nodes(content, additions)

    return content
