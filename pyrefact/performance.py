"""Fixes that aim to improve performance"""

import ast

from pyrefact import parsing, performance_numpy, processing


def _is_contains_comparison(node) -> bool:
    if not isinstance(node, ast.Compare):
        return False
    if len(node.ops) != 1:
        return False
    if not isinstance(node.ops[0], ast.In):
        return False
    return True


def _can_be_evaluated_safe(node: ast.AST) -> bool:
    """Check if a node can be evaluated at "compile-time"

    Args:
        node (ast.AST): Node to check

    Returns:
        bool: True if the node can be evaluated
    """
    try:
        ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return False

    return True


@processing.fix
def optimize_contains_types(source: str) -> str:
    """Replace inlined lists with sets.

    Args:
        source (str): Python source code

    Returns:
        str: Modified python source code
    """
    root = parsing.parse(source)

    sorted_list_tuple_call_template = ast.Call(
        func=ast.Name(id=("sorted", "list", "tuple"), ctx=ast.Load),
        args=[object],
        keywords=[],
    )

    for node in filter(_is_contains_comparison, parsing.walk(root, ast.Compare)):
        for comp in node.comparators:
            if isinstance(comp, ast.ListComp):
                yield comp, ast.GeneratorExp(elt=comp.elt, generators=comp.generators)

            elif isinstance(comp, ast.DictComp):
                yield comp, ast.SetComp(elt=comp.key, generators=comp.generators)

            elif isinstance(comp, (ast.List, ast.Tuple)):
                preferred_type = (
                    ast.Set if _can_be_evaluated_safe(ast.Set(elts=comp.elts)) else ast.Tuple
                )
                if not isinstance(comp, preferred_type):
                    yield comp, preferred_type(elts=comp.elts)

            elif parsing.match_template(comp, sorted_list_tuple_call_template):
                yield comp, comp.args[0]


@processing.fix
def remove_redundant_iter(source: str) -> str:
    root = parsing.parse(source)
    iter_template = ast.Call(func=ast.Name(id=("iter", "list", "tuple")), args=[object])
    template = (ast.For(iter=iter_template), ast.comprehension(iter=iter_template))

    for node in parsing.walk(root, template):
        yield node.iter, node.iter.args[0]


@processing.fix(restart_on_replace=True)
def remove_redundant_chained_calls(source: str) -> str:
    root = parsing.parse(source)

    function_chain_redundancy_mapping = {
        "sorted": {"list", "sorted", "tuple", "iter", "reversed"},
        "list": {"list", "tuple", "iter"},
        "set": {"set", "list", "sorted", "tuple", "iter", "reversed"},
        "iter": {"list", "tuple", "iter"},
        "reversed": {"list", "tuple"},
        "tuple": {"list", "tuple", "iter"},
        "sum": {"list", "tuple", "iter", "sorted", "reversed"},
    }

    templates = tuple(
        ast.Call(func=ast.Name(id=key), args=[ast.Call(func=ast.Name(id=value), args=[object])])
        for key, values in function_chain_redundancy_mapping.items()
        for value in values
    )

    for node in parsing.walk(root, templates):
        arg = node.args[0].args[0]
        while parsing.match_template(arg, templates):
            arg = arg.args[0].args[0]
        yield node, ast.Call(func=node.func, args=[arg], keywords=[])


@processing.fix
def replace_sorted_heapq(source: str) -> str:
    root = parsing.parse(source)

    heapq_nlargest = ast.Attribute(
        value=ast.Name(id="heapq", ctx=ast.Load()), attr="nlargest", ctx=ast.Load()
    )
    heapq_nsmallest = ast.Attribute(
        value=ast.Name(id="heapq", ctx=ast.Load()), attr="nsmallest", ctx=ast.Load()
    )
    builtin_max = ast.Name(id="max", ctx=ast.Load())
    builtin_min = ast.Name(id="min", ctx=ast.Load())
    builtin_list = ast.Name(id="list", ctx=ast.Load())
    builtin_reversed = ast.Name(id="reversed", ctx=ast.Load())

    template_sorted_subscript = ast.Subscript(
        value=ast.Call(func=ast.Name(id="sorted"), args=[object], keywords={ast.keyword(arg="key")})
    )

    # Slice templates
    template_first_element = ast.Constant(value=0)
    template_last_element = ast.UnaryOp(op=ast.USub, operand=ast.Constant(value=1))
    template_first_n = ast.Slice(lower=None, upper=ast.AST)
    template_last_n = ast.Slice(lower=ast.UnaryOp(op=ast.USub), upper=None)

    for node in parsing.walk(root, template_sorted_subscript):

        args = node.value.args
        keywords = node.value.keywords
        node_slice = parsing.slice_of(node)
        if parsing.match_template(node_slice, template_first_element):
            replacement = ast.Call(
                func=builtin_min, args=args, keywords=keywords, lineno=node.lineno
            )
            yield node, replacement
        elif parsing.match_template(node_slice, template_last_element):
            replacement = ast.Call(
                func=builtin_max, args=args, keywords=keywords, lineno=node.lineno
            )
            yield node, replacement
        elif parsing.match_template(node_slice, template_first_n) and not isinstance(
            node_slice.upper, ast.UnaryOp
        ):
            func = heapq_nsmallest
            value = node_slice.upper
            replacement = ast.Call(func=func, args=[value] + args, keywords=keywords)
            yield node, replacement
        elif parsing.match_template(node_slice, template_last_n):
            func = heapq_nlargest
            value = node_slice.lower.operand
            replacement = ast.Call(
                func=builtin_list,
                keywords=[],
                args=[
                    ast.Call(
                        func=builtin_reversed,
                        keywords=[],
                        args=[ast.Call(func=func, args=[value] + args, keywords=keywords)],
                    )
                ],
            )
            yield node, replacement


def _wrap_transpose(node: ast.AST) -> ast.Call:
    return ast.Call(func=ast.Name(id="zip"), args=[ast.Starred(value=node)], keywords=[])


def replace_subscript_looping(source: str) -> str:
    root = parsing.parse(source)

    replacements = {}

    generator_comp_template = ast.comprehension(
        target=ast.Name, iter=ast.Call(func=ast.Name(id="range"))
    )

    generator_template = (
        ast.ListComp(generators=[generator_comp_template]),
        ast.SetComp(generators=[generator_comp_template]),
        ast.GeneratorExp(generators=[generator_comp_template]),
    )

    for comp in parsing.walk(root, generator_template):
        wrapper_function = parsing.get_comp_wrapper_func_equivalent(comp)

        elt_subscripts = {
            node
            for node in parsing.walk(comp.elt, ast.Subscript)
            if isinstance(node.value, (ast.Attribute, ast.Name))
            and any(
                {name.id for name in parsing.walk(node, ast.Name)}
                & {name.id for name in parsing.walk(comp.generators[0], ast.Name)}
            )
        }
        if not all(
            parsing.match_template(
                subscript, ast.Subscript(value=(ast.Name, ast.Attribute(value=ast.Name)))
            )
            for subscript in elt_subscripts
        ):
            continue

        subscript_name = comp.generators[0].target.id
        subscripted_names = {
            subscript.value.id
            if isinstance(subscript.value, ast.Name)
            else subscript.value.value.id
            for subscript in elt_subscripts
        }
        if len(subscripted_names) != 1:
            continue
        subscripted_name = subscripted_names.pop()

        if not all(
            parsing.match_template(subscript, ast.Subscript(value=ast.Name(id=subscripted_name)))
            and (
                parsing.match_template(
                    parsing.slice_of(subscript),
                    (
                        ast.Name(id=subscript_name),
                        ast.Tuple(
                            elts=[
                                (
                                    ast.Slice,
                                    ast.Name(id=subscript_name),
                                    ast.Index(value=ast.Name(id=subscript_name)),
                                )
                            ]
                            * 2
                        ),
                    ),
                )
            )
            for subscript in elt_subscripts
        ):
            # All subscripts are not a[i, :], a[:, i] or a[i]
            continue

        uses_of_subscript_name = set(parsing.walk(comp.elt, ast.Name(id=subscript_name)))
        for subscript in parsing.walk(comp.elt, ast.Subscript):
            for value in parsing.walk(parsing.slice_of(subscript), ast.Name):
                uses_of_subscript_name.discard(value)

        if uses_of_subscript_name:
            # i is used for something other than subscripting a
            continue

        iterated_node = comp.generators[0].iter

        if parsing.match_template(
            iterated_node.args,
            [ast.Call(func=ast.Name(id="len"), args=[ast.Name(id=subscripted_name)])],
        ):
            replacements[comp] = ast.Call(
                func=ast.Name(id=wrapper_function),
                args=[ast.Name(id=subscripted_name)],
                keywords=[],
            )
            break

        if not parsing.match_template(
            iterated_node.args,
            [ast.Subscript(value=ast.Attribute(value=ast.Name(id=subscripted_name), attr="shape"))],
        ):
            continue

        if parsing.match_template(parsing.slice_of(iterated_node.args[0]), ast.Constant(value=1)):
            if performance_numpy.uses_numpy(root):
                transposed_name = performance_numpy.wrap_transpose(ast.Name(id=subscripted_name))
            else:
                transposed_name = _wrap_transpose(ast.Name(id=subscripted_name))
            replacements[comp.generators[0].iter] = transposed_name
            target_name = ast.Name(id=f"{subscripted_name}_")
            replacements[comp.generators[0].target] = target_name
            for subscript in elt_subscripts:
                replacements[subscript] = target_name
            break

        if parsing.match_template(parsing.slice_of(iterated_node.args[0]), ast.Constant(value=0)):
            replacements[comp.generators[0].iter] = ast.Name(id=subscripted_name)
            target_name = ast.Name(id=f"{subscripted_name}_")
            replacements[comp.generators[0].target] = target_name
            for subscript in elt_subscripts:
                replacements[subscript] = target_name
            break

    source = processing.replace_nodes(source, replacements)

    if replacements:
        return replace_subscript_looping(source)

    return source
