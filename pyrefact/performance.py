"""Fixes that aim to improve performance"""

import ast

from pyrefact import constants, parsing, performance_numpy, processing


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


def optimize_contains_types(content: str) -> str:
    """Replace inlined lists with sets.

    Args:
        content (str): Python source code

    Returns:
        str: Modified python source code
    """
    root = parsing.parse(content)

    replacements = {}

    for node in filter(_is_contains_comparison, parsing.walk(root, ast.Compare)):

        for comp in node.comparators:
            if isinstance(comp, (ast.ListComp)):
                replacement = ast.GeneratorExp(elt=comp.elt, generators=comp.generators)
            elif isinstance(comp, ast.DictComp):
                replacement = ast.SetComp(elt=comp.key, generators=comp.generators)
            elif isinstance(comp, (ast.List, ast.Tuple)):
                preferred_type = (
                    ast.Set if _can_be_evaluated_safe(ast.Set(elts=comp.elts)) else ast.Tuple
                )
                if isinstance(comp, preferred_type):
                    continue
                replacement = preferred_type(elts=comp.elts)
            elif (
                parsing.is_call(comp, ("sorted", "list", "tuple"))
                and isinstance(comp.func.ctx, ast.Load)
                and len(comp.args) == 1
                and not comp.keywords
            ):
                replacement = comp.args[0]
            else:
                continue

            replacements[comp] = replacement

    if replacements:
        content = processing.replace_nodes(content, replacements)

    return content


def remove_redundant_iter(content: str) -> str:
    root = parsing.parse(content)
    replacements = {
        node.iter: node.iter.args[0]
        for node in parsing.walk(root, (ast.For, ast.comprehension))
        if isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and (node.iter.func.id in {"iter", "list", "tuple"})
        and (len(node.iter.args) == 1)
    }

    if replacements:
        content = processing.replace_nodes(content, replacements)

    return content


def remove_redundant_chained_calls(content: str) -> str:
    root = parsing.parse(content)

    function_chain_redundancy_mapping = {
        "sorted": {"list", "sorted", "tuple", "iter", "reversed"},
        "list": {"list", "tuple", "iter"},
        "set": {"set", "list", "sorted", "tuple", "iter", "reversed"},
        "iter": {"list", "tuple", "iter"},
        "reversed": {"list", "tuple"},
        "tuple": {"list", "tuple", "iter"},
        "sum": {"list", "tuple", "iter", "sorted", "reversed"},
    }

    replacements = {}
    touched_linenos = set()

    for node in parsing.walk(root, ast.Call):
        if not (isinstance(node.func, ast.Name) and node.args):
            continue

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
            and (modified_node.args[0].func.id in redundant_call_names)
        ):
            modified_node = replacements[node] = ast.Call(
                func=node.func, args=modified_node.args[0].args, keywords=[]
            )
            touched_linenos.update(node_lineno_range)

    if replacements:
        content = processing.replace_nodes(content, replacements)

    return content


def _is_sorted_subscript(node) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    if not isinstance(node.value, ast.Call):
        return False
    if not isinstance(node.value.func, ast.Name):
        return False
    if node.value.func.id != "sorted":
        return False
    return True


def replace_sorted_heapq(content: str) -> str:
    root = parsing.parse(content)

    replacements = {}
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

    for node in filter(_is_sorted_subscript, parsing.walk(root, ast.Subscript)):

        args = node.value.args
        keywords = node.value.keywords
        if len(args) > 1:
            continue
        if len(keywords) > 1 or any(kw.arg != "key" for kw in keywords):
            continue
        node_slice = parsing.slice_of(node)
        if isinstance(node_slice, ast.Constant):
            value = node_slice.value
            if value != 0:
                continue
            replacement = ast.Call(
                func=builtin_min, args=args, keywords=keywords, lineno=node.lineno
            )
        elif (
            isinstance(node_slice, ast.UnaryOp)
            and isinstance(node_slice.op, ast.USub)
            and isinstance(node_slice.operand, ast.Constant)
        ):
            value = node_slice.operand.value
            if value != 1:
                continue
            replacement = ast.Call(
                func=builtin_max, args=args, keywords=keywords, lineno=node.lineno
            )
        elif isinstance(node_slice, ast.Slice):
            lower = node_slice.lower
            upper = node_slice.upper
            if lower is None and upper is not None and not isinstance(upper, ast.UnaryOp):
                func = heapq_nsmallest
                value = upper
                replacement = ast.Call(func=func, args=[value] + args, keywords=keywords)
            elif (
                lower is not None
                and upper is None
                and isinstance(lower, ast.UnaryOp)
                and isinstance(lower.op, ast.USub)
            ):
                func = heapq_nlargest
                value = lower.operand
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
            else:
                continue
        else:
            continue
        replacements[node] = replacement

    if replacements:
        content = processing.replace_nodes(content, replacements)

    return content


def _wrap_transpose(node: ast.AST) -> ast.Call:
    return ast.Call(func=ast.Name(id="zip"), args=[ast.Starred(value=node)], keywords=[])


def replace_subscript_looping(content: str) -> str:
    root = parsing.parse(content)

    replacements = {}

    for comp in parsing.walk(root, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        wrapper_function = parsing.get_comp_wrapper_func_equivalent(comp)

        if len(comp.generators) != 1:
            continue

        if not isinstance(comp.generators[0].target, ast.Name):
            continue

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
            isinstance(subscript.value, ast.Name)
            or (
                isinstance(subscript.value, ast.Attribute)
                and isinstance(subscript.value.value, ast.Name)
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
            isinstance(subscript.value, ast.Name)
            and subscript.value.id == subscripted_name
            and (
                (
                    isinstance(parsing.slice_of(subscript), ast.Tuple)
                    and len(parsing.slice_of(subscript).elts) == 2
                    and all(
                        isinstance(elt, ast.Slice)
                        or (isinstance(elt, ast.Name) and elt.id == subscript_name)
                        or (
                            constants.PYTHON_VERSION < (3, 9)
                            and isinstance(elt, ast.Index)
                            and isinstance(elt.value, ast.Name)
                            and elt.value.id == subscript_name
                        )
                        for elt in parsing.slice_of(subscript).elts
                    )
                )
                or (
                    isinstance(parsing.slice_of(subscript), ast.Name)
                    and parsing.slice_of(subscript).id == subscript_name
                )
            )
            for subscript in elt_subscripts
        ):
            # All subscripts are not a[i, :], a[:, i] or a[i]
            continue

        uses_of_subscript_name = {
            name for name in parsing.walk(comp.elt, ast.Name) if name.id == subscript_name
        }
        for subscript in parsing.walk(comp.elt, ast.Subscript):
            for value in parsing.walk(parsing.slice_of(subscript), ast.Name):
                uses_of_subscript_name.discard(value)

        if uses_of_subscript_name:
            # i is used for something other than subscripting a
            continue

        if len(comp.generators) != 1:
            continue

        iterated_node = comp.generators[0].iter

        if not parsing.is_call(iterated_node, "range"):
            continue

        if (
            len(iterated_node.args) == 1
            and isinstance(iterated_node.args[0], ast.Call)
            and isinstance(iterated_node.args[0].func, ast.Name)
            and iterated_node.args[0].func.id == "len"
            and len(iterated_node.args[0].args) == 1
            and isinstance(iterated_node.args[0].args[0], ast.Name)
            and iterated_node.args[0].args[0].id == subscripted_name
        ):
            replacements[comp] = ast.Call(
                func=ast.Name(id=wrapper_function),
                args=[ast.Name(id=subscripted_name)],
                keywords=[],
            )
            break

        if not (
            len(iterated_node.args) == 1
            and isinstance(iterated_node.args[0], ast.Subscript)
            and isinstance(iterated_node.args[0].value, ast.Attribute)
            and isinstance(iterated_node.args[0].value.value, ast.Name)
            and iterated_node.args[0].value.value.id == subscripted_name
            and iterated_node.args[0].value.attr == "shape"
        ):
            continue

        if (
            isinstance(parsing.slice_of(iterated_node.args[0]), ast.Constant)
            and parsing.slice_of(iterated_node.args[0]).value == 1
        ):
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

        if not (
            isinstance(parsing.slice_of(iterated_node.args[0]), ast.Constant)
            and parsing.slice_of(iterated_node.args[0]).value == 0
        ):
            continue

        replacements[comp.generators[0].iter] = ast.Name(id=subscripted_name)
        target_name = ast.Name(id=f"{subscripted_name}_")
        replacements[comp.generators[0].target] = target_name
        for subscript in elt_subscripts:
            replacements[subscript] = target_name
        break

    content = processing.replace_nodes(content, replacements)

    if replacements:
        return replace_subscript_looping(content)

    return content
