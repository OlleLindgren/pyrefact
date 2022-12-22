"""Fixes that aim to improve performance"""

import ast
from typing import Collection

from pyrefact import parsing, processing


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
    except (ValueError, SyntaxError):
        return False

    return True


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


def replace_with_sets(content: str) -> str:
    """Replace inlined lists with sets.

    Args:
        content (str): Python source code

    Returns:
        str: Modified python source code
    """
    root = parsing.parse(content)
    safe_callables = parsing.safe_callable_names(root)

    replacements = {}

    for node in parsing.walk(root, ast.Compare):
        if not _is_contains_comparison(node):
            continue

        for comp in node.comparators:
            if isinstance(comp, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
                preferred_type = (
                    ast.SetComp
                    if _can_be_evaluated_safe(ast.Expression(body=comp))
                    else ast.GeneratorExp
                )
                if isinstance(node, preferred_type):
                    continue
                replacement = preferred_type(elt=comp.elt, generators=comp.generators)
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
                    func=ast.Name(id="set", ctx=ast.Load()),
                    args=comp.args,
                    keywords=comp.keywords,
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


def remove_redundant_iter(content: str) -> str:
    root = parsing.parse(content)
    replacements = {}
    for node in parsing.walk(root, (ast.For, ast.comprehension)):
        if (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id in {"iter", "list", "tuple"}
            and len(node.iter.args) == 1
        ):
            replacements[node.iter] = node.iter.args[0]

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

    for node in parsing.walk(root, ast.Subscript):
        if not _is_sorted_subscript(node):
            continue

        args = node.value.args
        keywords = node.value.keywords
        if len(args) > 1:
            continue
        if len(keywords) > 1 or any(kw.arg != "key" for kw in keywords):
            continue
        if isinstance(node.slice, ast.Constant):
            value = node.slice.value
            if value != 0:
                continue
            replacement = ast.Call(
                func=builtin_min, args=args, keywords=keywords, lineno=node.lineno
            )
        elif (
            isinstance(node.slice, ast.UnaryOp)
            and isinstance(node.slice.op, ast.USub)
            and isinstance(node.slice.operand, ast.Constant)
        ):
            value = node.slice.operand.value
            if value != 1:
                continue
            replacement = ast.Call(
                func=builtin_max, args=args, keywords=keywords, lineno=node.lineno
            )
        elif isinstance(node.slice, ast.Slice):
            lower = node.slice.lower
            upper = node.slice.upper
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
