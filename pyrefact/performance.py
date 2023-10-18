"""Fixes that aim to improve performance"""

import ast

from pyrefact import constants, core, processing


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
    root = core.parse(source)
    find = "{{element}} in {{wrapper}}({{collection}})"
    replace = "{{element}} in {{collection}}"
    wrapper_names = ("sorted", "list", "tuple", "set", "iter", "reversed")
    template = core.compile_template(find, wrapper=ast.Name(id=wrapper_names))

    yield from processing.find_replace(source, template, replace)

    sorted_list_tuple_call_template = ast.Call(
        func=ast.Name(id=("sorted", "list", "tuple"), ctx=ast.Load), args=[object], keywords=[]
    )

    template = core.compile_template(
        "{{element}} in {{collection}}",
        collection=(ast.ListComp, ast.DictComp, ast.SetComp, ast.List, ast.Tuple),
    )

    for node in core.walk(root, template):
        for comp in node.comparators:
            if isinstance(comp, (ast.ListComp, ast.SetComp)):
                yield comp, ast.GeneratorExp(elt=comp.elt, generators=comp.generators)

            elif isinstance(comp, ast.DictComp):
                yield comp, ast.GeneratorExp(elt=comp.key, generators=comp.generators)

            elif isinstance(comp, (ast.List, ast.Tuple)):
                preferred_type = (
                    ast.Set if _can_be_evaluated_safe(ast.Set(elts=comp.elts)) else ast.Tuple
                )
                if not isinstance(comp, preferred_type):
                    yield comp, preferred_type(elts=comp.elts)

            elif core.match_template(comp, sorted_list_tuple_call_template):
                yield comp, comp.args[0]


@processing.fix
def remove_redundant_iter(source: str) -> str:
    root = core.parse(source)
    iter_template = ast.Call(func=ast.Name(id=("iter", "list", "tuple")), args=[object])
    template = (ast.For(iter=iter_template), ast.comprehension(iter=iter_template))

    for node in core.walk(root, template):
        yield node.iter, node.iter.args[0]


@processing.fix
def remove_redundant_chained_calls(source: str) -> str:
    root = core.parse(source)

    # If outer is present, inner is redundant
    outer_inner_redundancy_mapping = {
        "sorted": {"list", "sorted", "tuple", "iter", "reversed"},
        "list": {"list", "tuple", "iter"},
        "set": {"set", "list", "sorted", "tuple", "iter", "reversed"},
        "iter": {"list", "tuple", "iter"},
        "reversed": {"list", "tuple"},
        "tuple": {"list", "tuple", "iter"},
        "sum": {"list", "tuple", "iter", "sorted", "reversed"},
    }

    templates = tuple(
        ast.Call(
            func=ast.Name(id=key), args=[ast.Call(func=ast.Name(id=tuple(values)), args=[object])]
        )
        for key, values in outer_inner_redundancy_mapping.items()
    )

    for node in core.walk(root, templates):
        arg = node.args[0].args[0]
        while core.match_template(arg, templates):
            arg = arg.args[0].args[0]
        yield node, ast.Call(func=node.func, args=[arg], keywords=[])

    # If inner is present, outer is redundant
    inner_outer_redundancy_mapping = {
        "sorted": {"list", "sorted"},
        "list": {"list"},
        "set": {"set"},
        "iter": {"iter"},
        "tuple": {"tuple"},
    }

    templates = tuple(
        ast.Call(
            func=ast.Name(id=tuple(values)), args=[ast.Call(func=ast.Name(id=key), args=[object])]
        )
        for key, values in inner_outer_redundancy_mapping.items()
    )

    for node in core.walk(root, templates):
        yield node, node.args[0]

    reversed_sorted_template = ast.Call(
        func=ast.Name(id="reversed"), args=[ast.Call(func=ast.Name(id="sorted"))]
    )
    for node in core.walk(root, reversed_sorted_template):
        replacement = node.args[0]
        for i, kw in enumerate(replacement.keywords):
            if kw.arg == "reverse":
                kw.value = ast.UnaryOp(op=ast.Not(), operand=kw.value)
                break
        else:
            i = len(replacement.keywords)
            replacement.keywords.append(
                ast.keyword(arg="reverse", value=ast.Constant(value=True, kind=None))
            )

        try:
            value = core.literal_value(replacement.keywords[i].value)
        except ValueError:
            pass
        else:
            if value:
                replacement.keywords[i].value = ast.Constant(value=bool(value), kind=None)
            else:
                del replacement.keywords[i]

        yield node, replacement


def _slice_of(node: ast.Subscript) -> ast.AST:
    node_slice = node.slice
    if constants.PYTHON_VERSION < (3, 9):
        if isinstance(node_slice, ast.Index):
            return node_slice.value
        if isinstance(node_slice, ast.ExtSlice):
            return ast.Tuple(elts=node_slice.dims)

    return node_slice


@processing.fix
def replace_sorted_heapq(source: str) -> str:
    root = core.parse(source)

    heapq_nlargest = core.compile_template("heapq.nlargest")
    heapq_nsmallest = core.compile_template("heapq.nsmallest")
    builtin_max = core.compile_template("max")
    builtin_min = core.compile_template("min")
    builtin_list = core.compile_template("list")
    builtin_reversed = core.compile_template("reversed")

    template_sorted_subscript = ast.Subscript(
        value=ast.Call(func=ast.Name(id="sorted"), args=[object], keywords={ast.keyword(arg="key")})
    )

    # Slice templates
    template_first_element = core.compile_template("0")
    template_last_element = core.compile_template("-1")
    template_first_n = core.compile_template("sequence[:{{n}}]", n=ast.AST).slice
    template_last_n = core.compile_template("sequence[-{{n}}:]", n=ast.AST).slice

    for node in core.walk(root, template_sorted_subscript):
        args = node.value.args
        keywords = node.value.keywords
        node_slice = _slice_of(node)
        if core.match_template(node_slice, template_first_element):
            replacement = ast.Call(
                func=builtin_min, args=args, keywords=keywords, lineno=node.lineno
            )
            yield node, replacement
        elif core.match_template(node_slice, template_last_element):
            replacement = ast.Call(
                func=builtin_max, args=args, keywords=keywords, lineno=node.lineno
            )
            yield node, replacement
        elif core.match_template(node_slice, template_first_n) and not isinstance(
            node_slice.upper, ast.UnaryOp
        ):
            func = heapq_nsmallest
            value = node_slice.upper
            replacement = ast.Call(func=func, args=[value] + args, keywords=keywords)
            yield node, replacement
        elif core.match_template(node_slice, template_last_n):
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
            )],)
            yield node, replacement


def _wrap_transpose(node: ast.AST) -> ast.Call:
    return ast.Call(func=ast.Name(id="zip"), args=[ast.Starred(value=node)], keywords=[])


@processing.fix
def _replace_subscript_looping_simple_cases(source: str) -> str:
    yield from processing.find_replace(
        source,
        "[{{sequence}}[{{index}}] for {{index}} in range(len({{sequence}}))]",
        "list({{sequence}})",
    )
    yield from processing.find_replace(
        source,
        "[{{sequence}}[{{index}}, :] for {{index}} in range(len({{sequence}}))]",
        "list({{sequence}})",
    )
    yield from processing.find_replace(
        source,
        "[{{sequence}}[{{index}}] for {{index}} in range({{sequence}}.shape[0])]",
        "list({{sequence}})",
    )
    yield from processing.find_replace(
        source,
        "[{{sequence}}[{{index}}, :] for {{index}} in range({{sequence}}.shape[0])]",
        "list({{sequence}})",
    )
    yield from processing.find_replace(
        source,
        "[{{sequence}}[:, {{index}}] for {{index}} in range({{sequence}}.shape[1])]",
        "list({{sequence}}.T)",
    )
    yield from processing.find_replace(
        source,
        "({{sequence}}[{{index}}] for {{index}} in range(len({{sequence}})))",
        "iter({{sequence}})",
    )
    yield from processing.find_replace(
        source,
        "({{sequence}}[{{index}}, :] for {{index}} in range(len({{sequence}})))",
        "iter({{sequence}})",
    )
    yield from processing.find_replace(
        source,
        "({{sequence}}[{{index}}] for {{index}} in range({{sequence}}.shape[0]))",
        "iter({{sequence}})",
    )
    yield from processing.find_replace(
        source,
        "({{sequence}}[{{index}}, :] for {{index}} in range({{sequence}}.shape[0]))",
        "iter({{sequence}})",
    )
    yield from processing.find_replace(
        source,
        "({{sequence}}[:, {{index}}] for {{index}} in range({{sequence}}.shape[1]))",
        "iter({{sequence}}.T)",
    )


@processing.fix
def _replace_subscript_looping_complex_cases(source: str) -> str:
    target_template = core.Wildcard("target", ast.Name, common=True)
    index_template = core.Wildcard("index", ast.Name, common=True)
    target_length_template = core.compile_template(
        ("len({{target}})", "{{target}}.shape[0]"), target=target_template
    )
    target_length_template_transpose = core.compile_template(
        ("len({{target}}.T)", "{{target}}.shape[1]"), target=target_template
    )
    target_indexed_template = core.compile_template(
        "{{target}}[{{index}}]", target=target_template, index=index_template
    )
    comprehension_template = ast.comprehension(
        target=index_template,
        iter=ast.Call(
            func=ast.Name(id="range"),
            args=[(target_length_template, target_length_template_transpose)],
        ),
        ifs=[],
    )
    comp_template = (
        ast.ListComp(generators=[comprehension_template]),
        ast.GeneratorExp(generators=[comprehension_template]),
        ast.SetComp(generators=[comprehension_template]),
        ast.DictComp(generators=[comprehension_template]),
    )
    root = core.parse(source)
    for template_match in core.walk_wildcard(root, comp_template):
        comprehension = template_match.root.generators[0]
        target_indexed_template = core.compile_template(
            ("{{target}}[{{index}}]", "{{target}}[{{index}}, :]", "{{target}}[:, {{index}}]"),
            target=template_match.target,
            index=template_match.index,
        )
        target_indexed_nodes = set(core.walk(template_match.root, target_indexed_template))
        target_used_nodes = set(core.walk(template_match.root, template_match.target)) - {
            comprehension.target
        }

        if len(target_indexed_nodes) != len(target_used_nodes):
            continue

        new_index_name = f"{template_match.target.id}_{template_match.index.id}"

        yield comprehension.target, ast.Name(id=new_index_name)
        if core.match_template(comprehension.iter.args[0], target_length_template):
            yield comprehension.iter, template_match.target
        else:
            yield comprehension.iter, _wrap_transpose(template_match.target)

        for node in target_indexed_nodes:
            yield node, ast.Name(id=new_index_name)


@processing.fix
def replace_subscript_looping(source: str) -> str:
    yield from _replace_subscript_looping_simple_cases._fix_func(source)
    yield from _replace_subscript_looping_complex_cases._fix_func(source)
