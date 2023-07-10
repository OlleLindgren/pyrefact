from __future__ import annotations

import ast

from pyrefact import core, parsing, processing


def _uses_numpy(root: ast.Module) -> bool:
    if "numpy" in parsing.module_dependencies(root):
        return True

    # If np.something is referenced anywhere, assume it uses numpy as well.
    template = core.compile_template(("np.{{something}}", "numpy.{{something}}"))
    return any(core.walk(root, template))


def _is_sum_call(call: ast.Call):
    return parsing.is_call(call, ("sum", "np.sum", "numpy.sum"))


def _is_zip_product(comp: ast.ListComp | ast.GeneratorExp):
    template = core.compile_template((
        "[{{left}} * {{right}} for {{left}}, {{right}} in zip({{left_iterable}}, {{right_iterable}})]",
        "({{left}} * {{right}} for {{left}}, {{right}} in zip({{left_iterable}}, {{right_iterable}}))",
    ))
    return core.match_template(comp, template)


def _wrap_np_dot(*args: ast.AST) -> ast.Call:
    return ast.Call(
        func=ast.Attribute(value=ast.Name(id="np"), attr="dot"), args=list(args), keywords=[]
    )


def _wrap_np_matmul(*args: ast.AST) -> ast.Call:
    return ast.Call(
        func=ast.Attribute(value=ast.Name(id="np"), attr="matmul"), args=list(args), keywords=[]
    )


def _wrap_transpose(node: ast.AST) -> ast.Attribute:
    return ast.Attribute(value=node, attr="T")


@processing.fix
def simplify_matmul_transposes(source: str) -> str:
    """Replace np.matmul(a.T, b.T).T with np.matmul(b, a), if found."""

    root = core.parse(source)

    target_template = ast.Call(
        func=ast.Attribute(value=ast.Name(id=("np", "numpy")), attr="matmul"), args=[object, object]
    )
    for node in filter(parsing.is_transpose_operation, core.walk(root, ast.Attribute)):
        target = parsing.transpose_target(node)
        if (
            core.match_template(target, target_template)
            and not any(isinstance(arg, ast.Starred) for arg in target.args)
            and all(parsing.is_transpose_operation(arg) for arg in target.args)
        ):
            left, right = target.args
            matmul = _wrap_np_matmul(_wrap_transpose(right), _wrap_transpose(left))
            matmul.func = target.func
            matmul.keywords = target.keywords
            yield node, matmul


@processing.fix
def replace_implicit_dot(source: str) -> str:
    root = core.parse(source)
    if not _uses_numpy(root):
        return

    template = ast.Call(args=[(ast.ListComp, ast.GeneratorExp)], keywords=[])
    for call in core.walk(root, template):
        if _is_sum_call(call) and _is_zip_product(call.args[0]):
            zip_args = call.args[0].generators[0].iter.args
            yield call, _wrap_np_dot(*zip_args)


@processing.fix
def replace_implicit_matmul(source: str) -> str:
    find = """
    for {{i}} in range(len({{left}})):
        for {{j}} in range(len({{right}}[0])):
            for {{k}} in range(len({{right}})):
                {{result}}[{{i}}][{{j}}] += {{left}}[{{i}}][{{k}}] * {{right}}[{{k}}][{{j}}]
    """
    replace = "{{result}} = np.matmul({{left}}, {{right}})"
    yield from processing.find_replace(source, find, replace)

    find = """
    for {{i}} in range(len({{left}})):
        for {{j}} in range(len({{right}}[0])):
            {{result}}[{{i}}][{{j}}] = np.dot({{left}}[{{i}}] * {{right}}.T[{{j}}])
    """
    replace = "{{result}} = np.matmul({{left}}, {{right}})"
    yield from processing.find_replace(source, find, replace)

    find = """
    {{result}} = [[
        sum(
            {{left}}[{{i}}][{{k}}] * {{right}}[{{k}}][{{j}}]
            for {{k}} in range(len({{right}}))
        )
        for {{j}} in range(len({{right}}[0]))
        ]
        for {{i}} in range(len({{left}}))
    ]
    """
    replace = "{{result}} = np.matmul({{left}}, {{right}})"
    yield from processing.find_replace(source, find, replace)

    find = """
    {{result}} = [[
        np.dot({{left}}[{{i}}] * {{right}}.T[{{j}}])
        for {{j}} in range(len({{right}}[0]))
        ]
        for {{i}} in range(len({{left}}))
    ]
    """
    replace = "{{result}} = np.matmul({{left}}, {{right}})"
    yield from processing.find_replace(source, find, replace)

    find = "[[np.dot({{left_row}}, {{right_row}}) for {{left_row}} in {{left}}] for {{right_row}} in {{right}}.T]"
    replace = "np.matmul({{left}}, {{right}}).T"
    yield from processing.find_replace(source, find, replace)

    find = "[[np.dot({{left_row}}, {{right_row}}) for {{left_row}} in {{left}}.T] for {{right_row}} in {{right}}]"
    replace = "np.matmul({{right}}, {{left}})"
    yield from processing.find_replace(source, find, replace)

    find = "[[np.dot({{left_row}}, {{right_row}}) for {{left_row}} in {{left}}.T] for {{right_row}} in {{right}}.T]"
    replace = "np.matmul({{left}}.T, {{right}}).T"
    yield from processing.find_replace(source, find, replace)

    find = "[[np.dot({{left_row}}, {{right_row}}) for {{left_row}} in {{left}}] for {{right_row}} in {{right}}]"
    replace = "np.matmul({{right}}.T, {{left}}.T)"
    yield from processing.find_replace(source, find, replace)

    find = """
    np.array([[
        np.dot({{b_mat}}[:, {{b_index}}], {{a_mat}}[{{a_index}}, :])
        for {{b_index}} in range({{b_mat}}.shape[1])
        ]
        for {{a_index}} in range({{a_mat}}.shape[0])
    ])"""
    replace = "np.matmul({{a_mat}}, {{b_mat}})"
    yield from processing.find_replace(source, find, replace)

    find = """
    np.array([[
        np.dot({{c_mat}}[{{c_index}}, :], {{a_mat}}[{{a_index}}, :])
        for {{c_index}} in range({{c_mat}}.shape[0])
        ]
        for {{a_index}} in range({{a_mat}}.shape[0])
    ])"""
    replace = "np.matmul({{c_mat}}, {{a_mat}}.T).T"
    yield from processing.find_replace(source, find, replace)

    find = """
    np.array([[
        np.dot({{b_mat}}[:, {{b_index}}], {{d_mat}}[:, {{d_index}}])
        for {{b_index}} in range({{b_mat}}.shape[1])
        ]
        for {{d_index}} in range({{d_mat}}.shape[1])
    ])"""
    replace = "np.matmul({{b_mat}}.T, {{d_mat}}).T"
    yield from processing.find_replace(source, find, replace)

    find = """
    np.array([[
        np.dot({{a_mat}}[{{a_index}}, :], {{b_mat}}[:, {{b_index}}])
        for {{a_index}} in range({{a_mat}}.shape[0])
        ]
        for {{b_index}} in range({{b_mat}}.shape[1])
    ])"""
    replace = "np.matmul({{a_mat}}, {{b_mat}}).T"
    yield from processing.find_replace(source, find, replace)
