import ast
from typing import Callable, Union

from pyrefact import parsing, processing


def uses_numpy(root: ast.Module) -> bool:
    if "numpy" in parsing.module_dependencies(root):
        return True

    # If np.something is referenced anywhere, assume it uses numpy as well.
    return any(
        isinstance(node.value, ast.Name) and node.value.id in {"numpy", "np"}
        for node in parsing.walk(root, ast.Attribute)
    )


def _only_if_uses_numpy(f: Callable) -> Callable:
    def wrapper(content: str) -> str:
        root = parsing.parse(content)
        if not uses_numpy(root):
            return content

        return f(content)

    return wrapper


def _is_sum_call(call: ast.Call):
    return parsing.is_call(call, ("sum", "np.sum", "numpy.sum"))


def _is_np_array_call(call: ast.Call) -> bool:
    return parsing.is_call(call, ("np.array", "numpy.array"))


def _is_zip_product(comp: Union[ast.ListComp, ast.GeneratorExp]):
    return (
        isinstance(comp.elt, ast.BinOp)
        and isinstance(comp.elt.op, ast.Mult)
        and isinstance(comp.elt.left, ast.Name)
        and isinstance(comp.elt.right, ast.Name)
        and len(comp.generators) == 1
        and not any(gen.ifs for gen in comp.generators)
        and isinstance(comp.generators[0].target, ast.Tuple)
        and all(isinstance(x, ast.Name) for x in comp.generators[0].target.elts)
        and {x.id for x in comp.generators[0].target.elts} == {comp.elt.left.id, comp.elt.right.id}
        and parsing.is_call(comp.generators[0].iter, "zip")
    )


def _wrap_np_dot(*args: ast.AST) -> ast.Call:
    return ast.Call(
        func=ast.Attribute(value=ast.Name(id="np"), attr="dot"),
        args=list(args),
        keywords=[],
    )


def _wrap_np_matmul(*args: ast.AST) -> ast.Call:
    return ast.Call(
        func=ast.Attribute(value=ast.Name(id="np"), attr="matmul"),
        args=list(args),
        keywords=[],
    )


def wrap_transpose(node: ast.AST) -> ast.Attribute:
    return ast.Attribute(value=node, attr="T")


def simplify_matmul_transposes(content: str) -> str:
    """Replace np.matmul(a.T, b.T).T with np.matmul(b, a), if found."""

    root = parsing.parse(content)
    replacements = {}

    for node in filter(parsing.is_transpose_operation, parsing.walk(root, ast.Attribute)):

        target = parsing.transpose_target(node)
        if parsing.is_call(target, ("np.matmul", "numpy.matmul")):
            if (
                len(target.args) == 2
                and not any(isinstance(arg, ast.Starred) for arg in target.args)
                and all(parsing.is_transpose_operation(arg) for arg in target.args)
            ):
                left, right = target.args
                matmul = _wrap_np_matmul(wrap_transpose(right), wrap_transpose(left))
                matmul.func = target.func
                matmul.keywords = target.keywords
                replacements[node] = matmul

    content = processing.replace_nodes(content, replacements)

    return content


@_only_if_uses_numpy
def replace_implicit_dot(content: str) -> str:
    root = parsing.parse(content)

    replacements = {}

    template = ast.Call(args=[(ast.ListComp, ast.GeneratorExp)], keywords=[])
    for call in parsing.walk(root, template):
        if _is_sum_call(call) and _is_zip_product(call.args[0]):
            zip_args = call.args[0].generators[0].iter.args
            replacements[call] = _wrap_np_dot(*zip_args)

    content = processing.replace_nodes(content, replacements)

    return content


@_only_if_uses_numpy
def replace_implicit_matmul(content: str) -> str:
    root = parsing.parse(content)

    replacements = {}

    comp_template = ast.ListComp(
        generators=[
            ast.comprehension(ifs=[], target=ast.Name, iter=(ast.Name, ast.Attribute(attr="T")))
        ]
    )

    for call in filter(_is_np_array_call, parsing.walk(root, ast.Call)):
        if parsing.match_template(
            call, ast.Call(args=[ast.ListComp(elt=ast.ListComp)], keywords=[])
        ):
            comp_outer = call.args[0]
            comp_inner = comp_outer.elt
            if parsing.match_template(comp_outer, comp_template) and parsing.match_template(
                comp_inner, comp_template
            ):
                if parsing.is_call(comp_inner.elt, ("numpy.dot", "np.dot")):
                    left_id = (
                        comp_inner.generators[0].target.id
                        if isinstance(comp_inner.generators[0].target, ast.Name)
                        else comp_inner.generators[0].target.value.id
                    )
                    right_id = (
                        comp_outer.generators[0].target.id
                        if isinstance(comp_outer.generators[0].target, ast.Name)
                        else comp_outer.generators[0].target.value.id
                    )
                    if (
                        left_id == comp_inner.generators[0].target.id
                        and right_id == comp_outer.generators[0].target.id
                        or (
                            right_id == comp_inner.generators[0].target.id
                            and left_id == comp_outer.generators[0].target.id
                        )
                    ):
                        replacements[call] = wrap_transpose(
                            _wrap_np_matmul(
                                comp_inner.generators[0].iter,
                                wrap_transpose(comp_outer.generators[0].iter),
                            )
                        )

    content = processing.replace_nodes(content, replacements)

    return content
