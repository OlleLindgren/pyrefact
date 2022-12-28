import ast
from typing import Callable, Union

from pyrefact import parsing, processing


def _uses_numpy(root: ast.Module) -> bool:
    if "numpy" in parsing.module_dependencies(root):
        return True

    # If np.something is referenced anywhere, assume it uses numpy as well.
    return any(
        isinstance(node.value, ast.Name) and node.value.id in ("numpy", "np")
        for node in parsing.walk(root, ast.Attribute)
    )


def only_if_uses_numpy(f: Callable) -> Callable:
    def wrapper(content: str) -> str:
        root = parsing.parse(content)
        if not _uses_numpy(root):
            return content

        return f(content)

    return wrapper


def _is_sum_call(call: ast.Call):
    return (isinstance(call.func, ast.Name) and call.func.id == "sum") or (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "sum"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in ("np", "numpy")
    )


def _is_np_array_call(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "array"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in ("np", "numpy")
    )


def _is_np_dot_call(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "dot"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in ("np", "numpy")
    )


def _is_zip_call(call: ast.Call):
    return isinstance(call.func, ast.Name) and call.func.id == "zip"


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
        and {x.id for x in comp.generators[0].target.elts}
        == {comp.elt.left.id, comp.elt.right.id}
        and _is_zip_call(comp.generators[0].iter)
    )


def _wrap_np_array(node: ast.AST) -> ast.Call:
    return ast.Call(
        func=ast.Attribute(value=ast.Name(id="np"), attr="array"),
        args=[node],
        keywords=[],
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


def _wrap_transpose(node: ast.AST) -> ast.Attribute:
    return ast.Attribute(value=node, attr="T")


def _simplify_transposes(content: str) -> str:
    root = parsing.parse(content)

    replacements = {}

    for node in parsing.walk(root, ast.Attribute):
        if node.attr == "T" and isinstance(node.value, ast.Attribute) and node.value.attr == "T":
            replacements[node] = node.value.value

    content = processing.replace_nodes(content, replacements)

    return content


@only_if_uses_numpy
def replace_implicit_dot(content: str) -> str:
    root = parsing.parse(content)

    replacements = {}

    for call in filter(_is_sum_call, parsing.walk(root, ast.Call)):
        if (
            len(call.args) == 1
            and not call.keywords
            and isinstance(call.args[0], (ast.ListComp, ast.GeneratorExp))
        ):
            if _is_zip_product(call.args[0]):
                zip_args = call.args[0].generators[0].iter.args
                replacements[call] = _wrap_np_dot(
                    *zip_args
                )

    content = processing.replace_nodes(content, replacements)
    content = _simplify_transposes(content)

    return content


@only_if_uses_numpy
def replace_implicit_matmul(content: str) -> str:
    root = parsing.parse(content)

    replacements = {}

    for call in filter(_is_np_array_call, parsing.walk(root, ast.Call)):
        if len(call.args) == 1 and not call.keywords:
            comp_outer = call.args[0]
            if isinstance(comp_outer, ast.ListComp) and len(comp_outer.generators) == 1 and not any(gen.ifs for gen in comp_outer.generators):
                comp_inner = comp_outer.elt
                if isinstance(comp_inner, ast.ListComp) and len(comp_inner.generators) == 1 and not any(gen.ifs for gen in comp_inner.generators):
                    if _is_np_dot_call(comp_inner.elt):
                        replacements[call] = _wrap_transpose(_wrap_np_matmul(
                            comp_inner.generators[0].iter,
                            _wrap_transpose(comp_outer.generators[0].iter),
                        ))

    content = processing.replace_nodes(content, replacements)
    content = _simplify_transposes(content)

    return content
