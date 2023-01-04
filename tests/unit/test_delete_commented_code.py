#!/usr/bin/env python3


import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (
        (
            """
def f() -> int:
    # return 100 ** 100
    return 2

# def q() -> int:
#     if x > 1:
#         return 3
#     else:
#

#         return -3

y = 10

# def h() -> int:
#     if x > 1:
#         # return 3
#         return 99
#     else:
#         return -3

            """,
            """
def f() -> int:
    return 2

y = 10
            """,
        ),
        (
            """
# x: int = 100
# x: int = 100  # pylint: disable=invalid-name
# pylint: disable=some-important-error
# type: ignore
# flake8: noqa
# noqa
# x: int = 100  # pylint: disable=invalid-name
# noqa: abcdef
            """,
            """
# pylint: disable=some-important-error
# type: ignore
# flake8: noqa
# noqa
# noqa: abcdef
            """,
        ),
        (
            """
def f() -> int:
    # Normal comment
    return 2
# Normal comment
def h() -> int:
    return 2  # Normal comment
# z = lambda: 3
            """,
            """
def f() -> int:
    # Normal comment
    return 2
# Normal comment
def h() -> int:
    return 2  # Normal comment
            """,
        ),
        (
            """
def f() -> int:
    # Normal comment
    # return 99
    # q > v > 99 > 0
    return 2
# Normal comment
# def h() -> int:
#     return 2  # Normal comment
# This is not valid python
# z = lambda: 3
# lambda: 3
            """,
            """
def f() -> int:
    # Normal comment
    return 2
# Normal comment
# This is not valid python
            """,
        ),
        (
            """
# Comment about something interesting
# lambda: 2
# Another comment about something interesting
# lambda: 1 if x > y else z < 100
# comment comment comment
# name_of_thing
            """,
            """
# Comment about something interesting
# lambda: 2
# Another comment about something interesting
# comment comment comment
# name_of_thing
            """,
        ),
        (
            """
print('''
# x = 3
# y = z
'''
)

# true comment
# x = 3 > y  # this should be removed

z = 2
print(f'''
# x = {3}
# y = {z}
'''
)

print('''
# x = {x}
# y = {z}
'''.format(x=z, z=z)
)
            """,
            """
print('''
# x = 3
# y = z
'''
)

# true comment

z = 2
print(f'''
# x = {3}
# y = {z}
'''
)

print('''
# x = {x}
# y = {z}
'''.format(x=z, z=z)
)
            """,
        ),
        (
            """
#import ast
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
    return (isinstance(call.func, ast.Name) and call.func.id == "sum") or (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "sum"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in {"np", "numpy"}
    )


def _is_np_array_call(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "array"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in {"np", "numpy"}
    )


def _is_np_dot_call(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "dot"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in {"np", "numpy"}
    )


def _is_np_matmul_call(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "matmul"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in {"np", "numpy"}
    )


# def _is_zip_call(call: ast.Call):
#     return isinstance(call.func, ast.Name) and call.func.id == "zip"


### Code with some particular purpose ###


# def _is_zip_product(comp: Union[ast.ListComp, ast.GeneratorExp]):
#     return (
#         isinstance(comp.elt, ast.BinOp)
#         and isinstance(comp.elt.op, ast.Mult)
#         and isinstance(comp.elt.left, ast.Name)
#         and isinstance(comp.elt.right, ast.Name)
#         and len(comp.generators) == 1
#         and not any(gen.ifs for gen in comp.generators)
#         and isinstance(comp.generators[0].target, ast.Tuple)
#         and all(isinstance(x, ast.Name) for x in comp.generators[0].target.elts)
#         and {x.id for x in comp.generators[0].target.elts} == {comp.elt.left.id, comp.elt.right.id}
#         and _is_zip_call(comp.generators[0].iter)
#     )


# def _wrap_np_dot(*args: ast.AST) -> ast.Call:
#     return ast.Call(
#         func=ast.Attribute(value=ast.Name(id="np"), attr="dot"),
#         args=list(args),
#         keywords=[],
#     )


#*#*#*#*#*#*      ##Code with some particular purpose ###


# def _wrap_np_matmul(*args: ast.AST) -> ast.Call:
#     return ast.Call(
#         func=ast.Attribute(value=ast.Name(id="np"), attr="matmul"),
#         args=list(args),
#         keywords=[],
#     )


def wrap_transpose(node: ast.AST) -> ast.Attribute:
    return ast.Attribute(value=node, attr="T")


def simplify_matmul_transposes(content: str) -> str:
    '''Replace np.matmul(a.T, b.T).T with np.matmul(b, a), if found.'''

    root = parsing.parse(content)
    replacements = {}

    for node in parsing.walk(root, ast.Attribute):
        if parsing.is_transpose_operation(node):
            target = parsing.transpose_target(node)
            if isinstance(target, ast.Call) and _is_np_matmul_call(target):
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


# @_only_if_uses_numpy
# def replace_implicit_dot(content: str) -> str:
#     root = parsing.parse(content)

#     replacements = {}

#     for call in filter(_is_sum_call, parsing.walk(root, ast.Call)):
#         if (
#             len(call.args) == 1
#             and not call.keywords
#             and isinstance(call.args[0], (ast.ListComp, ast.GeneratorExp))
#         ):
#             if _is_zip_product(call.args[0]):
#                 zip_args = call.args[0].generators[0].iter.args
#                 replacements[call] = _wrap_np_dot(*zip_args)

#     content = processing.replace_nodes(content, replacements)

#     return content


@_only_if_uses_numpy
def replace_implicit_matmul(content: str) -> str:
    root = parsing.parse(content)

    replacements = {}

    for call in filter(_is_np_array_call, parsing.walk(root, ast.Call)):
        if len(call.args) == 1 and not call.keywords:
            comp_outer = call.args[0]
            if (
                isinstance(comp_outer, ast.ListComp)
                and len(comp_outer.generators) == 1
                and not any(gen.ifs for gen in comp_outer.generators)
            ):
                comp_inner = comp_outer.elt
                if (
                    isinstance(comp_inner, ast.ListComp)
                    and len(comp_inner.generators) == 1
                    and not any(gen.ifs for gen in comp_inner.generators)
                ):
                    # if not isinstance(comp_inner.generators[0].target, ast.Name):
                    #     continue
                    # if not isinstance(comp_outer.generators[0].target, ast.Name):
                    #     continue
                    # if not isinstance(comp_inner.generators[0].iter, ast.Name) and not (
                    #     isinstance(comp_inner.generators[0].iter, ast.Attribute)
                    #     and comp_inner.generators[0].iter.attr == "T"
                    # ):
                    #     continue
                    if not isinstance(comp_outer.generators[0].iter, ast.Name) and not (
                        isinstance(comp_outer.generators[0].iter, ast.Attribute)
                        and comp_outer.generators[0].iter.attr == "T"
                    ):
                        continue
                    if _is_np_dot_call(comp_inner.elt):
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
                        # if (
                        #     left_id == comp_inner.generators[0].target.id
                        #     and right_id == comp_outer.generators[0].target.id
                        #     or (
                        #         right_id == comp_inner.generators[0].target.id
                        #         and left_id == comp_outer.generators[0].target.id
                        #     )
                        # ):
                        #     replacements[call] = wrap_transpose(
                        #         _wrap_np_matmul(
                        #             comp_inner.generators[0].iter,
                        #             wrap_transpose(comp_outer.generators[0].iter),
                        #         )
                        #     )

    content = processing.replace_nodes(content, replacements)

    return content

            """,
            """
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
    return (isinstance(call.func, ast.Name) and call.func.id == "sum") or (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "sum"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in {"np", "numpy"}
    )


def _is_np_array_call(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "array"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in {"np", "numpy"}
    )


def _is_np_dot_call(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "dot"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in {"np", "numpy"}
    )


def _is_np_matmul_call(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "matmul"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in {"np", "numpy"}
    )


#*#*#*#*#*#*      ##Code with some particular purpose ###


def wrap_transpose(node: ast.AST) -> ast.Attribute:
    return ast.Attribute(value=node, attr="T")


def simplify_matmul_transposes(content: str) -> str:
    '''Replace np.matmul(a.T, b.T).T with np.matmul(b, a), if found.'''

    root = parsing.parse(content)
    replacements = {}

    for node in parsing.walk(root, ast.Attribute):
        if parsing.is_transpose_operation(node):
            target = parsing.transpose_target(node)
            if isinstance(target, ast.Call) and _is_np_matmul_call(target):
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
def replace_implicit_matmul(content: str) -> str:
    root = parsing.parse(content)

    replacements = {}

    for call in filter(_is_np_array_call, parsing.walk(root, ast.Call)):
        if len(call.args) == 1 and not call.keywords:
            comp_outer = call.args[0]
            if (
                isinstance(comp_outer, ast.ListComp)
                and len(comp_outer.generators) == 1
                and not any(gen.ifs for gen in comp_outer.generators)
            ):
                comp_inner = comp_outer.elt
                if (
                    isinstance(comp_inner, ast.ListComp)
                    and len(comp_inner.generators) == 1
                    and not any(gen.ifs for gen in comp_inner.generators)
                ):
                    if not isinstance(comp_outer.generators[0].iter, ast.Name) and not (
                        isinstance(comp_outer.generators[0].iter, ast.Attribute)
                        and comp_outer.generators[0].iter.attr == "T"
                    ):
                        continue
                    if _is_np_dot_call(comp_inner.elt):
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

    content = processing.replace_nodes(content, replacements)

    return content

            """,
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = fixes.delete_commented_code(content)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
