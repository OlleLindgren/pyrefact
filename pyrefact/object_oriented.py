"""Fixes related to improving classes and object-oriented code."""

import ast
import copy
import re
from typing import Collection, Iterable

from pyrefact import core, parsing, processing


@processing.fix
def remove_unused_self_cls(source: str) -> str:
    """Remove unused self and cls arguments from classes.

    Args:
        source (str): Python source code

    Returns:
        str: Python source code without any unused self or cls arguments.
    """
    root = core.parse(source)

    for classdef in parsing.iter_classdefs(root):
        class_non_instance_methods = {
            funcdef.name
            for funcdef in parsing.iter_funcdefs(classdef)
            if any(_decorators_of_type(funcdef, "staticmethod"))
            or any(_decorators_of_type(funcdef, "classmethod"))
        }
        for funcdef in parsing.iter_funcdefs(classdef):
            arguments = funcdef.args.posonlyargs + funcdef.args.args
            if not arguments:
                continue
            first_arg_name = arguments[0].arg

            first_arg_accesses = set()
            static_accesses = set()
            for node in funcdef.body:
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and child.id == first_arg_name:
                        first_arg_accesses.add(child)
                    elif parsing.is_call(
                        child, [f"{first_arg_name}.{attr}" for attr in class_non_instance_methods]
                    ):
                        static_accesses.add(child.func.value)

            instance_access_names = {node.id for node in first_arg_accesses - static_accesses}
            if first_arg_name in instance_access_names:
                # Should be non-static and non-classmethod
                continue
            static_access_names = {
                node.id for node in static_accesses if node.id not in instance_access_names
            }
            delete_decorators = set()
            if first_arg_name in static_access_names:
                # Add classmethod at the top
                if any(_decorators_of_type(funcdef, "classmethod")):
                    continue
                decorator = "classmethod"
            else:
                # Add staticmethod at the top, remove classmethod
                if any(_decorators_of_type(funcdef, "staticmethod")):
                    continue
                decorator = "staticmethod"
                delete_decorators.add("classmethod")
            funcdef_copy = copy.copy(funcdef)
            funcdef_copy.lineno = min(x.lineno for x in ast.walk(funcdef) if hasattr(x, "lineno"))
            funcdef_copy.decorator_list = [
                dec
                for dec in funcdef.decorator_list
                if not (isinstance(dec, ast.Name) and dec.id in delete_decorators)
            ]
            funcdef_copy.decorator_list.insert(
                0,
                ast.Name(
                    id=decorator,
                    ctx=ast.Load(),
                    lineno=funcdef.lineno - 1,
                    col_offset=funcdef.col_offset,
            ),)
            args = funcdef.args.posonlyargs or funcdef.args.args
            if args:
                del args[0]
            if decorator == "classmethod":
                args.insert(0, ast.arg(arg="cls", annotation=None))

            if decorator == "classmethod":
                for node in funcdef_copy.body:
                    for child in ast.walk(node):
                        if isinstance(child, ast.Name) and child.id == first_arg_name:
                            child.id = "cls"

            yield funcdef, funcdef_copy


def _decorators_of_type(node: ast.FunctionDef, name: str) -> Iterable[ast.AST]:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == name:
            yield decorator


@processing.fix
def move_staticmethod_static_scope(source: str, preserve: Collection[str]) -> str:
    root = core.parse(source)

    attributes_to_preserve = set()
    for name in preserve:
        if "." in name:
            *_, property_name = name.split(".")
            attributes_to_preserve.add(property_name)

    class_function_names = set()
    class_attribute_accesses = set()
    for classdef in parsing.iter_classdefs(root):
        for funcdef in parsing.iter_funcdefs(classdef):
            class_function_names.add((classdef.name, funcdef.name))

    for node in core.walk(root, ast.Attribute):
        if (
            core.match_template(node.value, ast.Call(func=ast.Name))
            and (node.value.func.id, node.attr) in class_function_names
        ):
            class_attribute_accesses.add(node)
        elif isinstance(node.value, ast.Name):
            if (
                node.value.id in {"self", "cls"}
                or (node.value.id, node.attr) in class_function_names
            ):
                class_attribute_accesses.add(node)
            else:
                attributes_to_preserve.add(node.value.id)

    static_names = {funcdef.name for funcdef in parsing.iter_funcdefs(root)} | preserve
    name_replacements = {}

    replacements = {}
    for classdef in sorted(parsing.iter_classdefs(root), key=lambda cd: cd.lineno, reverse=True):
        for funcdef in parsing.iter_funcdefs(classdef):
            if funcdef.name in attributes_to_preserve:
                continue
            if f"{classdef.name}.{funcdef.name}" in preserve:
                continue
            if parsing.is_magic_method(funcdef):
                continue
            if not set(_decorators_of_type(funcdef, "staticmethod")):
                continue
            new_name = funcdef.name
            if not parsing.is_private(new_name):
                new_name = f"_{new_name}"
            if new_name in static_names:
                new_name = re.sub("^_{2,}", "", f"_{classdef.name}{new_name}")
            if new_name in static_names:
                continue
            name_replacements[(classdef.name, funcdef.name)] = new_name

        moved_function_names = {fname: name for ((_, fname), name) in name_replacements.items()}

        for node in class_attribute_accesses:
            classdef_aliases = [classdef.name]
            if classdef.lineno < node.lineno < classdef.end_lineno:
                classdef_aliases.extend(("self", "cls"))

            template = ast.Attribute(
                value=(
                    ast.Name(id=tuple(classdef_aliases)),
                    ast.Call(func=ast.Name(id=classdef.name)),
                ),
                attr=tuple(moved_function_names),
            )

            if core.match_template(node, template):
                replacements[node] = ast.Name(
                    id=moved_function_names[node.attr], ctx=node.ctx, lineno=node.lineno
                )

    if not name_replacements:
        return

    if len(name_replacements) != len(set(name_replacements.values())):
        return

    transaction = 0
    for before, after in replacements.items():
        yield before, after, transaction

    transaction = 1
    for classdef in parsing.iter_classdefs(root):
        for funcdef in parsing.iter_funcdefs(classdef):
            new_name = name_replacements.get((classdef.name, funcdef.name))
            if new_name is None:
                continue

            staticmethod_decorators = set(_decorators_of_type(funcdef, "staticmethod"))
            static_names.add(new_name)

            funcdef_static = ast.FunctionDef(
                name=new_name,
                args=funcdef.args,
                body=funcdef.body,
                decorator_list=[
                    dec for dec in funcdef.decorator_list if dec not in staticmethod_decorators
                ],
                type_params=[],
                returns=funcdef.returns,
                lineno=classdef.lineno - 1,
                col_offset=classdef.col_offset,
            )
            yield funcdef, None, transaction
            yield None, funcdef_static, transaction

            transaction += 1
