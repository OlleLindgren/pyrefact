"""Fixes related to improving classes and object-oriented code."""
import ast
import re
from typing import Collection, Iterable

from pyrefact import parsing, processing


def remove_unused_self_cls(content: str) -> str:
    """Remove unused self and cls arguments from classes.

    Args:
        content (str): Python source code

    Returns:
        str: Python source code without any unused self or cls arguments.
    """
    root = parsing.parse(content)

    replacements = {}

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
                    elif isinstance(child, ast.Call):
                        func = child.func
                        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                            if (
                                func.value.id == first_arg_name
                                and func.attr in class_non_instance_methods
                            ):
                                static_accesses.add(func.value)

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
            funcdef.lineno = min(x.lineno for x in ast.walk(funcdef) if hasattr(x, "lineno"))
            funcdef.decorator_list = [
                dec
                for dec in funcdef.decorator_list
                if not (isinstance(dec, ast.Name) and dec.id in delete_decorators)
            ]
            funcdef.decorator_list.insert(
                0,
                ast.Name(
                    id=decorator,
                    ctx=ast.Load(),
                    lineno=funcdef.lineno - 1,
                    col_offset=funcdef.col_offset,
                ),
            )
            args = funcdef.args.posonlyargs or funcdef.args.args
            del args[0]
            if decorator == "classmethod":
                args.insert(0, ast.arg(arg="cls", annotation=None))

            if decorator == "classmethod":
                for node in funcdef.body:
                    for child in ast.walk(node):
                        if isinstance(child, ast.Name) and child.id == first_arg_name:
                            child.id = "cls"

            replacements[funcdef] = funcdef

    if replacements:
        content = processing.replace_nodes(content, replacements)

    return content


def _decorators_of_type(node: ast.FunctionDef, name: str) -> Iterable[ast.AST]:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == name:
            yield decorator


def move_staticmethod_static_scope(content: str, preserve: Collection[str]) -> str:
    root = parsing.parse(content)

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

    for node in parsing.walk(root, ast.Attribute):
        if isinstance(node.value, ast.Name):
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
            if funcdef.name in attributes_to_preserve or parsing.is_magic_method(funcdef):
                continue
            staticmethod_decorators = set(_decorators_of_type(funcdef, "staticmethod"))
            if not staticmethod_decorators:
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
            if (node.value.id == classdef.name and node.attr in moved_function_names) or (
                classdef.lineno < node.lineno < classdef.end_lineno
                and node.value.id in {"self", "cls"}
                and node.attr in moved_function_names
            ):
                replacements[node] = ast.Name(
                    id=moved_function_names[node.attr], ctx=node.ctx, lineno=node.lineno
                )

    if not name_replacements:
        return content

    if len(name_replacements) != len(set(name_replacements.values())):
        return content

    if replacements:
        content = processing.replace_nodes(content, replacements)
        root = parsing.parse(content)

    for classdef in sorted(parsing.iter_classdefs(root), key=lambda cd: cd.lineno, reverse=True):
        delete = []
        additions = []

        for funcdef in parsing.iter_funcdefs(classdef):
            new_name = name_replacements.get((classdef.name, funcdef.name))
            if new_name is None:
                continue
            staticmethod_decorators = set(_decorators_of_type(funcdef, "staticmethod"))
            static_names.add(new_name)
            delete.append(funcdef)
            additions.append(
                ast.FunctionDef(
                    name=new_name,
                    args=funcdef.args,
                    body=funcdef.body,
                    decorator_list=[
                        dec for dec in funcdef.decorator_list if dec not in staticmethod_decorators
                    ],
                    returns=funcdef.returns,
                    lineno=classdef.lineno - 1,
                )
            )

        if delete or additions:
            content = processing.remove_nodes(content, delete, root)
            content = processing.insert_nodes(content, reversed(additions))

    return content
