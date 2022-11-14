"""Fixes related to improving classes and object-oriented code."""
import ast

from pyrefact import parsing, processing


def remove_unused_self_cls(content: str) -> str:
    """Remove unused self and cls arguments from classes.

    Args:
        content (str): Python source code

    Returns:
        str: Python source code without any unused self or cls arguments.
    """
    root = ast.parse(content)

    replacements = {}

    for classdef in parsing.iter_classdefs(root):
        class_non_instance_methods = {
            funcdef.name
            for funcdef in parsing.iter_funcdefs(classdef)
            if any(
                decorator.id in {"staticmethod", "classmethod"}
                for decorator in funcdef.decorator_list
                if isinstance(decorator, ast.Name)
            )
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
            decorators = {
                decorator.id
                for decorator in funcdef.decorator_list
                if isinstance(decorator, ast.Name)
            }
            delete_decorators = set()
            if first_arg_name in static_access_names:
                # Add classmethod at the top
                if "classmethod" in decorators:
                    continue
                decorator = "classmethod"
            else:
                # Add staticmethod at the top, remove classmethod
                if "staticmethod" in decorators:
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
