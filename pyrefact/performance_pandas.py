import ast
from typing import Callable, Union

from pyrefact import constants, parsing, processing


@processing.fix
def replace_loc_at_iloc_iat(source: str) -> str:
    if constants.PYTHON_VERSION >= (3, 9):
        pattern = ast.Subscript(
            value=parsing.Wildcard("attribute", ast.Attribute(attr=('loc', 'iloc'))),
            slice=(
                ast.Tuple(elts=[ast.Constant, ast.Constant]),
                ast.Constant,
            ))
    else:
        pattern = ast.Subscript(
            value=parsing.Wildcard("attribute", ast.Attribute(attr=('loc', 'iloc'))),
            slice=ast.Index(value=(
                ast.Tuple(elts=[ast.Constant, ast.Constant]),
                ast.Constant,
            )))

    root = parsing.parse(source)
    for node, attribute in parsing.walk_wildcard(root, pattern):
        if attribute.attr == "loc":
            attr = "at"
        elif attribute.attr == "iloc":
            attr = "iat"
        else:
            continue

        yield attribute, ast.Attribute(value=attribute.value, attr=attr)
