import ast

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
    for _, attribute in parsing.walk_wildcard(root, pattern):
        if attribute.attr == "loc":
            attr = "at"
        elif attribute.attr == "iloc":
            attr = "iat"
        else:
            continue

        yield attribute, ast.Attribute(value=attribute.value, attr=attr)


@processing.fix
def replace_iterrows_index(source: str) -> str:
    root = parsing.parse(source)
    
    target_template = ast.Tuple(elts=[
        parsing.Wildcard("new_target", object),
        ast.Name(id="_")])
    iter_template = ast.Call(
        func=ast.Attribute(
            value=parsing.Wildcard("underlying_object", object),
            attr="iterrows"),
        args=[], keywords=[])

    template = (
        ast.For(target=target_template, iter=iter_template),
        ast.comprehension(target=target_template, iter=iter_template),
    )

    for node, new_target, underlying_object in parsing.walk_wildcard(root, template):
        yield node.target, new_target
        yield node.iter, ast.Attribute(value=underlying_object, attr="index")

