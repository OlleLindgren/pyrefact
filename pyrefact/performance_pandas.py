import ast

from pyrefact import core, processing


@processing.fix
def replace_loc_at_iloc_iat(source: str) -> str:
    yield from processing.find_replace(
        source,
        "{{value}}.loc[{{i}}]",
        "{{value}}.at[{{i}}]",
        i=ast.Constant,
        transaction=0
    )
    yield from processing.find_replace(
        source,
        "{{value}}.loc[{{i}}, {{j}}]",
        "{{value}}.at[{{i}}, {{j}}]",
        i=ast.Constant,
        j=ast.Constant,
        transaction=1
    )
    yield from processing.find_replace(
        source,
        "{{value}}.iloc[{{i}}]",
        "{{value}}.iat[{{i}}]",
        i=ast.Constant,
        transaction=2
    )
    yield from processing.find_replace(
        source,
        "{{value}}.iloc[{{i}}, {{j}}]",
        "{{value}}.iat[{{i}}, {{j}}]",
        i=ast.Constant,
        j=ast.Constant,
        transaction=3
    )


@processing.fix
def replace_iterrows_index(source: str) -> str:
    root = core.parse(source)

    target_template = ast.Tuple(elts=[core.Wildcard("new_target", object), ast.Name(id="_")])
    iter_template = ast.Call(
        func=ast.Attribute(value=core.Wildcard("underlying_object", object), attr="iterrows"),
        args=[],
        keywords=[],
    )

    template = (
        ast.For(target=target_template, iter=iter_template),
        ast.comprehension(target=target_template, iter=iter_template),
    )

    for node, new_target, underlying_object in core.walk_wildcard(root, template):
        yield node.target, new_target
        yield node.iter, ast.Attribute(value=underlying_object, attr="index")


@processing.fix
def replace_iterrows_itertuples(source: str) -> str:
    root = core.parse(source)
    replacements = {}
    target_template = ast.Tuple(
        elts=[ast.Name(id="_"), ast.Name(id=core.Wildcard("new_target_id", str))]
    )
    iter_template = ast.Call(
        func=ast.Attribute(value=core.Wildcard("underlying_object", object), attr="iterrows"),
        args=[],
        keywords=[],
    )
    template = ast.For(target=target_template, iter=iter_template)
    for node, new_target_id, underlying_object in core.walk_wildcard(root, template):
        # If new_target is modified, the underlying dataframe is also modified. This cannot be
        # done with itertuples() since tuples are immutable.

        # Look for and skip these patterns:
        # new_target[...] = ... -> continue
        # new_target.anything[...] = ... -> continue
        # new_target.anything = ... -> continue
        new_target = ast.Name(id=new_target_id)
        new_target_altered_template = (
            ast.Subscript(value=new_target, ctx=(ast.Store, ast.Del)),
            ast.Subscript(value=ast.Attribute(value=new_target), ctx=(ast.Store, ast.Del)),
            ast.Attribute(value=new_target, ctx=ast.Store),
        )
        if any(any(core.walk(child, new_target_altered_template)) for child in node.body):
            continue

        # replace_loc_at_iloc_iat() will replace .loc[] with .at[] when confident that the key
        # is a single constant. Hence we do not need to support .loc/iloc[], and if we find it the
        # checks in replace_loc_at_iloc_iat() were not certain that the key was a single constant.
        # Therefore we skip it. We also skip [name], since we'd need getattr to retrieve the
        # serialized name, and we don't like getattr.
        # Look for and skip these patterns:
        # new_target.loc[] -> continue
        # new_target.iloc[] -> continue
        # new_target[name] -> continue
        unsupported_new_target_access_template = (
            ast.Subscript(value=new_target, slice=(ast.Index(value=ast.Name), ast.Name)),
            ast.Subscript(
                value=ast.Attribute(value=new_target, attr=("loc", "iloc")),
                slice=(ast.Index(value=ast.Name), ast.Name),
        ),)
        if any(
            any(core.walk(child, unsupported_new_target_access_template)) for child in node.body
        ):
            continue

        # new_target["name"] -> new_target.name
        # new_target.at["name"] -> new_target.name
        # new_target.iat[index] -> new_target[index + 1]
        target_get_access_template = ast.Subscript(
            value=new_target,
            slice=(
                ast.Index(value=ast.Constant(value=core.Wildcard("attr", str))),
                ast.Constant(value=core.Wildcard("attr", str)),
        ),)
        target_at_access_template = ast.Subscript(
            value=ast.Attribute(value=new_target, attr="at"),
            slice=(
                ast.Index(value=ast.Constant(value=core.Wildcard("attr", str))),
                ast.Constant(value=core.Wildcard("attr", str)),
        ),)
        target_iat_access_template = ast.Subscript(
            value=ast.Attribute(value=new_target, attr="iat")
        )
        node_replacements = {}
        for child in node.body:
            for target_get_access, attr in core.walk_wildcard(child, target_get_access_template):
                node_replacements[target_get_access] = ast.Attribute(value=new_target, attr=attr)
            for target_at_access, attr in core.walk_wildcard(child, target_at_access_template):
                node_replacements[target_at_access] = ast.Attribute(value=new_target, attr=attr)
            for target_iat_access in core.walk(child, target_iat_access_template):
                node_replacements[target_iat_access] = ast.Subscript(
                    value=new_target,
                    slice=ast.BinOp(
                        left=target_iat_access.slice,
                        op=ast.Add(),
                        right=ast.Constant(value=1, kind=None),
                ),)
        # All mentions should have been replaced, otherwise something is wrong.

        # I'm aware that attribute accesses don't necessarily need to be replaced, but
        # since there are over 200 "normal" attributes of a series/df, I consider this sort
        # of access bad practice as you'll be confused about what is what. And then I have
        # to maintain a list of those and, ... well, it's just a bad idea.
        n_new_target_mentions = len(
            [child for child in node.body for _ in core.walk(child, new_target)]
        )
        if n_new_target_mentions != len(node_replacements):
            continue

        new_iter = ast.Call(
            func=ast.Attribute(value=underlying_object, attr="itertuples"), args=[], keywords=[]
        )
        node_replacements[node.iter] = new_iter
        node_replacements[node.target] = new_target

        replacements.update(node_replacements)
    template = ast.comprehension(target=target_template, iter=iter_template)

    for node, replacement in replacements.items():
        yield node, replacement, 0
