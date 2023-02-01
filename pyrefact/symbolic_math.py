"""Features related to symbolic mathematics."""


import ast
import math
from typing import Callable, Sequence, Tuple

import sympy
import sympy.parsing

from pyrefact import constants, parsing, processing


def _get_range_start_end(rng: ast.Call) -> Tuple[ast.AST, ast.AST]:
    if not (isinstance(rng.func, ast.Name) and rng.func.id == "range"):
        raise ValueError(f"Expected range call, not {rng}")

    if len(rng.args) == 1:
        return ast.Constant(value=0, kind=None), rng.args[0], ast.Constant(value=1, kind=None)

    if len(rng.args) == 2:
        return *rng.args, ast.Constant(value=1, kind=None)

    if len(rng.args) == 3:
        return tuple(rng.args)

    raise NotImplementedError(f"Cannot parse range with arg count of {len(rng.args)}")


def _parse_sympy_expr(expression):
    return sympy.parsing.sympy_parser.parse_expr(expression)


def _simplify_math(f: Callable) -> ast.AST:
    def wrapper(*args, **kwargs):
        expression = f(*args, **kwargs)
        source = parsing.unparse(expression).strip()

        # TODO substitute constant calls, attributes and other stuff with variables

        source = str(sympy.simplify(source))
        return parsing.parse(source)

    return wrapper


@_simplify_math
def _sum_int_squares_to(value: ast.AST) -> ast.AST:
    return ast.BinOp(
        left=ast.BinOp(
            left=ast.BinOp(left=value, op=ast.Sub(), right=ast.Constant(value=1, kind=None)),
            op=ast.Mult(),
            right=value,
        ),
        op=ast.Div(),
        right=ast.Constant(value=2, kind=None),
    )


@_simplify_math
def _sum_range(rng: ast.Call) -> ast.AST:
    start, end, step = _get_range_start_end(rng)

    if not parsing.match_template(step, ast.Constant(value=1)):
        return rng

    if parsing.match_template(end, ast.Constant(value=0)):
        return _sum_int_squares_to(end)

    return ast.BinOp(
        left=_sum_int_squares_to(end),
        op=ast.Sub(),
        right=_sum_int_squares_to(start),
    )


@_simplify_math
def _sum_constants(values: Sequence[ast.AST]) -> ast.AST:
    expr = " + ".join(parsing.unparse(node).strip() for node in values)
    return parsing.parse(expr)


def _integrate_over(expr: ast.AST, generators: Sequence[ast.comprehension]) -> ast.AST:
    source = parsing.unparse(expr).strip()
    sym_expr = _parse_sympy_expr(source)
    for comprehension in generators:
        integrand = _parse_sympy_expr(parsing.unparse(comprehension.target).strip())
        if isinstance(comprehension.iter, ast.Call):
            start, end, step = _get_range_start_end(comprehension.iter)
            lower = _parse_sympy_expr(parsing.unparse(start).strip())
            upper = _parse_sympy_expr(parsing.unparse(end).strip())
            step = _parse_sympy_expr(parsing.unparse(step).strip())

            if step == 1:
                upper -= 1
            else:
                n_steps = math.floor((upper - lower) / step)
                lower = lower / step
                upper = lower + n_steps
                sym_expr = sym_expr.subs(integrand, step * integrand)

            sym_expr = sympy.Sum(sym_expr, (integrand, lower, upper))

        elif isinstance(comprehension.iter, (ast.Tuple, ast.List, ast.Set)):
            values = [
                _parse_sympy_expr(parsing.unparse(value).strip())
                for value in comprehension.iter.elts
            ]
            if isinstance(comprehension.iter, ast.Set):
                values = set(values)

            sym_expr = sum(sym_expr.subs(integrand, value) for value in values)

        else:
            raise NotImplementedError(f"Cannot parse iterator: {comprehension.iter}")

    sym_expr = sym_expr.doit()
    sym_expr = sympy.simplify(sym_expr)

    return parsing.parse(str(sym_expr))


@processing.fix
def simplify_math_iterators(source: str) -> str:

    root = parsing.parse(source)

    template = ast.Call(
        func=ast.Name(id=tuple(constants.MATH_FUNCTIONS)),
        keywords=[],
        args=[object],
    )
    basic_types_template = {ast.Name, ast.Constant, ast.UnaryOp, ast.BinOp}
    basic_iter_template = (
        ast.Call(func=ast.Name(id="range"), args=basic_types_template),
        ast.Set(elts=basic_types_template),
        ast.List(elts=basic_types_template),
        ast.Tuple(elts=basic_types_template),
    )
    basic_generator_template = ast.comprehension(iter=basic_iter_template, ifs=[], target=ast.Name)
    basic_comprehension_template = (
        ast.GeneratorExp(generators={basic_generator_template}),
        ast.ListComp(generators={basic_generator_template}),
    )
    basic_collection_template = (
        ast.Tuple(elts={ast.Constant, ast.UnaryOp, ast.BinOp}),
        ast.List(elts={ast.Constant, ast.UnaryOp, ast.BinOp}),
    )

    for node in parsing.walk(root, template):
        arg = node.args[0]
        if parsing.match_template(arg, ast.Call(func=ast.Name(id="range"))):
            if any((node is not arg for node in parsing.walk(arg, (ast.Attribute, ast.Call)))):
                continue
            if node.func.id != "sum":
                continue
            yield node, _sum_range(arg)

        elif parsing.match_template(arg, basic_collection_template):
            if any(parsing.walk(arg, ast.Attribute)):
                continue
            if not all(
                parsing.match_template(node.func, ast.Name(id="range"))
                for node in parsing.walk(arg, ast.Call)
            ):
                continue
            yield node, _sum_constants(arg.elts)

        elif parsing.match_template(arg, basic_comprehension_template):
            if any(parsing.walk(arg, ast.Attribute)):
                continue
            if not all(
                parsing.match_template(node.func, ast.Name(id="range"))
                for node in parsing.walk(arg, ast.Call)
            ):
                continue
            yield node, _integrate_over(arg.elt, arg.generators)
