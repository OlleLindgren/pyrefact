"""Features related to symbolic mathematics."""
from __future__ import annotations


import ast
import collections
import itertools
import math
from typing import Callable, Mapping, Sequence, Tuple

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


def _ast_to_symmath_expr_conversion(node, conversion):
    if isinstance(node, ast.BoolOp):
        values = []
        for value in node.values:
            expression, node_conversion = _ast_to_symmath_expr_conversion(value, conversion)
            conversion.update(node_conversion)
            values.append(expression)
        if isinstance(node.op, ast.And):
            return sympy.And(*values), conversion
        if isinstance(node.op, ast.Or):
            return sympy.Or(*values), conversion

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        operand, conversion = _ast_to_symmath_expr_conversion(node.operand, conversion)
        return sympy.Not(operand), conversion

    if isinstance(node, ast.Name):
        return sympy.Symbol(node.id), conversion

    unparse = parsing.unparse(node)
    if unparse in conversion:
        expression, _ = conversion[unparse]
        return expression, conversion

    variable_name = f"__unparse_var_{len(conversion)}__"
    if conversion:
        _, variable_names = zip(*conversion.values())
        assert variable_name not in variable_names

    expression = sympy.Symbol(variable_name)
    conversion[unparse] = expression, node
    return expression, conversion


def _ast_to_symmath_expr(node):
    expression, conversion = _ast_to_symmath_expr_conversion(node, {})
    conversion = dict(conversion.values())
    return expression, conversion


def _symmath_expr_to_ast(expression, conversion: Mapping[str, ast.AST]):
    if isinstance(expression, sympy.Symbol):
        if expression in conversion:
            return conversion[expression]
        return ast.Name(id=expression.name)
    if isinstance(expression, sympy.And):
        return ast.BoolOp(op=ast.And(), values=[_symmath_expr_to_ast(v, conversion) for v in expression.args])
    if isinstance(expression, sympy.Or):
        return ast.BoolOp(op=ast.Or(), values=[_symmath_expr_to_ast(v, conversion) for v in expression.args])
    if isinstance(expression, sympy.Not):
        return ast.UnaryOp(op=ast.Not(), operand=_symmath_expr_to_ast(expression.args[0], conversion))
    if isinstance(expression, sympy.logic.boolalg.BooleanTrue):
        return ast.Constant(value=True, kind=None)
    if isinstance(expression, sympy.logic.boolalg.BooleanFalse):
        return ast.Constant(value=False, kind=None)

    raise ValueError(f"Unsupported expression: {expression} of type {type(expression)}")


def simplify_ast_boolop(node: ast.BoolOp | ast.UnaryOp) -> ast.BoolOp | ast.UnaryOp:
    """Simplify boolean AST expression.

    Args:
        node (ast.BoolOp | ast.UnaryOp): Boolean AST expression

    Returns:
        ast.BoolOp | ast.UnaryOp: Simplified boolean AST expression

    """
    expression, conversion = _ast_to_symmath_expr(node)
    expression = expression.simplify()

    simplified_node = _symmath_expr_to_ast(expression, conversion)
    return simplified_node


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


@processing.fix
def simplify_boolean_expressions(source: str) -> str:
    root = parsing.parse(source)

    regular_compare_template = ast.Compare(
        left=object,
        ops=[(ast.Eq, ast.NotEq, ast.Gt, ast.Lt, ast.GtE, ast.LtE)],
        comparators=[object])

    for node in parsing.walk(root, ast.BoolOp):

        if isinstance(node.op, (ast.And, ast.Or)):
            # Find opposite expressions
            expression_conditions = collections.defaultdict(set)
            for value in node.values:
                if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not):
                    expression_conditions[parsing.unparse(value.operand)].add(False)
                else:
                    expression_conditions[parsing.unparse(value)].add(True)

            if {True, False} in expression_conditions.values():
                # Something can (in the or case) or must (in the and case) be both True and False
                if isinstance(node.op, ast.Or):
                    yield node, ast.Constant(value=True, kind=None)
                else:
                    yield node, ast.Constant(value=False, kind=None)

                continue

            # Find redundant combinations of >, >=, <, >=, ==, !=
            constant_bounds = collections.defaultdict(lambda: collections.defaultdict(set))
            # Should contain for example:
            # {"f(x)": {ast.Gt: {(4, <ast>)}, ast.Lt: {(6, <ast>)}, ast.Eq: {(5, <ast>)}}}
            # Which would mean that f(x) is between 4 and 6, and is equal to 5, so this
            # could be simplified to f(x) == 5. Since the asts that provided the Gt, Lt
            # and Eq constraints are also known, we know that we can remove the ones that
            # gave the gt and lt constraints.
            constraint_values = list(node.values)
            for value in constraint_values:
                if parsing.match_template(value, ast.BoolOp(op=type(node.op))):
                    constraint_values.extend(value.values)
                    continue

            for value in parsing.filter_nodes(constraint_values, regular_compare_template):
                left_is_unparsed = False
                right_is_unparsed = False
                try:
                    left = parsing.literal_value(value.left)
                except ValueError:
                    left = parsing.unparse(value.left)
                    left_is_unparsed = True

                try:
                    right = parsing.literal_value(value.comparators[0])
                except ValueError:
                    right = parsing.unparse(value.comparators[0])
                    right_is_unparsed = True

                if left_is_unparsed and right_is_unparsed:
                    continue

                if not left_is_unparsed and not right_is_unparsed:
                    # Pure literal comparison is handled in the next step
                    continue

                op_type = type(value.ops[0])
                opposite_op_mapping = {
                    ast.Eq: ast.Eq,
                    ast.NotEq: ast.NotEq,
                    ast.Gt: ast.Lt,
                    ast.Lt: ast.Gt,
                    ast.GtE: ast.LtE,
                    ast.LtE: ast.GtE,
                }
                if right_is_unparsed:
                    op_type = opposite_op_mapping[op_type]
                    left, right = right, left

                if not isinstance(right, (int, float, bool)):
                    continue

                try:
                    constant_bounds[left][op_type].add((right, value))
                except TypeError:
                    pass

            redundant_and_values = set()
            redundant_or_values = set()
            always_true = False
            always_false = False
            for left, bounds in constant_bounds.items():
                # Check for identical constraints
                for (thr1, thr1_value), (thr2, thr2_value) in itertools.chain.from_iterable(
                    itertools.combinations(subset, 2)
                    for subset in bounds.values()
                ):
                    if thr1 == thr2:
                        if thr1_value in node.values:
                            redundant_and_values.add(thr1_value)
                            redundant_or_values.add(thr1_value)
                        else:
                            redundant_and_values.add(thr2_value)
                            redundant_or_values.add(thr2_value)

                # Check for redundant constraints, where one is stronger than the other.
                # These checks purposefully do not cover the case where the two constraints
                # are equal, since that is already covered by the previous check.
                if len(bounds[ast.Eq]) >= 2:
                    for (eq1, _), (eq2, eq2_value) in itertools.combinations(bounds[ast.Eq], 2):
                        if eq1 == eq2:
                            redundant_and_values.add(eq2_value)
                            redundant_or_values.add(eq2_value)
                        if eq1 != eq2:
                            always_false |= isinstance(node.op, ast.And)

                if len(bounds[ast.Gt]) >= 2:
                    for (gt1, gt1_value), (gt2, gt2_value) in itertools.combinations(bounds[ast.Gt], 2):
                        if gt1 > gt2:
                            redundant_and_values.add(gt2_value)
                            redundant_or_values.add(gt1_value)
                        if gt1 < gt2:
                            redundant_and_values.add(gt1_value)
                            redundant_or_values.add(gt2_value)

                if len(bounds[ast.Lt]) >= 2:
                    for (lt1, lt1_value), (lt2, lt2_value) in itertools.combinations(bounds[ast.Lt], 2):
                        if lt1 < lt2:
                            redundant_and_values.add(lt2_value)
                            redundant_or_values.add(lt1_value)
                        if lt1 > lt2:
                            redundant_and_values.add(lt1_value)
                            redundant_or_values.add(lt2_value)

                if len(bounds[ast.GtE]) >= 2:
                    for (gte1, gte1_value), (gte2, gte2_value) in itertools.combinations(bounds[ast.GtE], 2):
                        if gte1 > gte2:
                            redundant_and_values.add(gte2_value)
                            redundant_or_values.add(gte1_value)
                        if gte1 < gte2:
                            redundant_and_values.add(gte1_value)
                            redundant_or_values.add(gte2_value)

                if len(bounds[ast.LtE]) >= 2:
                    for (lte1, lte1_value), (lte2, lte2_value) in itertools.combinations(bounds[ast.LtE], 2):
                        if lte1 < lte2:
                            redundant_and_values.add(lte2_value)
                            redundant_or_values.add(lte1_value)
                        if lte1 > lte2:
                            redundant_and_values.add(lte1_value)
                            redundant_or_values.add(lte2_value)

                # eq vs everything else
                for eq, eq_value in bounds[ast.Eq]:
                    for neq, neq_value in bounds[ast.NotEq]:
                        if eq == neq:
                            always_true |= isinstance(node.op, ast.Or)
                            always_false |= isinstance(node.op, ast.And)
                        if eq != neq:
                            redundant_or_values.add(eq_value)
                            redundant_and_values.add(neq_value)

                    for gt, gt_value in bounds[ast.Gt]:
                        if eq <= gt:
                            always_false |= isinstance(node.op, ast.And)
                        if eq > gt:
                            redundant_or_values.add(eq_value)
                            redundant_and_values.add(gt_value)

                    for lt, lt_value in bounds[ast.Lt]:
                        if eq >= lt:
                            always_false |= isinstance(node.op, ast.And)
                        if eq < lt:
                            redundant_or_values.add(eq_value)
                            redundant_and_values.add(lt_value)

                    for gte, gte_value in bounds[ast.GtE]:
                        if eq < gte:
                            always_false |= isinstance(node.op, ast.And)
                        if eq >= gte:
                            redundant_or_values.add(eq_value)
                            redundant_and_values.add(gte_value)

                    for lte, lte_value in bounds[ast.LtE]:
                        if eq > lte:
                            always_false |= isinstance(node.op, ast.And)
                        if eq <= lte:
                            redundant_or_values.add(eq_value)
                            redundant_and_values.add(lte_value)

                # neq vs everything else, except eq
                for neq, neq_value in bounds[ast.NotEq]:
                    for gt, gt_value in bounds[ast.Gt]:
                        if neq <= gt:
                            redundant_and_values.add(neq_value)
                            redundant_or_values.add(gt_value)
                        if neq > gt:
                            always_true |= isinstance(node.op, ast.Or)

                    for lt, lt_value in bounds[ast.Lt]:
                        if neq >= lt:
                            redundant_and_values.add(neq_value)
                            redundant_or_values.add(lt_value)
                        if neq < lt:
                            always_true |= isinstance(node.op, ast.Or)

                    for gte, gte_value in bounds[ast.GtE]:
                        if neq < gte:
                            redundant_and_values.add(neq_value)
                            redundant_or_values.add(gte_value)
                        if neq >= gte:
                            always_true |= isinstance(node.op, ast.Or)

                    for lte, lte_value in bounds[ast.LtE]:
                        if neq > lte:
                            redundant_and_values.add(neq_value)
                            redundant_or_values.add(lte_value)
                        if neq <= lte:
                            always_true |= isinstance(node.op, ast.Or)

                # gt vs everything else, except eq and neq
                for gt, gt_value in bounds[ast.Gt]:
                    for lt, lt_value in bounds[ast.Lt]:
                        if gt >= lt:
                            always_false |= isinstance(node.op, ast.And)
                        if gt < lt:
                            always_true |= isinstance(node.op, ast.Or)

                        if gt == lt:
                            for eq, eq_value in bounds[ast.Eq]:
                                if gt == eq:
                                    always_true |= isinstance(node.op, ast.Or)
                                    always_false |= isinstance(node.op, ast.And)

                    for lte, lte_value in bounds[ast.LtE]:
                        if gt >= lte:
                            always_false |= isinstance(node.op, ast.And)
                        if gt < lte:
                            always_true |= isinstance(node.op, ast.Or)

                    for gte, gte_value in bounds[ast.GtE]:
                        if gt > gte:
                            redundant_and_values.add(gte_value)
                            redundant_or_values.add(gt_value)
                        if gt <= gte:
                            redundant_and_values.add(gt_value)
                            redundant_or_values.add(gte_value)

                # gte vs everything else, except eq, neq and gt
                for gte, gte_value in bounds[ast.GtE]:
                    for lt, lt_value in bounds[ast.Lt]:
                        if gte >= lt:
                            always_false |= isinstance(node.op, ast.And)
                        if gte < lt:
                            always_true |= isinstance(node.op, ast.Or)

                    for lte, lte_value in bounds[ast.LtE]:
                        if gte > lte:
                            always_false |= isinstance(node.op, ast.And)
                        if gte <= lte:
                            always_true |= isinstance(node.op, ast.Or)

                # lt vs everything else, except eq, neq, gt and gte (i.e. only lte)
                for lt, lt_value in bounds[ast.Lt]:
                    for lte, lte_value in bounds[ast.LtE]:
                        if lt <= lte:
                            redundant_and_values.add(lte_value)
                            redundant_or_values.add(lt_value)
                        if lt > lte:
                            redundant_and_values.add(lt_value)
                            redundant_or_values.add(lte_value)

            if always_false:
                yield node, ast.Constant(value=False, kind=None)
                continue
            
            if always_true:
                yield node, ast.Constant(value=True, kind=None)
                continue

            if redundant_and_values and isinstance(node.op, ast.And):
                values = [value for value in node.values if value not in redundant_and_values]
                if len(values) == 1:
                    yield node, values[0]
                    continue
                if values != node.values:
                    yield node, ast.BoolOp(op=node.op, values=values)
                    continue

            if redundant_or_values and isinstance(node.op, ast.Or):
                values = [value for value in node.values if value not in redundant_or_values]
                if len(values) == 1:
                    yield node, values[0]
                    continue
                if values != node.values:
                    yield node, ast.BoolOp(op=node.op, values=values)
                    continue

        if isinstance(node.op, ast.And):
            # One of node.values is always False => Expression is always False
            if any(isinstance(value, ast.Constant) and not value.value for value in node.values):
                yield node, ast.Constant(value=False, kind=None)
                continue

            # Remove all True values
            values = [value for value in node.values if not parsing.match_template(value, ast.Constant(value=True))]
            if not values:
                yield node, ast.Constant(value=True, kind=None)
                continue

            if len({parsing.unparse(value) for value in values}) == 1:
                yield node, values[0]
                continue

            if len(values) < len(node.values):
                yield node, ast.BoolOp(op=ast.And(), values=values)
                continue

        elif isinstance(node.op, ast.Or):
            # One of node.values is always True => Expression is always True
            if any(isinstance(value, ast.Constant) and value.value for value in node.values):
                yield node, ast.Constant(value=True, kind=None)
                continue

            # Remove all False values
            values = [value for value in node.values if not parsing.match_template(value, ast.Constant(value=False))]
            if not values:
                yield node, ast.Constant(value=False, kind=None)
                continue

            if len({parsing.unparse(value) for value in values}) == 1:
                yield node, values[0]
                continue

            if len(values) < len(node.values):
                yield node, ast.BoolOp(op=ast.Or(), values=values)
                continue

    for node in parsing.walk(root, ast.UnaryOp):
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Constant):
            yield node, ast.Constant(value=not node.operand.value, kind=None)

    for node in parsing.walk(root, regular_compare_template):
        operator = node.ops[0]
        comparator = node.comparators[0]
        try:
            left = parsing.literal_value(node.left)
            right = parsing.literal_value(comparator)
        except ValueError:
            if isinstance(operator, ast.Eq) and parsing.unparse(node.left) == parsing.unparse(comparator):
                yield node, ast.Constant(value=True, kind=None)

            continue

        if isinstance(operator, ast.Eq):
            yield node, ast.Constant(value=left == right, kind=None)

        elif isinstance(operator, ast.NotEq):
            yield node, ast.Constant(value=left != right, kind=None)

        elif isinstance(operator, ast.Gt):
            yield node, ast.Constant(value=left > right, kind=None)
        
        elif isinstance(operator, ast.Lt):
            yield node, ast.Constant(value=left < right, kind=None)
        
        elif isinstance(operator, ast.GtE):
            yield node, ast.Constant(value=left >= right, kind=None)

        elif isinstance(operator, ast.LtE):
            yield node, ast.Constant(value=left <= right, kind=None)


@processing.fix(restart_on_replace=True)
def simplify_boolean_expressions_symmath(source: str) -> str:
    root = parsing.parse(source)
    for node in parsing.walk(root, (ast.BoolOp, ast.UnaryOp)):
        try:
            simplified = simplify_ast_boolop(node)
        except ValueError:
            continue

        node_complexity = sum(len(x.values) for x in parsing.walk(node, ast.BoolOp))
        simplified_complexity = sum(len(x.values) for x in parsing.walk(simplified, ast.BoolOp))

        if simplified_complexity < node_complexity:
            yield node, simplified
