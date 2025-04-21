import ast
import unittest

from pyrefact import core


def parse_one_node(source: str) -> ast.AST:
    module = ast.parse(source)
    assert len(module.body) == 1
    assert isinstance(module.body[0], ast.Expr)
    return module.body[0].value


class TestLiteralValue(unittest.TestCase):
    def test_basic_datatypes(self):
        node = parse_one_node("1")
        assert core.literal_value(node) == 1

        node = parse_one_node("1.5")
        assert core.literal_value(node) == 1.5

        node = parse_one_node("False")
        assert core.literal_value(node) == False

        node = parse_one_node("()")
        assert core.literal_value(node) == ()

        node = parse_one_node("[]")
        assert core.literal_value(node) == []

        node = parse_one_node("{}")
        assert core.literal_value(node) == {}

        node = parse_one_node("'abcd'")
        assert core.literal_value(node) == "abcd"

        node = parse_one_node("None")
        assert core.literal_value(node) == None

    def test_nested_data(self):
        node = parse_one_node("[1, 2, 3]")
        assert core.literal_value(node) == [1, 2, 3]

        node = parse_one_node("{'a': 1, 'b': 2}")
        assert core.literal_value(node) == {"a": 1, "b": 2}

        node = parse_one_node("{'a': [1, 2, 3], 'b': {'c': 4}}")
        assert core.literal_value(node) == {"a": [1, 2, 3], "b": {"c": 4}}

    def test_comparisons(self):
        node = parse_one_node("1 < 2")
        assert core.literal_value(node) == True

        node = parse_one_node("1 > 2")
        assert core.literal_value(node) == False

        node = parse_one_node("1 == 2")
        assert core.literal_value(node) == False

        node = parse_one_node("1 != 2")
        assert core.literal_value(node) == True

        node = parse_one_node("1 <= 2")
        assert core.literal_value(node) == True

        node = parse_one_node("1 >= 2")
        assert core.literal_value(node) == False

        node = parse_one_node("1 < 2 < 3 < 4 == 4 < 4.5")
        assert core.literal_value(node) == True

        node = parse_one_node("not True")
        assert core.literal_value(node) == False

        node = parse_one_node("[] or ()")
        assert core.literal_value(node) == ()

        node = parse_one_node("[] and ()")
        assert core.literal_value(node) == []

        node = parse_one_node("not []")
        assert core.literal_value(node) == True

        node = parse_one_node("1 and 2")
        assert core.literal_value(node) == 2

        node = parse_one_node("1 or 2")
        assert core.literal_value(node) == 1

    def test_arithmetic(self):
        node = parse_one_node("1 + 2")
        assert core.literal_value(node) == 3

        node = parse_one_node("1 - 2")
        assert core.literal_value(node) == -1

        node = parse_one_node("[1] + [2]")
        assert core.literal_value(node) == [1, 2]

    def test_literal_calls(self):
        node = parse_one_node("''.join(['a', 'b'])")
        assert core.literal_value(node) == "ab"

        node = parse_one_node("len('abc')")
        assert core.literal_value(node) == 3

        node = parse_one_node("len([1, 2, 3])")
        assert core.literal_value(node) == 3

    def runTest(self):
        self.test_basic_datatypes()
        self.test_nested_data()
        self.test_comparisons()
        self.test_arithmetic()
        self.test_literal_calls()


def main() -> int:
    # For use with ./tests/main.py, which looks for these main functions.
    # unittest.main() will do sys.exit() or something, it quits the whole
    # program after done and prevents further tests from running.

    test_result = TestLiteralValue().run()
    if not test_result.wasSuccessful():
        test_result.printErrors()
        return 1

    return 0


if __name__ == "__main__":
    unittest.main()
