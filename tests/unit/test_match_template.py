import ast
import textwrap
import unittest

from pyrefact import core


class TestMatchTemplate(unittest.TestCase):

    def test_template_matches_self(self):
        template = "x = 1"
        compiled_template = core.compile_template(template)
        node = ast.parse(template).body[0]

        m = core.match_template(node, compiled_template)
        self.assertIs(m[0], node)
        self.assertEqual(m, (node,))

        template = """
        class Foo:
            pass
        """
        template = textwrap.dedent(template)
        compiled_template = core.compile_template(template)
        node = ast.parse(template).body[0]

        m = core.match_template(node, compiled_template)
        self.assertIs(m[0], node)
        self.assertEqual(m, (node,))

        template = """
        class Foo(bar):
            x: int = 10
            z: str = "hello"
            def __init__(self, y: int | str):
                self.y = y
        """
        template = textwrap.dedent(template)
        compiled_template = core.compile_template(template)
        node = ast.parse(template).body[0]

        m = core.match_template(node, compiled_template)
        self.assertIs(m[0], node)
        self.assertEqual(m, (node,))

        template = '''
def format_with_black(source: str, *, line_length: int = 100) -> str:
    """Format code with black.

    Args:
        source (str): Python source code

    Returns:
        str: Formatted source code.
    """
    original_source = source
    indent = indentation_level(source)
    if indent > 0:
        source = textwrap.dedent(source)

    try:
        source = black.format_str(
            source, mode=black.Mode(line_length=max(60, line_length - indent))
        )
    except (SyntaxError, black.parsing.InvalidInput):
        logger.error("Black raised InvalidInput on code:{}", source)
        return original_source

    if indent > 0:
        source = textwrap.indent(source, " " * indent)

    return source
'''
        compiled_template = core.compile_template(template)
        node = ast.parse(template).body[0]

        m = core.match_template(node, compiled_template)
        self.assertIs(m[0], node)
        self.assertEqual(m, (node,))

        template = '''
class Range(NamedTuple):
    start: int  # Character number
    end: int  # Character number

    def overlaps(self, other: "Range") -> bool:
        return self.start < other.end and other.start < self.end

    # Use & operator for overlaps()
    def __and__(self, other: "Range") -> bool:
        return self.overlaps(other)
'''
        compiled_template = core.compile_template(template)
        node = ast.parse(template).body[0]

        m = core.match_template(node, compiled_template)
        self.assertIs(m[0], node)
        self.assertEqual(m, (node,))

    def test_basic_wildcard(self):
        template = "x = {{anything}}"
        code = "x = 1337 ** 99 - 100 ** qwerty"

        compiled_template = core.compile_template(template)
        node = ast.parse(code).body[0]
        m = core.match_template(node, compiled_template)
        self.assertIs(m.root, node)

        compiled_template = core.compile_template(template, anything=str)
        node = ast.parse(code).body[0]
        m = core.match_template(node, compiled_template)
        self.assertEqual(m, ())

        compiled_template = core.compile_template(template, anything=ast.BinOp)
        node = ast.parse(code).body[0]
        m = core.match_template(node, compiled_template)
        self.assertIs(m.root, node)

        template = """
        def f({{variable_name}}):
            foo.{{variable_name}} = {{variable_name}}
        """
        code = """
        def f(y):
            foo.y = y
        """
        compiled_template = core.compile_template(template)
        node = ast.parse(textwrap.dedent(code)).body[0]
        m = core.match_template(node, compiled_template)
        self.assertIs(m.root, node)

        template = """
        class Foo({{superclass}}):
            x: int = 10
            z: str = {{z_type}}
            def __init__(self, y: int | str):
                self.y = y
        """
        code = """
        class Foo(bar):
            x: int = 10
            z: str = "asdf"
            def __init__(self, y: int | str):
                self.y = y
        """
        compiled_template = core.compile_template(template)
        node = ast.parse(textwrap.dedent(code)).body[0]
        m = core.match_template(node, compiled_template)
        self.assertIs(m.root, node)

    def runTest(self):
        self.test_template_matches_self()
        self.test_basic_wildcard()


def main() -> int:
    # For use with ./tests/main.py, which looks for these main functions.
    # unittest.main() will do sys.exit() or something, it quits the whole
    # program after done and prevents further tests from running.
    test_result = TestMatchTemplate().run()
    if test_result.wasSuccessful():
        return 0

    test_result.printErrors()
    return 1


if __name__ == "__main__":
    unittest.main()
