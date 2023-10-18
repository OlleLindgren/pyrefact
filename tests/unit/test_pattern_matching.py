import unittest
from typing import Iterable, Sequence

from pyrefact import core, pattern_matching


class TestFinditer(unittest.TestCase):
    def test_basic_code(self):
        pattern = source = "x = 1"
        iterator = pattern_matching.finditer(pattern, source)
        self.assertIsInstance(iterator, Iterable)

        matches = list(iterator)
        self.assertEqual(len(matches), 1)

        m = matches[0]
        expected = core.Match(
            span=core.Range(start=0, end=5), source=source, groups=(core.parse(source).body[0],)
        )

        self.assertEqual(m.span, expected.span)
        self.assertEqual(m.source, expected.source)
        assert core.match_template(m.groups[0], expected.groups[0])
        assert core.match_template(expected.groups[0], m.groups[0])

    def test_complex_code(self):
        source = """
class Foo(bar):
    x: int = 10
    z: str = "hello"

    def __init__(self, y: int | str):
        self.y = y

class Spam(eggs):
    x: int = 10
    z: int = 3223

    def __init__(self, y: int | str):
        self.y = y

if __name__ == "__main__":
    x = 1
        """
        pattern = """
class {{name}}({{base}}):
    x: {{x_type}} = {{x_value}}
    z: {{z_type}} = {{z_value}}

    def __init__(self, y: int | str):
        self.y = y
        """
        iterator = pattern_matching.finditer(pattern, source)
        self.assertIsInstance(iterator, Iterable)

        matches = list(iterator)
        self.assertEqual(len(matches), 2)

        m = matches[0]
        expected = core.Match(
            span=core.Range(start=1, end=111), source=source, groups=(core.parse(source).body[0],)
        )

        self.assertEqual(m.span, expected.span)
        self.assertEqual(m.source, expected.source)
        assert core.match_template(m.groups[0], expected.groups[0])
        assert core.match_template(expected.groups[0], m.groups[0])

        m = matches[1]
        expected = core.Match(
            span=core.Range(start=113, end=222), source=source, groups=(core.parse(source).body[1],)
        )

        self.assertEqual(m.span, expected.span)
        self.assertEqual(m.source, expected.source)
        assert core.match_template(m.groups[0], expected.groups[0])
        assert core.match_template(expected.groups[0], m.groups[0])

    def runTest(self):
        self.test_basic_code()
        self.test_complex_code()


class TestFindall(unittest.TestCase):
    def test_basic_code(self):
        pattern = source = "x = 1"
        iterator = pattern_matching.findall(pattern, source)
        self.assertIsInstance(iterator, Iterable)

        matches = list(iterator)
        self.assertEqual(len(matches), 1)

        m = matches[0]
        self.assertEqual(m, source)

    def test_complex_code(self):
        source = """
class Foo(bar):
    x: int = 10
    z: str = "hello"

    def __init__(self, y: int | str):
        self.y = y

class Spam(eggs):
    x: int = 10
    z: int = 3223

    def __init__(self, y: int | str):
        self.y = y

if __name__ == "__main__":
    x = 1
        """
        pattern = """
class {{name}}({{base}}):
    x: {{x_type}} = {{x_value}}
    z: {{z_type}} = {{z_value}}

    def __init__(self, y: int | str):
        self.y = y
        """
        iterator = pattern_matching.findall(pattern, source)
        self.assertIsInstance(iterator, Sequence)

        matches = list(iterator)
        self.assertEqual(len(matches), 2)

        self.assertEqual(matches[0], source[1:111])
        self.assertEqual(matches[1], source[113:222])

    def runTest(self):
        self.test_basic_code()
        self.test_complex_code()


class TestSearch(unittest.TestCase):
    def test_basic_code(self):
        pattern = source = "x = 1"
        m = pattern_matching.search(pattern, source)
        self.assertIsInstance(m, core.Match)

        expected = core.Match(
            span=core.Range(start=0, end=5), source=source, groups=(core.parse(source).body[0],)
        )

        self.assertEqual(m.span, expected.span)
        self.assertEqual(m.source, expected.source)
        assert core.match_template(m.groups[0], expected.groups[0])
        assert core.match_template(expected.groups[0], m.groups[0])

        # Some tests of the Match type itself
        # TODO move these to a separate test file/class/function or something
        self.assertEqual(m.span.start, m.start)
        self.assertEqual(m.span.end, m.end)
        self.assertEqual(m.string, m.source[m.start : m.end])
        self.assertEqual(m.lineno, 1)
        self.assertEqual(m.col_offset, 0)
        self.assertIs(m.root, m.groups[0])

    def test_complex_code(self):
        source = """
class Foo(bar):
    x: int = 10
    z: str = "hello"

    def __init__(self, y: int | str):
        self.y = y

class Spam(eggs):
    x: int = 10
    z: int = 3223

    def __init__(self, y: int | str):
        self.y = y

if __name__ == "__main__":
    x = 1
        """
        pattern = """
class {{name}}({{base}}):
    x: {{x_type}} = {{x_value}}
    z: {{z_type}} = {{z_value}}

    def __init__(self, y: int | str):
        self.y = y
        """
        m = pattern_matching.search(pattern, source)
        self.assertIsInstance(m, core.Match)

        expected = core.Match(
            span=core.Range(start=1, end=111), source=source, groups=(core.parse(source).body[0],)
        )

        self.assertEqual(m.span, expected.span)
        self.assertEqual(m.source, expected.source)
        assert core.match_template(m.groups[0], expected.groups[0])
        assert core.match_template(expected.groups[0], m.groups[0])

    def runTest(self):
        self.test_basic_code()
        self.test_complex_code()


class TestSub(unittest.TestCase):
    def test_basic_code(self):
        pattern = source = "x = 1"
        replacement = "x = 2"
        result = pattern_matching.sub(pattern, replacement, source)
        self.assertIsInstance(result, str)
        self.assertEqual(result, replacement)

    def test_complex_code(self):
        source = """
class Foo(bar):
    x: int = 10
    z: str = "hello"

    def __init__(self, y: int | str):
        self.y = y

class Spam(eggs):
    x: int = 10
    z: int = 3223

    def __init__(self, y: int | str):
        self.y = y

if __name__ == "__main__":
    x = 1
        """
        pattern = """
class {{name}}({{base}}):
    x: {{x_type}} = {{x_value}}
    z: {{z_type}} = {{z_value}}

    def __init__(self, y: int | str):
        self.y = y
        """
        replacement = """
class {{name}}({{base}}):
    '''{{name}} serves an important purpose.'''

    x: {{x_type}} = {{x_value}} - 1
    z: {{z_type}} = 2 ** {{z_value}} - 3 ** {{x_value}}

    def __init__(self, y: int | str):
        self.y = y
        """
        expected = """
class Foo(bar):
    '''Foo serves an important purpose.'''

    x: int = 10 - 1
    z: str = 2 ** "hello" - 3 ** 10

    def __init__(self, y: int | str):
        self.y = y

class Spam(eggs):
    '''Spam serves an important purpose.'''

    x: int = 10 - 1
    z: int = 2 ** 3223 - 3 ** 10

    def __init__(self, y: int | str):
        self.y = y

if __name__ == "__main__":
    x = 1
        """

        result = pattern_matching.sub(pattern, replacement, source)
        self.assertIsInstance(result, str)

        self.assertEqual(result, expected)

    def runTest(self):
        self.test_basic_code()
        self.test_complex_code()


def main() -> int:
    # For use with ./tests/main.py, which looks for these main functions.
    # unittest.main() will do sys.exit() or something, it quits the whole
    # program after done and prevents further tests from running.
    test_result = TestFinditer().run()
    if not test_result.wasSuccessful():
        test_result.printErrors()
        return 1

    test_result = TestFindall().run()
    if not test_result.wasSuccessful():
        test_result.printErrors()
        return 1

    test_result = TestSearch().run()
    if not test_result.wasSuccessful():
        test_result.printErrors()
        return 1

    test_result = TestSub().run()
    if not test_result.wasSuccessful():
        test_result.printErrors()
        return 1

    return 0


if __name__ == "__main__":
    unittest.main()
