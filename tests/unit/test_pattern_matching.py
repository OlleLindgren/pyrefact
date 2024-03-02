#!/usr/bin/env python3
import ast
import platform
import sys
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


class TestMatch(unittest.TestCase):
    def test_basic_code(self):
        pattern = source = "x = 1"
        m = pattern_matching.match(pattern, source)
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
        m = pattern_matching.match(pattern, source)
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


class TestFullMatch(unittest.TestCase):
    def test_basic_code(self):
        pattern = source = "x = 1"
        m = pattern_matching.fullmatch(pattern, source)
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
        m = pattern_matching.fullmatch(pattern, source)
        self.assertIs(m, None)

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
        """
        pattern = """
class {{name1}}({{base1}}):
    x: {{x_type1}} = {{x_value1}}
    z: {{z_type1}} = {{z_value1}}

    def __init__(self, y: int | str):
        self.y = y

class {{name2}}({{base2}}):
    x: {{x_type2}} = {{x_value2}}
    z: {{z_type2}} = {{z_value2}}

    def __init__(self, y: int | str):
        self.y = y

        """
        m = pattern_matching.fullmatch(pattern, source)
        self.assertIsInstance(m, core.Match)

        expected = core.Match(
            span=core.Range(start=1, end=222),
            source=source,
            groups=(core.parse(source),),
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

        # count=0 is the same as not passing count
        result = pattern_matching.sub(pattern, replacement, source, count=0)
        self.assertIsInstance(result, str)

        self.assertEqual(result, expected)

        result = pattern_matching.sub(pattern, replacement, source, count=2)
        self.assertIsInstance(result, str)

        self.assertEqual(result, expected)

        result = pattern_matching.sub(pattern, replacement, source, count=100)
        self.assertIsInstance(result, str)

        self.assertEqual(result, expected)

        expected = """
class Foo(bar):
    '''Foo serves an important purpose.'''

    x: int = 10 - 1
    z: str = 2 ** "hello" - 3 ** 10

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

        result = pattern_matching.sub(pattern, replacement, source, count=1)
        self.assertIsInstance(result, str)

        self.assertEqual(result, expected)

    def test_for_recursion(self):
        source = "x = y - z"
        pattern = "{{some}} - {{other}}"
        replacement = "{{other}} - {{some}}"
        expected = "x = z - y"

        result = pattern_matching.sub(pattern, replacement, source)
        self.assertIsInstance(result, str)

        self.assertEqual(result, expected)

    def runTest(self):
        self.test_basic_code()
        self.test_complex_code()
        self.test_for_recursion()


class TestCompile(unittest.TestCase):

    def test_function_alias(self):
        assert pattern_matching.compile is core.compile_template

    def test_basic_code(self):

        source = "x = 10"
        pattern = core.compile_template(source)
        expected = ast.Assign(
            targets=[ast.Name(id='x')],
            value=ast.Constant(value=10, kind=None),
            type_comment=None,
        )
        assert core.match_template(pattern, expected)
        assert core.match_template(expected, pattern)

    def test_wildcard_code(self):
        source = "{{x}} = 10"
        pattern = core.compile_template(source)
        expected = ast.Assign(
            targets=[core.Wildcard(name="x", template=object)],
            value=ast.Constant(value=10, kind=None),
            type_comment=None,
        )
        assert pattern.targets == expected.targets

    def test_indented_code(self):
        source = """
        x = 10
        """
        pattern = core.compile_template(source)
        expected = ast.Assign(
            targets=[ast.Name(id='x')],
            value=ast.Constant(value=10, kind=None),
            type_comment=None,
        )
        assert core.match_template(pattern, expected)
        assert core.match_template(expected, pattern)

    def test_multiline_code(self):
        source = """
        import os
        class Foo(bar):
            x: int = 10

        h = 100
        """
        pattern = core.compile_template(source)
        expected = [
            ast.Import(names=[ast.alias(name='os', asname=None)]),
            ast.ClassDef(
                name='Foo',
                bases=[ast.Name(id='bar')],
                keywords=[],
                body=[
                    ast.AnnAssign(
                        target=ast.Name(id='x'),
                        annotation=ast.Name(id='int'),
                        value=ast.Constant(value=10, kind=None),
                        simple=1,
                    ),
                ],
                decorator_list=[],
            ),
            ast.Assign(
                targets=[ast.Name(id='h')],
                value=ast.Constant(value=100, kind=None),
                type_comment=None,
            ),
        ]
        assert isinstance(pattern, list)
        assert len(pattern) == len(expected)
        assert core.match_template(pattern, expected)
        assert core.match_template(expected, pattern)

    def test_zeroorone(self):
        source = """
        x = 10
        {{...?}}
        z = 1
        {{name?}}
        """
        pattern = core.compile_template(source)
        expected = [
            ast.Assign(
                targets=[ast.Name(id='x')],
                value=ast.Constant(value=10, kind=None),
                type_comment=None,
            ),
            core.ZeroOrOne(object),
            ast.Assign(
                targets=[ast.Name(id='z')],
                value=ast.Constant(value=1, kind=None),
                type_comment=None,
            ),
            core.ZeroOrOne(core.Wildcard(name="name", template=object, common=False)),
        ]
        assert isinstance(pattern, list)
        assert len(pattern) == len(expected)
        assert core.match_template(pattern[0], expected[0])
        assert core.match_template(pattern[2], expected[2])
        assert pattern[1] == expected[1]
        assert pattern[3] == expected[3]

    def test_one(self):
        source = """
        x = 10
        {{...}}
        z = 1
        {{name}}
        """
        pattern = core.compile_template(source)
        expected = [
            ast.Assign(
                targets=[ast.Name(id='x')],
                value=ast.Constant(value=10, kind=None),
                type_comment=None,
            ),
            core.Wildcard(name="Ellipsis_anything", template=object, common=False),
            ast.Assign(
                targets=[ast.Name(id='z')],
                value=ast.Constant(value=1, kind=None),
                type_comment=None,
            ),
            core.Wildcard(name="name", template=object, common=True),
        ]
        assert isinstance(pattern, list)
        assert len(pattern) == len(expected)
        assert core.match_template(pattern[0], expected[0])
        assert core.match_template(pattern[2], expected[2])
        assert pattern[1] == expected[1]
        assert pattern[3] == expected[3]

    def test_zeroormany(self):
        source = """
        x = 10
        {{...*}}
        z = 1
        {{name*}}
        """
        pattern = core.compile_template(source)
        expected = [
            ast.Assign(
                targets=[ast.Name(id='x')],
                value=ast.Constant(value=10, kind=None),
                type_comment=None,
            ),
            core.ZeroOrMany(object),
            ast.Assign(
                targets=[ast.Name(id='z')],
                value=ast.Constant(value=1, kind=None),
                type_comment=None,
            ),
            core.ZeroOrMany(core.Wildcard(name="name", template=object, common=False)),
        ]
        assert isinstance(pattern, list)
        assert len(pattern) == len(expected)
        assert core.match_template(pattern[0], expected[0])
        assert core.match_template(pattern[2], expected[2])
        assert pattern[1] == expected[1]

    def test_oneormany(self):
        source = """
        x = 10
        {{...+}}
        z = 1
        {{name+}}
        """
        pattern = core.compile_template(source)
        expected = [
            ast.Assign(
                targets=[ast.Name(id='x')],
                value=ast.Constant(value=10, kind=None),
                type_comment=None,
            ),
            core.OneOrMany(object),
            ast.Assign(
                targets=[ast.Name(id='z')],
                value=ast.Constant(value=1, kind=None),
                type_comment=None,
            ),
            core.OneOrMany(core.Wildcard(name="name", template=object, common=False)),
        ]
        assert isinstance(pattern, list)
        assert len(pattern) == len(expected)
        assert core.match_template(pattern[0], expected[0])
        assert core.match_template(pattern[2], expected[2])
        assert pattern[1] == expected[1]

    def test_zeroorone_list(self):
        source = "[1, 2, {{...?}}, 3, 4, {{...?}}]"
        pattern = core.compile_template(source)
        expected = ast.List(
            elts=[
                ast.Constant(value=1, kind=None),
                ast.Constant(value=2, kind=None),
                core.ZeroOrOne(object),
                ast.Constant(value=3, kind=None),
                ast.Constant(value=4, kind=None),
                core.ZeroOrOne(object),
            ]
        )
        assert isinstance(pattern, ast.List)

        assert len(pattern.elts) == len(expected.elts)
        assert core.match_template(pattern.elts[0], expected.elts[0])
        assert core.match_template(pattern.elts[1], expected.elts[1])
        assert pattern.elts[2] == expected.elts[2]
        assert core.match_template(pattern.elts[3], expected.elts[3])
        assert core.match_template(pattern.elts[4], expected.elts[4])
        assert pattern.elts[5] == expected.elts[5]

    def test_one_list(self):
        source = "[1, 2, {{...}}, 3, 4, {{...}}]"
        pattern = core.compile_template(source)
        expected = ast.List(
            elts=[
                ast.Constant(value=1, kind=None),
                ast.Constant(value=2, kind=None),
                core.Wildcard(name="Ellipsis_anything", template=object, common=False),
                ast.Constant(value=3, kind=None),
                ast.Constant(value=4, kind=None),
                core.Wildcard(name="Ellipsis_anything", template=object, common=False),
            ]
        )
        assert isinstance(pattern, ast.List)

        assert len(pattern.elts) == len(expected.elts)
        assert core.match_template(pattern.elts[0], expected.elts[0])
        assert core.match_template(pattern.elts[1], expected.elts[1])
        assert pattern.elts[2] == expected.elts[2]
        assert core.match_template(pattern.elts[3], expected.elts[3])
        assert core.match_template(pattern.elts[4], expected.elts[4])
        assert pattern.elts[5] == expected.elts[5]

    def test_zeroormany_list(self):
        source = "[1, 2, {{...*}}, 3, 4, {{...*}}]"
        pattern = core.compile_template(source)
        expected = ast.List(
            elts=[
                ast.Constant(value=1, kind=None),
                ast.Constant(value=2, kind=None),
                core.ZeroOrMany(object),
                ast.Constant(value=3, kind=None),
                ast.Constant(value=4, kind=None),
                core.ZeroOrMany(object),
            ]
        )
        assert isinstance(pattern, ast.List)

        assert len(pattern.elts) == len(expected.elts)
        assert core.match_template(pattern.elts[0], expected.elts[0])
        assert core.match_template(pattern.elts[1], expected.elts[1])
        assert pattern.elts[2] == expected.elts[2]
        assert core.match_template(pattern.elts[3], expected.elts[3])
        assert core.match_template(pattern.elts[4], expected.elts[4])
        assert pattern.elts[5] == expected.elts[5]

    def test_oneormany_list(self):
        source = "[1, 2, {{...+}}, 3, 4, {{...+}}]"
        pattern = core.compile_template(source)
        expected = ast.List(
            elts=[
                ast.Constant(value=1, kind=None),
                ast.Constant(value=2, kind=None),
                core.OneOrMany(object),
                ast.Constant(value=3, kind=None),
                ast.Constant(value=4, kind=None),
                core.OneOrMany(object),
            ]
        )
        assert isinstance(pattern, ast.List)

        assert len(pattern.elts) == len(expected.elts)
        assert core.match_template(pattern.elts[0], expected.elts[0])
        assert core.match_template(pattern.elts[1], expected.elts[1])
        assert pattern.elts[2] == expected.elts[2]
        assert core.match_template(pattern.elts[3], expected.elts[3])
        assert core.match_template(pattern.elts[4], expected.elts[4])
        assert pattern.elts[5] == expected.elts[5]

    def test_one_import_from(self):
        source = "from foo import {{...}}"
        pattern = core.compile_template(source)
        expected = ast.ImportFrom(
            module="foo",
            names=[
                ast.alias(
                    name=core.Wildcard(
                        name="Ellipsis_anything",
                        template=object,
                        common=False
                    ),
                    asname=None,
            )],
            level=0,
        )
        assert isinstance(pattern, ast.ImportFrom)
        assert pattern.module == expected.module
        assert len(pattern.names) == len(expected.names) == 1
        assert vars(pattern.names[0]) == vars(expected.names[0])

        source = "from foo import {{...}} as {{...}}"
        pattern = core.compile_template(source)
        expected = ast.ImportFrom(
            module="foo",
            names=[
                ast.alias(
                    name=core.Wildcard(
                        name="Ellipsis_anything",
                        template=object,
                        common=False
                    ),
                    asname=core.Wildcard(
                        name="Ellipsis_anything",
                        template=object,
                        common=False
                    ),
            )],
            level=0,
        )
        assert isinstance(pattern, ast.ImportFrom)
        assert pattern.module == expected.module
        assert len(pattern.names) == len(expected.names) == 1
        assert vars(pattern.names[0]) == vars(expected.names[0])

    def test_oneormany_import_from(self):
        source = "from foo import {{...+}}"
        pattern = core.compile_template(source)
        expected = ast.ImportFrom(
            module="foo",
            names=[
                core.OneOrMany(
                    ast.alias(
                        name=object,
                        asname=object,
            ))],
            level=0,
        )
        assert isinstance(pattern, ast.ImportFrom)
        assert pattern.module == expected.module
        assert len(pattern.names) == len(expected.names) == 1
        assert isinstance(pattern.names[0], core.OneOrMany)
        assert vars(pattern.names[0].template) == vars(expected.names[0].template)

        source = "from foo import {{name+}}"
        pattern = core.compile_template(source)
        expected = ast.ImportFrom(
            module="foo",
            names=[
                core.OneOrMany(
                    ast.alias(
                        name=core.Wildcard(
                            name="name",
                            template=object,
                            common=False
                        ),
                        asname=object,
            ))],
            level=0,
        )
        assert isinstance(pattern, ast.ImportFrom)
        assert pattern.module == expected.module
        assert len(pattern.names) == len(expected.names) == 1
        assert isinstance(pattern.names[0], core.OneOrMany)
        assert vars(pattern.names[0].template) == vars(expected.names[0].template)

    def runTest(self):
        self.test_function_alias()
        self.test_basic_code()
        self.test_wildcard_code()
        self.test_indented_code()
        self.test_multiline_code()
        self.test_zeroorone()
        self.test_one()
        self.test_zeroormany()
        self.test_oneormany()
        self.test_zeroorone_list()
        self.test_one_list()
        self.test_zeroormany_list()
        self.test_oneormany_list()
        self.test_one_import_from()


def main() -> int:
    # For use with ./tests/main.py, which looks for these main functions.
    # unittest.main() will do sys.exit() or something, it quits the whole
    # program after done and prevents further tests from running.

    # These tests are known to be broken on PyPy. Should maybe be fixed,
    # but I will not prioritize it.
    if platform.python_implementation() == "PyPy":
        return 0

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

    test_result = TestMatch().run()
    if not test_result.wasSuccessful():
        test_result.printErrors()
        return 1

    test_result = TestFullMatch().run()
    if not test_result.wasSuccessful():
        test_result.printErrors()
        return 1

    test_result = TestSub().run()
    if not test_result.wasSuccessful():
        test_result.printErrors()
        return 1

    test_result = TestCompile().run()
    if not test_result.wasSuccessful():
        test_result.printErrors()
        return 1

    return 0


if __name__ == "__main__":
    if platform.python_implementation() == "PyPy":
        print("Skipping tests for PyPy")
        sys.exit(0)

    unittest.main()
