#!/usr/bin/env python3
import ast
import sys
import unittest
from pathlib import Path

from pyrefact import core, tracing

tracing_test_files_path = Path(__file__).parent / "tracing_test_files"
a_py = tracing_test_files_path / "a.py"
b_py = tracing_test_files_path / "b.py"
c_py = tracing_test_files_path / "c.py"
d_py = tracing_test_files_path / "d.py"
e_py = tracing_test_files_path / "e.py"

sys.path.append(str(tracing_test_files_path))


class TestTraceImports(unittest.TestCase):
    @staticmethod
    def test_basic_trace():
        result = tracing.trace_origin("sys", d_py.read_text())
        assert isinstance(result, tracing._TraceResult)
        assert result.source == "import sys"
        assert result.lineno == 6
        assert core.match_template(result.ast, ast.Import(names=[ast.alias(name="sys", asname=None)]))

        result = tracing.trace_origin("x", d_py.read_text())
        assert isinstance(result, tracing._TraceResult)
        assert result.source == "x = 1"
        assert result.lineno == 3
        assert core.match_template(result.ast, ast.Assign(targets=[ast.Name(id="x")], value=ast.Constant(value=1)))

        result = tracing.trace_origin("y", d_py.read_text())
        assert isinstance(result, tracing._TraceResult)
        assert result.source == "y = 100"
        assert result.lineno == 4
        assert core.match_template(result.ast, ast.Assign(targets=[ast.Name(id="y")], value=ast.Constant(value=100)))

    @staticmethod
    def test_cross_file_trace():
        result = tracing.trace_origin("x", c_py.read_text())
        assert isinstance(result, tracing._TraceResult)
        assert result.source == "from d import x, y as z"
        assert result.lineno == 3
        assert core.match_template(result.ast, ast.ImportFrom(module="d", names=[ast.alias(name="x", asname=None), ast.alias(name="y", asname="z")], level=0))

        traced_source_file = tracing._trace_module_source_file(result.ast.module)
        assert traced_source_file == str(d_py)

        result = tracing.trace_origin("e", d_py.read_text())
        assert isinstance(result, tracing._TraceResult)
        assert result.source == "import e"
        assert result.lineno == 8
        assert core.match_template(result.ast, ast.Import(names=[ast.alias(name="e", asname=None)]))

        traced_source_file = tracing._trace_module_source_file("e")
        assert traced_source_file == str(e_py)

        result = tracing.trace_origin("hh", a_py.read_text())
        assert isinstance(result, tracing._TraceResult)
        assert result.source == "from e import hh"
        assert result.lineno == 6
        assert core.match_template(result.ast, ast.ImportFrom(module="e", names=[ast.alias(name="hh", asname=None)], level=0))

        traced_source_file = tracing._trace_module_source_file("e")
        assert traced_source_file == str(e_py)

    @staticmethod
    def test_nested_cross_file_trace():
        result = tracing.trace_origin("k", a_py.read_text())
        assert isinstance(result, tracing._TraceResult)
        assert result.source == "from b import x as k"
        assert result.lineno == 4
        assert core.match_template(result.ast, ast.ImportFrom(module="b", names=[ast.alias(name="x", asname="k")], level=0))

        traced_source_file = tracing._trace_module_source_file(result.ast.module)
        assert traced_source_file == str(b_py)

        result = tracing.trace_origin("x", b_py.read_text())
        assert isinstance(result, tracing._TraceResult)
        assert result.source == "from c import *"
        assert result.lineno == 3
        assert core.match_template(result.ast, ast.ImportFrom(module="c", names=[ast.alias(name="*", asname=None)], level=0))

        traced_source_file = tracing._trace_module_source_file(result.ast.module)
        assert traced_source_file == str(c_py)

        result = tracing.trace_origin("ww", b_py.read_text())
        assert isinstance(result, tracing._TraceResult)
        assert result.source == "from e import *"
        assert result.lineno == 4
        assert core.match_template(result.ast, ast.ImportFrom(module="e", names=[ast.alias(name="*", asname=None)], level=0))

        traced_source_file = tracing._trace_module_source_file(result.ast.module)
        assert traced_source_file == str(e_py)

        result = tracing.trace_origin("hh", b_py.read_text())
        assert isinstance(result, tracing._TraceResult)
        assert result.source == "from e import *"
        assert result.lineno == 4
        assert core.match_template(result.ast, ast.ImportFrom(module="e", names=[ast.alias(name="*", asname=None)], level=0))

        traced_source_file = tracing._trace_module_source_file(result.ast.module)
        assert traced_source_file == str(e_py)

        result = tracing.trace_origin("hh", e_py.read_text())
        assert isinstance(result, tracing._TraceResult)
        assert result.source == "hh = aabb"
        assert result.lineno == 5
        assert core.match_template(result.ast, ast.Assign(targets=[ast.Name(id="hh")], value=ast.Name(id="aabb")))

        result = tracing.trace_origin("x", c_py.read_text())
        assert isinstance(result, tracing._TraceResult)
        assert result.source == "from d import x, y as z"
        assert result.lineno == 3
        assert core.match_template(result.ast, ast.ImportFrom(module="d", names=[ast.alias(name="x", asname=None), ast.alias(name="y", asname="z")], level=0))

        traced_source_file = tracing._trace_module_source_file(result.ast.module)
        assert traced_source_file == str(d_py)

    def runTest(self):
        self.test_basic_trace()
        self.test_cross_file_trace()
        self.test_nested_cross_file_trace()


def main() -> int:
    test_result = TestTraceImports().run()

    if not test_result.wasSuccessful():
        print("FAILED")
        return 1

    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
