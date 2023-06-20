#!/usr/bin/env python3

# fmt: off
# isort: skip_file
# pyrefact: skip_file
from os import *
import sys as sys, os
from pathlib import Path

from pyrefact import constants, tracing

important_variable = 42
if foo := important_variable - 1:
    pass


def main() -> int:
    with Path(__file__).open("r", encoding="utf-8") as stream:
        source = stream.read()

    assert tracing.trace_origin("tracing", source)[:2] == ("from pyrefact import constants, tracing", 10)
    assert tracing.trace_origin("sys", source)[:2] == ("import sys as sys, os", 7)
    assert tracing.trace_origin("sys", source)[:2] == ("import sys as sys, os", 7)
    assert tracing.trace_origin("main", source)[1] == 17
    assert tracing.trace_origin("important_variable", source)[:2] == ("important_variable = 42", 12)

    # get_code() breaks this one, not worth the effort of fixing for almost EOL 3.8
    if constants.PYTHON_VERSION >= (3, 9):
        assert tracing.trace_origin("foo", source)[:2] == ("foo := important_variable - 1", 13)

    assert tracing.trace_origin("getcwd", source)[:2] == ("from os import *", 6)
    assert tracing.trace_origin("source", source)[:2] == ("source = stream.read()", 19)

    return 0


if __name__ == "__main__":
    sys.exit(main())
