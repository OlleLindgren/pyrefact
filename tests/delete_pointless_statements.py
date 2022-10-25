import sys

from pyrefact.fixes import delete_pointless_statements

CODE = """

12





variable = "asdf"

var2 = 'asdfasdf'

var3 = '''
this should not be deleted'''


def f():
    '''
    This is a docstring and should be left alone
    '''


"""


EXPECTED = """

12





variable = "asdf"

var2 = 'asdfasdf'

var3 = '''
this should not be deleted'''


def f():
    '''
    This is a docstring and should be left alone
    '''


"""


SHEBANG = """#!/usr/bin/env python3


"""


REGULAR_MODULE_DOCSTRING = '''"""A normal module docstring"""

def f() -> int:
    return 0

import sys
if __name__ == "__main__":
    sys.exit(f())
'''


def main() -> int:

    got = delete_pointless_statements(CODE)
    assert got == EXPECTED, "\n".join(("Wrong result: (got, expected)", got, EXPECTED))
    assert SHEBANG == delete_pointless_statements(SHEBANG)
    assert REGULAR_MODULE_DOCSTRING == delete_pointless_statements(REGULAR_MODULE_DOCSTRING)

    return 0


if __name__ == "__main__":
    sys.exit(main())
