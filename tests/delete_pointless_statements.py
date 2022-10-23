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


def main() -> int:

    got = delete_pointless_statements(CODE)
    assert got == EXPECTED, "\n".join(("Wrong result: (got, expected)", got, EXPECTED))

    return 0


if __name__ == "__main__":
    sys.exit(main())
