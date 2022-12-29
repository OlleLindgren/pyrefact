import sys

from pyrefact.fixes import delete_pointless_statements

CODE = """12

def pointless_function() -> None:
    555


1234
{1: sum((2, 3, 6, 0)), "asdf": 13-12}

"""


EXPECTED = """

def pointless_function() -> None:
    pass

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
    assert got.strip() == EXPECTED.strip(), "\n".join(
        ("Wrong result: (got, expected)", got, "\n***\n", EXPECTED)
    )
    assert SHEBANG == delete_pointless_statements(SHEBANG)
    assert REGULAR_MODULE_DOCSTRING == delete_pointless_statements(REGULAR_MODULE_DOCSTRING)

    return 0


if __name__ == "__main__":
    sys.exit(main())
