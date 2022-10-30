import sys

from pyrefact.fixes import delete_unused_functions_and_classes

CODE = """

def function() -> int:
    return 0

def _private_function() -> bool:
    return 11

def _used_function() -> str:
    '''Docstring mentioning _private_function()'''
    return "this function is used"

def _user_of_used_function() -> str:
    return _used_function()

if __name__ == "__main__":
    _user_of_used_function()

"""


EXPECTED = """

def _used_function() -> str:
    '''Docstring mentioning _private_function()'''
    return "this function is used"

def _user_of_used_function() -> str:
    return _used_function()

if __name__ == "__main__":
    _user_of_used_function()

"""


def main() -> int:

    got = delete_unused_functions_and_classes(CODE)
    assert got.strip() == EXPECTED.strip(), "\n".join(
        ("Wrong result: (got, expected)", got, EXPECTED)
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
