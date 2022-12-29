import sys

from pyrefact.fixes import undefine_unused_variables

CODE = """
def some_python_code() -> None:
    x = 3

    a, asdf, *args, _, _, q = source_of_stuff()

    do_stuff_with(args)

    a, asdf, *args2, _, _, q = source_of_stuff()

    t, _, k, _ = source_of_stuff()

    print(t + k)

    return 0

FUBAR = 420
_FUBAR_PRIVATE = 69

"""


EXPECTED_FORMATTING = """
def some_python_code() -> None:
    3

    _, _, *args, _, _, _ = source_of_stuff()

    do_stuff_with(args)

    source_of_stuff()

    t, _, k, _ = source_of_stuff()

    print(t + k)

    return 0

420
69

"""


def main() -> int:

    got = undefine_unused_variables(CODE)
    assert got == EXPECTED_FORMATTING, "\n".join(
        ("Wrong result: (got, expected)", got, EXPECTED_FORMATTING)
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
