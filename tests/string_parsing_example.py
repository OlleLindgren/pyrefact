#!/usr/bin/env python3

import sys

from pyrefact.parsing import get_is_code_mask

_TEST_STRINGS = {
    '''f"{_VARIABLE + ''}' is part of an f string"''': "___xxxxxxxxxxxx____________________________",
    '''f"""{_VARIABLE + ''}' is part of an f string"""''': "_____xxxxxxxxxxxx______________________________",
    """f'{_VARIABLE + ""}" is part of an f string'""": "___xxxxxxxxxxxx____________________________",
    """f'''{_VARIABLE + ""}" is part of an f string'''""": "_____xxxxxxxxxxxx______________________________",
    '''f"""this {qwerty + f'is a nested {"f string" + some_variable + "1" + auto()}, which is'}also valid."""''': "__________xxxxxxxxx_________________________xxxxxxxxxxxxxxxxxxx___xxxxxxxxx___________________________",
    '''f"""
This is a very nasty{"1" + f'{ asdf }'}{{ unbalanced thing
    """''': "_____________________________xxx___xxxxxx______________________________",
}


def main():
    for f_string, expected in _TEST_STRINGS.items():
        actual = "".join("x" if mask else "_" for mask in get_is_code_mask(f_string))
        if actual != expected:
            print("Wrong result. (code, expected, actual):")
            print(f_string)
            print(expected)
            print(actual)


if __name__ == "__main__":
    sys.exit(main())
