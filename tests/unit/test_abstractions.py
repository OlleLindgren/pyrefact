#!/usr/bin/env python3


import sys
from pathlib import Path

from pyrefact import abstractions

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (
        (
            """
for x in range(11):
    print(x > 7)
    if x == 3:
        continue
    if x == 5:
        continue
    if x == 8:
        continue

    print(x)
    """,
            """
def _pyrefact_abstraction_1(x) -> bool:
    if x == 3:
        return False
    if x == 5:
        return False
    if x == 8:
        return False

    return True


for x in range(11):
    print(x > 7)
    if not _pyrefact_abstraction_1(x):
        continue

    print(x)
    """,
        ),
        (
            """
for x in range(11):
    print(x > 7)
    if x == 3:
        break
    if x == 5:
        break
    if x == 8:
        break

    print(x)
    """,
            """
def _pyrefact_abstraction_1(x) -> bool:
    if x == 3:
        return True
    if x == 5:
        return True
    if x == 8:
        return True

    return False


for x in range(11):
    print(x > 7)
    if _pyrefact_abstraction_1(x):
        break

    print(x)
    """,
        ),
        (
            """
for x in range(11):
    if x == 3:
        a = 12
    elif x == 5:
        a = x
    elif x == 8:
        a = sum((x, 2, 3))
    else:
        a = x + 1

    print(x)
    print(x, 1, 2)
    print(x, x, x)
    """,
            """
def _pyrefact_abstraction_1(x):
    if x == 3:
        return 12
    elif x == 5:
        return x
    elif x == 8:
        return sum((x, 2, 3))
    else:
        return x + 1


for x in range(11):
    a = _pyrefact_abstraction_1(x)

    print(x)
    print(x, 1, 2)
    print(x, x, x)
    """,
        ),
        (
            """
for var in range(11):
    print(22)
    params = {"password": 11, "username": 22}

    if var == 2:
        params["x"] = True
    elif var == 11 and s == 3:
        params["y"] = var, s
    else:
        params["xxx"] = 3

    if foo:
        params["is_foo"] = True

    response = requests.get(url, params)
    assert response.status_code == 200, "got a non-200 response"
        """,
            """
def _pyrefact_abstraction_1(foo, s, var):
    params = {'password': 11, 'username': 22}

    if var == 2:
        params['x'] = True
    elif var == 11 and s == 3:
        params['y'] = (var, s)
    else:
        params['xxx'] = 3

    if foo:
        params['is_foo'] = True

    return params


for var in range(11):
    print(22)

    params = _pyrefact_abstraction_1(foo, s, var)
    response = requests.get(url, params)
    assert response.status_code == 200, "got a non-200 response"
        """,
        ),
        (
            """
def f(x):
    '''This is a docstring'''
    if x == 3:
        y = AAA
    elif x > 9:
        y = 3
    else:
        if x >= 2:
            y = 5
        else:
            y = 8
    return y
        """,
            """
def f(x):
    '''This is a docstring'''
    if x == 3:
        y = AAA
    elif x > 9:
        y = 3
    else:
        if x >= 2:
            y = 5
        else:
            y = 8
    return y
        """,
        ),
        (
            """
AAA = 3
def f(x):
    '''This is a docstring'''
    for x in range(13):
        print(x > 5)
        if x == 3:
            y = AAA
        elif x > 9:
            y = 3
        else:
            if x >= 2:
                y = 5
            else:
                y = 8
        yield y
        """,
            """
AAA = 3
def _pyrefact_abstraction_1(x):
    if x == 3:
        return AAA
    elif x > 9:
        return 3
    elif x >= 2:
        return 5
    else:
        return 8
def f(x):
    '''This is a docstring'''
    for x in range(13):
        print(x > 5)
        yield _pyrefact_abstraction_1(x)
        """,
        ),
        (
            """
x = 9

if x == 3:
    y = 2
elif x == 2:
    y = 3
else:
    y = 99

print(x, y)

x = 2

if x == 3:
    y = 2
elif x == 2:
    y = 3
else:
    y = 99

print(x, y)



q = 8

if q == 3:
    z = 2
elif q == 2:
    z = 3
else:
    z = 99

print(q, z)
        """,
            """
def _pyrefact_abstraction_1(x):
    if x == 3:
        return 2
    elif x == 2:
        return 3
    else:
        return 99

def _pyrefact_abstraction_2(x):
    if x == 3:
        return 2
    elif x == 2:
        return 3
    else:
        return 99

def _pyrefact_abstraction_3(q):
    if q == 3:
        return 2
    elif q == 2:
        return 3
    else:
        return 99


x = 9

y = _pyrefact_abstraction_1(x)

print(x, y)


x = 2

y = _pyrefact_abstraction_2(x)

print(x, y)


q = 8

z = _pyrefact_abstraction_3(q)

print(q, z)

        """,  # Duplicate functions should be removed, but not by create_abstractions().
        ),
    )

    for content, expected_abstraction in test_cases:

        processed_content = abstractions.create_abstractions(content)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
