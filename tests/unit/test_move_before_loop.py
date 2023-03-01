#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (
        (  # constant expression that depends on nothing in loop
            """
for i in range(100):
    x = 10
    print(x + i)
            """,
            """
x = 10
for i in range(100):
    print(x + i)
            """,
        ),
        (  # constant expressions that depend on loop
            """
x = 10
for i in range(100):
    y = 100 - x
    x = 22
    print(x + i - y)
            """,
            """
x = 10
for i in range(100):
    y = 100 - x
    x = 22
    print(x + i - y)
            """,
        ),
        (  # multiple constant expressions that do not depend on loop
            """
for i in range(100):
    print(100)
    x = 26
    print(1 + i + 10)
    x = 11
    x = 20
    print(x > 1 > 10)
    print(x + i)
            """,
            """
x = 26
x = 11
x = 20
for i in range(100):
    print(100)
    print(1 + i + 10)
    print(x > 1 > 10)
    print(x + i)
            """,
        ),
        (  # more complicated constant expressions, some of which depend on loop
            """
for i in range(100):
    print(100)
    x = 26
    print(1 + i + 10)
    x = 11
    if i > 10:
        x = 2
    x = 20
    print(x > 1 > 10)
    print(x + i)
            """,
            """
x = 26
x = 11
for i in range(100):
    print(100)
    print(1 + i + 10)
    if i > 10:
        x = 2
    x = 20
    print(x > 1 > 10)
    print(x + i)
            """,
        ),
        (  # AugAssign should not be moved
            """
x = 0
for i in range(100):
    x += 1
    print(x + i)
            """,
            """
x = 0
for i in range(100):
    x += 1
    print(x + i)
            """,
        ),
        (  # AnnAssign can be moved
            """
x = 0
for i in range(100):
    x: int = 1
    print(x + i)
            """,
            """
x = 0
x: int = 1
for i in range(100):
    print(x + i)
            """,
        ),
        (  # Multi-assign can be moved if independent
            """
for i in range(100):
    x, y = 10, 22
    print(x, y)
            """,
            """
x, y = 10, 22
for i in range(100):
    print(x, y)
            """,
        ),
        (  # Multi-assign can not be moved if not independent
            """
for i in range(100):
    x, y, z = 10, 22, i
    print(x, y, z)
            """,
            """
for i in range(100):
    x, y, z = 10, 22, i
    print(x, y, z)
            """,
        ),
        (  # x depends on the for loop's iter
            """
for i in range(100):
    x = i
    print(x + i)
            """,
            """
for i in range(100):
    x = i
    print(x + i)
            """,
        ),
        (  # x depends on the while loop's test
            """
while (x := 10) > 2:
    print(x + i)
            """,
            """
while (x := 10) > 2:
    print(x + i)
            """,
        ),
        (  # the while loop's test depends on x
            """
x = 0
i = 0
while x < 10:
    x = 100
    print(x)
            """,
            """
x = 0
i = 0
while x < 10:
    x = 100
    print(x)
            """,
        ),
        (  # the while loop's test depends on x
            """
while parsing.match_template(w, templates):
    w = w.args[0].args[0]
            """,
            """
while parsing.match_template(w, templates):
    w = w.args[0].args[0]
            """,
        ),
        (
            """
for a in b:
    variable = True
    for c in d:
        try:
            do_stuff(c, d)
        except ValueError:
            continue

        variable = False

    if_orelse_template = ast.If(body=[object], orelse=[object])
    for condition in parsing.walk(ast.Module(body=n2.body), if_orelse_template):
        if condition in loop_replacements:
            continue

        try:
            h, q, z = foo()
        except ValueError:
            continue

        subscript_calls.add(t_call)
        if r (a, b, c, d):
            if isinstance(f_value, ast.List) == (t_call == "append"):
                loop_replacements[condition] = on_true
                continue
            variable = False
            break

    if not variable:
        continue

    print(1)
            """,
            """
for a in b:
    variable = True
    for c in d:
        try:
            do_stuff(c, d)
        except ValueError:
            continue

        variable = False

    if_orelse_template = ast.If(body=[object], orelse=[object])
    for condition in parsing.walk(ast.Module(body=n2.body), if_orelse_template):
        if condition in loop_replacements:
            continue

        try:
            h, q, z = foo()
        except ValueError:
            continue

        subscript_calls.add(t_call)
        if r (a, b, c, d):
            if isinstance(f_value, ast.List) == (t_call == "append"):
                loop_replacements[condition] = on_true
                continue
            variable = False
            break

    if not variable:
        continue

    print(1)
            """,
        ),
        (
            """
for user_id in iter_user_logins():
    date = '2022-03-01'
    start_time = '13:08:19'
    end_time = '19:01:31'
    start = f"{date}T{start_time}"
    end = f"{date}T{end_time}"
    cursor.execute(
        '''
        INSERT INTO wonderland(start, end, user_id) VALUES (%s, %s, %s)
        '''
        [start, end, user_id],
    )
            """,
            """
date = '2022-03-01'
start_time = '13:08:19'
end_time = '19:01:31'
start = f'{date}T{start_time}'
end = f'{date}T{end_time}'
for user_id in iter_user_logins():
    cursor.execute(
        '''
        INSERT INTO wonderland(start, end, user_id) VALUES (%s, %s, %s)
        '''
        [start, end, user_id],
    )
            """,
        ),
        (
            """
for i in range(10):
    x = (1, 2, 3)
    for y in x:
        print(y)
            """,
            """
x = (1, 2, 3)
for i in range(10):
    for y in x:
        print(y)
            """,
        ),
        (
            """
for i in range(10):
    x = {1: 2}
    for y in x:
        w = {**x, 1 + y: 3 - i}
    w.update(x)
    print(w)
            """,
            """
x = {1: 2}
for i in range(10):
    for y in x:
        w = {**x, 1 + y: 3 - i}
    w.update(x)
    print(w)
            """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.move_before_loop(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
