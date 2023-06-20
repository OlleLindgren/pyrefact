"""Integration test cases, used in test_format_code and test_format_file"""

import testing_infra

INTEGRATION_TEST_CASES = (
    ("", ""),
    (
        """
from os import *
from typing import *
def foo() -> Tuple[int, Sequence[int]]:
    for x in range(100):
        if x > 10:
            y = 13
        else:
            x += 1
            x *= 12
            print(x > 30)
            y = 100 - sum(x, 2, 3)
            print(getcwd())
    return 1, (1, 2, 3)
    print(x)

if __name__ == "__main__":
    foo()
        """,
        """
from os import getcwd
from typing import Sequence, Tuple
def _foo() -> Tuple[int, Sequence[int]]:
    for x in range(100):
        if x <= 10:
            x += 1
            x *= 12
            print(x > 30)
            print(getcwd())
    return 1, (1, 2, 3)

if __name__ == "__main__":
    _foo()
        """,
    ),
    (
        """
x = 100
z = list()
def foo() -> bool:
    if x <= 1:
        if x < -500:
            if x < -600:
                return 0
            elif x >= -550:
                return -2
            elif x % 3 == 0:
                if x % 2 == 0:
                    if x % 5 == 0:
                        if x % 61 == 0:
                            if x % (x - 4) == 0:
                                return 61
                    return 21
                return 33
            elif x % 7 == 0:
                return -33
            elif x % 18 == 0:
                if x % 66 == 0:
                    return -8
                return 3
            return 222
        elif x > -400:
            return 3
        elif x > -300 and x % 2 == 0:
            if x % 3 == 0:
                return -3
            else:
                return -2
        return 2
    elif x == 2:
        return 3
    else:
        # if x > 0:
        #     return 100 * x < 1000
        if z:
            return -3
        else:
            return 2
print(foo())
        """,
        """
X = 100
Z = []
def _foo() -> bool:
    if X > 1:
        if X == 2:
            return 3
        if Z:
            return -3
        return 2
    if X >= -500:
        if X > -400:
            return 3
        if X <= -300 or X % 2 != 0:
            return 2
        if X % 3 == 0:
            return -3
        return -2
    if X < -600:
        return 0
    if X >= -550:
        return -2
    if X % 3 == 0:
        if X % 2 != 0:
            return 33
        if X % 5 == 0:
            if X % 61 == 0:
                if X % (X - 4) == 0:
                    return 61
        return 21
    if X % 7 == 0:
        return -33
    if X % 18 != 0:
        return 222
    if X % 66 == 0:
        return -8
    return 3
print(_foo())
        """,
    ),
    (
        """
x = sorted(list(range(100)))[::3]
z = []
for a in x:
    if a % 3 == 0 and a % 4 == 2:
        for w in x:
            if w > len(x) // 2:
                z.append(w ** 3)
if all(y in [1, 2, 5] for y in sorted(set(list(z)))):
    print(z, x)
        """,
        """
X = sorted(range(100))[::3]
Z = [w ** 3 for a in X if a % 3 == 0 and a % 4 == 2 for w in X if w > len(X) // 2]
if all(y in {1, 2, 5} for y in sorted(set(Z))):
    print(Z, X)
        """,
    ),
    testing_infra.ignore_on_version(3, 8)(
        """
x = sorted(list(range(100)))[::3]
z = {ww: ww + 1 for ww in range(12)}
z.update({22: 33, 99: 88, 1: 23232})
for a in x:
    if a % 3 == 0 and a % 4 == 2:
        for w in x:
            if w > len(x) // 2:
                z[(w ** 3)] = w ** 2
for a in x:
    if a % 5 == 0 and a % 9 == 2:
        z[(w ** -1)] = w ** -2
z[1] = 333
z[1] = 333
if all(y in [1, 2, 5] for y in set(sorted(list(z)))):
    print(z, x)
        """,
        """
X = sorted(range(100))[::3]
Z = {
    **{ww: ww + 1 for ww in range(12)},
    22: 33,
    99: 88,
    **{w**3: w**2 for a in X if a % 3 == 0 and a % 4 == 2 for w in X if w > len(X) // 2},
    **{w ** (-1): w ** (-2) for a in X if a % 5 == 0 and a % 9 == 2},
    1: 333,
}
if all(y in {1, 2, 5} for y in set(Z)):
    print(Z, X)
        """,
    ),
    (
        """
z = {a for a in range(10)}
x = sum(z)
print(x)
        """,
        """
X = sum(set(range(10)))
print(X)
        """,
    ),
    (
        """
q = 3
w = list()
for a in range(-1, 10):
    for k in range(-1, 1):
        w.append(a ** 2 + q + k ** 2)
y = sum(w)
print(y)
            """,
        """
Q = 3
Y = 22 * Q + 583
print(Y)
        """,
    ),
    (
        """
x = 0
a = 1
for i in range(100):
    x += i**2
    a -= i**3

print(x)
            """,
        """
X = 328350
print(X)
        """,
    ),
    (
        """
x = 0
a = 1
for i in range(100):
    x += i**2
    a -= i**3

print(x, a)
            """,
        """
X = 0
A = 1
for i in range(100):
    X += i**2
    A -= i**3

print(X, A)
        """,
    ),
    (
        """
x = list()
for i in range(100):
    x += i**2

print(x)
            """,
        """
X = [i ** 2 for i in range(100)]
print(X)
        """,
    ),
    (
        """
import numpy as np

i, j, k = 10, 11, 12

a = np.random.random((i, j))
b = np.random.random((j, k))

u = np.array(
    [
        [
            np.sum(
                a__ * b__
                for a__, b__ in zip(a_, b_)
            )
            for a_ in a
        ]
        for b_ in b.T
    ]
).T

print(u)
            """,
        """
import numpy as np

I, J, K = 10, 11, 12

A = np.random.random((I, J))
B = np.random.random((J, K))

U = np.matmul(A, B)

print(U)
        """,
    ),
    (
        """
import numpy as np

i, j, k = 10, 11, 12

a = np.random.random((i, j))
b = np.random.random((j, k))

u = np.array(
    [
        [
            np.dot(b[:, i], a[j, :])
            for i in range(b.shape[1])
        ]
        for j in range(a.shape[0])
    ]
)

print(u)
            """,
        """
import numpy as np

I, J, K = 10, 11, 12

A = np.random.random((I, J))
B = np.random.random((J, K))

U = np.matmul(A, B)

print(U)
        """,
    ),
    (
        """
a = np.random.random(100)
b = np.random.random(100)

u = 0
for a_, b_ in zip(a, b):
    u += a_ * b_

print(u)
            """,
        """
import numpy as np

A = np.random.random(100)
B = np.random.random(100)

U = np.dot(A, B)

print(U)
        """,
    ),
    (
        """
import pandas as pd
df = pd.DataFrame([{"foo": 10 + i, "bar": 10 - i} for i in list(range(10))])
x = 0
for _, row in df.iterrows():
    x += row.iloc[1] + row.loc["foo"] - row["bar"]
for ix, _ in df.iterrows():
    x += ix
print(x)
            """,
        """
import pandas as pd
DF = pd.DataFrame([{"foo": 10 + i, "bar": 10 - i} for i in range(10)])
X = sum((row[1 + 1] + row.foo - row.bar for row in DF.itertuples())) + sum(DF.index)
print(X)
        """,
    ),
    (
        """
    a = np.random.random(100)
    b = np.random.random(100)

    u = 0
    for a_, b_ in zip(a, b):
        u += a_ * b_

    print(u)
            """,
        """
    a = np.random.random(100)
    b = np.random.random(100)

    u = np.dot(a, b)

    print(u)
        """,
    ),
    (
        """
def f(x) -> int:
    if x == 1:
        x = 2
    else:
        if x == 3:
            x = 7
        else:
            if x != 8:
                if x >= 912:
                    x = -2
                elif x ** x > x ** 3:
                    x = -1
                else:
                    x = 14
            else:
                x = 99

    return x

print(f(11))
            """,
        """
def _f(x) -> int:
    if x == 1:
        return 2
    if x == 3:
        return 7
    if x == 8:
        return 99
    if x >= 912:
        return -2
    if x ** x > x ** 3:
        return -1

    return 14

print(_f(11))
            """,
    ),
    (
        """
import random
import sys
def f(x: int) -> int:
    import heapq
    y = e = 112
    if x >= 2:
        d = 12
    if []:
        x *= 99
    if x == 3:
        y = x ** 13
        return 8
    else:
        return 8
print(f(12))
while False:
    sys.exit(0)
            """,
        """
def _f(x: int) -> int:
    return 8
print(_f(12))
            """,
    ),
    (
        """
class Foo:
    def asdf(self):
        x = None
        if 2 in {1, 2, 3}:
            print(3)
def wsdf():
    z = ()
    if 2 in {1, 2, 3}:
        print(3)
wsdf()
Foo().asdf()
Foo.asdf()
        """,
        """
def _asdf():
    print(3)
_asdf()
_asdf()
_asdf()
        """,
    ),
    (
        """
e = [
    x**2
    for x in (x for (_, x) in zip([i for (_, i) in enumerate(range(102, 202))], range(100)))
    if x > 7
]
print(e)
        """,
        """
E = [x ** 2 for x in range(8, 100)]
print(E)
        """,
    ),
    (
        """
def foo():
    e = sum([
        x**2
        for x in (x for (_, x) in zip([i for (_, i) in enumerate(range(102, 202))], range(100)))
    ])
    return e

print(foo())
        """,
        """
def _foo():
    return 328350

print(_foo())
        """,
    ),
    (
        """
import numpy  ;      import re; import heapq;
d = {"x": 100, "y": 10}
for x in d.keys():
    print(d[x])
    print(re.findall(str(d[x]), "asdf"))
        """,
        """
import re
D = {"x": 100, "y": 10}
for d_x in D.values():
    print(d_x)
    print(re.findall(str(d_x), "asdf"))
        """,
    ),
    (
        """
x = [a for a in range(10) if a % 2 == 0 and a > 5 and a % 2 == 0]
print(sum(x))
        """,
        """
print(sum([a for a in range(6, 10) if a % 2 == 0]))
        """,
    ),
)
