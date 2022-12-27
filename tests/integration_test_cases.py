"""Integration test cases, used in test_format_code and test_format_file"""

INTEGRATION_TEST_CASES = (
    ("", ""),
    (
        """
for x in range(100):
    if x > 10:
        y = 13
    else:
        x += 1
        x *= 12
        print(x > 30)
        y = 100 - sum(x, 2, 3)
print(x)
        """,
        """
for x in range(100):
    if x <= 10:
        x += 1
        x *= 12
        print(x > 30)
print(x)
        """,
    ),
    (
        """
x = 100
z = list()
def foo() -> bool:
    if x == 1:
        return False
    elif x == 2:
        return True
    else:
        if z:
            return False
        else:
            return True
print(foo())
        """,
        """
X = 100
Z = []
def _foo() -> bool:
    if X == 1:
        return False
    if X == 2:
        return True
    if Z:
        return False

    return True
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
Z = [w**3 for a in X if a % 3 == 0 and a % 4 == 2 for w in X if w > len(X) // 2]
if all(y in {1, 2, 5} for y in sorted(set(Z))):
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
for i in range(100):
    x += i**2

print(x)
            """,
        """
X = 0
for i in range(100):
    X += i**2

print(X)
        """,
    ),
)