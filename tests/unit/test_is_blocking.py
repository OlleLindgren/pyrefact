import ast
import sys

from pyrefact import parsing


def main() -> int:
    """Test parsing.is_blocking

    Returns:
        int: 1 if the function behaves incorrectly, otherwise 0
    """
    for source in (
        """
for i in range(10):
    continue
""",
        """
for i in range(10):
    break
""",
        """
for i in range(10):
    print(1)
    print(2)
    with x as y:
        continue
""",
        """
for i in x:
    while True:
        break
""",
        """
for i in x:
    while True:
        break
    continue
""",
        """
while True:
    break
    raise Exception()
""",
        """
while statement:
    if x:
        statement = False
    if random.random():
        statement = False
""",
        """
while False:
    print(2)
    raise RuntimeError()
""",
        """
if 66:
    pass
""",
        """
if 66:
    print(6)
else:
    return 22
""",
        """
if None:
    break
else:
    f = 2 + x()
""",
        """
for i in []:
    raise ValueError()
""",
        """
for i in something:
    raise ValueError()
""",
    ):
        node = parsing.parse(source).body[0]
        if not parsing.is_blocking(node):
            continue

        print("Ast is blocking, but should not be:")
        print(source)
        print("Ast structure:")
        print(ast.dump(node, indent=2))
        return 1

    for source in (
        """
for i in [1, 2, 3]:
    raise ValueError()
""",
        """
for i in (1, 2, 3):
    print(1)
    print(2)
    with x as y:
        raise RuntimeError()
""",
        """
for i in [None, False]:
    while True:
        break
    assert False
""",
        """
if x:
    raise RuntimeError()
elif y:
    if z:
        for a in range(10):
            continue
        return 1
    break
else:
    print(2)
    return 99
""",
        """
while True:
    while True:
        while True:
            print(3)
""",
        """
while True:
    while True:
        while True:
            raise Exception()
""",
        """
while 1:
    while False:
        pass
""",
        """
if 66:
    return 0
""",
        """
if 66:
    return 0
else:
    print(8)
""",
        """
if None:
    pass
else:
    return 0
""",
    ):
        node = parsing.parse(source).body[0]
        if parsing.is_blocking(node):
            continue

        print("Ast is not blocking, but should be:")
        print(source)
        print("Ast structure:")
        print(ast.dump(node, indent=2))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
