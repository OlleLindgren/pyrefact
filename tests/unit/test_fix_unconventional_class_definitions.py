#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import object_oriented

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (
        (
        """
class Foo:
    pass

Foo.x = 1
    """,
        """
class Foo:
    pass
    x = 1
    """,
        ),
        (
        """
class Foo:
    pass

Foo.x = (1, 2, 3)
    """,
        """
class Foo:
    pass
    x = (1, 2, 3)
    """,
        ),
        (
        """
class Foo:
    pass

Bar.x = 1
    """,
        """
class Foo:
    pass

Bar.x = 1
    """,
        ),
        (
        """
class Foo(object):
    pass

Foo.x = 1
    """,
        """
class Foo(object):
    pass
    x = 1
    """,
        ),
        (
        """
@a
@bunch
@of
@decorators
class Foo(object, list, set, tuple, []):
    pass

Foo.x = 1
    """,
        """
@a
@bunch
@of
@decorators
class Foo(object, list, set, tuple, []):
    pass
    x = 1
    """,
        ),
        (
        """
class Foo:
    pass

Foo.x = 1
Foo.y = z
Foo.z = func()
    """,
        """
class Foo:
    pass
    x = 1
    y = z
    z = func()
    """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = object_oriented.fix_unconventional_class_definitions(source)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
